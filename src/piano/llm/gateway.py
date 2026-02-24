"""LLM Gateway for centralized request management.

Provides queue-based request processing with priority, concurrency limiting,
per-agent and global rate limiting, request deduplication, circuit breaker,
and cost aggregation.

Reference: docs/implementation/02-llm-integration.md Section 1.5
"""

from __future__ import annotations

__all__ = [
    "CircuitBreaker",
    "LLMGateway",
    "QueuedRequest",
    "RequestPriority",
]

import asyncio
import hashlib
import json
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from enum import IntEnum
from typing import TYPE_CHECKING
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from piano.core.types import LLMRequest, LLMResponse  # noqa: TC001
from piano.llm.provider import RateLimitExceededError

if TYPE_CHECKING:
    from piano.llm.provider import LLMProvider

logger = structlog.get_logger(__name__)


class RequestPriority(IntEnum):
    """Priority levels for LLM requests (lower value = higher priority)."""

    HIGH = 1
    NORMAL = 2
    LOW = 3


class QueuedRequest(BaseModel):
    """A queued LLM request with priority and metadata."""

    request: LLMRequest = Field(..., description="The LLM request to execute")
    priority: RequestPriority = Field(
        default=RequestPriority.NORMAL, description="Request priority"
    )
    agent_id: str = Field(..., description="ID of the agent making the request")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    future_id: str = Field(
        default_factory=lambda: uuid4().hex, description="Unique ID for result matching"
    )

    def __lt__(self, other: QueuedRequest) -> bool:
        """Compare by priority, then by creation time (FIFO within same priority)."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


class CircuitBreaker:
    """Circuit breaker for LLM provider failure management.

    Tracks failures per provider and temporarily stops sending requests
    to providers that exceed the failure threshold.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        reset_timeout_seconds: float = 60.0,
    ) -> None:
        """Initialise the circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening circuit.
            reset_timeout_seconds: Time before attempting to reset circuit.
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds
        self._failures: dict[str, deque[float]] = defaultdict(deque)

    def record_failure(self, provider_id: str) -> None:
        """Record a failure for a provider.

        Args:
            provider_id: Identifier for the provider (e.g., model name).
        """
        now = time.monotonic()
        self._failures[provider_id].append(now)
        logger.warning(
            "circuit_breaker_failure",
            provider_id=provider_id,
            failures=len(self._failures[provider_id]),
        )

    def record_success(self, provider_id: str) -> None:
        """Record a success for a provider, clearing its failure history.

        Args:
            provider_id: Identifier for the provider.
        """
        if provider_id in self._failures:
            del self._failures[provider_id]

    def is_open(self, provider_id: str) -> bool:
        """Check if the circuit is open (provider should be avoided).

        Args:
            provider_id: Identifier for the provider.

        Returns:
            True if the circuit is open (too many failures).
        """
        if provider_id not in self._failures:
            return False

        failures = self._failures[provider_id]
        now = time.monotonic()

        # Remove failures outside the reset timeout window
        while failures and (now - failures[0]) > self.reset_timeout_seconds:
            failures.popleft()

        # Circuit is open if we have too many recent failures
        is_open = len(failures) >= self.failure_threshold
        if is_open:
            logger.warning(
                "circuit_breaker_open",
                provider_id=provider_id,
                failures=len(failures),
                reset_timeout_seconds=self.reset_timeout_seconds,
            )
        return is_open

    def reset(self, provider_id: str) -> None:
        """Manually reset the circuit for a provider.

        Args:
            provider_id: Identifier for the provider.
        """
        if provider_id in self._failures:
            del self._failures[provider_id]
            logger.info("circuit_breaker_reset", provider_id=provider_id)


class LLMGateway:
    """Centralized LLM request gateway with queueing and rate limiting.

    Manages all LLM requests with priority queueing, concurrency control,
    per-agent and global rate limiting, request deduplication, circuit
    breaker pattern, and cost tracking.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_concurrent: int = 10,
        requests_per_minute: int = 100,
        cost_alert_threshold: float = 50.0,
    ) -> None:
        """Initialise the gateway.

        Args:
            provider: The LLM provider to use for requests.
            max_concurrent: Maximum number of concurrent LLM requests.
            requests_per_minute: Global rate limit (requests per minute).
            cost_alert_threshold: Cost threshold (USD) for alerting.
        """
        self._provider = provider
        self._max_concurrent = max_concurrent
        self._requests_per_minute = requests_per_minute
        self._cost_alert_threshold = cost_alert_threshold

        # Queue and concurrency control
        self._queue: asyncio.PriorityQueue[QueuedRequest] = asyncio.PriorityQueue()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._futures: dict[str, asyncio.Future[LLMResponse]] = {}

        # Rate limiting
        self._global_timestamps: deque[float] = deque()
        self._agent_timestamps: dict[str, deque[float]] = defaultdict(deque)

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker()

        # Cost tracking
        self._total_cost_usd: float = 0.0
        self._total_requests: int = 0
        self._failed_requests: int = 0
        self._cost_history: list[tuple[float, float]] = []  # (timestamp, cost)

        # Deduplication cache (short-lived)
        self._dedup_cache: dict[str, asyncio.Future[LLMResponse]] = {}
        self._dedup_ttl_seconds = 5.0

        # Background tasks (prevent GC of fire-and-forget tasks)
        self._background_tasks: set[asyncio.Task[None]] = set()

        # Cost exceeded flag
        self._cost_exceeded = False

        # Background task
        self._processor_task: asyncio.Task[None] | None = None
        self._running = False

    @staticmethod
    def _make_dedup_key(request: LLMRequest) -> str:
        """Create a deduplication key from request fields."""
        payload = json.dumps(
            {
                "prompt": request.prompt,
                "system_prompt": request.system_prompt,
                "model": request.model,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "json_mode": request.json_mode,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _check_rate_limit(self, agent_id: str) -> None:
        """Check if rate limit has been exceeded for global or per-agent.

        Args:
            agent_id: The agent making the request.

        Raises:
            Exception: If rate limit is exceeded.
        """
        now = time.monotonic()
        window_start = now - 60.0

        # Clean up old timestamps
        while self._global_timestamps and self._global_timestamps[0] < window_start:
            self._global_timestamps.popleft()

        agent_ts = self._agent_timestamps[agent_id]
        while agent_ts and agent_ts[0] < window_start:
            agent_ts.popleft()

        # Check global limit
        if len(self._global_timestamps) >= self._requests_per_minute:
            raise RateLimitExceededError(
                f"Global rate limit exceeded: {len(self._global_timestamps)} requests in last "
                f"minute (limit: {self._requests_per_minute})"
            )

        # Check per-agent limit (20% of global limit per agent)
        agent_limit = max(1, int(self._requests_per_minute * 0.2))
        if len(agent_ts) >= agent_limit:
            raise RateLimitExceededError(
                f"Per-agent rate limit exceeded for {agent_id}: {len(agent_ts)} requests in "
                f"last minute (limit: {agent_limit})"
            )

        # Record timestamps
        self._global_timestamps.append(now)
        agent_ts.append(now)

    async def submit(
        self,
        request: LLMRequest,
        agent_id: str,
        priority: RequestPriority = RequestPriority.NORMAL,
    ) -> LLMResponse:
        """Submit a request to the gateway and return the response.

        Args:
            request: The LLM request to execute.
            agent_id: ID of the agent making the request.
            priority: Request priority (default: NORMAL).

        Returns:
            The LLM response.

        Raises:
            Exception: If rate limit is exceeded or gateway is not running.
        """
        if not self._running:
            raise RuntimeError("Gateway is not running. Call start() first.")

        # Check cost exceeded
        if self._cost_exceeded:
            raise RuntimeError(
                f"Cost threshold exceeded (${self._total_cost_usd:.2f} >= "
                f"${self._cost_alert_threshold:.2f}). New requests are rejected."
            )

        # Check rate limits
        self._check_rate_limit(agent_id)

        # Check deduplication cache
        dedup_key = self._make_dedup_key(request)
        if dedup_key in self._dedup_cache:
            logger.debug("request_deduplicated", dedup_key=dedup_key[:16])
            return await self._dedup_cache[dedup_key]

        # Create queued request
        queued = QueuedRequest(request=request, priority=priority, agent_id=agent_id)

        # Create future for result
        future: asyncio.Future[LLMResponse] = asyncio.Future()
        self._futures[queued.future_id] = future
        self._dedup_cache[dedup_key] = future

        # Schedule cleanup of dedup cache
        async def _cleanup_dedup() -> None:
            await asyncio.sleep(self._dedup_ttl_seconds)
            if dedup_key in self._dedup_cache:
                del self._dedup_cache[dedup_key]

        task = asyncio.create_task(_cleanup_dedup())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        # Add to queue
        await self._queue.put(queued)
        logger.debug(
            "request_queued",
            priority=priority.name,
            agent_id=agent_id,
            future_id=queued.future_id[:8],
        )

        # Wait for result
        return await future

    async def _process_queue(self) -> None:
        """Background task that processes queued requests.

        Runs continuously while the gateway is active, respecting
        concurrency limits and circuit breaker state. Each request is
        dispatched to its own task for concurrent processing.
        """
        logger.info("gateway_processor_started")

        while self._running:
            try:
                # Get next request (with timeout to allow clean shutdown)
                try:
                    queued = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                # Check circuit breaker
                provider_id = queued.request.model or "default"
                if self._circuit_breaker.is_open(provider_id):
                    error_msg = f"Circuit breaker open for provider {provider_id}"
                    logger.error("circuit_breaker_rejected", provider_id=provider_id)
                    future = self._futures.pop(queued.future_id, None)
                    if future and not future.done():
                        future.set_exception(Exception(error_msg))
                    self._failed_requests += 1
                    continue

                # Dispatch to its own task for concurrent processing
                task = asyncio.create_task(self._handle_item(queued))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

            except Exception as exc:
                logger.exception("gateway_processor_error", error=str(exc))

        logger.info("gateway_processor_stopped")

    async def _handle_item(self, queued: QueuedRequest) -> None:
        """Handle a single queued request with concurrency limiting.

        Acquires the semaphore before processing, and ensures the future
        is resolved even on unexpected errors.

        Args:
            queued: The queued request to handle.
        """
        async with self._semaphore:
            try:
                await self._process_request(queued)
            except Exception as exc:
                # Ensure future is resolved even on unexpected errors
                future = self._futures.pop(queued.future_id, None)
                if future and not future.done():
                    future.set_exception(exc)
                logger.exception("handle_item_error", error=str(exc))

    async def _process_request(self, queued: QueuedRequest) -> None:
        """Process a single queued request.

        Args:
            queued: The queued request to process.
        """
        future = self._futures.pop(queued.future_id, None)
        if future is None or future.done():
            return

        start_time = time.monotonic()
        provider_id = queued.request.model or "default"

        try:
            # Call the provider
            response = await self._provider.complete(queued.request)

            # Track success
            self._circuit_breaker.record_success(provider_id)
            self._total_cost_usd += response.cost_usd
            self._total_requests += 1
            self._cost_history.append((time.monotonic(), response.cost_usd))

            # Check cost threshold
            if self._total_cost_usd >= self._cost_alert_threshold:
                logger.warning(
                    "cost_threshold_exceeded",
                    total_cost_usd=self._total_cost_usd,
                    threshold_usd=self._cost_alert_threshold,
                )
                self._cost_exceeded = True

            # Set result
            future.set_result(response)

            latency = (time.monotonic() - start_time) * 1000
            logger.debug(
                "request_completed",
                agent_id=queued.agent_id,
                latency_ms=round(latency, 1),
                cost_usd=response.cost_usd,
            )

        except Exception as exc:
            # Track failure
            self._circuit_breaker.record_failure(provider_id)
            self._failed_requests += 1

            # Set exception
            future.set_exception(exc)

            logger.error(
                "request_failed",
                agent_id=queued.agent_id,
                error=str(exc),
            )

    def get_stats(self) -> dict[str, object]:
        """Get gateway statistics.

        Returns:
            Dictionary with current statistics including queue depth,
            total cost, request counts, and error rates.
        """
        return {
            "pending_requests": self._queue.qsize(),
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "total_cost_usd": round(self._total_cost_usd, 4),
            "error_rate": (
                round(self._failed_requests / self._total_requests, 3)
                if self._total_requests > 0
                else 0.0
            ),
            "running": self._running,
        }

    @property
    def total_cost_usd(self) -> float:
        """Return the total cost in USD across all requests."""
        return self._total_cost_usd

    @property
    def pending_count(self) -> int:
        """Return the number of pending requests in the queue."""
        return self._queue.qsize()

    async def start(self) -> None:
        """Start the gateway background processor."""
        if self._running:
            logger.warning("gateway_already_running")
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_queue())
        logger.info("gateway_started")

    async def stop(self) -> None:
        """Stop the gateway and drain the queue."""
        if not self._running:
            return

        logger.info("gateway_stopping")
        self._running = False

        # Wait for processor to finish
        if self._processor_task is not None:
            await self._processor_task

        # Cancel any remaining futures
        for future in self._futures.values():
            if not future.done():
                future.cancel()

        self._futures.clear()
        self._dedup_cache.clear()

        logger.info(
            "gateway_stopped",
            total_requests=self._total_requests,
            total_cost_usd=round(self._total_cost_usd, 2),
        )
