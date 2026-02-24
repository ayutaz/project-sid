"""Resource limiter with backpressure for per-agent and per-worker throttling.

Manages concurrent LLM calls and memory usage on a per-agent and per-worker
basis using asyncio semaphores. When resource usage exceeds configurable
thresholds, backpressure is applied by throttling the offending agent.

Reference: docs/implementation/08-infrastructure.md
"""

from __future__ import annotations

__all__ = [
    "BackpressurePolicy",
    "ResourceLimiter",
    "ResourceUsage",
]

import asyncio
from enum import StrEnum

import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Resource types
# ---------------------------------------------------------------------------


class ResourceType(StrEnum):
    """Types of resources that can be limited."""

    LLM = "llm"
    MEMORY = "memory"


# ---------------------------------------------------------------------------
# ResourceUsage model
# ---------------------------------------------------------------------------


class ResourceUsage(BaseModel):
    """Snapshot of current resource usage for a single agent."""

    agent_id: str
    concurrent_llm: int = 0
    memory_mb: float = 0.0
    throttled: bool = False


# ---------------------------------------------------------------------------
# BackpressurePolicy
# ---------------------------------------------------------------------------


class BackpressurePolicy:
    """Policy that determines when an agent should be throttled.

    Thresholds are expressed as fractions of the configured maximums.

    * ``warning_threshold`` (default 0.8) -- log a warning when usage
      exceeds this fraction of the limit.
    * ``throttle_threshold`` (default 0.9) -- throttle the agent when
      usage exceeds this fraction of the limit.
    """

    def __init__(
        self,
        *,
        warning_threshold: float = 0.8,
        throttle_threshold: float = 0.9,
    ) -> None:
        if not 0.0 < warning_threshold <= 1.0:
            msg = f"warning_threshold must be in (0, 1], got {warning_threshold}"
            raise ValueError(msg)
        if not 0.0 < throttle_threshold <= 1.0:
            msg = f"throttle_threshold must be in (0, 1], got {throttle_threshold}"
            raise ValueError(msg)
        if warning_threshold > throttle_threshold:
            msg = (
                f"warning_threshold ({warning_threshold}) must be "
                f"<= throttle_threshold ({throttle_threshold})"
            )
            raise ValueError(msg)
        self.warning_threshold = warning_threshold
        self.throttle_threshold = throttle_threshold

    def should_warn(
        self,
        usage: ResourceUsage,
        *,
        max_concurrent_llm: int,
        max_memory_mb: int,
    ) -> bool:
        """Return ``True`` if the agent's usage exceeds the warning threshold."""
        llm_over = (
            max_concurrent_llm > 0
            and usage.concurrent_llm > max_concurrent_llm * self.warning_threshold
        )
        mem_over = max_memory_mb > 0 and usage.memory_mb > max_memory_mb * self.warning_threshold
        return llm_over or mem_over

    def should_throttle(
        self,
        usage: ResourceUsage,
        *,
        max_concurrent_llm: int,
        max_memory_mb: int,
    ) -> bool:
        """Return ``True`` if the agent's usage exceeds the throttle threshold."""
        llm_over = (
            max_concurrent_llm > 0
            and usage.concurrent_llm > max_concurrent_llm * self.throttle_threshold
        )
        mem_over = max_memory_mb > 0 and usage.memory_mb > max_memory_mb * self.throttle_threshold
        return llm_over or mem_over


# ---------------------------------------------------------------------------
# ResourceLimiter
# ---------------------------------------------------------------------------


class _AgentState:
    """Internal per-agent resource tracking state."""

    __slots__ = ("llm_available", "llm_in_use", "llm_semaphore", "memory_mb", "throttled")

    def __init__(self, max_concurrent_llm: int) -> None:
        self.llm_semaphore = asyncio.Semaphore(max_concurrent_llm)
        self.llm_available: int = max_concurrent_llm
        self.llm_in_use: int = 0
        self.memory_mb: float = 0.0
        self.throttled: bool = False


class ResourceLimiter:
    """Manage per-agent and per-worker resource limits.

    Uses asyncio semaphores for concurrency control and a pluggable
    :class:`BackpressurePolicy` for throttling decisions.

    Parameters:
        max_concurrent_llm_per_agent: Maximum simultaneous LLM calls per agent.
        max_concurrent_llm_per_worker: Maximum simultaneous LLM calls across
            all agents on this worker.
        max_memory_per_agent_mb: Memory ceiling per agent in megabytes.
        policy: Optional :class:`BackpressurePolicy` (defaults to 80%/90%).
    """

    def __init__(
        self,
        *,
        max_concurrent_llm_per_agent: int = 3,
        max_concurrent_llm_per_worker: int = 50,
        max_memory_per_agent_mb: int = 64,
        policy: BackpressurePolicy | None = None,
    ) -> None:
        self.max_concurrent_llm_per_agent = max_concurrent_llm_per_agent
        self.max_concurrent_llm_per_worker = max_concurrent_llm_per_worker
        self.max_memory_per_agent_mb = max_memory_per_agent_mb
        self.policy = policy or BackpressurePolicy()

        self._agents: dict[str, _AgentState] = {}
        self._worker_semaphore = asyncio.Semaphore(max_concurrent_llm_per_worker)
        self._worker_llm_available: int = max_concurrent_llm_per_worker
        self._worker_llm_in_use: int = 0
        self._lock = asyncio.Lock()

    # -- internal helpers ---------------------------------------------------

    async def _get_agent_state(self, agent_id: str) -> _AgentState:
        """Return (or create) the tracking state for *agent_id*."""
        async with self._lock:
            if agent_id not in self._agents:
                self._agents[agent_id] = _AgentState(self.max_concurrent_llm_per_agent)
                logger.debug("resource_limiter.agent_registered", agent_id=agent_id)
            return self._agents[agent_id]

    def _get_agent_state_sync(self, agent_id: str) -> _AgentState | None:
        """Return existing agent state without creating, or ``None``."""
        return self._agents.get(agent_id)

    # -- public API ---------------------------------------------------------

    async def acquire(self, agent_id: str, resource_type: str) -> bool:
        """Attempt to acquire a resource slot.

        The entire check-and-acquire sequence is protected by a single lock
        to prevent race conditions between concurrent callers.

        Args:
            agent_id: The agent requesting the resource.
            resource_type: ``"llm"`` or ``"memory"``.

        Returns:
            ``True`` if the resource was acquired, ``False`` if the agent
            is throttled or at capacity.
        """
        state = await self._get_agent_state(agent_id)

        if resource_type == ResourceType.LLM:
            async with self._lock:
                # Check throttle status first
                usage = self._build_usage(agent_id, state)
                throttled = self.policy.should_throttle(
                    usage,
                    max_concurrent_llm=self.max_concurrent_llm_per_agent,
                    max_memory_mb=self.max_memory_per_agent_mb,
                )
                if throttled:
                    state.throttled = True
                    logger.warning(
                        "resource_limiter.throttled",
                        agent_id=agent_id,
                        concurrent_llm=state.llm_in_use,
                    )
                    return False

                # Try agent-level semaphore (non-blocking)
                if state.llm_available <= 0:
                    logger.debug(
                        "resource_limiter.agent_limit_reached",
                        agent_id=agent_id,
                        in_use=state.llm_in_use,
                    )
                    return False

                # Try worker-level semaphore (non-blocking)
                if self._worker_llm_available <= 0:
                    logger.debug(
                        "resource_limiter.worker_limit_reached",
                        worker_llm_in_use=self._worker_llm_in_use,
                    )
                    return False

                # Acquire both semaphores atomically with counter updates
                await state.llm_semaphore.acquire()
                state.llm_available -= 1
                await self._worker_semaphore.acquire()
                self._worker_llm_available -= 1
                state.llm_in_use += 1
                self._worker_llm_in_use += 1

                # Update throttle status after acquiring
                state.throttled = False
                self._check_and_log_warning(agent_id, state)

                logger.debug(
                    "resource_limiter.acquired",
                    agent_id=agent_id,
                    resource_type=resource_type,
                    concurrent_llm=state.llm_in_use,
                )
                return True

        if resource_type == ResourceType.MEMORY:
            # Memory is tracked but not semaphore-gated; see set_memory_usage
            usage = self._build_usage(agent_id, state)
            if self.policy.should_throttle(
                usage,
                max_concurrent_llm=self.max_concurrent_llm_per_agent,
                max_memory_mb=self.max_memory_per_agent_mb,
            ):
                state.throttled = True
                return False
            return True

        logger.warning("resource_limiter.unknown_resource_type", resource_type=resource_type)
        return False

    async def release(self, agent_id: str, resource_type: str) -> None:
        """Release a previously acquired resource slot.

        Args:
            agent_id: The agent releasing the resource.
            resource_type: ``"llm"`` or ``"memory"``.
        """
        state = await self._get_agent_state(agent_id)

        if resource_type == ResourceType.LLM:
            async with self._lock:
                if state.llm_in_use <= 0:
                    logger.warning(
                        "resource_limiter.release_without_acquire",
                        agent_id=agent_id,
                    )
                    return

                state.llm_in_use -= 1
                self._worker_llm_in_use = max(0, self._worker_llm_in_use - 1)
                state.llm_semaphore.release()
                state.llm_available += 1
                self._worker_semaphore.release()
                self._worker_llm_available += 1

                # Re-evaluate throttle status
                usage = self._build_usage(agent_id, state)
                state.throttled = self.policy.should_throttle(
                    usage,
                    max_concurrent_llm=self.max_concurrent_llm_per_agent,
                    max_memory_mb=self.max_memory_per_agent_mb,
                )

                logger.debug(
                    "resource_limiter.released",
                    agent_id=agent_id,
                    resource_type=resource_type,
                    concurrent_llm=state.llm_in_use,
                )
        elif resource_type == ResourceType.MEMORY:
            # Memory release is handled via set_memory_usage
            pass
        else:
            logger.warning("resource_limiter.unknown_resource_type", resource_type=resource_type)

    async def set_memory_usage(self, agent_id: str, memory_mb: float) -> None:
        """Update the reported memory usage for an agent.

        Args:
            agent_id: The agent whose memory usage changed.
            memory_mb: Current memory usage in megabytes.
        """
        state = await self._get_agent_state(agent_id)
        state.memory_mb = memory_mb

        # Re-evaluate throttle status
        usage = self._build_usage(agent_id, state)
        was_throttled = state.throttled
        state.throttled = self.policy.should_throttle(
            usage,
            max_concurrent_llm=self.max_concurrent_llm_per_agent,
            max_memory_mb=self.max_memory_per_agent_mb,
        )

        if state.throttled and not was_throttled:
            logger.warning(
                "resource_limiter.memory_throttle",
                agent_id=agent_id,
                memory_mb=memory_mb,
                max_memory_mb=self.max_memory_per_agent_mb,
            )
        self._check_and_log_warning(agent_id, state)

    def get_usage(self, agent_id: str) -> ResourceUsage:
        """Return current resource usage for an agent.

        Args:
            agent_id: The agent to query.

        Returns:
            A :class:`ResourceUsage` snapshot. Returns zeroed usage if the
            agent has not been registered yet.
        """
        state = self._get_agent_state_sync(agent_id)
        if state is None:
            return ResourceUsage(agent_id=agent_id)
        return self._build_usage(agent_id, state)

    async def deregister_agent(self, agent_id: str) -> None:
        """Remove all tracking state for an agent.

        Any in-use LLM slots are released back to the worker pool.

        Args:
            agent_id: The agent to deregister.
        """
        async with self._lock:
            state = self._agents.pop(agent_id, None)
            if state is None:
                return
            # Release any in-use worker slots
            for _ in range(state.llm_in_use):
                self._worker_llm_in_use = max(0, self._worker_llm_in_use - 1)
                self._worker_semaphore.release()
                self._worker_llm_available += 1
            logger.debug("resource_limiter.agent_deregistered", agent_id=agent_id)

    def is_throttled(self, agent_id: str) -> bool:
        """Return whether the agent is currently throttled.

        Args:
            agent_id: The agent to check.
        """
        state = self._get_agent_state_sync(agent_id)
        if state is None:
            return False
        return state.throttled

    @property
    def worker_llm_in_use(self) -> int:
        """Return the number of LLM slots currently in use across the worker."""
        return self._worker_llm_in_use

    # -- internal -----------------------------------------------------------

    def _build_usage(self, agent_id: str, state: _AgentState) -> ResourceUsage:
        """Build a :class:`ResourceUsage` from internal state."""
        return ResourceUsage(
            agent_id=agent_id,
            concurrent_llm=state.llm_in_use,
            memory_mb=state.memory_mb,
            throttled=state.throttled,
        )

    def _check_and_log_warning(self, agent_id: str, state: _AgentState) -> None:
        """Log a warning if usage is above the warning threshold."""
        usage = self._build_usage(agent_id, state)
        if self.policy.should_warn(
            usage,
            max_concurrent_llm=self.max_concurrent_llm_per_agent,
            max_memory_mb=self.max_memory_per_agent_mb,
        ):
            logger.warning(
                "resource_limiter.high_usage",
                agent_id=agent_id,
                concurrent_llm=state.llm_in_use,
                memory_mb=state.memory_mb,
            )
