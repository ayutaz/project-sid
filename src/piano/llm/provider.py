"""LLM provider abstraction layer.

Defines the LLMProvider protocol and a concrete OpenAI-based implementation
with retry logic, timeout handling, and cost tracking.

Reference: docs/implementation/02-llm-integration.md
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Protocol, runtime_checkable

from openai import AsyncOpenAI
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import BadRequestError as OpenAIBadRequestError
from openai import PermissionDeniedError as OpenAIPermissionDeniedError

from piano.core.types import LLMRequest, LLMResponse, ModuleTier

# Non-transient OpenAI errors that should not be retried
_NON_TRANSIENT_ERRORS = (
    OpenAIAuthenticationError,
    OpenAIBadRequestError,
    OpenAIPermissionDeniedError,
)

logger = logging.getLogger(__name__)


# --- Custom Exceptions ---


class CostLimitExceededError(Exception):
    """Raised when the cost limit is exceeded."""

    pass


class RateLimitExceededError(Exception):
    """Raised when the rate limit is exceeded."""

    pass

# Tier-based default models
DEFAULT_MODELS: dict[ModuleTier, str] = {
    ModuleTier.FAST: "gpt-4o-mini",
    ModuleTier.MID: "gpt-4o-mini",
    ModuleTier.SLOW: "gpt-4o",
}

# Static cost table: model -> (cost_per_1k_input, cost_per_1k_output)
# Maintenance: update this table when adding new models or when OpenAI changes pricing.
# Prices sourced from https://openai.com/api/pricing/
_OPENAI_COSTS: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.010),
    "gpt-4o-mini": (0.000150, 0.000600),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4.1-nano": (0.0001, 0.0004),
}


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers.

    All LLM providers must implement the ``complete`` method. This enables
    swapping between real providers, mocks, and cached wrappers without
    changing calling code.
    """

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request and return the response."""
        ...


class OpenAIProvider:
    """LLM provider backed by the OpenAI SDK.

    Supports OpenAI models with automatic retry and exponential backoff.
    """

    def __init__(
        self,
        *,
        default_models: dict[ModuleTier, str] | None = None,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        cost_limit_usd: float = 100.0,
        calls_per_minute_limit: int = 100,
        api_key: str | None = None,
    ) -> None:
        """Initialise the provider.

        Args:
            default_models: Override the default model mapping per tier.
            max_retries: Maximum number of retry attempts on transient errors.
            timeout_seconds: Per-request timeout in seconds.
            cost_limit_usd: Maximum allowed cost in USD before raising CostLimitExceededError.
            calls_per_minute_limit: Maximum calls per minute before raising RateLimitExceededError.
            api_key: Optional OpenAI API key. If None, uses OPENAI_API_KEY env var.
        """
        self.default_models = default_models or dict(DEFAULT_MODELS)
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self._cost_limit_usd = cost_limit_usd
        self._calls_per_minute_limit = calls_per_minute_limit
        self._total_cost_usd: float = 0.0
        self._call_count: int = 0
        self._call_timestamps: deque[float] = deque()
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_seconds)

    def _resolve_model(self, request: LLMRequest) -> str:
        """Return the model string, falling back to tier default."""
        if request.model:
            return request.model
        return self.default_models[request.tier]

    def _build_messages(self, request: LLMRequest) -> list[dict[str, str]]:
        """Build the messages list from the request."""
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        return messages

    def _check_cost_limit(self) -> None:
        """Check if cost limit has been exceeded.

        Raises:
            CostLimitExceededError: If total cost exceeds the configured limit.
        """
        if self._total_cost_usd >= self._cost_limit_usd:
            raise CostLimitExceededError(
                f"Cost limit exceeded: ${self._total_cost_usd:.4f} >= ${self._cost_limit_usd:.4f}"
            )

    def _check_rate_limit(self) -> None:
        """Check if rate limit has been exceeded using a sliding window.

        Raises:
            RateLimitExceededError: If calls per minute exceeds the configured limit.
        """
        now = time.monotonic()
        window_start = now - 60.0  # 60 seconds ago

        # Remove timestamps outside the sliding window
        while self._call_timestamps and self._call_timestamps[0] < window_start:
            self._call_timestamps.popleft()

        # Check if we've exceeded the rate limit
        if len(self._call_timestamps) >= self._calls_per_minute_limit:
            raise RateLimitExceededError(
                f"Rate limit exceeded: {len(self._call_timestamps)} calls in the last minute "
                f"(limit: {self._calls_per_minute_limit})"
            )

    @property
    def total_cost_usd(self) -> float:
        """Return the total cost in USD across all calls."""
        return self._total_cost_usd

    @property
    def call_count(self) -> int:
        """Return the total number of successful calls."""
        return self._call_count

    def reset_cost_tracking(self) -> None:
        """Reset cost and call count tracking to zero."""
        self._total_cost_usd = 0.0
        self._call_count = 0
        self._call_timestamps.clear()

    @staticmethod
    def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost using the static cost table.

        Args:
            model: The model ID.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD, or 0.0 if model is not in the table.
        """
        costs = _OPENAI_COSTS.get(model)
        if costs is None:
            return 0.0
        input_cost, output_cost = costs
        return (prompt_tokens / 1000.0) * input_cost + (completion_tokens / 1000.0) * output_cost

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request with retry and backoff.

        Args:
            request: The LLM request to execute.

        Returns:
            An LLMResponse with content, usage, cost, and latency info.

        Raises:
            CostLimitExceededError: If cost limit is exceeded.
            RateLimitExceededError: If rate limit is exceeded.
            Exception: If all retry attempts are exhausted.
        """
        # Check limits before making the call
        self._check_cost_limit()
        self._check_rate_limit()

        model = self._resolve_model(request)
        messages = self._build_messages(request)

        kwargs: dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_exception: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                start = time.monotonic()
                response = await self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
                latency_ms = (time.monotonic() - start) * 1000

                content = response.choices[0].message.content or ""
                usage = {}
                prompt_tokens = 0
                completion_tokens = 0
                if response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }

                cost = self._estimate_cost(model, prompt_tokens, completion_tokens)

                # Update tracking after successful call
                self._total_cost_usd += cost
                self._call_count += 1
                self._call_timestamps.append(time.monotonic())

                return LLMResponse(
                    content=content,
                    model=model,
                    usage=usage,
                    latency_ms=latency_ms,
                    cost_usd=cost,
                )
            except _NON_TRANSIENT_ERRORS:
                # Non-transient errors should not be retried
                raise
            except Exception as exc:
                last_exception = exc
                if attempt < self.max_retries - 1:
                    wait = 2**attempt  # 1s, 2s, 4s
                    logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        self.max_retries,
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)

        # This should never happen if max_retries > 0, but guard for type safety
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("LLM call failed with no exception captured")
