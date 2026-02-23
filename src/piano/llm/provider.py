"""LLM provider abstraction layer.

Defines the LLMProvider protocol and a concrete LiteLLM-based implementation
with retry logic, timeout handling, and cost tracking.

Reference: docs/implementation/02-llm-integration.md
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Protocol, runtime_checkable

import litellm

from piano.core.types import LLMRequest, LLMResponse, ModuleTier

logger = logging.getLogger(__name__)

# Tier-based default models
DEFAULT_MODELS: dict[ModuleTier, str] = {
    ModuleTier.FAST: "gpt-4o-mini",
    ModuleTier.MID: "gpt-4o-mini",
    ModuleTier.SLOW: "gpt-4o",
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


class LiteLLMProvider:
    """LLM provider backed by litellm.

    Supports all models that litellm supports (OpenAI, Anthropic, Google, etc.)
    with automatic retry and exponential backoff.
    """

    def __init__(
        self,
        *,
        default_models: dict[ModuleTier, str] | None = None,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialise the provider.

        Args:
            default_models: Override the default model mapping per tier.
            max_retries: Maximum number of retry attempts on transient errors.
            timeout_seconds: Per-request timeout in seconds.
        """
        self.default_models = default_models or dict(DEFAULT_MODELS)
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

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

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request with retry and backoff.

        Args:
            request: The LLM request to execute.

        Returns:
            An LLMResponse with content, usage, cost, and latency info.

        Raises:
            Exception: If all retry attempts are exhausted.
        """
        model = self._resolve_model(request)
        messages = self._build_messages(request)

        kwargs: dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "timeout": self.timeout_seconds,
        }
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_exception: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                start = time.monotonic()
                response = await litellm.acompletion(**kwargs)  # type: ignore[arg-type]
                latency_ms = (time.monotonic() - start) * 1000

                content = response.choices[0].message.content or ""
                usage = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }

                try:
                    cost = litellm.completion_cost(completion_response=response)
                except Exception:
                    cost = 0.0

                return LLMResponse(
                    content=content,
                    model=model,
                    usage=usage,
                    latency_ms=latency_ms,
                    cost_usd=cost,
                )
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

        raise last_exception  # type: ignore[misc]
