"""Tests for LiteLLMProvider.

All tests mock litellm.acompletion to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from piano.core.types import LLMRequest, ModuleTier
from piano.llm.provider import (
    DEFAULT_MODELS,
    CostLimitExceededError,
    LiteLLMProvider,
    LLMProvider,
    RateLimitExceededError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_litellm_response(
    content: str = "Hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
) -> MagicMock:
    """Create a mock litellm response object."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.usage.total_tokens = total_tokens
    return response


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestLLMProviderProtocol:
    def test_litellm_provider_is_llm_provider(self) -> None:
        provider = LiteLLMProvider()
        assert isinstance(provider, LLMProvider)


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

class TestModelResolution:
    def test_uses_explicit_model(self) -> None:
        provider = LiteLLMProvider()
        request = LLMRequest(prompt="hi", model="claude-sonnet-4-5-20250929")
        assert provider._resolve_model(request) == "claude-sonnet-4-5-20250929"

    def test_falls_back_to_tier_default_fast(self) -> None:
        provider = LiteLLMProvider()
        request = LLMRequest(prompt="hi", tier=ModuleTier.FAST)
        assert provider._resolve_model(request) == DEFAULT_MODELS[ModuleTier.FAST]

    def test_falls_back_to_tier_default_mid(self) -> None:
        provider = LiteLLMProvider()
        request = LLMRequest(prompt="hi", tier=ModuleTier.MID)
        assert provider._resolve_model(request) == DEFAULT_MODELS[ModuleTier.MID]

    def test_falls_back_to_tier_default_slow(self) -> None:
        provider = LiteLLMProvider()
        request = LLMRequest(prompt="hi", tier=ModuleTier.SLOW)
        assert provider._resolve_model(request) == DEFAULT_MODELS[ModuleTier.SLOW]

    def test_custom_default_models(self) -> None:
        custom = {ModuleTier.FAST: "my-fast", ModuleTier.MID: "my-mid", ModuleTier.SLOW: "my-slow"}
        provider = LiteLLMProvider(default_models=custom)
        request = LLMRequest(prompt="hi", tier=ModuleTier.FAST)
        assert provider._resolve_model(request) == "my-fast"


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

class TestMessageBuilding:
    def test_user_only(self) -> None:
        provider = LiteLLMProvider()
        request = LLMRequest(prompt="hello")
        messages = provider._build_messages(request)
        assert messages == [{"role": "user", "content": "hello"}]

    def test_system_and_user(self) -> None:
        provider = LiteLLMProvider()
        request = LLMRequest(prompt="hello", system_prompt="you are helpful")
        messages = provider._build_messages(request)
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "you are helpful"}
        assert messages[1] == {"role": "user", "content": "hello"}


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

class TestCompletion:
    @pytest.mark.asyncio
    async def test_successful_completion(self) -> None:
        mock_resp = _make_litellm_response("world")
        provider = LiteLLMProvider()

        with (
            patch("piano.llm.provider.litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch("piano.llm.provider.litellm.completion_cost", return_value=0.001),
        ):
            mock_ac.return_value = mock_resp
            result = await provider.complete(LLMRequest(prompt="hello"))

        assert result.content == "world"
        assert result.model == "gpt-4o"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 15
        assert result.cost_usd == pytest.approx(0.001)
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_json_mode_sets_response_format(self) -> None:
        mock_resp = _make_litellm_response('{"key": "value"}')
        provider = LiteLLMProvider()

        with (
            patch("piano.llm.provider.litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch("piano.llm.provider.litellm.completion_cost", return_value=0.0),
        ):
            mock_ac.return_value = mock_resp
            await provider.complete(LLMRequest(prompt="json", json_mode=True))

        call_kwargs = mock_ac.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_temperature_and_max_tokens_passed(self) -> None:
        mock_resp = _make_litellm_response()
        provider = LiteLLMProvider()

        with (
            patch("piano.llm.provider.litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch("piano.llm.provider.litellm.completion_cost", return_value=0.0),
        ):
            mock_ac.return_value = mock_resp
            await provider.complete(LLMRequest(prompt="hi", temperature=0.7, max_tokens=512))

        call_kwargs = mock_ac.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_cost_calculation_fallback_on_error(self) -> None:
        mock_resp = _make_litellm_response()
        provider = LiteLLMProvider()

        with (
            patch("piano.llm.provider.litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch("piano.llm.provider.litellm.completion_cost", side_effect=Exception("no cost")),
        ):
            mock_ac.return_value = mock_resp
            result = await provider.complete(LLMRequest(prompt="hi"))

        assert result.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_none_content_becomes_empty_string(self) -> None:
        mock_resp = _make_litellm_response()
        mock_resp.choices[0].message.content = None
        provider = LiteLLMProvider()

        with (
            patch("piano.llm.provider.litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch("piano.llm.provider.litellm.completion_cost", return_value=0.0),
        ):
            mock_ac.return_value = mock_resp
            result = await provider.complete(LLMRequest(prompt="hi"))

        assert result.content == ""


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------

class TestRetry:
    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self) -> None:
        mock_resp = _make_litellm_response("ok")
        provider = LiteLLMProvider(max_retries=3)

        call_count = 0

        async def flaky_call(**kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")
            return mock_resp

        with (
            patch("piano.llm.provider.litellm.acompletion", side_effect=flaky_call),
            patch("piano.llm.provider.litellm.completion_cost", return_value=0.0),
            patch("piano.llm.provider.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await provider.complete(LLMRequest(prompt="hi"))

        assert result.content == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_exhausted(self) -> None:
        provider = LiteLLMProvider(max_retries=3)

        with (
            patch(
                "piano.llm.provider.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=RuntimeError("permanent"),
            ),
            patch("piano.llm.provider.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError, match="permanent"),
        ):
            await provider.complete(LLMRequest(prompt="hi"))

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self) -> None:
        provider = LiteLLMProvider(max_retries=3)

        with (
            patch(
                "piano.llm.provider.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=RuntimeError("fail"),
            ),
            patch("piano.llm.provider.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            pytest.raises(RuntimeError),
        ):
            await provider.complete(LLMRequest(prompt="hi"))

        # Backoff delays: 2^0=1, 2^1=2 (3rd attempt fails, no sleep after)
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0].args[0] == 1
        assert mock_sleep.call_args_list[1].args[0] == 2


# ---------------------------------------------------------------------------
# Cost tracking and limits
# ---------------------------------------------------------------------------

class TestCostTracking:
    @pytest.mark.asyncio
    async def test_cost_tracking_increments(self) -> None:
        """Cost should increase after successful calls."""
        mock_resp = _make_litellm_response("ok")
        provider = LiteLLMProvider(cost_limit_usd=10.0)

        with (
            patch("piano.llm.provider.litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch("piano.llm.provider.litellm.completion_cost", return_value=0.5),
        ):
            mock_ac.return_value = mock_resp
            await provider.complete(LLMRequest(prompt="hello"))
            await provider.complete(LLMRequest(prompt="world"))

        assert provider.total_cost_usd == pytest.approx(1.0)
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_cost_limit_exceeded_raises(self) -> None:
        """Should raise CostLimitExceededError when limit is hit."""
        mock_resp = _make_litellm_response("ok")
        provider = LiteLLMProvider(cost_limit_usd=1.0)

        with (
            patch("piano.llm.provider.litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch("piano.llm.provider.litellm.completion_cost", return_value=0.6),
        ):
            mock_ac.return_value = mock_resp
            await provider.complete(LLMRequest(prompt="first"))
            # Second call should put us over the limit
            await provider.complete(LLMRequest(prompt="second"))

            # Third call should fail before making API call
            with pytest.raises(CostLimitExceededError, match="Cost limit exceeded"):
                await provider.complete(LLMRequest(prompt="third"))

        # Should have only 2 successful calls
        assert provider.call_count == 2
        assert provider.total_cost_usd == pytest.approx(1.2)

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_raises(self) -> None:
        """Should raise RateLimitExceededError when rate limit is exceeded."""
        mock_resp = _make_litellm_response("ok")
        provider = LiteLLMProvider(calls_per_minute_limit=2)

        with (
            patch("piano.llm.provider.litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch("piano.llm.provider.litellm.completion_cost", return_value=0.01),
        ):
            mock_ac.return_value = mock_resp
            await provider.complete(LLMRequest(prompt="first"))
            await provider.complete(LLMRequest(prompt="second"))

            # Third call should fail before making API call
            with pytest.raises(RateLimitExceededError, match="Rate limit exceeded"):
                await provider.complete(LLMRequest(prompt="third"))

        # Should have only 2 successful calls
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_reset_cost_tracking(self) -> None:
        """Reset should clear cost and call count tracking."""
        mock_resp = _make_litellm_response("ok")
        provider = LiteLLMProvider(cost_limit_usd=10.0)

        with (
            patch("piano.llm.provider.litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch("piano.llm.provider.litellm.completion_cost", return_value=0.5),
        ):
            mock_ac.return_value = mock_resp
            await provider.complete(LLMRequest(prompt="hello"))
            await provider.complete(LLMRequest(prompt="world"))

        assert provider.total_cost_usd == pytest.approx(1.0)
        assert provider.call_count == 2

        provider.reset_cost_tracking()

        assert provider.total_cost_usd == 0.0
        assert provider.call_count == 0

    @pytest.mark.asyncio
    async def test_cost_properties(self) -> None:
        """total_cost_usd and call_count should be accessible."""
        provider = LiteLLMProvider()

        # Initial state
        assert provider.total_cost_usd == 0.0
        assert provider.call_count == 0

        mock_resp = _make_litellm_response("ok")
        with (
            patch("piano.llm.provider.litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch("piano.llm.provider.litellm.completion_cost", return_value=0.25),
        ):
            mock_ac.return_value = mock_resp
            await provider.complete(LLMRequest(prompt="test"))

        # After one call
        assert provider.total_cost_usd == pytest.approx(0.25)
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_sliding_window(self) -> None:
        """Rate limit should use a sliding window (timestamps older than 60s don't count)."""
        mock_resp = _make_litellm_response("ok")
        provider = LiteLLMProvider(calls_per_minute_limit=2)

        with (
            patch("piano.llm.provider.litellm.acompletion", new_callable=AsyncMock) as mock_ac,
            patch("piano.llm.provider.litellm.completion_cost", return_value=0.01),
            patch("piano.llm.provider.time.monotonic") as mock_time,
        ):
            mock_ac.return_value = mock_resp

            # First call at t=0
            mock_time.return_value = 0.0
            await provider.complete(LLMRequest(prompt="first"))

            # Second call at t=1
            mock_time.return_value = 1.0
            await provider.complete(LLMRequest(prompt="second"))

            # Third call at t=61 (first call has expired from window)
            mock_time.return_value = 61.0
            await provider.complete(LLMRequest(prompt="third"))

        # All three calls should succeed
        assert provider.call_count == 3

    @pytest.mark.asyncio
    async def test_cost_tracking_ignores_failed_calls(self) -> None:
        """Failed calls should not increment cost or call count."""
        provider = LiteLLMProvider(max_retries=1, cost_limit_usd=10.0)

        with (
            patch(
                "piano.llm.provider.litellm.acompletion",
                new_callable=AsyncMock,
                side_effect=RuntimeError("fail"),
            ),
            patch("piano.llm.provider.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError),
        ):
            await provider.complete(LLMRequest(prompt="fail"))

        # Cost and count should remain at 0
        assert provider.total_cost_usd == 0.0
        assert provider.call_count == 0
