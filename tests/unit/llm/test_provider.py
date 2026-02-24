"""Tests for OpenAIProvider.

All tests mock the OpenAI SDK client to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import BadRequestError as OpenAIBadRequestError

from piano.core.types import LLMRequest, ModuleTier
from piano.llm.provider import (
    DEFAULT_MODELS,
    CostLimitExceededError,
    LLMProvider,
    OpenAIProvider,
    RateLimitExceededError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_response(
    content: str = "Hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
) -> MagicMock:
    """Create a mock OpenAI ChatCompletion response object."""
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
    def test_openai_provider_is_llm_provider(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        assert isinstance(provider, LLMProvider)


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


class TestModelResolution:
    def test_uses_explicit_model(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        request = LLMRequest(prompt="hi", model="gpt-4o")
        assert provider._resolve_model(request) == "gpt-4o"

    def test_falls_back_to_tier_default_fast(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        request = LLMRequest(prompt="hi", tier=ModuleTier.FAST)
        assert provider._resolve_model(request) == DEFAULT_MODELS[ModuleTier.FAST]

    def test_falls_back_to_tier_default_mid(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        request = LLMRequest(prompt="hi", tier=ModuleTier.MID)
        assert provider._resolve_model(request) == DEFAULT_MODELS[ModuleTier.MID]

    def test_falls_back_to_tier_default_slow(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        request = LLMRequest(prompt="hi", tier=ModuleTier.SLOW)
        assert provider._resolve_model(request) == DEFAULT_MODELS[ModuleTier.SLOW]

    def test_custom_default_models(self) -> None:
        custom = {ModuleTier.FAST: "my-fast", ModuleTier.MID: "my-mid", ModuleTier.SLOW: "my-slow"}
        provider = OpenAIProvider(default_models=custom, api_key="test-key")
        request = LLMRequest(prompt="hi", tier=ModuleTier.FAST)
        assert provider._resolve_model(request) == "my-fast"


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


class TestMessageBuilding:
    def test_user_only(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        request = LLMRequest(prompt="hello")
        messages = provider._build_messages(request)
        assert messages == [{"role": "user", "content": "hello"}]

    def test_system_and_user(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
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
        mock_resp = _make_openai_response("world")
        provider = OpenAIProvider(api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)
        result = await provider.complete(LLMRequest(prompt="hello"))

        assert result.content == "world"
        assert result.model == "gpt-4o"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 15
        assert result.cost_usd >= 0.0
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_json_mode_sets_response_format(self) -> None:
        mock_resp = _make_openai_response('{"key": "value"}')
        provider = OpenAIProvider(api_key="test-key")

        mock_create = AsyncMock(return_value=mock_resp)
        provider._client.chat.completions.create = mock_create
        await provider.complete(LLMRequest(prompt="json", json_mode=True))

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_temperature_and_max_tokens_passed(self) -> None:
        mock_resp = _make_openai_response()
        provider = OpenAIProvider(api_key="test-key")

        mock_create = AsyncMock(return_value=mock_resp)
        provider._client.chat.completions.create = mock_create
        await provider.complete(LLMRequest(prompt="hi", temperature=0.7, max_tokens=512))

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_cost_estimation_for_known_model(self) -> None:
        mock_resp = _make_openai_response(
            prompt_tokens=1000, completion_tokens=500, total_tokens=1500
        )
        provider = OpenAIProvider(api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)
        result = await provider.complete(LLMRequest(prompt="hi"))

        # gpt-4o: input=0.0025/1k, output=0.010/1k
        # cost = 1000/1000 * 0.0025 + 500/1000 * 0.010 = 0.0025 + 0.005 = 0.0075
        assert result.cost_usd == pytest.approx(0.0075)

    @pytest.mark.asyncio
    async def test_cost_estimation_for_unknown_model(self) -> None:
        mock_resp = _make_openai_response()
        provider = OpenAIProvider(api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)
        result = await provider.complete(LLMRequest(prompt="hi", model="unknown-model"))

        assert result.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_cost_estimation_for_gpt4o_mini(self) -> None:
        mock_resp = _make_openai_response(
            prompt_tokens=2000, completion_tokens=1000, total_tokens=3000
        )
        provider = OpenAIProvider(api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)
        result = await provider.complete(
            LLMRequest(prompt="hi", tier=ModuleTier.FAST)
        )

        # gpt-4o-mini: input=0.000150/1k, output=0.000600/1k
        # cost = 2000/1000 * 0.000150 + 1000/1000 * 0.000600 = 0.0003 + 0.0006 = 0.0009
        assert result.cost_usd == pytest.approx(0.0009)
        assert result.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_model_passed_in_kwargs(self) -> None:
        mock_resp = _make_openai_response("ok")
        provider = OpenAIProvider(api_key="test-key")

        mock_create = AsyncMock(return_value=mock_resp)
        provider._client.chat.completions.create = mock_create
        await provider.complete(LLMRequest(prompt="hi", model="gpt-4.1-nano"))

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4.1-nano"

    @pytest.mark.asyncio
    async def test_response_without_usage(self) -> None:
        """response.usage=None should result in empty usage and zero cost."""
        mock_resp = _make_openai_response("ok")
        mock_resp.usage = None
        provider = OpenAIProvider(api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)
        result = await provider.complete(LLMRequest(prompt="hi"))

        assert result.usage == {}
        assert result.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_none_content_becomes_empty_string(self) -> None:
        mock_resp = _make_openai_response()
        mock_resp.choices[0].message.content = None
        provider = OpenAIProvider(api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)
        result = await provider.complete(LLMRequest(prompt="hi"))

        assert result.content == ""


# ---------------------------------------------------------------------------
# Cost estimation (static method)
# ---------------------------------------------------------------------------


class TestCostEstimation:
    def test_estimate_cost_gpt4o(self) -> None:
        cost = OpenAIProvider._estimate_cost("gpt-4o", 1000, 1000)
        # 1000/1000 * 0.0025 + 1000/1000 * 0.010 = 0.0125
        assert cost == pytest.approx(0.0125)

    def test_estimate_cost_gpt4o_mini(self) -> None:
        cost = OpenAIProvider._estimate_cost("gpt-4o-mini", 1000, 1000)
        # 1000/1000 * 0.000150 + 1000/1000 * 0.000600 = 0.00075
        assert cost == pytest.approx(0.00075)

    def test_estimate_cost_gpt41(self) -> None:
        cost = OpenAIProvider._estimate_cost("gpt-4.1", 1000, 1000)
        # 1000/1000 * 0.002 + 1000/1000 * 0.008 = 0.010
        assert cost == pytest.approx(0.010)

    def test_estimate_cost_gpt41_mini(self) -> None:
        cost = OpenAIProvider._estimate_cost("gpt-4.1-mini", 1000, 1000)
        # 1000/1000 * 0.0004 + 1000/1000 * 0.0016 = 0.002
        assert cost == pytest.approx(0.002)

    def test_estimate_cost_gpt41_nano(self) -> None:
        cost = OpenAIProvider._estimate_cost("gpt-4.1-nano", 1000, 1000)
        # 1000/1000 * 0.0001 + 1000/1000 * 0.0004 = 0.0005
        assert cost == pytest.approx(0.0005)

    def test_estimate_cost_unknown_model_returns_zero(self) -> None:
        cost = OpenAIProvider._estimate_cost("unknown-model", 5000, 3000)
        assert cost == 0.0

    def test_estimate_cost_zero_tokens(self) -> None:
        cost = OpenAIProvider._estimate_cost("gpt-4o", 0, 0)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


class TestRetry:
    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self) -> None:
        mock_resp = _make_openai_response("ok")
        provider = OpenAIProvider(max_retries=3, api_key="test-key")

        call_count = 0

        async def flaky_call(**kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")
            return mock_resp

        provider._client.chat.completions.create = flaky_call  # type: ignore[assignment]

        with patch("piano.llm.provider.asyncio.sleep", new_callable=AsyncMock):
            result = await provider.complete(LLMRequest(prompt="hi"))

        assert result.content == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_exhausted(self) -> None:
        provider = OpenAIProvider(max_retries=3, api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("permanent"),
        )

        with (
            patch("piano.llm.provider.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError, match="permanent"),
        ):
            await provider.complete(LLMRequest(prompt="hi"))

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self) -> None:
        provider = OpenAIProvider(max_retries=3, api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("fail"),
        )

        with (
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
        mock_resp = _make_openai_response("ok")
        provider = OpenAIProvider(cost_limit_usd=10.0, api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)
        await provider.complete(LLMRequest(prompt="hello"))
        await provider.complete(LLMRequest(prompt="world"))

        assert provider.total_cost_usd > 0.0
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_cost_limit_exceeded_raises(self) -> None:
        """Should raise CostLimitExceededError when limit is hit."""
        # Use a very large response to push cost up quickly
        mock_resp = _make_openai_response(
            "ok", prompt_tokens=100000, completion_tokens=100000, total_tokens=200000
        )
        provider = OpenAIProvider(cost_limit_usd=0.001, api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)
        await provider.complete(LLMRequest(prompt="first"))

        # Next call should fail because cost exceeded
        with pytest.raises(CostLimitExceededError, match="Cost limit exceeded"):
            await provider.complete(LLMRequest(prompt="second"))

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_raises(self) -> None:
        """Should raise RateLimitExceededError when rate limit is exceeded."""
        mock_resp = _make_openai_response("ok")
        provider = OpenAIProvider(calls_per_minute_limit=2, api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)
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
        mock_resp = _make_openai_response("ok")
        provider = OpenAIProvider(cost_limit_usd=10.0, api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)
        await provider.complete(LLMRequest(prompt="hello"))
        await provider.complete(LLMRequest(prompt="world"))

        assert provider.total_cost_usd > 0.0
        assert provider.call_count == 2

        provider.reset_cost_tracking()

        assert provider.total_cost_usd == 0.0
        assert provider.call_count == 0

    @pytest.mark.asyncio
    async def test_cost_properties(self) -> None:
        """total_cost_usd and call_count should be accessible."""
        provider = OpenAIProvider(api_key="test-key")

        # Initial state
        assert provider.total_cost_usd == 0.0
        assert provider.call_count == 0

        mock_resp = _make_openai_response("ok")
        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)
        await provider.complete(LLMRequest(prompt="test"))

        # After one call
        assert provider.total_cost_usd >= 0.0
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_sliding_window(self) -> None:
        """Rate limit should use a sliding window (timestamps older than 60s don't count)."""
        mock_resp = _make_openai_response("ok")
        provider = OpenAIProvider(calls_per_minute_limit=2, api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        with patch("piano.llm.provider.time.monotonic") as mock_time:
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
        provider = OpenAIProvider(max_retries=1, cost_limit_usd=10.0, api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("fail"),
        )

        with pytest.raises(RuntimeError):
            await provider.complete(LLMRequest(prompt="fail"))

        # Cost and count should remain at 0
        assert provider.total_cost_usd == 0.0
        assert provider.call_count == 0

    @pytest.mark.asyncio
    async def test_rate_limit_timestamp_recorded_after_success(self) -> None:
        """Rate limit timestamp should only be recorded after a successful call."""
        mock_resp = _make_openai_response("ok")
        provider = OpenAIProvider(calls_per_minute_limit=10, api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)
        await provider.complete(LLMRequest(prompt="hello"))

        # Timestamp should be recorded
        assert len(provider._call_timestamps) == 1

    @pytest.mark.asyncio
    async def test_rate_limit_timestamp_not_recorded_on_failure(self) -> None:
        """Rate limit timestamp should NOT be recorded when the call fails."""
        provider = OpenAIProvider(max_retries=1, calls_per_minute_limit=10, api_key="test-key")

        provider._client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("fail"),
        )

        with pytest.raises(RuntimeError):
            await provider.complete(LLMRequest(prompt="fail"))

        # No timestamp recorded since call failed
        assert len(provider._call_timestamps) == 0


# ---------------------------------------------------------------------------
# Non-transient error handling
# ---------------------------------------------------------------------------


class TestNonTransientErrors:
    @pytest.mark.asyncio
    async def test_authentication_error_not_retried(self) -> None:
        """AuthenticationError should not be retried."""
        provider = OpenAIProvider(max_retries=3, api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        exc = OpenAIAuthenticationError(
            message="Invalid API key",
            response=mock_response,
            body=None,
        )
        provider._client.chat.completions.create = AsyncMock(side_effect=exc)

        with (
            patch("piano.llm.provider.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            pytest.raises(OpenAIAuthenticationError),
        ):
            await provider.complete(LLMRequest(prompt="hi"))

        # Sleep should never have been called (no retry)
        assert mock_sleep.call_count == 0

    @pytest.mark.asyncio
    async def test_bad_request_error_not_retried(self) -> None:
        """BadRequestError should not be retried."""
        provider = OpenAIProvider(max_retries=3, api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {}
        exc = OpenAIBadRequestError(
            message="Bad request",
            response=mock_response,
            body=None,
        )
        provider._client.chat.completions.create = AsyncMock(side_effect=exc)

        with (
            patch("piano.llm.provider.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            pytest.raises(OpenAIBadRequestError),
        ):
            await provider.complete(LLMRequest(prompt="hi"))

        # Sleep should never have been called (no retry)
        assert mock_sleep.call_count == 0

    @pytest.mark.asyncio
    async def test_transient_error_still_retried(self) -> None:
        """RuntimeError (transient) should still be retried."""
        mock_resp = _make_openai_response("ok")
        provider = OpenAIProvider(max_retries=3, api_key="test-key")

        call_count = 0

        async def fail_then_succeed(**kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("transient")
            return mock_resp

        provider._client.chat.completions.create = fail_then_succeed  # type: ignore[assignment]

        with patch("piano.llm.provider.asyncio.sleep", new_callable=AsyncMock):
            result = await provider.complete(LLMRequest(prompt="hi"))

        assert result.content == "ok"
        assert call_count == 2
