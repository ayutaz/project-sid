"""Tests for LiteLLMProvider.

All tests mock litellm.acompletion to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from piano.core.types import LLMRequest, ModuleTier
from piano.llm.provider import DEFAULT_MODELS, LiteLLMProvider, LLMProvider

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
