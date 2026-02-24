"""Tests for MockLLMProvider."""

from __future__ import annotations

import pytest

from piano.core.types import LLMRequest, ModuleTier
from piano.llm.mock import MockLLMProvider
from piano.llm.provider import LLMProvider


class TestMockBasic:
    def test_implements_llm_provider_protocol(self) -> None:
        mock = MockLLMProvider()
        assert isinstance(mock, LLMProvider)

    @pytest.mark.asyncio
    async def test_default_response(self) -> None:
        mock = MockLLMProvider()
        result = await mock.complete(LLMRequest(prompt="anything"))
        assert result.content == '{"action": "idle"}'
        assert result.model == "mock"
        assert result.latency_ms == 0.0
        assert result.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_custom_default_response(self) -> None:
        mock = MockLLMProvider()
        mock.set_default_response("custom default")
        result = await mock.complete(LLMRequest(prompt="anything"))
        assert result.content == "custom default"


class TestPatternMatching:
    @pytest.mark.asyncio
    async def test_pattern_match(self) -> None:
        mock = MockLLMProvider()
        mock.add_response("hello", "world")
        result = await mock.complete(LLMRequest(prompt="say hello please"))
        assert result.content == "world"

    @pytest.mark.asyncio
    async def test_first_match_wins(self) -> None:
        mock = MockLLMProvider()
        mock.add_response("greet", "first")
        mock.add_response("greet", "second")
        result = await mock.complete(LLMRequest(prompt="greet me"))
        assert result.content == "first"

    @pytest.mark.asyncio
    async def test_no_match_returns_default(self) -> None:
        mock = MockLLMProvider()
        mock.add_response("xyz", "matched")
        result = await mock.complete(LLMRequest(prompt="abc"))
        assert result.content == '{"action": "idle"}'

    @pytest.mark.asyncio
    async def test_multiple_patterns(self) -> None:
        mock = MockLLMProvider()
        mock.add_response("plan", "planning response")
        mock.add_response("goal", "goal response")
        r1 = await mock.complete(LLMRequest(prompt="make a plan"))
        r2 = await mock.complete(LLMRequest(prompt="set a goal"))
        assert r1.content == "planning response"
        assert r2.content == "goal response"


class TestCallHistory:
    @pytest.mark.asyncio
    async def test_records_calls(self) -> None:
        mock = MockLLMProvider()
        req = LLMRequest(prompt="test", tier=ModuleTier.FAST)
        await mock.complete(req)
        assert len(mock.call_history) == 1
        assert mock.call_history[0].prompt == "test"

    @pytest.mark.asyncio
    async def test_records_multiple_calls(self) -> None:
        mock = MockLLMProvider()
        await mock.complete(LLMRequest(prompt="first"))
        await mock.complete(LLMRequest(prompt="second"))
        await mock.complete(LLMRequest(prompt="third"))
        assert len(mock.call_history) == 3

    @pytest.mark.asyncio
    async def test_assert_called_with_success(self) -> None:
        mock = MockLLMProvider()
        await mock.complete(LLMRequest(prompt="hello world"))
        mock.assert_called_with("hello")

    @pytest.mark.asyncio
    async def test_assert_called_with_failure(self) -> None:
        mock = MockLLMProvider()
        await mock.complete(LLMRequest(prompt="hello world"))
        with pytest.raises(AssertionError, match="No call with pattern"):
            mock.assert_called_with("nonexistent")

    def test_assert_called_with_no_history(self) -> None:
        mock = MockLLMProvider()
        with pytest.raises(AssertionError):
            mock.assert_called_with("anything")


class TestDemoProvider:
    """Tests for MockLLMProvider.create_demo_provider()."""

    def test_create_demo_provider(self) -> None:
        provider = MockLLMProvider.create_demo_provider()
        assert provider._demo_mode is True
        assert len(provider._demo_responses) > 0

    @pytest.mark.asyncio
    async def test_demo_cycles_through_responses(self) -> None:
        provider = MockLLMProvider.create_demo_provider()
        responses = []
        for _ in range(len(provider._demo_responses)):
            result = await provider.complete(LLMRequest(prompt="test"))
            responses.append(result.content)

        # Should have gotten each demo response exactly once
        assert len(responses) == len(provider._demo_responses)
        for i, resp in enumerate(responses):
            assert resp == provider._demo_responses[i]

    @pytest.mark.asyncio
    async def test_demo_wraps_around(self) -> None:
        provider = MockLLMProvider.create_demo_provider()
        n = len(provider._demo_responses)
        # Exhaust all, then get first one again
        for _ in range(n):
            await provider.complete(LLMRequest(prompt="test"))
        result = await provider.complete(LLMRequest(prompt="test"))
        assert result.content == provider._demo_responses[0]

    @pytest.mark.asyncio
    async def test_demo_model_name(self) -> None:
        provider = MockLLMProvider.create_demo_provider()
        result = await provider.complete(LLMRequest(prompt="test"))
        assert result.model == "mock-demo"

    @pytest.mark.asyncio
    async def test_demo_records_call_history(self) -> None:
        provider = MockLLMProvider.create_demo_provider()
        await provider.complete(LLMRequest(prompt="hello"))
        assert len(provider.call_history) == 1
        assert provider.call_history[0].prompt == "hello"

    @pytest.mark.asyncio
    async def test_demo_responses_are_valid_json(self) -> None:
        import json

        provider = MockLLMProvider.create_demo_provider()
        for _ in range(len(provider._demo_responses)):
            result = await provider.complete(LLMRequest(prompt="test"))
            data = json.loads(result.content)
            assert "action" in data
            assert "action_params" in data
            assert "reasoning" in data

    @pytest.mark.asyncio
    async def test_non_demo_mode_unaffected(self) -> None:
        """Normal provider should not use demo responses."""
        provider = MockLLMProvider()
        result = await provider.complete(LLMRequest(prompt="test"))
        assert result.content == '{"action": "idle"}'
        assert result.model == "mock"
