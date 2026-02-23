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
