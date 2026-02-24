"""Tests for ChatBroadcaster module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from piano.bridge.chat_broadcaster import ChatBroadcaster
from piano.core.types import CCDecision, ModuleTier
from tests.helpers import InMemorySAS


@pytest.fixture
def mock_bridge() -> AsyncMock:
    bridge = AsyncMock()
    bridge.chat = AsyncMock(return_value={"success": True})
    return bridge


@pytest.fixture
def sas() -> InMemorySAS:
    return InMemorySAS()


def _make_talking_section(content: str = "Hello world") -> dict:
    return {"latest_utterance": {"content": content, "tone": "neutral"}}


class TestChatBroadcaster:
    def test_name_and_tier(self, mock_bridge: AsyncMock) -> None:
        cb = ChatBroadcaster(bridge=mock_bridge)
        assert cb.name == "chat_broadcaster"
        assert cb.tier == ModuleTier.FAST

    async def test_on_broadcast_with_speaking_sends_chat(
        self, mock_bridge: AsyncMock, sas: InMemorySAS
    ) -> None:
        await sas.update_section("talking", _make_talking_section("Hi there"))
        cb = ChatBroadcaster(bridge=mock_bridge, sas=sas)
        decision = CCDecision(speaking="greet nearby player")

        await cb.on_broadcast(decision)

        mock_bridge.chat.assert_awaited_once_with("Hi there")

    async def test_on_broadcast_no_speaking_noop(
        self, mock_bridge: AsyncMock, sas: InMemorySAS
    ) -> None:
        await sas.update_section("talking", _make_talking_section("Hi"))
        cb = ChatBroadcaster(bridge=mock_bridge, sas=sas)
        decision = CCDecision(speaking=None)

        await cb.on_broadcast(decision)

        mock_bridge.chat.assert_not_awaited()

    async def test_on_broadcast_no_sas_noop(self, mock_bridge: AsyncMock) -> None:
        cb = ChatBroadcaster(bridge=mock_bridge, sas=None)
        decision = CCDecision(speaking="say hello")

        await cb.on_broadcast(decision)

        mock_bridge.chat.assert_not_awaited()

    async def test_tick_sends_unsent_utterance(
        self, mock_bridge: AsyncMock, sas: InMemorySAS
    ) -> None:
        await sas.update_section("talking", _make_talking_section("tick msg"))
        cb = ChatBroadcaster(bridge=mock_bridge)

        result = await cb.tick(sas)

        mock_bridge.chat.assert_awaited_once_with("tick msg")
        assert result.module_name == "chat_broadcaster"
        assert result.data["sent_count"] == 1

    async def test_tick_clears_utterance_after_send(
        self, mock_bridge: AsyncMock, sas: InMemorySAS
    ) -> None:
        await sas.update_section("talking", _make_talking_section("will clear"))
        cb = ChatBroadcaster(bridge=mock_bridge)

        await cb.tick(sas)

        section = await sas.get_section("talking")
        assert section.get("latest_utterance") is None

    async def test_bridge_error_handled(self, mock_bridge: AsyncMock, sas: InMemorySAS) -> None:
        mock_bridge.chat.side_effect = ConnectionError("bridge down")
        await sas.update_section("talking", _make_talking_section("fail msg"))
        cb = ChatBroadcaster(bridge=mock_bridge)

        result = await cb.tick(sas)

        # Should not crash, sent_count stays 0
        assert result.data["sent_count"] == 0
        # Utterance should NOT be cleared on error
        section = await sas.get_section("talking")
        assert section["latest_utterance"]["content"] == "fail msg"

    async def test_sent_count_increments(self, mock_bridge: AsyncMock, sas: InMemorySAS) -> None:
        cb = ChatBroadcaster(bridge=mock_bridge)

        await sas.update_section("talking", _make_talking_section("msg1"))
        await cb.tick(sas)
        assert cb._sent_count == 1

        await sas.update_section("talking", _make_talking_section("msg2"))
        await cb.tick(sas)
        assert cb._sent_count == 2

    async def test_tick_no_utterance_noop(self, mock_bridge: AsyncMock, sas: InMemorySAS) -> None:
        cb = ChatBroadcaster(bridge=mock_bridge)

        result = await cb.tick(sas)

        mock_bridge.chat.assert_not_awaited()
        assert result.data["sent_count"] == 0

    async def test_concurrent_broadcast_and_tick_no_double_send(
        self, mock_bridge: AsyncMock, sas: InMemorySAS
    ) -> None:
        """Concurrent on_broadcast + tick should not double-send the same utterance."""
        await sas.update_section("talking", _make_talking_section("once only"))
        cb = ChatBroadcaster(bridge=mock_bridge, sas=sas)
        decision = CCDecision(speaking="say something")

        # Run on_broadcast and tick concurrently
        await asyncio.gather(
            cb.on_broadcast(decision),
            cb.tick(sas),
        )

        # The lock ensures only one path sends the message
        assert mock_bridge.chat.await_count == 1
