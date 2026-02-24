"""Tests for BridgePerceptionModule - converts Bridge events to SAS PerceptData."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from piano.bridge.perception import _MAX_CHAT_MESSAGES, BridgePerceptionModule
from piano.core.types import BridgeEvent, ModuleTier
from tests.helpers import InMemorySAS


def _make_bridge() -> MagicMock:
    """Create a mock BridgeClient."""
    bridge = MagicMock()
    bridge.start_event_listener = AsyncMock()
    bridge.stop_event_listener = AsyncMock()
    return bridge


def _make_event(event_type: str, data: dict | None = None) -> BridgeEvent:
    return BridgeEvent(event_type=event_type, data=data or {})


@pytest.fixture
def bridge() -> MagicMock:
    return _make_bridge()


@pytest.fixture
def module(bridge: MagicMock) -> BridgePerceptionModule:
    return BridgePerceptionModule(bridge)


@pytest.fixture
def sas() -> InMemorySAS:
    return InMemorySAS()


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


async def test_name_and_tier(module: BridgePerceptionModule) -> None:
    assert module.name == "bridge_perception"
    assert module.tier == ModuleTier.FAST


async def test_initialize_starts_listener(
    module: BridgePerceptionModule, bridge: MagicMock
) -> None:
    await module.initialize()
    bridge.start_event_listener.assert_awaited_once()
    # Second call should be no-op
    await module.initialize()
    assert bridge.start_event_listener.await_count == 1


# ---------------------------------------------------------------------------
# Empty buffer
# ---------------------------------------------------------------------------


async def test_empty_buffer_noop(module: BridgePerceptionModule, sas: InMemorySAS) -> None:
    result = await module.tick(sas)
    assert result.data["events_processed"] == 0
    percepts = await sas.get_percepts()
    assert percepts.health == 20.0  # unchanged default


# ---------------------------------------------------------------------------
# Perception events
# ---------------------------------------------------------------------------


async def test_perception_event_updates_position(
    module: BridgePerceptionModule, sas: InMemorySAS
) -> None:
    await module._on_event(
        _make_event("perception", {"position": {"x": 10.0, "y": 64.0, "z": -5.0}})
    )
    result = await module.tick(sas)
    assert result.data["events_processed"] == 1
    percepts = await sas.get_percepts()
    assert percepts.position == {"x": 10.0, "y": 64.0, "z": -5.0}


async def test_perception_event_updates_health(
    module: BridgePerceptionModule, sas: InMemorySAS
) -> None:
    await module._on_event(_make_event("perception", {"health": 15, "food": 18}))
    await module.tick(sas)
    percepts = await sas.get_percepts()
    assert percepts.health == 15.0
    assert percepts.hunger == 18.0


async def test_perception_event_updates_players(
    module: BridgePerceptionModule, sas: InMemorySAS
) -> None:
    await module._on_event(
        _make_event(
            "perception",
            {"nearby_players": [{"name": "Alice"}, "Bob"]},
        )
    )
    await module.tick(sas)
    percepts = await sas.get_percepts()
    assert percepts.nearby_players == ["Alice", "Bob"]


async def test_perception_event_updates_weather(
    module: BridgePerceptionModule, sas: InMemorySAS
) -> None:
    await module._on_event(_make_event("perception", {"is_raining": True, "time_of_day": 6000}))
    await module.tick(sas)
    percepts = await sas.get_percepts()
    assert percepts.weather == "rain"
    assert percepts.time_of_day == 6000


async def test_perception_event_updates_inventory_list(
    module: BridgePerceptionModule, sas: InMemorySAS
) -> None:
    await module._on_event(
        _make_event(
            "perception",
            {
                "inventory": [
                    {"name": "diamond", "count": 3},
                    {"name": "stick", "count": 10},
                ]
            },
        )
    )
    await module.tick(sas)
    percepts = await sas.get_percepts()
    assert percepts.inventory == {"diamond": 3, "stick": 10}


async def test_perception_event_updates_inventory_dict(
    module: BridgePerceptionModule, sas: InMemorySAS
) -> None:
    await module._on_event(_make_event("perception", {"inventory": {"iron_ingot": 5}}))
    await module.tick(sas)
    percepts = await sas.get_percepts()
    assert percepts.inventory == {"iron_ingot": 5}


# ---------------------------------------------------------------------------
# Chat events
# ---------------------------------------------------------------------------


async def test_chat_event_adds_message(module: BridgePerceptionModule, sas: InMemorySAS) -> None:
    await module._on_event(_make_event("chat", {"username": "Steve", "message": "Hello!"}))
    await module.tick(sas)
    percepts = await sas.get_percepts()
    assert len(percepts.chat_messages) == 1
    assert percepts.chat_messages[0]["username"] == "Steve"
    assert percepts.chat_messages[0]["message"] == "Hello!"


async def test_chat_messages_capped(module: BridgePerceptionModule, sas: InMemorySAS) -> None:
    for i in range(_MAX_CHAT_MESSAGES + 5):
        await module._on_event(_make_event("chat", {"username": f"user{i}", "message": f"msg{i}"}))
    await module.tick(sas)
    percepts = await sas.get_percepts()
    assert len(percepts.chat_messages) == _MAX_CHAT_MESSAGES
    # Oldest messages trimmed - first message should be user5
    assert percepts.chat_messages[0]["username"] == "user5"


# ---------------------------------------------------------------------------
# Death event
# ---------------------------------------------------------------------------


async def test_death_event_sets_health_zero(
    module: BridgePerceptionModule, sas: InMemorySAS
) -> None:
    await module._on_event(_make_event("death"))
    await module.tick(sas)
    percepts = await sas.get_percepts()
    assert percepts.health == 0.0


# ---------------------------------------------------------------------------
# Multiple events & error handling
# ---------------------------------------------------------------------------


async def test_multiple_events_drained(module: BridgePerceptionModule, sas: InMemorySAS) -> None:
    await module._on_event(_make_event("perception", {"health": 10}))
    await module._on_event(_make_event("chat", {"username": "Alex", "message": "Hi"}))
    await module._on_event(_make_event("death"))
    result = await module.tick(sas)
    assert result.data["events_processed"] == 3
    percepts = await sas.get_percepts()
    # Death was last, so health should be 0
    assert percepts.health == 0.0
    assert len(percepts.chat_messages) == 1


async def test_malformed_event_handled(module: BridgePerceptionModule, sas: InMemorySAS) -> None:
    """Bad data in an event should not crash the module."""
    # nearby_players expects iterable but gets an int
    await module._on_event(_make_event("perception", {"nearby_players": 42}))
    # Should not raise
    result = await module.tick(sas)
    assert result.data["events_processed"] == 1


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


async def test_buffer_overflow_drops_oldest_events(
    module: BridgePerceptionModule, sas: InMemorySAS
) -> None:
    """When >100 events arrive before tick, only the latest 100 are kept."""
    for i in range(120):
        await module._on_event(_make_event("perception", {"health": float(i)}))
    # deque(maxlen=100) keeps only the latest 100 items
    assert len(module._buffer) == 100

    result = await module.tick(sas)
    assert result.data["events_processed"] == 100

    # Last event had health=119, so that's what SAS should have
    percepts = await sas.get_percepts()
    assert percepts.health == 119.0


async def test_unknown_event_type_ignored(module: BridgePerceptionModule, sas: InMemorySAS) -> None:
    """Unknown event types (e.g. 'spawn') are drained but don't change percepts."""
    original_percepts = await sas.get_percepts()
    original_health = original_percepts.health

    await module._on_event(_make_event("spawn", {"entity": "zombie"}))
    result = await module.tick(sas)

    # Event was processed (counted) but percepts unchanged
    assert result.data["events_processed"] == 1
    percepts = await sas.get_percepts()
    assert percepts.health == original_health


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


async def test_position_validation_missing_keys(
    module: BridgePerceptionModule, sas: InMemorySAS
) -> None:
    """Position dict without x/y/z keys is rejected."""
    original = await sas.get_percepts()
    original_pos = original.position

    # Missing 'z' key
    await module._on_event(
        _make_event("perception", {"position": {"x": 1.0, "y": 2.0}})
    )
    await module.tick(sas)
    percepts = await sas.get_percepts()
    assert percepts.position == original_pos  # unchanged


async def test_position_validation_not_dict(
    module: BridgePerceptionModule, sas: InMemorySAS
) -> None:
    """Position that is not a dict is rejected."""
    original = await sas.get_percepts()
    original_pos = original.position

    await module._on_event(
        _make_event("perception", {"position": [1.0, 2.0, 3.0]})
    )
    await module.tick(sas)
    percepts = await sas.get_percepts()
    assert percepts.position == original_pos  # unchanged


async def test_shutdown_resets_listener_flag(
    module: BridgePerceptionModule, bridge: MagicMock
) -> None:
    await module.initialize()
    assert module._listener_started is True
    await module.shutdown()
    assert module._listener_started is False

    # Verify stop_event_listener was called on the bridge
    bridge.stop_event_listener.assert_awaited_once()
