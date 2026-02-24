"""Integration tests for perception event flow."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from piano.bridge.perception import BridgePerceptionModule
from piano.main import _create_sas


@pytest.fixture
def sas():
    return _create_sas("test-agent", sas_backend="memory")


@pytest.fixture
def mock_bridge():
    bridge = AsyncMock()
    bridge.start_event_listener = AsyncMock()
    return bridge


class TestPerceptionFlow:
    async def test_perception_event_updates_sas_position(self, sas, mock_bridge, perception_event):
        """Perception event updates SAS position via BridgePerceptionModule."""
        module = BridgePerceptionModule(mock_bridge)
        await module.initialize()

        # Simulate receiving an event
        await module._on_event(perception_event)
        result = await module.tick(sas)

        percepts = await sas.get_percepts()
        assert percepts.position["x"] == 10.0
        assert percepts.health == 18.0
        assert result.data["events_processed"] == 1

    async def test_chat_event_adds_to_messages(self, sas, mock_bridge, chat_event):
        """Chat event adds message to SAS percepts."""
        module = BridgePerceptionModule(mock_bridge)
        await module._on_event(chat_event)
        await module.tick(sas)

        percepts = await sas.get_percepts()
        assert len(percepts.chat_messages) == 1
        assert percepts.chat_messages[0]["username"] == "PlayerA"

    async def test_death_event_sets_health_zero(self, sas, mock_bridge, death_event):
        """Death event sets health to 0."""
        module = BridgePerceptionModule(mock_bridge)
        await module._on_event(death_event)
        await module.tick(sas)

        percepts = await sas.get_percepts()
        assert percepts.health == 0.0

    async def test_multiple_events_in_one_tick(
        self,
        sas,
        mock_bridge,
        perception_event,
        chat_event,
    ):
        """Multiple events processed in a single tick."""
        module = BridgePerceptionModule(mock_bridge)
        await module._on_event(perception_event)
        await module._on_event(chat_event)

        result = await module.tick(sas)
        assert result.data["events_processed"] == 2

        percepts = await sas.get_percepts()
        assert percepts.position["x"] == 10.0
        assert len(percepts.chat_messages) == 1

    async def test_scheduler_runs_perception_module(self, sas, mock_bridge, perception_event):
        """Perception module works within the scheduler tick loop."""
        import asyncio

        from piano.core.scheduler import ModuleScheduler

        module = BridgePerceptionModule(mock_bridge)
        scheduler = ModuleScheduler(tick_interval=0.01)
        scheduler.register(module)

        await module._on_event(perception_event)

        # Start the scheduler which will run module ticks
        await scheduler.start(sas)

        # Poll for the module result instead of fixed sleep
        for _ in range(30):
            result = scheduler.get_latest_result("bridge_perception")
            if result is not None:
                break
            await asyncio.sleep(0.02)

        await scheduler.stop()

        # Verify the module was executed via scheduler
        result = scheduler.get_latest_result("bridge_perception")
        assert result is not None
        assert result.module_name == "bridge_perception"

    async def test_empty_tick_is_noop(self, sas, mock_bridge):
        """Tick with no events does nothing."""
        module = BridgePerceptionModule(mock_bridge)
        result = await module.tick(sas)
        assert result.data["events_processed"] == 0
