"""Integration test fixtures for PIANO tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from piano.core.scheduler import ModuleScheduler
from piano.core.types import AgentId, BridgeEvent, ModuleTier

from ..helpers import DummyModule, InMemorySAS


@pytest.fixture
def integration_sas() -> InMemorySAS:
    """In-memory SAS for integration tests."""
    return InMemorySAS(agent_id="integration-agent-001")


@pytest.fixture
def fast_scheduler() -> ModuleScheduler:
    """Scheduler with short tick interval for fast test execution."""
    return ModuleScheduler(tick_interval=0.05)


@pytest.fixture
def integration_agent_id() -> AgentId:
    """Agent ID for integration tests."""
    return "integration-agent-001"


@pytest.fixture
def fast_module() -> DummyModule:
    """FAST tier dummy module."""
    return DummyModule(module_name="fast_module", tier=ModuleTier.FAST)


@pytest.fixture
def mid_module() -> DummyModule:
    """MID tier dummy module."""
    return DummyModule(module_name="mid_module", tier=ModuleTier.MID)


@pytest.fixture
def slow_module() -> DummyModule:
    """SLOW tier dummy module."""
    return DummyModule(module_name="slow_module", tier=ModuleTier.SLOW)


# --- Simulation integration fixtures ---


@pytest.fixture
def mock_bridge() -> AsyncMock:
    """Create a mock BridgeClient."""
    bridge = AsyncMock()
    bridge.connect = AsyncMock()
    bridge.disconnect = AsyncMock()
    bridge.ping = AsyncMock(return_value=True)
    bridge.send_command = AsyncMock(return_value={"success": True, "data": {}})
    bridge.chat = AsyncMock(return_value={"success": True})
    bridge.start_event_listener = AsyncMock()
    bridge.status = "connected"
    return bridge


@pytest.fixture
def perception_event() -> BridgeEvent:
    """A sample perception event."""
    return BridgeEvent(
        event_type="perception",
        data={
            "position": {"x": 10.0, "y": 64.0, "z": -5.0},
            "health": 18.0,
            "food": 15.0,
            "nearby_players": [
                {"name": "PlayerA", "distance": 5, "position": {"x": 12, "y": 64, "z": -3}}
            ],
            "time_of_day": 6000,
            "is_raining": False,
        },
    )


@pytest.fixture
def chat_event() -> BridgeEvent:
    """A sample chat event."""
    return BridgeEvent(
        event_type="chat",
        data={"username": "PlayerA", "message": "Hello there!"},
    )


@pytest.fixture
def death_event() -> BridgeEvent:
    """A sample death event."""
    return BridgeEvent(
        event_type="death",
        data={},
    )
