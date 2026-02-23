"""Integration test fixtures for PIANO tests."""

from __future__ import annotations

import pytest

from piano.core.scheduler import ModuleScheduler
from piano.core.types import AgentId, ModuleTier

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
