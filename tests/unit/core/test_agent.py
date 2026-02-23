"""Unit tests for the Agent runtime."""

from __future__ import annotations

from piano.core.agent import Agent
from piano.core.scheduler import ModuleScheduler
from piano.core.types import ModuleTier

from ...helpers import DummyModule, InMemorySAS


class TestAgentUnit:
    """Unit tests for Agent properties and registration."""

    def test_agent_id(self) -> None:
        """Agent should expose the configured agent_id."""
        sas = InMemorySAS(agent_id="unit-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="unit-001", sas=sas, scheduler=scheduler)
        assert agent.agent_id == "unit-001"

    def test_running_initially_false(self) -> None:
        """Agent.running should be False before run() is called."""
        sas = InMemorySAS()
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="unit-001", sas=sas, scheduler=scheduler)
        assert not agent.running

    def test_register_module_adds_to_list(self) -> None:
        """register_module should add the module to both agent and scheduler."""
        sas = InMemorySAS()
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="unit-001", sas=sas, scheduler=scheduler)

        mod = DummyModule(module_name="test_mod", tier=ModuleTier.FAST)
        agent.register_module(mod)

        assert len(agent.modules) == 1
        assert agent.modules[0].name == "test_mod"
        assert "test_mod" in scheduler.modules

    def test_sas_property(self) -> None:
        """Agent.sas should return the configured SAS instance."""
        sas = InMemorySAS(agent_id="unit-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="unit-001", sas=sas, scheduler=scheduler)
        assert agent.sas is sas

    def test_scheduler_property(self) -> None:
        """Agent.scheduler should return the configured scheduler."""
        sas = InMemorySAS()
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="unit-001", sas=sas, scheduler=scheduler)
        assert agent.scheduler is scheduler

    def test_modules_returns_copy(self) -> None:
        """Agent.modules should return a copy, not the internal list."""
        sas = InMemorySAS()
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="unit-001", sas=sas, scheduler=scheduler)
        mod = DummyModule(module_name="copy_test", tier=ModuleTier.FAST)
        agent.register_module(mod)

        modules = agent.modules
        modules.clear()
        assert len(agent.modules) == 1  # Internal list not affected
