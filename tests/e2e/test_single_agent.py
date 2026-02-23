"""E2E test: Single agent lifecycle.

Runs a single agent for a small number of ticks and verifies
that the basic simulation loop works end-to-end.
"""
from __future__ import annotations

import asyncio

import pytest

from piano.core.agent import Agent
from piano.core.scheduler import ModuleScheduler
from piano.core.types import ModuleTier
from tests.helpers import DummyModule, InMemorySAS


@pytest.mark.e2e
class TestSingleAgentE2E:
    """End-to-end tests for a single agent simulation."""

    async def test_agent_runs_10_ticks(
        self,
        e2e_sas: InMemorySAS,
        e2e_scheduler: ModuleScheduler,
        e2e_fast_module: DummyModule,
    ) -> None:
        """Agent should run 10 ticks and stop normally."""
        agent = Agent(
            agent_id="e2e-agent-001",
            sas=e2e_sas,
            scheduler=e2e_scheduler,
        )
        agent.register_module(e2e_fast_module)

        await agent.initialize()
        await agent.run(max_ticks=10)

        assert not agent.running
        assert e2e_scheduler.tick_count >= 10
        assert e2e_fast_module.tick_count >= 10

    async def test_agent_with_multiple_modules(
        self,
        e2e_sas: InMemorySAS,
        e2e_scheduler: ModuleScheduler,
    ) -> None:
        """Agent with multiple tier modules should run correctly."""
        fast_mod = DummyModule(module_name="e2e_fast", tier=ModuleTier.FAST)
        mid_mod = DummyModule(module_name="e2e_mid", tier=ModuleTier.MID)
        slow_mod = DummyModule(module_name="e2e_slow", tier=ModuleTier.SLOW)

        agent = Agent(
            agent_id="e2e-agent-001",
            sas=e2e_sas,
            scheduler=e2e_scheduler,
        )
        agent.register_module(fast_mod)
        agent.register_module(mid_mod)
        agent.register_module(slow_mod)

        await agent.initialize()
        await agent.run(max_ticks=10)

        assert not agent.running
        # FAST runs every tick, MID every 3, SLOW every 10
        assert fast_mod.tick_count >= 10
        assert mid_mod.tick_count >= 1
        assert slow_mod.tick_count >= 1

    async def test_agent_sas_has_data(
        self,
        e2e_sas: InMemorySAS,
        e2e_scheduler: ModuleScheduler,
        e2e_fast_module: DummyModule,
    ) -> None:
        """SAS should have been initialized after agent run."""
        agent = Agent(
            agent_id="e2e-agent-001",
            sas=e2e_sas,
            scheduler=e2e_scheduler,
        )
        agent.register_module(e2e_fast_module)

        await agent.initialize()
        await agent.run(max_ticks=5)

        # SAS was initialized
        snapshot = await e2e_sas.snapshot()
        assert "percepts" in snapshot
        assert "goals" in snapshot

    async def test_agent_graceful_shutdown(
        self,
        e2e_sas: InMemorySAS,
    ) -> None:
        """Agent should handle graceful shutdown during run."""
        scheduler = ModuleScheduler(tick_interval=0.02)
        fast_mod = DummyModule(module_name="e2e_fast", tier=ModuleTier.FAST)

        agent = Agent(
            agent_id="e2e-agent-001",
            sas=e2e_sas,
            scheduler=scheduler,
        )
        agent.register_module(fast_mod)
        await agent.initialize()

        # Start agent in background
        run_task = asyncio.create_task(agent.run())

        # Let it run for a bit
        await asyncio.sleep(0.1)

        # Shutdown
        await agent.shutdown()
        await run_task

        assert not agent.running
        assert fast_mod.tick_count > 0
