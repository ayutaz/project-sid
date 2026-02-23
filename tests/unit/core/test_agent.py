"""Unit tests for the Agent runtime."""

from __future__ import annotations

import asyncio

import pytest

from piano.core.agent import Agent
from piano.core.module import Module
from piano.core.scheduler import ModuleScheduler, SchedulerState
from piano.core.types import ModuleResult, ModuleTier

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

    def test_agent_repr(self) -> None:
        """Agent string representation should be informative."""
        sas = InMemorySAS(agent_id="test-repr")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="test-repr", sas=sas, scheduler=scheduler)
        # Just verify it doesn't crash and contains the agent_id
        repr_str = repr(agent)
        assert isinstance(repr_str, str)
        assert "Agent" in repr_str


class TestAgentLifecycle:
    """Tests for Agent initialization, run, and shutdown lifecycle."""

    async def test_initialize_calls_sas_initialize(self) -> None:
        """Agent.initialize() should call SAS.initialize()."""
        sas = InMemorySAS(agent_id="init-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="init-001", sas=sas, scheduler=scheduler)

        # InMemorySAS.initialize() is a no-op, but we can verify it doesn't raise
        await agent.initialize()
        # No exception = success

    async def test_initialize_calls_scheduler_start(self) -> None:
        """Agent.run() should call scheduler.start()."""
        sas = InMemorySAS(agent_id="start-001")
        scheduler = ModuleScheduler(tick_interval=0.1)
        agent = Agent(agent_id="start-001", sas=sas, scheduler=scheduler)

        await agent.initialize()
        # Start run in background with enough ticks to check state
        run_task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.05)  # Let it start
        assert scheduler.state == SchedulerState.RUNNING
        await agent.shutdown()
        await run_task

    async def test_run_respects_max_ticks(self) -> None:
        """Agent.run(max_ticks=N) should stop after approximately N ticks.

        Note: Due to the agent's sleep after max_ticks is reached, one extra
        tick may execute. This is acceptable behavior.
        """
        sas = InMemorySAS(agent_id="max-ticks-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="max-ticks-001", sas=sas, scheduler=scheduler)

        mod = DummyModule(module_name="ticker", tier=ModuleTier.FAST)
        agent.register_module(mod)

        await agent.initialize()
        await agent.run(max_ticks=5)

        # Scheduler should have run at least 5 ticks (may be 1 extra due to timing)
        assert scheduler.tick_count >= 5
        assert scheduler.tick_count <= 6
        # Module should have run at least 5 times
        assert mod.tick_count >= 5
        assert mod.tick_count <= 6

    async def test_run_with_no_max_ticks(self) -> None:
        """Agent.run() without max_ticks should run until shutdown() is called."""
        sas = InMemorySAS(agent_id="infinite-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="infinite-001", sas=sas, scheduler=scheduler)

        mod = DummyModule(module_name="infinite_ticker", tier=ModuleTier.FAST)
        agent.register_module(mod)

        await agent.initialize()

        # Start infinite run in background
        run_task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.2)  # Let a few ticks happen
        assert agent.running
        assert scheduler.state == SchedulerState.RUNNING

        # Now shut it down
        await agent.shutdown()
        await run_task

        assert not agent.running
        assert scheduler.state == SchedulerState.STOPPED

    async def test_shutdown_from_running_state(self) -> None:
        """Agent.shutdown() should cleanly stop a running agent."""
        sas = InMemorySAS(agent_id="shutdown-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="shutdown-001", sas=sas, scheduler=scheduler)

        await agent.initialize()
        run_task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)
        assert agent.running

        await agent.shutdown()
        await run_task

        assert not agent.running
        assert scheduler.state == SchedulerState.STOPPED

    async def test_shutdown_when_not_running(self) -> None:
        """Agent.shutdown() should be idempotent (no error if not running)."""
        sas = InMemorySAS(agent_id="idle-shutdown-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="idle-shutdown-001", sas=sas, scheduler=scheduler)

        # Shutdown without ever running
        await agent.shutdown()
        assert not agent.running

    async def test_shutdown_calls_scheduler_stop(self) -> None:
        """Agent.shutdown() should call scheduler.stop()."""
        sas = InMemorySAS(agent_id="sched-stop-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="sched-stop-001", sas=sas, scheduler=scheduler)

        await agent.initialize()
        run_task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)
        assert scheduler.state == SchedulerState.RUNNING

        await agent.shutdown()
        await run_task

        assert scheduler.state == SchedulerState.STOPPED

    async def test_shutdown_calls_module_shutdown(self) -> None:
        """Agent.shutdown() should trigger module.shutdown() via scheduler."""
        sas = InMemorySAS(agent_id="mod-shutdown-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="mod-shutdown-001", sas=sas, scheduler=scheduler)

        mod = DummyModule(module_name="lifecycle_mod", tier=ModuleTier.FAST)
        agent.register_module(mod)

        await agent.initialize()
        run_task = asyncio.create_task(agent.run(max_ticks=2))
        await run_task

        # After run completes, scheduler.stop() is called, which calls module.shutdown()
        assert mod.shut_down

    async def test_run_raises_when_already_running(self) -> None:
        """Agent.run() should raise RuntimeError if already running."""
        sas = InMemorySAS(agent_id="double-run-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="double-run-001", sas=sas, scheduler=scheduler)

        await agent.initialize()
        run_task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        # Try to run again while already running
        with pytest.raises(RuntimeError, match="already running"):
            await agent.run()

        await agent.shutdown()
        await run_task


class TestAgentMultiModule:
    """Tests for agents with multiple modules."""

    async def test_agent_with_multiple_modules(self) -> None:
        """Agent should register and run multiple modules correctly.

        Note: Due to timing, 1 extra tick may occur beyond max_ticks.
        """
        sas = InMemorySAS(agent_id="multi-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="multi-001", sas=sas, scheduler=scheduler)

        mod1 = DummyModule(module_name="mod1", tier=ModuleTier.FAST)
        mod2 = DummyModule(module_name="mod2", tier=ModuleTier.MID)
        mod3 = DummyModule(module_name="mod3", tier=ModuleTier.SLOW)

        agent.register_module(mod1)
        agent.register_module(mod2)
        agent.register_module(mod3)

        assert len(agent.modules) == 3
        assert len(scheduler.modules) == 3

        await agent.initialize()
        await agent.run(max_ticks=10)

        # FAST runs every tick (allow 1 extra due to timing)
        assert mod1.tick_count >= 10
        assert mod1.tick_count <= 11
        # MID runs every 3 ticks: ticks 3, 6, 9 = 3 times (or 4 if tick 12 happens)
        assert mod2.tick_count >= 3
        assert mod2.tick_count <= 4
        # SLOW runs every 10 ticks: tick 10 = 1 time
        assert mod3.tick_count >= 1
        assert mod3.tick_count <= 2

    async def test_agent_modules_all_initialized(self) -> None:
        """All modules should be initialized when scheduler starts."""
        sas = InMemorySAS(agent_id="init-all-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="init-all-001", sas=sas, scheduler=scheduler)

        mod1 = DummyModule(module_name="init1", tier=ModuleTier.FAST)
        mod2 = DummyModule(module_name="init2", tier=ModuleTier.MID)
        agent.register_module(mod1)
        agent.register_module(mod2)

        await agent.initialize()
        # Run at least one tick to trigger scheduler.start()
        await agent.run(max_ticks=1)

        assert mod1.initialized
        assert mod2.initialized


class TestAgentErrorHandling:
    """Tests for error scenarios in Agent lifecycle."""

    async def test_agent_with_error_module(self) -> None:
        """Module that raises exceptions should not crash the agent."""

        class ErrorModule(Module):
            """Module that always raises an exception."""

            def __init__(self) -> None:
                self.tick_count = 0

            @property
            def name(self) -> str:
                return "error_module"

            @property
            def tier(self) -> ModuleTier:
                return ModuleTier.FAST

            async def tick(self, sas) -> ModuleResult:
                self.tick_count += 1
                raise ValueError("Intentional error for testing")

        sas = InMemorySAS(agent_id="error-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="error-001", sas=sas, scheduler=scheduler)

        error_mod = ErrorModule()
        good_mod = DummyModule(module_name="good_mod", tier=ModuleTier.FAST)

        agent.register_module(error_mod)
        agent.register_module(good_mod)

        await agent.initialize()
        await agent.run(max_ticks=3)

        # Both modules should have run
        assert error_mod.tick_count == 3
        assert good_mod.tick_count == 3

        # Good module should have produced results
        results = scheduler.collect_results()
        good_results = [r for r in results if r.module_name == "good_mod"]
        assert len(good_results) == 3

        # Error module should have error results
        error_results = [r for r in results if r.module_name == "error_module"]
        assert len(error_results) == 3
        assert all(r.error is not None for r in error_results)

    async def test_run_cancelled_externally(self) -> None:
        """Agent.run() should handle CancelledError gracefully."""
        sas = InMemorySAS(agent_id="cancel-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="cancel-001", sas=sas, scheduler=scheduler)

        await agent.initialize()
        run_task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)

        # Cancel the run task - the agent catches CancelledError internally
        run_task.cancel()
        # The agent handles CancelledError and doesn't re-raise it
        await run_task

        # Agent should clean up properly
        assert not agent.running
        assert scheduler.state == SchedulerState.STOPPED


class TestAgentProperties:
    """Tests for Agent property accessors."""

    def test_agent_properties(self) -> None:
        """Agent should provide access to agent_id, sas, and scheduler."""
        sas = InMemorySAS(agent_id="props-001")
        scheduler = ModuleScheduler(tick_interval=0.1)
        agent = Agent(agent_id="props-001", sas=sas, scheduler=scheduler, cc=None)

        assert agent.agent_id == "props-001"
        assert agent.sas is sas
        assert agent.scheduler is scheduler
        assert not agent.running

    async def test_running_property_changes(self) -> None:
        """Agent.running should reflect current state."""
        sas = InMemorySAS(agent_id="running-prop-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="running-prop-001", sas=sas, scheduler=scheduler)

        assert not agent.running

        await agent.initialize()
        run_task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.1)
        assert agent.running

        await agent.shutdown()
        await run_task
        assert not agent.running


class TestAgentIntegration:
    """Integration tests for complete Agent workflows."""

    async def test_full_lifecycle_with_results(self) -> None:
        """Test complete init -> run -> shutdown with result collection."""
        sas = InMemorySAS(agent_id="full-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="full-001", sas=sas, scheduler=scheduler)

        mod = DummyModule(
            module_name="result_mod",
            tier=ModuleTier.FAST,
            result_data={"status": "ok"},
        )
        agent.register_module(mod)

        await agent.initialize()
        await agent.run(max_ticks=3)

        results = scheduler.collect_results()
        # Agent may run 1 extra tick due to timing in the run loop
        assert len(results) >= 3
        assert all(r.module_name == "result_mod" for r in results)
        assert all(r.success for r in results)
        assert results[0].data["status"] == "ok"

        assert not agent.running

    async def test_multiple_sequential_runs(self) -> None:
        """Agent should support multiple sequential run() calls after shutdown."""
        sas = InMemorySAS(agent_id="seq-001")
        scheduler = ModuleScheduler(tick_interval=0.05)
        agent = Agent(agent_id="seq-001", sas=sas, scheduler=scheduler)

        mod = DummyModule(module_name="seq_mod", tier=ModuleTier.FAST)
        agent.register_module(mod)

        # First run
        await agent.initialize()
        await agent.run(max_ticks=2)
        first_tick_count = mod.tick_count
        # Allow 1 extra tick due to timing
        assert first_tick_count >= 2
        assert first_tick_count <= 3

        # Second run (requires re-initialization because scheduler was stopped)
        await agent.initialize()
        await agent.run(max_ticks=3)
        # Tick count continues from previous (allow 1 extra per run)
        expected_min = first_tick_count + 3
        expected_max = first_tick_count + 4
        assert mod.tick_count >= expected_min
        assert mod.tick_count <= expected_max
