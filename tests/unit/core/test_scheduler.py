"""Tests for the ModuleScheduler."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from piano.core.module import Module
from piano.core.scheduler import ModuleScheduler, SchedulerState
from piano.core.types import ModuleResult, ModuleTier


# --- Dummy Module Implementations ---


class DummyFastModule(Module):
    """A FAST-tier module that counts its tick invocations."""

    def __init__(self, name: str = "fast_module") -> None:
        self._name = name
        self.tick_count = 0
        self.initialized = False
        self.shut_down = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> ModuleTier:
        return ModuleTier.FAST

    async def tick(self, sas: Any) -> ModuleResult:
        self.tick_count += 1
        return ModuleResult(module_name=self.name, tier=self.tier, data={"count": self.tick_count})

    async def initialize(self) -> None:
        self.initialized = True

    async def shutdown(self) -> None:
        self.shut_down = True


class DummyMidModule(Module):
    """A MID-tier module that counts ticks."""

    def __init__(self, name: str = "mid_module") -> None:
        self._name = name
        self.tick_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> ModuleTier:
        return ModuleTier.MID

    async def tick(self, sas: Any) -> ModuleResult:
        self.tick_count += 1
        return ModuleResult(module_name=self.name, tier=self.tier, data={"count": self.tick_count})


class DummySlowModule(Module):
    """A SLOW-tier module that counts ticks."""

    def __init__(self, name: str = "slow_module") -> None:
        self._name = name
        self.tick_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> ModuleTier:
        return ModuleTier.SLOW

    async def tick(self, sas: Any) -> ModuleResult:
        self.tick_count += 1
        return ModuleResult(module_name=self.name, tier=self.tier, data={"count": self.tick_count})


class ErrorModule(Module):
    """A module that always raises an exception."""

    @property
    def name(self) -> str:
        return "error_module"

    @property
    def tier(self) -> ModuleTier:
        return ModuleTier.FAST

    async def tick(self, sas: Any) -> ModuleResult:
        raise RuntimeError("Intentional test error")


class SlowTickModule(Module):
    """A FAST-tier module with a slow tick (for timing tests)."""

    def __init__(self) -> None:
        self.tick_count = 0

    @property
    def name(self) -> str:
        return "slow_tick_module"

    @property
    def tier(self) -> ModuleTier:
        return ModuleTier.FAST

    async def tick(self, sas: Any) -> ModuleResult:
        self.tick_count += 1
        await asyncio.sleep(0.01)
        return ModuleResult(module_name=self.name, tier=self.tier)


# --- Helper ---


async def run_scheduler_ticks(scheduler: ModuleScheduler, sas: Any, num_ticks: int) -> None:
    """Run the scheduler for a specific number of ticks, then stop."""
    await scheduler.start(sas)
    # Wait for enough ticks to accumulate
    while scheduler.tick_count < num_ticks:
        await asyncio.sleep(0.05)
    await scheduler.stop()


# --- Tests ---


class TestModuleRegistration:
    """Tests for module registration and unregistration."""

    def test_register_module(self) -> None:
        scheduler = ModuleScheduler()
        module = DummyFastModule()
        scheduler.register(module)
        assert "fast_module" in scheduler.modules
        assert scheduler.modules["fast_module"] is module

    def test_register_duplicate_raises(self) -> None:
        scheduler = ModuleScheduler()
        scheduler.register(DummyFastModule())
        with pytest.raises(ValueError, match="already registered"):
            scheduler.register(DummyFastModule())

    def test_unregister_module(self) -> None:
        scheduler = ModuleScheduler()
        module = DummyFastModule()
        scheduler.register(module)
        removed = scheduler.unregister("fast_module")
        assert removed is module
        assert "fast_module" not in scheduler.modules

    def test_unregister_nonexistent_raises(self) -> None:
        scheduler = ModuleScheduler()
        with pytest.raises(KeyError, match="not registered"):
            scheduler.unregister("nonexistent")

    def test_register_multiple_modules(self) -> None:
        scheduler = ModuleScheduler()
        fast = DummyFastModule()
        mid = DummyMidModule()
        slow = DummySlowModule()
        scheduler.register(fast)
        scheduler.register(mid)
        scheduler.register(slow)
        assert len(scheduler.modules) == 3


class TestLifecycle:
    """Tests for scheduler lifecycle (start/stop/pause/resume)."""

    async def test_initial_state_is_idle(self) -> None:
        scheduler = ModuleScheduler()
        assert scheduler.state == SchedulerState.IDLE

    async def test_start_sets_running(self) -> None:
        scheduler = ModuleScheduler(tick_interval=0.05)
        scheduler.register(DummyFastModule())
        await scheduler.start(None)
        assert scheduler.state == SchedulerState.RUNNING
        await scheduler.stop()

    async def test_stop_sets_stopped(self) -> None:
        scheduler = ModuleScheduler(tick_interval=0.05)
        scheduler.register(DummyFastModule())
        await scheduler.start(None)
        await scheduler.stop()
        assert scheduler.state == SchedulerState.STOPPED

    async def test_start_when_running_raises(self) -> None:
        scheduler = ModuleScheduler(tick_interval=0.05)
        await scheduler.start(None)
        with pytest.raises(RuntimeError, match="already running"):
            await scheduler.start(None)
        await scheduler.stop()

    async def test_stop_when_idle_raises(self) -> None:
        scheduler = ModuleScheduler()
        with pytest.raises(RuntimeError, match="Cannot stop"):
            await scheduler.stop()

    async def test_pause_and_resume(self) -> None:
        scheduler = ModuleScheduler(tick_interval=0.05)
        scheduler.register(DummyFastModule())
        await scheduler.start(None)
        scheduler.pause()
        assert scheduler.state == SchedulerState.PAUSED
        scheduler.resume()
        assert scheduler.state == SchedulerState.RUNNING
        await scheduler.stop()

    async def test_pause_when_not_running_raises(self) -> None:
        scheduler = ModuleScheduler()
        with pytest.raises(RuntimeError, match="Cannot pause"):
            scheduler.pause()

    async def test_resume_when_not_paused_raises(self) -> None:
        scheduler = ModuleScheduler(tick_interval=0.05)
        await scheduler.start(None)
        with pytest.raises(RuntimeError, match="Cannot resume"):
            scheduler.resume()
        await scheduler.stop()

    async def test_initialize_called_on_start(self) -> None:
        scheduler = ModuleScheduler(tick_interval=0.05)
        module = DummyFastModule()
        scheduler.register(module)
        assert not module.initialized
        await scheduler.start(None)
        assert module.initialized
        await scheduler.stop()

    async def test_shutdown_called_on_stop(self) -> None:
        scheduler = ModuleScheduler(tick_interval=0.05)
        module = DummyFastModule()
        scheduler.register(module)
        await scheduler.start(None)
        assert not module.shut_down
        await scheduler.stop()
        assert module.shut_down

    async def test_stop_from_paused(self) -> None:
        scheduler = ModuleScheduler(tick_interval=0.05)
        scheduler.register(DummyFastModule())
        await scheduler.start(None)
        scheduler.pause()
        await scheduler.stop()
        assert scheduler.state == SchedulerState.STOPPED


class TestTierExecution:
    """Tests for tier-based execution frequency."""

    async def test_fast_module_every_tick(self) -> None:
        scheduler = ModuleScheduler(tick_interval=0.02)
        fast = DummyFastModule()
        scheduler.register(fast)
        await run_scheduler_ticks(scheduler, None, 5)
        # FAST runs every tick, so should have >= 5 ticks
        assert fast.tick_count >= 5

    async def test_mid_module_frequency(self) -> None:
        """MID modules run every 3 ticks by default."""
        scheduler = ModuleScheduler(tick_interval=0.02)
        fast = DummyFastModule()
        mid = DummyMidModule()
        scheduler.register(fast)
        scheduler.register(mid)
        await run_scheduler_ticks(scheduler, None, 9)
        # In 9 ticks: MID fires at ticks 3, 6, 9 = 3 times
        assert mid.tick_count == 3
        assert fast.tick_count == 9

    async def test_slow_module_frequency(self) -> None:
        """SLOW modules run every 10 ticks by default."""
        scheduler = ModuleScheduler(tick_interval=0.02)
        fast = DummyFastModule()
        slow = DummySlowModule()
        scheduler.register(fast)
        scheduler.register(slow)
        await run_scheduler_ticks(scheduler, None, 10)
        # In 10 ticks: SLOW fires at tick 10 = 1 time
        assert slow.tick_count == 1
        assert fast.tick_count == 10

    async def test_custom_tier_intervals(self) -> None:
        """Custom tier intervals override defaults."""
        custom = {ModuleTier.FAST: 1, ModuleTier.MID: 2, ModuleTier.SLOW: 5}
        scheduler = ModuleScheduler(tick_interval=0.02, tier_intervals=custom)
        mid = DummyMidModule()
        scheduler.register(mid)
        await run_scheduler_ticks(scheduler, None, 6)
        # MID interval=2: fires at ticks 2, 4, 6 = 3 times
        assert mid.tick_count == 3


class TestErrorHandling:
    """Tests for error handling during module execution."""

    async def test_error_module_does_not_crash_scheduler(self) -> None:
        """A module that raises should not stop the scheduler."""
        scheduler = ModuleScheduler(tick_interval=0.02)
        error_mod = ErrorModule()
        fast = DummyFastModule()
        scheduler.register(error_mod)
        scheduler.register(fast)
        await run_scheduler_ticks(scheduler, None, 5)
        # The fast module should still have executed
        assert fast.tick_count >= 5

    async def test_error_module_produces_error_result(self) -> None:
        """An erroring module should produce a ModuleResult with error set."""
        scheduler = ModuleScheduler(tick_interval=0.02)
        scheduler.register(ErrorModule())
        await run_scheduler_ticks(scheduler, None, 1)
        result = scheduler.get_latest_result("error_module")
        assert result is not None
        assert result.error is not None
        assert "Intentional test error" in result.error
        assert not result.success


class TestResultCollection:
    """Tests for result collection."""

    async def test_collect_results_returns_accumulated(self) -> None:
        scheduler = ModuleScheduler(tick_interval=0.02)
        scheduler.register(DummyFastModule())
        await scheduler.start(None)
        # Wait for a few ticks
        while scheduler.tick_count < 3:
            await asyncio.sleep(0.02)
        results = scheduler.collect_results()
        assert len(results) >= 3
        await scheduler.stop()

    async def test_collect_results_clears_buffer(self) -> None:
        scheduler = ModuleScheduler(tick_interval=0.02)
        scheduler.register(DummyFastModule())
        await scheduler.start(None)
        while scheduler.tick_count < 2:
            await asyncio.sleep(0.02)
        first = scheduler.collect_results()
        assert len(first) >= 2
        # Immediately collecting again should return empty or very few
        second = scheduler.collect_results()
        assert len(second) < len(first)
        await scheduler.stop()

    async def test_get_latest_result(self) -> None:
        scheduler = ModuleScheduler(tick_interval=0.02)
        scheduler.register(DummyFastModule())
        await run_scheduler_ticks(scheduler, None, 3)
        result = scheduler.get_latest_result("fast_module")
        assert result is not None
        assert result.module_name == "fast_module"
        assert result.success

    async def test_get_latest_result_unknown_module(self) -> None:
        scheduler = ModuleScheduler()
        assert scheduler.get_latest_result("nonexistent") is None

    async def test_results_include_all_tiers(self) -> None:
        scheduler = ModuleScheduler(tick_interval=0.02)
        scheduler.register(DummyFastModule())
        scheduler.register(DummyMidModule())
        scheduler.register(DummySlowModule())
        await run_scheduler_ticks(scheduler, None, 10)
        results = scheduler.collect_results()
        module_names = {r.module_name for r in results}
        assert "fast_module" in module_names
        assert "mid_module" in module_names
        assert "slow_module" in module_names


class TestTickInterval:
    """Tests for tick interval configuration."""

    def test_default_tick_interval(self) -> None:
        scheduler = ModuleScheduler()
        assert scheduler.tick_interval == 0.5

    def test_custom_tick_interval(self) -> None:
        scheduler = ModuleScheduler(tick_interval=1.0)
        assert scheduler.tick_interval == 1.0

    def test_set_tick_interval(self) -> None:
        scheduler = ModuleScheduler()
        scheduler.tick_interval = 0.25
        assert scheduler.tick_interval == 0.25

    def test_invalid_tick_interval_raises(self) -> None:
        scheduler = ModuleScheduler()
        with pytest.raises(ValueError, match="positive"):
            scheduler.tick_interval = 0

    async def test_pause_does_not_increment_ticks(self) -> None:
        """While paused, tick_count should not increase."""
        scheduler = ModuleScheduler(tick_interval=0.02)
        scheduler.register(DummyFastModule())
        await scheduler.start(None)
        while scheduler.tick_count < 3:
            await asyncio.sleep(0.02)
        scheduler.pause()
        count_at_pause = scheduler.tick_count
        await asyncio.sleep(0.1)
        # Tick count should not have increased (or by at most 1 due to race)
        assert scheduler.tick_count <= count_at_pause + 1
        await scheduler.stop()
