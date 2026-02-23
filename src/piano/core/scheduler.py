"""Module scheduler for the PIANO architecture.

Manages concurrent execution of modules at different tick frequencies
based on their tier (FAST/MID/SLOW). Provides lifecycle management
(start/stop/pause) and result collection for the Cognitive Controller.

Reference: docs/implementation/01-system-architecture.md
"""

from __future__ import annotations

__all__ = ["ModuleScheduler", "SchedulerState"]

import asyncio
import contextlib
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog

from piano.core.types import ModuleResult, ModuleTier

if TYPE_CHECKING:
    from piano.core.module import Module

logger = structlog.get_logger()

# Tier -> execute every N ticks
DEFAULT_TIER_INTERVALS: dict[ModuleTier, int] = {
    ModuleTier.FAST: 1,   # every tick
    ModuleTier.MID: 3,    # every 3 ticks
    ModuleTier.SLOW: 10,  # every 10 ticks
}


class SchedulerState(StrEnum):
    """Lifecycle state of the scheduler."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class ModuleScheduler:
    """Schedules and executes PIANO modules at tier-based frequencies.

    The scheduler runs an async tick loop. On each tick it determines
    which modules should execute (based on their tier interval) and
    runs them concurrently via asyncio tasks. Module exceptions are
    caught and logged without crashing the scheduler.

    Usage::

        scheduler = ModuleScheduler(tick_interval=0.5)
        scheduler.register(my_module)
        await scheduler.start(sas)
        ...
        results = scheduler.collect_results()
        ...
        await scheduler.stop()
    """

    def __init__(
        self,
        tick_interval: float = 0.5,
        tier_intervals: dict[ModuleTier, int] | None = None,
    ) -> None:
        """Initialize the scheduler.

        Args:
            tick_interval: Seconds between ticks (default 0.5s / 500ms).
            tier_intervals: Override for per-tier tick intervals. Defaults
                to FAST=1, MID=3, SLOW=10.
        """
        self._tick_interval = tick_interval
        self._tier_intervals = tier_intervals or dict(DEFAULT_TIER_INTERVALS)
        self._modules: dict[str, Module] = {}
        self._state = SchedulerState.IDLE
        self._tick_count: int = 0
        self._loop_task: asyncio.Task[None] | None = None

        # Latest result per module (overwritten each execution)
        self._latest_results: dict[str, ModuleResult] = {}
        # Uncollected results accumulated since last collect_results() call
        self._pending_results: list[ModuleResult] = []

    # --- Properties ---

    @property
    def state(self) -> SchedulerState:
        """Current lifecycle state."""
        return self._state

    @property
    def tick_count(self) -> int:
        """Number of ticks executed since start."""
        return self._tick_count

    @property
    def tick_interval(self) -> float:
        """Seconds between ticks."""
        return self._tick_interval

    @tick_interval.setter
    def tick_interval(self, value: float) -> None:
        """Update the tick interval."""
        if value <= 0:
            raise ValueError("tick_interval must be positive")
        self._tick_interval = value

    @property
    def modules(self) -> dict[str, Module]:
        """Registered modules (name -> Module)."""
        return dict(self._modules)

    # --- Module Registration ---

    def register(self, module: Module) -> None:
        """Register a module with the scheduler.

        Args:
            module: Module instance to register.

        Raises:
            ValueError: If a module with the same name is already registered.
        """
        if module.name in self._modules:
            raise ValueError(f"Module '{module.name}' is already registered")
        self._modules[module.name] = module
        logger.info("module_registered", module=module.name, tier=module.tier.value)

    def unregister(self, name: str) -> Module:
        """Remove a module by name.

        Args:
            name: Name of the module to remove.

        Returns:
            The removed Module instance.

        Raises:
            KeyError: If no module with the given name is registered.
        """
        if name not in self._modules:
            raise KeyError(f"Module '{name}' is not registered")
        module = self._modules.pop(name)
        self._latest_results.pop(name, None)
        logger.info("module_unregistered", module=name)
        return module

    # --- Lifecycle ---

    async def start(self, sas: Any) -> None:
        """Start the tick loop.

        Calls initialize() on all registered modules, then begins
        the main tick loop as a background asyncio task.

        Args:
            sas: SharedAgentState instance passed to modules on each tick.

        Raises:
            RuntimeError: If the scheduler is already running.
        """
        if self._state == SchedulerState.RUNNING:
            raise RuntimeError("Scheduler is already running")

        self._state = SchedulerState.RUNNING
        self._tick_count = 0

        # Initialize all modules
        for module in self._modules.values():
            try:
                await module.initialize()
            except Exception:
                logger.exception("module_initialize_error", module=module.name)

        self._loop_task = asyncio.create_task(self._tick_loop(sas))
        logger.info("scheduler_started", tick_interval=self._tick_interval)

    async def stop(self) -> None:
        """Stop the tick loop and shut down all modules.

        Raises:
            RuntimeError: If the scheduler is not running or paused.
        """
        if self._state not in (SchedulerState.RUNNING, SchedulerState.PAUSED):
            raise RuntimeError(f"Cannot stop scheduler in state {self._state.value}")

        self._state = SchedulerState.STOPPED

        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None

        # Shutdown all modules
        for module in self._modules.values():
            try:
                await module.shutdown()
            except Exception:
                logger.exception("module_shutdown_error", module=module.name)

        logger.info("scheduler_stopped", total_ticks=self._tick_count)

    def pause(self) -> None:
        """Pause the tick loop. Ticks stop executing but state is preserved.

        Raises:
            RuntimeError: If the scheduler is not running.
        """
        if self._state != SchedulerState.RUNNING:
            raise RuntimeError(f"Cannot pause scheduler in state {self._state.value}")
        self._state = SchedulerState.PAUSED
        logger.info("scheduler_paused", tick=self._tick_count)

    def resume(self) -> None:
        """Resume the tick loop after a pause.

        Raises:
            RuntimeError: If the scheduler is not paused.
        """
        if self._state != SchedulerState.PAUSED:
            raise RuntimeError(f"Cannot resume scheduler in state {self._state.value}")
        self._state = SchedulerState.RUNNING
        logger.info("scheduler_resumed", tick=self._tick_count)

    # --- Result Collection ---

    def get_latest_result(self, module_name: str) -> ModuleResult | None:
        """Get the most recent result for a specific module.

        Args:
            module_name: Name of the module.

        Returns:
            The latest ModuleResult, or None if the module hasn't run yet.
        """
        return self._latest_results.get(module_name)

    def collect_results(self) -> list[ModuleResult]:
        """Collect all results accumulated since the last call.

        Returns the list of ModuleResults produced since the previous
        call to collect_results(), then clears the internal buffer.
        This is the primary interface for the Cognitive Controller.

        Returns:
            List of ModuleResult objects.
        """
        results = list(self._pending_results)
        self._pending_results.clear()
        return results

    # --- Internal Tick Loop ---

    async def _tick_loop(self, sas: Any) -> None:
        """Main tick loop. Runs until cancelled or stopped."""
        try:
            while self._state == SchedulerState.RUNNING or self._state == SchedulerState.PAUSED:
                if self._state == SchedulerState.PAUSED:
                    await asyncio.sleep(self._tick_interval)
                    continue

                self._tick_count += 1
                await self._execute_tick(sas)
                await asyncio.sleep(self._tick_interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("tick_loop_fatal_error", tick=self._tick_count)
            self._state = SchedulerState.STOPPED

    async def _execute_tick(self, sas: Any) -> None:
        """Run all modules due for this tick concurrently."""
        due_modules = self._get_due_modules()
        if not due_modules:
            return

        tasks = [
            asyncio.create_task(self._run_module(module, sas))
            for module in due_modules
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for module, result in zip(due_modules, results, strict=True):
            if isinstance(result, BaseException):
                # This shouldn't happen since _run_module catches exceptions,
                # but handle it defensively.
                error_result = ModuleResult(
                    module_name=module.name,
                    tier=module.tier,
                    error=f"Unexpected error: {result}",
                )
                self._latest_results[module.name] = error_result
                self._pending_results.append(error_result)
            elif isinstance(result, ModuleResult):
                self._latest_results[module.name] = result
                self._pending_results.append(result)

    async def _run_module(self, module: Module, sas: Any) -> ModuleResult:
        """Execute a single module's tick with error handling."""
        try:
            result = await module.tick(sas)
            logger.debug(
                "module_tick_complete",
                module=module.name,
                tick=self._tick_count,
                success=result.success,
            )
            return result
        except Exception as exc:
            logger.exception("module_tick_error", module=module.name, tick=self._tick_count)
            return ModuleResult(
                module_name=module.name,
                tier=module.tier,
                error=str(exc),
            )

    def _get_due_modules(self) -> list[Module]:
        """Return modules that should execute on the current tick."""
        due: list[Module] = []
        for module in self._modules.values():
            interval = self._tier_intervals.get(module.tier, 1)
            if self._tick_count % interval == 0:
                due.append(module)
        return due
