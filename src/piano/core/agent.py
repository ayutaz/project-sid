"""Agent runtime for the PIANO architecture.

The Agent class orchestrates modules, the scheduler, and the cognitive
controller to run a single PIANO agent. It owns the main tick loop and
manages the full lifecycle: initialize -> run -> shutdown.

Reference: docs/implementation/01-system-architecture.md
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from piano.core.scheduler import ModuleScheduler, SchedulerState

if TYPE_CHECKING:
    from piano.core.module import Module
    from piano.core.sas import SharedAgentState

logger = structlog.get_logger()


class Agent:
    """A single PIANO agent runtime.

    Coordinates the scheduler, modules, and SAS to run the agent's
    cognitive loop. Supports finite runs (``max_ticks``) for testing
    and infinite runs for production.
    """

    def __init__(
        self,
        agent_id: str,
        sas: SharedAgentState,
        scheduler: ModuleScheduler,
        cc: Any = None,
    ) -> None:
        """Initialize the agent.

        Args:
            agent_id: Unique identifier for this agent.
            sas: SharedAgentState instance for this agent.
            scheduler: ModuleScheduler to manage module execution.
            cc: Cognitive Controller instance (optional, can be None for tests).
        """
        self._agent_id = agent_id
        self._sas = sas
        self._scheduler = scheduler
        self._cc = cc
        self._running = False
        self._modules: list[Module] = []

    @property
    def agent_id(self) -> str:
        """Unique agent identifier."""
        return self._agent_id

    @property
    def running(self) -> bool:
        """Whether the agent is currently running."""
        return self._running

    @property
    def modules(self) -> list[Module]:
        """List of registered modules."""
        return list(self._modules)

    @property
    def sas(self) -> SharedAgentState:
        """The agent's SharedAgentState."""
        return self._sas

    @property
    def scheduler(self) -> ModuleScheduler:
        """The agent's ModuleScheduler."""
        return self._scheduler

    def register_module(self, module: Module) -> None:
        """Register a module with the agent and its scheduler.

        Args:
            module: Module to register.
        """
        self._modules.append(module)
        self._scheduler.register(module)
        logger.info("agent_module_registered", agent=self._agent_id, module=module.name)

    async def initialize(self) -> None:
        """Initialize the agent: set up SAS and prepare modules.

        Must be called before ``run()``.
        """
        logger.info("agent_initializing", agent=self._agent_id)
        await self._sas.initialize()
        logger.info("agent_initialized", agent=self._agent_id, module_count=len(self._modules))

    async def run(self, max_ticks: int | None = None) -> None:
        """Run the agent's main loop.

        Args:
            max_ticks: If set, stop after this many scheduler ticks.
                       If None, run until ``shutdown()`` is called.
        """
        if self._running:
            raise RuntimeError(f"Agent '{self._agent_id}' is already running")

        self._running = True
        logger.info("agent_starting", agent=self._agent_id, max_ticks=max_ticks)

        await self._scheduler.start(self._sas)

        try:
            if max_ticks is not None:
                # Finite run: wait until the scheduler reaches the tick count
                poll_interval = min(self._scheduler.tick_interval * 0.25, 0.05)
                while (
                    self._running
                    and self._scheduler.tick_count < max_ticks
                    and self._scheduler.state == SchedulerState.RUNNING
                ):
                    await asyncio.sleep(poll_interval)
                # Allow the last tick's module executions to complete
                await asyncio.sleep(self._scheduler.tick_interval * 0.5)
            else:
                # Infinite run: wait until stopped externally
                while self._running and self._scheduler.state == SchedulerState.RUNNING:
                    await asyncio.sleep(self._scheduler.tick_interval)
        except asyncio.CancelledError:
            logger.info("agent_run_cancelled", agent=self._agent_id)
        finally:
            if self._scheduler.state in (SchedulerState.RUNNING, SchedulerState.PAUSED):
                await self._scheduler.stop()
            self._running = False
            logger.info(
                "agent_stopped",
                agent=self._agent_id,
                total_ticks=self._scheduler.tick_count,
            )

    async def shutdown(self) -> None:
        """Gracefully shut down the agent.

        Stops the scheduler and cleans up resources.
        """
        logger.info("agent_shutting_down", agent=self._agent_id)
        self._running = False

        if self._scheduler.state in (SchedulerState.RUNNING, SchedulerState.PAUSED):
            await self._scheduler.stop()

        logger.info("agent_shutdown_complete", agent=self._agent_id)
