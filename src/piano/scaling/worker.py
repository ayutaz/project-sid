"""Asyncio-based agent worker process for the PIANO architecture.

Manages up to 250 agents within a single OS process using asyncio tasks.
Each agent's lifecycle is wrapped in an individual asyncio.Task, allowing
concurrent execution without blocking.

Reference: docs/implementation/01-system-architecture.md
"""

from __future__ import annotations

__all__ = ["AgentWorkerProcess", "WorkerStats", "WorkerStatus"]

import asyncio
import contextlib
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from piano.core.agent import Agent

logger = structlog.get_logger(__name__)


class WorkerStatus(StrEnum):
    """Lifecycle state of the worker process."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


class WorkerStats(BaseModel):
    """Runtime statistics for a worker process.

    Attributes:
        worker_id: The worker's unique identifier.
        status: Current worker status.
        agent_count: Number of agents currently managed.
        max_agents: Maximum agent capacity.
        uptime_seconds: Time since the worker started running.
        tick_counts: Mapping of agent_id to their scheduler tick count.
    """

    worker_id: str
    status: WorkerStatus
    agent_count: int
    max_agents: int
    uptime_seconds: float = 0.0
    tick_counts: dict[str, int] = Field(default_factory=dict)


class AgentHealthInfo(BaseModel):
    """Health status for a single agent.

    Attributes:
        agent_id: The agent's unique identifier.
        running: Whether the agent's run loop is active.
        task_done: Whether the agent's asyncio task has completed.
        tick_count: Number of scheduler ticks executed.
        error: Error message if the agent's task failed.
    """

    agent_id: str
    running: bool = False
    task_done: bool = False
    tick_count: int = 0
    error: str | None = None


class AgentWorkerProcess:
    """Manages up to ``max_agents`` PIANO agents in a single asyncio event loop.

    Each agent is run as an independent asyncio.Task. The worker provides
    lifecycle management (start/stop), dynamic agent add/remove, health
    checks, and runtime statistics.

    Usage::

        worker = AgentWorkerProcess(worker_id="w-001", max_agents=250)
        worker.add_agent(agent)
        await worker.start()
        ...
        stats = worker.get_stats()
        health = await worker.health_check()
        ...
        await worker.stop()
    """

    def __init__(
        self,
        worker_id: str,
        max_agents: int = 250,
    ) -> None:
        """Initialize the worker process.

        Args:
            worker_id: Unique identifier for this worker.
            max_agents: Maximum number of agents this worker can manage.
        """
        self._worker_id = worker_id
        self._max_agents = max_agents
        self._status = WorkerStatus.IDLE
        self._agents: dict[str, Agent] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._start_time: float | None = None

    # --- Properties ---

    @property
    def worker_id(self) -> str:
        """Unique worker identifier."""
        return self._worker_id

    @property
    def max_agents(self) -> int:
        """Maximum number of agents this worker can manage."""
        return self._max_agents

    @property
    def status(self) -> WorkerStatus:
        """Current lifecycle status."""
        return self._status

    @property
    def is_full(self) -> bool:
        """Whether the worker is at capacity."""
        return len(self._agents) >= self._max_agents

    @property
    def agent_count(self) -> int:
        """Number of agents currently managed."""
        return len(self._agents)

    # --- Agent Management ---

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the worker.

        If the worker is already running, the agent's task is started
        immediately.

        Args:
            agent: The Agent instance to add.

        Raises:
            ValueError: If the worker is at capacity or the agent ID
                already exists, or the worker is stopped/stopping.
        """
        if self._status in (WorkerStatus.STOPPING, WorkerStatus.STOPPED):
            raise ValueError(
                f"Cannot add agent to worker '{self._worker_id}' in state {self._status.value}"
            )

        if self.is_full:
            raise ValueError(
                f"Worker '{self._worker_id}' is at capacity ({self._max_agents})"
            )

        if agent.agent_id in self._agents:
            raise ValueError(
                f"Agent '{agent.agent_id}' already exists in worker '{self._worker_id}'"
            )

        self._agents[agent.agent_id] = agent

        logger.info(
            "worker_agent_added",
            worker=self._worker_id,
            agent=agent.agent_id,
            agent_count=len(self._agents),
        )

        # If the worker is already running, start the agent immediately
        if self._status == WorkerStatus.RUNNING:
            self._tasks[agent.agent_id] = asyncio.create_task(
                self._run_agent(agent),
                name=f"agent-{agent.agent_id}",
            )

    async def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the worker.

        Cancels the agent's task (if running) and shuts down the agent
        gracefully.

        Args:
            agent_id: The agent's unique identifier.

        Raises:
            KeyError: If no agent with the given ID exists.
        """
        if agent_id not in self._agents:
            raise KeyError(
                f"Agent '{agent_id}' not found in worker '{self._worker_id}'"
            )

        # Cancel the task if it exists and is still running
        task = self._tasks.pop(agent_id, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Shut down the agent
        agent = self._agents.pop(agent_id)
        try:
            await agent.shutdown()
        except Exception:
            logger.exception(
                "worker_agent_shutdown_error",
                worker=self._worker_id,
                agent=agent_id,
            )

        logger.info(
            "worker_agent_removed",
            worker=self._worker_id,
            agent=agent_id,
            agent_count=len(self._agents),
        )

    def list_agents(self) -> list[str]:
        """Get list of all managed agent IDs.

        Returns:
            List of agent IDs.
        """
        return list(self._agents.keys())

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID.

        Args:
            agent_id: The agent's unique identifier.

        Returns:
            The Agent instance or None if not found.
        """
        return self._agents.get(agent_id)

    # --- Lifecycle ---

    async def start(self) -> None:
        """Start all agents concurrently.

        Initializes each agent and creates an asyncio task for its run loop.

        Raises:
            RuntimeError: If the worker is already running.
        """
        if self._status == WorkerStatus.RUNNING:
            raise RuntimeError(f"Worker '{self._worker_id}' is already running")

        self._status = WorkerStatus.STARTING
        self._start_time = time.monotonic()

        logger.info(
            "worker_starting",
            worker=self._worker_id,
            agent_count=len(self._agents),
        )

        # Initialize all agents concurrently
        init_results = await asyncio.gather(
            *[agent.initialize() for agent in self._agents.values()],
            return_exceptions=True,
        )
        for agent, result in zip(self._agents.values(), init_results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "worker_agent_init_error",
                    worker=self._worker_id,
                    agent=agent.agent_id,
                    error=str(result),
                )

        # Create tasks for all agents
        for agent in self._agents.values():
            self._tasks[agent.agent_id] = asyncio.create_task(
                self._run_agent(agent),
                name=f"agent-{agent.agent_id}",
            )

        self._status = WorkerStatus.RUNNING

        logger.info(
            "worker_started",
            worker=self._worker_id,
            agent_count=len(self._agents),
        )

    async def stop(self) -> None:
        """Gracefully stop all agents and the worker.

        Cancels all agent tasks and shuts down each agent.

        Raises:
            RuntimeError: If the worker is not running.
        """
        if self._status not in (WorkerStatus.RUNNING, WorkerStatus.STARTING):
            raise RuntimeError(
                f"Cannot stop worker '{self._worker_id}' in state {self._status.value}"
            )

        self._status = WorkerStatus.STOPPING

        logger.info(
            "worker_stopping",
            worker=self._worker_id,
            agent_count=len(self._agents),
        )

        # Cancel all agent tasks
        for task in self._tasks.values():
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        # Shut down all agents
        shutdown_results = await asyncio.gather(
            *[agent.shutdown() for agent in self._agents.values()],
            return_exceptions=True,
        )
        for agent, result in zip(self._agents.values(), shutdown_results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "worker_agent_shutdown_error",
                    worker=self._worker_id,
                    agent=agent.agent_id,
                    error=str(result),
                )

        self._tasks.clear()
        self._status = WorkerStatus.STOPPED

        logger.info(
            "worker_stopped",
            worker=self._worker_id,
        )

    # --- Statistics and Health ---

    def get_stats(self) -> WorkerStats:
        """Get runtime statistics for this worker.

        Returns:
            WorkerStats with current state information.
        """
        uptime = 0.0
        if self._start_time is not None:
            uptime = time.monotonic() - self._start_time

        tick_counts: dict[str, int] = {}
        for agent_id, agent in self._agents.items():
            tick_counts[agent_id] = agent.scheduler.tick_count

        return WorkerStats(
            worker_id=self._worker_id,
            status=self._status,
            agent_count=len(self._agents),
            max_agents=self._max_agents,
            uptime_seconds=uptime,
            tick_counts=tick_counts,
        )

    async def health_check(self) -> dict[str, Any]:
        """Check the health of all agents.

        Returns:
            Dictionary with overall worker health and per-agent status:
            ``{"healthy": bool, "worker_status": str, "agents": {id: AgentHealthInfo}}``.
        """
        agents_health: dict[str, dict[str, Any]] = {}
        all_healthy = True

        for agent_id, agent in self._agents.items():
            task = self._tasks.get(agent_id)
            task_done = task.done() if task is not None else True
            error: str | None = None

            if task is not None and task.done():
                exc = task.exception() if not task.cancelled() else None
                if exc is not None:
                    error = str(exc)
                    all_healthy = False
                elif task.cancelled():
                    error = "Task was cancelled"
                    all_healthy = False

            # An agent that should be running but whose task is done is unhealthy
            if self._status == WorkerStatus.RUNNING and task_done:
                all_healthy = False

            info = AgentHealthInfo(
                agent_id=agent_id,
                running=agent.running,
                task_done=task_done,
                tick_count=agent.scheduler.tick_count,
                error=error,
            )
            agents_health[agent_id] = info.model_dump()

        return {
            "healthy": all_healthy,
            "worker_id": self._worker_id,
            "worker_status": self._status.value,
            "agent_count": len(self._agents),
            "agents": agents_health,
        }

    # --- Internal ---

    async def _run_agent(self, agent: Agent) -> None:
        """Run a single agent's loop, catching exceptions to avoid crashing the worker.

        Args:
            agent: The Agent instance to run.
        """
        try:
            await agent.run()
        except asyncio.CancelledError:
            logger.debug(
                "worker_agent_task_cancelled",
                worker=self._worker_id,
                agent=agent.agent_id,
            )
            raise
        except Exception:
            logger.exception(
                "worker_agent_task_error",
                worker=self._worker_id,
                agent=agent.agent_id,
            )
