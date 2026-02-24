"""Multi-process agent supervisor for the PIANO architecture.

Manages multiple WorkerProcesses, distributing agents across workers
for horizontal scaling. Supports 1000+ agents via sharding:
e.g. 4 workers x 250 agents/worker.

The actual multiprocessing is abstracted behind the WorkerHandle protocol,
allowing mock implementations for testing and real subprocess implementations
for production.

Reference: docs/implementation/08-infrastructure.md
"""

from __future__ import annotations

__all__ = [
    "AgentSupervisor",
    "SupervisorConfig",
    "WorkerHandle",
    "WorkerInfo",
    "WorkerState",
    "WorkerStats",
]

import asyncio
import contextlib
import math
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# --- Configuration ---


class SupervisorConfig(BaseModel):
    """Configuration for the AgentSupervisor.

    Attributes:
        agents_per_worker: Maximum number of agents per worker process.
        max_workers: Maximum number of worker processes allowed.
        health_check_interval: Seconds between automatic health checks.
    """

    agents_per_worker: int = 250
    max_workers: int = 8
    health_check_interval: float = 30.0


# --- Worker Protocol ---


class WorkerState(StrEnum):
    """Lifecycle state of a worker process."""

    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    UNHEALTHY = "unhealthy"


class WorkerStats(BaseModel):
    """Statistics reported by a worker."""

    worker_id: str
    agent_count: int = 0
    state: WorkerState = WorkerState.IDLE
    tick_count: int = 0
    errors: int = 0


@runtime_checkable
class WorkerHandle(Protocol):
    """Protocol for interacting with a worker process.

    Implementations may use real multiprocessing, asyncio tasks,
    or simple mocks for testing.
    """

    @property
    def worker_id(self) -> str:
        """Unique identifier for this worker."""
        ...

    async def start(self) -> None:
        """Start the worker process."""
        ...

    async def stop(self) -> None:
        """Stop the worker process gracefully."""
        ...

    async def add_agent(self, agent_id: str, shard_id: str) -> None:
        """Add an agent to this worker.

        Args:
            agent_id: Unique agent identifier.
            shard_id: Shard/partition identifier for the agent.
        """
        ...

    async def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from this worker.

        Args:
            agent_id: The agent to remove.
        """
        ...

    async def get_stats(self) -> WorkerStats:
        """Get current worker statistics."""
        ...

    async def health_check(self) -> bool:
        """Check if the worker is healthy.

        Returns:
            True if the worker is responsive and functioning.
        """
        ...


# --- Worker Info (internal tracking) ---


class WorkerInfo(BaseModel):
    """Internal tracking data for a worker managed by the supervisor.

    Attributes:
        worker_id: Unique worker identifier.
        agent_ids: Set of agent IDs assigned to this worker.
        shard_ids: Mapping of agent_id -> shard_id.
        state: Current worker state.
    """

    worker_id: str
    agent_ids: set[str] = Field(default_factory=set)
    shard_ids: dict[str, str] = Field(default_factory=dict)
    state: WorkerState = WorkerState.IDLE


# --- Supervisor ---


class AgentSupervisor:
    """Manages multiple worker processes for large-scale agent simulations.

    Distributes agents across workers, handles auto-scaling based on agent
    count, and provides rebalancing when load becomes uneven.

    Usage::

        config = SupervisorConfig(agents_per_worker=250, max_workers=4)
        supervisor = AgentSupervisor(config, worker_factory=my_factory)

        # Add agents (workers created automatically)
        worker_id = await supervisor.assign_agent("agent-001", "shard-0")
        worker_id = await supervisor.assign_agent("agent-002", "shard-0")

        # Start all workers
        await supervisor.start_all()

        # Check stats
        stats = supervisor.get_worker_stats()

        # Rebalance if needed
        await supervisor.rebalance()

        # Clean shutdown
        await supervisor.stop_all()
    """

    def __init__(
        self,
        config: SupervisorConfig,
        worker_factory: Callable[[str], WorkerHandle] | None = None,
    ) -> None:
        """Initialize the supervisor.

        Args:
            config: Supervisor configuration.
            worker_factory: Callable that creates WorkerHandle instances
                given a worker_id. If None, create_worker() must be overridden
                or a factory must be set before use.
        """
        self._config = config
        self._worker_factory = worker_factory
        self._workers: dict[str, WorkerHandle] = {}
        self._worker_info: dict[str, WorkerInfo] = {}
        self._agent_to_worker: dict[str, str] = {}
        self._next_worker_index: int = 0
        self._agent_locks: dict[str, asyncio.Lock] = {}
        self._health_check_task: asyncio.Task[None] | None = None

        logger.info(
            "supervisor_initialized",
            agents_per_worker=config.agents_per_worker,
            max_workers=config.max_workers,
            health_check_interval=config.health_check_interval,
        )

    # --- Properties ---

    @property
    def config(self) -> SupervisorConfig:
        """Supervisor configuration."""
        return self._config

    @property
    def worker_count(self) -> int:
        """Number of active workers."""
        return len(self._workers)

    @property
    def agent_count(self) -> int:
        """Total number of agents across all workers."""
        return len(self._agent_to_worker)

    @property
    def worker_ids(self) -> list[str]:
        """List of all worker IDs."""
        return list(self._workers.keys())

    # --- Worker Management ---

    def create_worker(self, worker_id: str) -> WorkerHandle:
        """Create a new worker process.

        Args:
            worker_id: Unique identifier for the new worker.

        Returns:
            A WorkerHandle for the new worker.

        Raises:
            ValueError: If worker_id already exists or max_workers reached.
            RuntimeError: If no worker_factory is configured.
        """
        if worker_id in self._workers:
            raise ValueError(f"Worker '{worker_id}' already exists")

        if len(self._workers) >= self._config.max_workers:
            raise ValueError(
                f"Cannot create worker: max_workers limit ({self._config.max_workers}) reached"
            )

        if self._worker_factory is None:
            raise RuntimeError(
                "No worker_factory configured. Provide one in __init__ or override create_worker()."
            )

        handle = self._worker_factory(worker_id)
        self._workers[worker_id] = handle
        self._worker_info[worker_id] = WorkerInfo(worker_id=worker_id)

        logger.info(
            "worker_created",
            worker_id=worker_id,
            total_workers=len(self._workers),
        )

        return handle

    def _find_available_worker(self) -> str | None:
        """Find a worker with capacity for more agents.

        Returns:
            worker_id of an available worker, or None if all are full.
        """
        for worker_id, info in self._worker_info.items():
            if len(info.agent_ids) < self._config.agents_per_worker:
                return worker_id
        return None

    def _generate_worker_id(self) -> str:
        """Generate a unique worker ID.

        Returns:
            A new worker ID string.
        """
        while True:
            worker_id = f"worker-{self._next_worker_index:03d}"
            self._next_worker_index += 1
            if worker_id not in self._workers:
                return worker_id

    # --- Agent Assignment ---

    async def assign_agent(self, agent_id: str, shard_id: str) -> str:
        """Assign an agent to a worker, creating a new worker if needed.

        The supervisor finds a worker with available capacity or auto-creates
        a new one. The agent is then added to the selected worker.

        Args:
            agent_id: Unique agent identifier.
            shard_id: Shard/partition identifier for the agent.

        Returns:
            The worker_id the agent was assigned to.

        Raises:
            ValueError: If agent_id is already assigned or no capacity available.
        """
        if agent_id in self._agent_to_worker:
            raise ValueError(f"Agent '{agent_id}' is already assigned to a worker")

        # Find or create a worker with capacity
        worker_id = self._find_available_worker()

        if worker_id is None:
            # Auto-scale: create a new worker
            if len(self._workers) >= self._config.max_workers:
                raise ValueError(
                    f"Cannot assign agent: all {self._config.max_workers} workers "
                    f"are at capacity ({self._config.agents_per_worker} agents each)"
                )

            worker_id = self._generate_worker_id()
            self.create_worker(worker_id)

        # Add agent to worker
        handle = self._workers[worker_id]
        await handle.add_agent(agent_id, shard_id)

        # Update tracking
        info = self._worker_info[worker_id]
        info.agent_ids.add(agent_id)
        info.shard_ids[agent_id] = shard_id
        self._agent_to_worker[agent_id] = worker_id

        logger.info(
            "agent_assigned",
            agent_id=agent_id,
            shard_id=shard_id,
            worker_id=worker_id,
            worker_agent_count=len(info.agent_ids),
        )

        return worker_id

    async def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from its assigned worker.

        Args:
            agent_id: The agent to remove.

        Raises:
            ValueError: If agent_id is not assigned to any worker.
        """
        if agent_id not in self._agent_to_worker:
            raise ValueError(f"Agent '{agent_id}' is not assigned to any worker")

        worker_id = self._agent_to_worker[agent_id]
        handle = self._workers[worker_id]

        # Remove from worker
        await handle.remove_agent(agent_id)

        # Update tracking
        info = self._worker_info[worker_id]
        info.agent_ids.discard(agent_id)
        info.shard_ids.pop(agent_id, None)
        del self._agent_to_worker[agent_id]

        logger.info(
            "agent_removed",
            agent_id=agent_id,
            worker_id=worker_id,
            worker_agent_count=len(info.agent_ids),
        )

    def get_agent_worker(self, agent_id: str) -> str | None:
        """Get the worker ID for an agent.

        Args:
            agent_id: The agent to look up.

        Returns:
            The worker_id, or None if the agent is not assigned.
        """
        return self._agent_to_worker.get(agent_id)

    # --- Lifecycle ---

    async def start_all(self) -> None:
        """Start all worker processes in parallel.

        Workers that are already running are skipped.
        Also starts a background health check loop.
        """
        if not self._workers:
            logger.warning("start_all_called_with_no_workers")
            return

        logger.info("starting_all_workers", count=len(self._workers))

        async def _start_one(worker_id: str, handle: WorkerHandle) -> None:
            try:
                await handle.start()
                self._worker_info[worker_id].state = WorkerState.RUNNING
            except Exception:
                logger.exception("worker_start_error", worker_id=worker_id)
                self._worker_info[worker_id].state = WorkerState.UNHEALTHY

        await asyncio.gather(*[_start_one(wid, h) for wid, h in self._workers.items()])

        running_count = sum(
            1 for info in self._worker_info.values() if info.state == WorkerState.RUNNING
        )
        logger.info(
            "all_workers_started",
            total=len(self._workers),
            running=running_count,
        )

        # Start background health check loop
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(
                self._health_check_loop(),
                name="supervisor-health-check",
            )

    async def stop_all(self) -> None:
        """Stop all worker processes gracefully in parallel."""
        # Cancel background health check
        if self._health_check_task is not None and not self._health_check_task.done():
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task
            self._health_check_task = None

        if not self._workers:
            logger.warning("stop_all_called_with_no_workers")
            return

        logger.info("stopping_all_workers", count=len(self._workers))

        async def _stop_one(worker_id: str, handle: WorkerHandle) -> None:
            try:
                await handle.stop()
                self._worker_info[worker_id].state = WorkerState.STOPPED
            except Exception:
                logger.exception("worker_stop_error", worker_id=worker_id)

        await asyncio.gather(*[_stop_one(wid, h) for wid, h in self._workers.items()])

        logger.info("all_workers_stopped", count=len(self._workers))

    # --- Statistics ---

    def get_worker_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all workers.

        Returns:
            Dictionary mapping worker_id to stat dictionaries containing
            agent_count, state, and agent_ids.
        """
        stats: dict[str, dict[str, Any]] = {}

        for worker_id, info in self._worker_info.items():
            stats[worker_id] = {
                "agent_count": len(info.agent_ids),
                "state": info.state.value,
                "agent_ids": sorted(info.agent_ids),
                "capacity": self._config.agents_per_worker,
                "utilization": (
                    len(info.agent_ids) / self._config.agents_per_worker
                    if self._config.agents_per_worker > 0
                    else 0.0
                ),
            }

        return stats

    # --- Rebalancing ---

    async def rebalance(self) -> dict[str, list[str]]:
        """Rebalance agents across workers for even distribution.

        Moves agents from overloaded workers to underloaded ones to achieve
        a more balanced distribution. Uses per-agent locks to prevent
        concurrent operations on the same agent. If adding to the destination
        fails, the agent is re-added to the source worker.

        Returns:
            Dictionary mapping worker_id to list of agent_ids that were
            moved TO that worker. Empty dict if no moves needed.
        """
        if len(self._workers) <= 1:
            logger.debug("rebalance_skipped", reason="0 or 1 workers")
            return {}

        total_agents = self.agent_count
        num_workers = len(self._workers)
        target_per_worker = math.ceil(total_agents / num_workers)

        # Identify overloaded and underloaded workers
        over: list[tuple[str, list[str]]] = []  # (worker_id, excess_agent_ids)
        under: list[tuple[str, int]] = []  # (worker_id, deficit)

        for worker_id, info in self._worker_info.items():
            count = len(info.agent_ids)
            if count > target_per_worker:
                excess = sorted(info.agent_ids)[target_per_worker:]
                over.append((worker_id, excess))
            elif count < target_per_worker:
                deficit = target_per_worker - count
                under.append((worker_id, deficit))

        if not over or not under:
            logger.debug("rebalance_skipped", reason="already balanced")
            return {}

        moves: dict[str, list[str]] = {}

        # Move excess agents to underloaded workers
        under_idx = 0
        for _src_worker_id, excess_agents in over:
            for agent_id in excess_agents:
                if under_idx >= len(under):
                    break

                dst_worker_id, deficit = under[under_idx]

                if deficit <= 0:
                    under_idx += 1
                    if under_idx >= len(under):
                        break
                    dst_worker_id, deficit = under[under_idx]

                # Get or create a per-agent lock
                if agent_id not in self._agent_locks:
                    self._agent_locks[agent_id] = asyncio.Lock()

                async with self._agent_locks[agent_id]:
                    # Get shard_id before removing
                    src_worker_id_actual = self._agent_to_worker[agent_id]
                    shard_id = self._worker_info[src_worker_id_actual].shard_ids[agent_id]

                    # Remove from source
                    src_handle = self._workers[src_worker_id_actual]
                    await src_handle.remove_agent(agent_id)
                    self._worker_info[src_worker_id_actual].agent_ids.discard(agent_id)
                    self._worker_info[src_worker_id_actual].shard_ids.pop(agent_id, None)

                    # Add to destination; if it fails, re-add to source
                    dst_handle = self._workers[dst_worker_id]
                    try:
                        await dst_handle.add_agent(agent_id, shard_id)
                    except Exception:
                        logger.exception(
                            "rebalance_add_failed",
                            agent_id=agent_id,
                            dst_worker=dst_worker_id,
                        )
                        # Roll back: re-add to source
                        await src_handle.add_agent(agent_id, shard_id)
                        self._worker_info[src_worker_id_actual].agent_ids.add(agent_id)
                        self._worker_info[src_worker_id_actual].shard_ids[agent_id] = shard_id
                        continue

                    self._worker_info[dst_worker_id].agent_ids.add(agent_id)
                    self._worker_info[dst_worker_id].shard_ids[agent_id] = shard_id
                    self._agent_to_worker[agent_id] = dst_worker_id

                # Track the move
                if dst_worker_id not in moves:
                    moves[dst_worker_id] = []
                moves[dst_worker_id].append(agent_id)

                # Update deficit
                under[under_idx] = (dst_worker_id, deficit - 1)

                logger.debug(
                    "agent_rebalanced",
                    agent_id=agent_id,
                    from_worker=src_worker_id_actual,
                    to_worker=dst_worker_id,
                )

        if moves:
            total_moved = sum(len(ids) for ids in moves.values())
            logger.info(
                "rebalance_complete",
                agents_moved=total_moved,
                target_per_worker=target_per_worker,
            )
        else:
            logger.debug("rebalance_complete", agents_moved=0)

        return moves

    # --- Health Check ---

    async def health_check_all(self) -> dict[str, bool]:
        """Run health checks on all workers.

        Attempts to restart unhealthy workers.

        Returns:
            Dictionary mapping worker_id to health status (True = healthy).
        """
        results: dict[str, bool] = {}

        for worker_id, handle in self._workers.items():
            try:
                healthy = await handle.health_check()
                results[worker_id] = healthy
                if not healthy:
                    self._worker_info[worker_id].state = WorkerState.UNHEALTHY
                    logger.warning("worker_unhealthy", worker_id=worker_id)
                    # Attempt restart
                    with contextlib.suppress(Exception):
                        await handle.stop()
                    try:
                        await handle.start()
                        self._worker_info[worker_id].state = WorkerState.RUNNING
                        logger.info("worker_restarted", worker_id=worker_id)
                    except Exception:
                        logger.exception("worker_restart_failed", worker_id=worker_id)
            except Exception:
                results[worker_id] = False
                self._worker_info[worker_id].state = WorkerState.UNHEALTHY
                logger.exception("worker_health_check_error", worker_id=worker_id)

        return results

    async def _health_check_loop(self) -> None:
        """Background loop that periodically runs health checks."""
        try:
            while True:
                await asyncio.sleep(self._config.health_check_interval)
                await self.health_check_all()
        except asyncio.CancelledError:
            return
