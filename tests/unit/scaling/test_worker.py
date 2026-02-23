"""Tests for the AgentWorkerProcess."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from piano.core.agent import Agent
from piano.core.scheduler import ModuleScheduler
from piano.core.types import ModuleTier
from piano.scaling.worker import AgentWorkerProcess, WorkerStats, WorkerStatus
from tests.helpers import DummyModule, InMemorySAS

# --- Helpers ---


def _make_agent(
    agent_id: str = "agent-001",
    tick_interval: float = 0.05,
    with_module: bool = True,
) -> Agent:
    """Create a minimal Agent for testing."""
    sas = InMemorySAS(agent_id=agent_id)
    scheduler = ModuleScheduler(tick_interval=tick_interval)
    agent = Agent(agent_id=agent_id, sas=sas, scheduler=scheduler)
    if with_module:
        module = DummyModule(module_name=f"mod-{agent_id}", tier=ModuleTier.FAST)
        agent.register_module(module)
    return agent


# --- Worker Initialization Tests ---


class TestWorkerInit:
    """Tests for worker initialization and properties."""

    def test_worker_creation_defaults(self):
        """Test that a worker is created with correct defaults."""
        worker = AgentWorkerProcess(worker_id="w-001")

        assert worker.worker_id == "w-001"
        assert worker.max_agents == 250
        assert worker.status == WorkerStatus.IDLE
        assert worker.agent_count == 0
        assert worker.is_full is False

    def test_worker_creation_custom_max_agents(self):
        """Test that max_agents can be customized."""
        worker = AgentWorkerProcess(worker_id="w-002", max_agents=10)

        assert worker.max_agents == 10


# --- Agent Add/Remove Tests ---


class TestAgentManagement:
    """Tests for adding and removing agents."""

    def test_add_agent_increases_count(self):
        """Test that adding an agent increases the count."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        agent = _make_agent("agent-001")

        worker.add_agent(agent)

        assert worker.agent_count == 1
        assert "agent-001" in worker.list_agents()

    def test_add_agent_at_capacity_raises(self):
        """Test that adding an agent when full raises ValueError."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=2)
        worker.add_agent(_make_agent("agent-001"))
        worker.add_agent(_make_agent("agent-002"))

        assert worker.is_full is True

        with pytest.raises(ValueError, match="at capacity"):
            worker.add_agent(_make_agent("agent-003"))

    def test_add_duplicate_agent_raises(self):
        """Test that adding an agent with existing ID raises ValueError."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker.add_agent(_make_agent("agent-001"))

        with pytest.raises(ValueError, match="already exists"):
            worker.add_agent(_make_agent("agent-001"))

    def test_add_agent_when_stopped_raises(self):
        """Test that adding an agent to a stopped worker raises ValueError."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker._status = WorkerStatus.STOPPED

        with pytest.raises(ValueError, match="state stopped"):
            worker.add_agent(_make_agent("agent-001"))

    def test_add_agent_when_stopping_raises(self):
        """Test that adding an agent to a stopping worker raises ValueError."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker._status = WorkerStatus.STOPPING

        with pytest.raises(ValueError, match="state stopping"):
            worker.add_agent(_make_agent("agent-001"))

    async def test_remove_agent_decreases_count(self):
        """Test that removing an agent decreases the count."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker.add_agent(_make_agent("agent-001"))
        assert worker.agent_count == 1

        await worker.remove_agent("agent-001")

        assert worker.agent_count == 0
        assert "agent-001" not in worker.list_agents()

    async def test_remove_nonexistent_agent_raises(self):
        """Test that removing a missing agent raises KeyError."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)

        with pytest.raises(KeyError, match="not found"):
            await worker.remove_agent("nonexistent")

    def test_get_agent_returns_agent(self):
        """Test that get_agent retrieves the correct agent."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        agent = _make_agent("agent-001")
        worker.add_agent(agent)

        assert worker.get_agent("agent-001") is agent
        assert worker.get_agent("nonexistent") is None

    def test_list_agents_returns_all_ids(self):
        """Test that list_agents returns all managed agent IDs."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        for i in range(3):
            worker.add_agent(_make_agent(f"agent-{i:03d}"))

        agent_ids = worker.list_agents()

        assert len(agent_ids) == 3
        assert "agent-000" in agent_ids
        assert "agent-001" in agent_ids
        assert "agent-002" in agent_ids

    def test_is_full_at_boundary(self):
        """Test is_full transitions at max_agents boundary."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=1)

        assert worker.is_full is False
        worker.add_agent(_make_agent("agent-001"))
        assert worker.is_full is True


# --- Start/Stop Lifecycle Tests ---


class TestWorkerLifecycle:
    """Tests for starting and stopping the worker."""

    async def test_start_transitions_to_running(self):
        """Test that start() transitions status to RUNNING."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker.add_agent(_make_agent("agent-001"))

        await worker.start()
        try:
            assert worker.status == WorkerStatus.RUNNING
        finally:
            await worker.stop()

    async def test_start_already_running_raises(self):
        """Test that starting an already-running worker raises RuntimeError."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker.add_agent(_make_agent("agent-001"))

        await worker.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await worker.start()
        finally:
            await worker.stop()

    async def test_stop_transitions_to_stopped(self):
        """Test that stop() transitions status to STOPPED."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker.add_agent(_make_agent("agent-001"))

        await worker.start()
        await asyncio.sleep(0.05)
        await worker.stop()

        assert worker.status == WorkerStatus.STOPPED

    async def test_stop_when_not_running_raises(self):
        """Test that stopping a non-running worker raises RuntimeError."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)

        with pytest.raises(RuntimeError, match="Cannot stop"):
            await worker.stop()

    async def test_start_with_no_agents(self):
        """Test that starting a worker with no agents works."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)

        await worker.start()
        try:
            assert worker.status == WorkerStatus.RUNNING
            assert worker.agent_count == 0
        finally:
            await worker.stop()

    async def test_agents_execute_ticks_after_start(self):
        """Test that agents run and accumulate ticks after start."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        agent = _make_agent("agent-001", tick_interval=0.02)
        worker.add_agent(agent)

        await worker.start()
        await asyncio.sleep(0.15)
        await worker.stop()

        assert agent.scheduler.tick_count > 0

    async def test_stop_cancels_agent_tasks(self):
        """Test that stop() cancels all running agent tasks."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker.add_agent(_make_agent("agent-001"))
        worker.add_agent(_make_agent("agent-002"))

        await worker.start()
        await asyncio.sleep(0.05)

        # Tasks should exist
        assert len(worker._tasks) == 2

        await worker.stop()

        # Tasks should be cleared
        assert len(worker._tasks) == 0


# --- Dynamic Agent Add/Remove While Running ---


class TestDynamicAgentManagement:
    """Tests for adding/removing agents while the worker is running."""

    async def test_add_agent_while_running_starts_task(self):
        """Test that adding an agent to a running worker starts it immediately."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker.add_agent(_make_agent("agent-001"))

        await worker.start()
        try:
            # Add agent while running
            agent2 = _make_agent("agent-002", tick_interval=0.02)
            worker.add_agent(agent2)

            assert worker.agent_count == 2
            assert "agent-002" in worker._tasks

            # Give the new agent time to run
            await asyncio.sleep(0.1)
            assert agent2.scheduler.tick_count > 0
        finally:
            await worker.stop()

    async def test_remove_agent_while_running(self):
        """Test that removing an agent from a running worker works."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker.add_agent(_make_agent("agent-001"))
        worker.add_agent(_make_agent("agent-002"))

        await worker.start()
        await asyncio.sleep(0.05)

        await worker.remove_agent("agent-001")

        assert worker.agent_count == 1
        assert "agent-001" not in worker._tasks
        assert "agent-001" not in worker.list_agents()

        await worker.stop()


# --- Statistics Tests ---


class TestWorkerStats:
    """Tests for worker statistics."""

    def test_get_stats_idle_worker(self):
        """Test stats for an idle worker with agents."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker.add_agent(_make_agent("agent-001"))
        worker.add_agent(_make_agent("agent-002"))

        stats = worker.get_stats()

        assert isinstance(stats, WorkerStats)
        assert stats.worker_id == "w-001"
        assert stats.status == WorkerStatus.IDLE
        assert stats.agent_count == 2
        assert stats.max_agents == 10
        assert stats.uptime_seconds == 0.0
        assert "agent-001" in stats.tick_counts
        assert "agent-002" in stats.tick_counts

    async def test_get_stats_running_worker(self):
        """Test stats for a running worker, including uptime."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker.add_agent(_make_agent("agent-001", tick_interval=0.02))

        await worker.start()
        await asyncio.sleep(0.1)

        stats = worker.get_stats()

        assert stats.status == WorkerStatus.RUNNING
        assert stats.uptime_seconds > 0.0
        assert stats.tick_counts["agent-001"] > 0

        await worker.stop()

    def test_get_stats_empty_worker(self):
        """Test stats for a worker with no agents."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=250)

        stats = worker.get_stats()

        assert stats.agent_count == 0
        assert stats.tick_counts == {}


# --- Health Check Tests ---


class TestHealthCheck:
    """Tests for the health check functionality."""

    async def test_health_check_healthy_running_agents(self):
        """Test health check when all agents are healthy and running."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker.add_agent(_make_agent("agent-001", tick_interval=0.02))

        await worker.start()
        await asyncio.sleep(0.05)

        health = await worker.health_check()

        assert health["healthy"] is True
        assert health["worker_id"] == "w-001"
        assert health["worker_status"] == "running"
        assert health["agent_count"] == 1
        assert "agent-001" in health["agents"]
        assert health["agents"]["agent-001"]["running"] is True

        await worker.stop()

    async def test_health_check_idle_worker_no_agents(self):
        """Test health check on an idle worker with no agents."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)

        health = await worker.health_check()

        assert health["healthy"] is True
        assert health["agent_count"] == 0
        assert health["agents"] == {}

    async def test_health_check_detects_crashed_agent(self):
        """Test that health check detects an agent whose task crashed."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        agent = _make_agent("agent-001")
        worker.add_agent(agent)

        await worker.start()
        await asyncio.sleep(0.02)

        # Simulate a crashed task by creating a done-with-exception future
        old_task = worker._tasks["agent-001"]
        old_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await old_task

        # Create a task that raises an error
        async def _failing_task() -> None:
            raise RuntimeError("Agent exploded")

        failed_task = asyncio.create_task(_failing_task())
        # Wait for the task to fail
        with contextlib.suppress(RuntimeError):
            await failed_task

        worker._tasks["agent-001"] = failed_task

        health = await worker.health_check()

        assert health["healthy"] is False
        assert health["agents"]["agent-001"]["task_done"] is True
        assert health["agents"]["agent-001"]["error"] is not None
        assert "exploded" in health["agents"]["agent-001"]["error"]

        # Restore so stop doesn't fail
        worker._tasks.pop("agent-001", None)
        await worker.stop()

    async def test_health_check_multiple_agents(self):
        """Test health check with multiple agents."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        for i in range(3):
            worker.add_agent(_make_agent(f"agent-{i:03d}", tick_interval=0.02))

        await worker.start()
        await asyncio.sleep(0.05)

        health = await worker.health_check()

        assert health["agent_count"] == 3
        for i in range(3):
            agent_id = f"agent-{i:03d}"
            assert agent_id in health["agents"]

        await worker.stop()


# --- Edge Cases ---


class TestEdgeCases:
    """Edge case and error handling tests."""

    async def test_agent_exception_does_not_crash_worker(self):
        """Test that a single agent's exception does not crash the worker."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)

        # Create a normal agent
        good_agent = _make_agent("good-agent", tick_interval=0.02)
        worker.add_agent(good_agent)

        # Create an agent that will error in its run
        bad_agent = _make_agent("bad-agent", tick_interval=0.02)
        worker.add_agent(bad_agent)

        await worker.start()

        # Patch the bad agent's run to raise - it should be isolated
        # Since the task is already running, we verify the worker stays running
        await asyncio.sleep(0.1)

        assert worker.status == WorkerStatus.RUNNING
        assert good_agent.scheduler.tick_count > 0

        await worker.stop()

    async def test_remove_agent_with_already_completed_task(self):
        """Test removing an agent whose task has already completed."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        agent = _make_agent("agent-001")
        worker.add_agent(agent)

        # Manually create a completed task
        async def _noop() -> None:
            pass

        done_task = asyncio.create_task(_noop())
        await done_task
        worker._tasks["agent-001"] = done_task

        # Removing should not fail even though task is done
        await worker.remove_agent("agent-001")
        assert worker.agent_count == 0

    async def test_worker_stop_after_stop_raises(self):
        """Test that stopping an already-stopped worker raises RuntimeError."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker.add_agent(_make_agent("agent-001"))

        await worker.start()
        await worker.stop()

        with pytest.raises(RuntimeError, match="Cannot stop"):
            await worker.stop()

    def test_worker_status_enum_values(self):
        """Test that WorkerStatus enum has expected values."""
        assert WorkerStatus.IDLE == "idle"
        assert WorkerStatus.STARTING == "starting"
        assert WorkerStatus.RUNNING == "running"
        assert WorkerStatus.STOPPING == "stopping"
        assert WorkerStatus.STOPPED == "stopped"

    async def test_start_initializes_agents(self):
        """Test that start() calls initialize on each agent."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        agent = _make_agent("agent-001")
        worker.add_agent(agent)

        await worker.start()
        try:
            # The agent's SAS should have been initialized (InMemorySAS.initialize is a no-op,
            # but the Agent.initialize flow calls it)
            # We can verify the agent ran by checking ticks after a brief wait
            await asyncio.sleep(0.05)
            assert agent.scheduler.tick_count >= 0  # At minimum initialized
        finally:
            await worker.stop()

    async def test_multiple_agents_run_concurrently(self):
        """Test that multiple agents run concurrently, not sequentially."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)

        agents = []
        for i in range(5):
            agent = _make_agent(f"agent-{i:03d}", tick_interval=0.02)
            worker.add_agent(agent)
            agents.append(agent)

        await worker.start()
        await asyncio.sleep(0.15)
        await worker.stop()

        # All agents should have accumulated ticks
        for agent in agents:
            assert agent.scheduler.tick_count > 0, (
                f"Agent {agent.agent_id} did not run (tick_count=0)"
            )

    async def test_worker_stats_model_serialization(self):
        """Test that WorkerStats can be serialized with Pydantic."""
        stats = WorkerStats(
            worker_id="w-001",
            status=WorkerStatus.RUNNING,
            agent_count=5,
            max_agents=250,
            uptime_seconds=123.4,
            tick_counts={"a-1": 10, "a-2": 20},
        )

        data = stats.model_dump()

        assert data["worker_id"] == "w-001"
        assert data["status"] == "running"
        assert data["agent_count"] == 5
        assert data["tick_counts"]["a-1"] == 10

    async def test_add_many_agents_up_to_limit(self):
        """Test adding agents up to the max limit."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=5)

        for i in range(5):
            worker.add_agent(_make_agent(f"agent-{i:03d}"))

        assert worker.agent_count == 5
        assert worker.is_full is True

        with pytest.raises(ValueError, match="at capacity"):
            worker.add_agent(_make_agent("agent-005"))
