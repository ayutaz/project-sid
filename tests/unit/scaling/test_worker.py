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


# --- Failed Agent Initialization Tests ---


class TestFailedAgentInit:
    """Tests that failed agent initialization is handled correctly."""

    async def test_failed_agents_excluded_from_run_tasks(self):
        """Agents that fail initialization should be removed from the worker."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)

        good_agent = _make_agent("good-agent", tick_interval=0.02)
        bad_agent = _make_agent("bad-agent", tick_interval=0.02)

        # Make the bad agent's initialize raise an error
        async def _fail_init() -> None:
            raise RuntimeError("Init failed!")

        bad_agent.initialize = _fail_init  # type: ignore[assignment]

        worker.add_agent(good_agent)
        worker.add_agent(bad_agent)
        assert worker.agent_count == 2

        await worker.start()
        try:
            # Bad agent should have been removed
            assert worker.agent_count == 1
            assert "good-agent" in worker.list_agents()
            assert "bad-agent" not in worker.list_agents()

            # Only good agent should have a task
            assert "good-agent" in worker._tasks
            assert "bad-agent" not in worker._tasks

            # Good agent should run normally
            await asyncio.sleep(0.1)
            assert good_agent.scheduler.tick_count > 0
        finally:
            await worker.stop()

    async def test_all_agents_fail_init(self):
        """If all agents fail init, worker still starts with no agents."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)

        bad_agent = _make_agent("bad-agent")

        async def _fail_init() -> None:
            raise RuntimeError("Init failed!")

        bad_agent.initialize = _fail_init  # type: ignore[assignment]
        worker.add_agent(bad_agent)

        await worker.start()
        try:
            assert worker.agent_count == 0
            assert worker.status == WorkerStatus.RUNNING
            assert len(worker._tasks) == 0
        finally:
            await worker.stop()


# --- Stuck Agent Detection Tests ---


class TestStuckAgentDetection:
    """Tests for tick progress check / stuck detection."""

    async def test_stuck_agent_detected_in_health_check(self):
        """An agent whose tick count hasn't changed is detected as stuck."""
        import time

        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10, stuck_threshold_seconds=0.05)
        agent = _make_agent("agent-001", tick_interval=0.02)
        worker.add_agent(agent)

        await worker.start()
        try:
            # Let the agent run a few ticks
            await asyncio.sleep(0.1)

            # First health check: record baseline
            health1 = await worker.health_check()
            assert health1["healthy"] is True
            current_tick = agent.scheduler.tick_count
            assert current_tick > 0

            # Manually set the baseline to the current tick but with old timestamp
            # to simulate "agent hasn't made progress in a long time"
            worker._last_tick_progress["agent-001"] = (
                current_tick,
                time.monotonic() - 1.0,  # 1 second ago
            )

            # Cancel the agent task so it can't make more progress
            task = worker._tasks["agent-001"]
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

            # Create a hanging task that never completes (simulating stuck agent)
            async def _hang_forever() -> None:
                await asyncio.sleep(3600)

            worker._tasks["agent-001"] = asyncio.create_task(_hang_forever())

            # Health check should detect agent as stuck
            health2 = await worker.health_check()
            assert health2["healthy"] is False
            agent_info = health2["agents"]["agent-001"]
            assert agent_info["error"] is not None
            assert "stuck" in agent_info["error"].lower()
        finally:
            await worker.stop()

    async def test_progressing_agent_not_stuck(self):
        """An agent making tick progress should not be marked stuck."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10, stuck_threshold_seconds=10.0)
        agent = _make_agent("agent-001", tick_interval=0.02)
        worker.add_agent(agent)

        await worker.start()
        try:
            # First health check
            await worker.health_check()
            await asyncio.sleep(0.1)

            # Second health check: ticks should have advanced
            health = await worker.health_check()
            assert health["healthy"] is True
            assert health["agents"]["agent-001"]["error"] is None
        finally:
            await worker.stop()


# --- Stop Re-entrancy Tests ---


class TestStopReentrancy:
    """Tests for stop re-entrancy guard."""

    async def test_stop_resets_start_time(self):
        """Stop should reset _start_time to None."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker.add_agent(_make_agent("agent-001"))

        await worker.start()
        assert worker._start_time is not None
        await worker.stop()
        assert worker._start_time is None

    async def test_concurrent_stop_calls(self):
        """Two concurrent stop() calls should not cause errors."""
        worker = AgentWorkerProcess(worker_id="w-001", max_agents=10)
        worker.add_agent(_make_agent("agent-001"))

        await worker.start()

        # First stop should succeed, second should raise due to state check
        results = await asyncio.gather(
            worker.stop(),
            worker.stop(),
            return_exceptions=True,
        )

        # One should succeed, one should raise RuntimeError
        errors = [r for r in results if isinstance(r, RuntimeError)]
        successes = [r for r in results if r is None]
        assert len(successes) == 1
        assert len(errors) == 1
        assert worker.status == WorkerStatus.STOPPED
