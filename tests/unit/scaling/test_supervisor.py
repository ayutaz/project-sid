"""Tests for the AgentSupervisor multi-process worker manager."""

from __future__ import annotations

import pytest

from piano.scaling.supervisor import (
    AgentSupervisor,
    SupervisorConfig,
    WorkerHandle,
    WorkerInfo,
    WorkerState,
    WorkerStats,
)

# --- Mock Worker ---


class MockWorker:
    """Mock WorkerHandle implementation for testing.

    Tracks all method calls without requiring real multiprocessing.
    """

    def __init__(self, worker_id: str) -> None:
        self._worker_id = worker_id
        self._agents: dict[str, str] = {}  # agent_id -> shard_id
        self._started = False
        self._stopped = False
        self._healthy = True
        self.start_count = 0
        self.stop_count = 0
        self.add_agent_calls: list[tuple[str, str]] = []
        self.remove_agent_calls: list[str] = []

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def start(self) -> None:
        self._started = True
        self._stopped = False
        self.start_count += 1

    async def stop(self) -> None:
        self._stopped = True
        self._started = False
        self.stop_count += 1

    async def add_agent(self, agent_id: str, shard_id: str) -> None:
        self._agents[agent_id] = shard_id
        self.add_agent_calls.append((agent_id, shard_id))

    async def remove_agent(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)
        self.remove_agent_calls.append(agent_id)

    async def get_stats(self) -> WorkerStats:
        return WorkerStats(
            worker_id=self._worker_id,
            agent_count=len(self._agents),
            state=WorkerState.RUNNING if self._started else WorkerState.IDLE,
        )

    async def health_check(self) -> bool:
        return self._healthy

    def set_unhealthy(self) -> None:
        """Make this worker report unhealthy for testing."""
        self._healthy = False


class FailingWorker(MockWorker):
    """A worker that raises on start()."""

    async def start(self) -> None:
        raise RuntimeError("Worker failed to start")


# --- Fixtures ---


@pytest.fixture
def mock_workers() -> dict[str, MockWorker]:
    """Registry of created mock workers for inspection."""
    return {}


@pytest.fixture
def worker_factory(mock_workers: dict[str, MockWorker]):
    """Factory that creates MockWorker instances and tracks them."""

    def factory(worker_id: str) -> MockWorker:
        worker = MockWorker(worker_id)
        mock_workers[worker_id] = worker
        return worker

    return factory


@pytest.fixture
def config() -> SupervisorConfig:
    """Default test supervisor configuration with small limits."""
    return SupervisorConfig(agents_per_worker=3, max_workers=4, health_check_interval=5.0)


@pytest.fixture
def supervisor(config, worker_factory) -> AgentSupervisor:
    """Create a supervisor with mock worker factory."""
    return AgentSupervisor(config, worker_factory=worker_factory)


# --- SupervisorConfig Tests ---


async def test_supervisor_config_defaults():
    """Test that SupervisorConfig has correct default values."""
    cfg = SupervisorConfig()

    assert cfg.agents_per_worker == 250
    assert cfg.max_workers == 8
    assert cfg.health_check_interval == 30.0


async def test_supervisor_config_custom_values():
    """Test that SupervisorConfig accepts custom values."""
    cfg = SupervisorConfig(agents_per_worker=100, max_workers=16, health_check_interval=10.0)

    assert cfg.agents_per_worker == 100
    assert cfg.max_workers == 16
    assert cfg.health_check_interval == 10.0


# --- WorkerHandle Protocol Tests ---


async def test_mock_worker_satisfies_protocol():
    """Test that MockWorker satisfies the WorkerHandle protocol."""
    worker = MockWorker("test-worker")
    assert isinstance(worker, WorkerHandle)


# --- Worker Creation Tests ---


async def test_create_worker(supervisor, mock_workers):
    """Test that create_worker creates and registers a worker."""
    handle = supervisor.create_worker("worker-A")

    assert handle is not None
    assert handle.worker_id == "worker-A"
    assert supervisor.worker_count == 1
    assert "worker-A" in supervisor.worker_ids
    assert "worker-A" in mock_workers


async def test_create_worker_duplicate_raises_error(supervisor):
    """Test that creating a worker with duplicate ID raises ValueError."""
    supervisor.create_worker("worker-A")

    with pytest.raises(ValueError, match="already exists"):
        supervisor.create_worker("worker-A")


async def test_create_worker_exceeds_max_workers(supervisor):
    """Test that exceeding max_workers raises ValueError."""
    for i in range(4):  # max_workers=4
        supervisor.create_worker(f"worker-{i}")

    with pytest.raises(ValueError, match="max_workers limit"):
        supervisor.create_worker("worker-extra")


async def test_create_worker_no_factory_raises_error():
    """Test that create_worker without factory raises RuntimeError."""
    cfg = SupervisorConfig(max_workers=4)
    supervisor = AgentSupervisor(cfg, worker_factory=None)

    with pytest.raises(RuntimeError, match="No worker_factory configured"):
        supervisor.create_worker("worker-0")


# --- Agent Assignment Tests ---


async def test_assign_agent_creates_worker_automatically(supervisor, mock_workers):
    """Test that assign_agent auto-creates a worker when none exist."""
    worker_id = await supervisor.assign_agent("agent-001", "shard-0")

    assert worker_id is not None
    assert supervisor.worker_count == 1
    assert supervisor.agent_count == 1
    assert supervisor.get_agent_worker("agent-001") == worker_id


async def test_assign_agent_reuses_existing_worker(supervisor):
    """Test that assign_agent reuses an existing worker with capacity."""
    w1 = await supervisor.assign_agent("agent-001", "shard-0")
    w2 = await supervisor.assign_agent("agent-002", "shard-0")

    assert w1 == w2
    assert supervisor.worker_count == 1
    assert supervisor.agent_count == 2


async def test_assign_agent_creates_new_worker_when_full(supervisor):
    """Test auto-scaling: new worker created when current one is full."""
    # agents_per_worker=3 in fixture
    await supervisor.assign_agent("agent-001", "shard-0")
    await supervisor.assign_agent("agent-002", "shard-0")
    await supervisor.assign_agent("agent-003", "shard-0")

    # 4th agent should trigger new worker
    w4 = await supervisor.assign_agent("agent-004", "shard-1")

    assert supervisor.worker_count == 2
    assert supervisor.agent_count == 4
    # New agent should be on a different worker than first 3
    w1 = supervisor.get_agent_worker("agent-001")
    assert w4 != w1


async def test_assign_agent_duplicate_raises_error(supervisor):
    """Test that assigning the same agent twice raises ValueError."""
    await supervisor.assign_agent("agent-001", "shard-0")

    with pytest.raises(ValueError, match="already assigned"):
        await supervisor.assign_agent("agent-001", "shard-0")


async def test_assign_agent_at_full_capacity_raises_error(supervisor):
    """Test that assigning when all workers are at capacity raises ValueError."""
    # 4 workers x 3 agents = 12 max
    for i in range(12):
        await supervisor.assign_agent(f"agent-{i:03d}", "shard-0")

    assert supervisor.worker_count == 4
    assert supervisor.agent_count == 12

    with pytest.raises(ValueError, match="at capacity"):
        await supervisor.assign_agent("agent-overflow", "shard-0")


async def test_assign_agent_passes_shard_id_to_worker(supervisor, mock_workers):
    """Test that shard_id is passed through to the worker handle."""
    await supervisor.assign_agent("agent-001", "shard-42")

    # Find the worker that was created
    worker = next(iter(mock_workers.values()))
    assert ("agent-001", "shard-42") in worker.add_agent_calls


# --- Agent Removal Tests ---


async def test_remove_agent(supervisor):
    """Test basic agent removal."""
    await supervisor.assign_agent("agent-001", "shard-0")
    assert supervisor.agent_count == 1

    await supervisor.remove_agent("agent-001")

    assert supervisor.agent_count == 0
    assert supervisor.get_agent_worker("agent-001") is None


async def test_remove_agent_nonexistent_raises_error(supervisor):
    """Test that removing non-existent agent raises ValueError."""
    with pytest.raises(ValueError, match="not assigned"):
        await supervisor.remove_agent("agent-nonexistent")


async def test_remove_agent_calls_worker(supervisor, mock_workers):
    """Test that remove_agent delegates to the worker handle."""
    await supervisor.assign_agent("agent-001", "shard-0")
    worker = next(iter(mock_workers.values()))

    await supervisor.remove_agent("agent-001")

    assert "agent-001" in worker.remove_agent_calls


async def test_remove_agent_frees_capacity(supervisor):
    """Test that removing an agent frees capacity on the worker."""
    # Fill a worker (capacity=3)
    await supervisor.assign_agent("agent-001", "shard-0")
    await supervisor.assign_agent("agent-002", "shard-0")
    await supervisor.assign_agent("agent-003", "shard-0")
    worker_count_before = supervisor.worker_count

    # Remove one agent
    await supervisor.remove_agent("agent-002")

    # Next agent should go to existing worker (not a new one)
    await supervisor.assign_agent("agent-004", "shard-0")
    assert supervisor.worker_count == worker_count_before


# --- Lifecycle Tests ---


async def test_start_all(supervisor, mock_workers):
    """Test that start_all starts all workers."""
    supervisor.create_worker("worker-A")
    supervisor.create_worker("worker-B")

    await supervisor.start_all()

    assert mock_workers["worker-A"].start_count == 1
    assert mock_workers["worker-B"].start_count == 1


async def test_start_all_no_workers(supervisor):
    """Test that start_all with no workers does not raise."""
    await supervisor.start_all()  # Should not raise


async def test_start_all_sets_running_state(supervisor):
    """Test that start_all sets worker state to RUNNING."""
    supervisor.create_worker("worker-A")
    await supervisor.start_all()

    stats = supervisor.get_worker_stats()
    assert stats["worker-A"]["state"] == "running"


async def test_start_all_handles_worker_failure(config):
    """Test that start_all handles workers that fail to start."""
    workers: dict[str, MockWorker] = {}

    def failing_factory(worker_id: str) -> MockWorker:
        worker = FailingWorker(worker_id)
        workers[worker_id] = worker
        return worker

    supervisor = AgentSupervisor(config, worker_factory=failing_factory)
    supervisor.create_worker("worker-fail")

    # Should not raise even if worker fails
    await supervisor.start_all()

    stats = supervisor.get_worker_stats()
    assert stats["worker-fail"]["state"] == "unhealthy"


async def test_stop_all(supervisor, mock_workers):
    """Test that stop_all stops all workers."""
    supervisor.create_worker("worker-A")
    supervisor.create_worker("worker-B")

    await supervisor.start_all()
    await supervisor.stop_all()

    assert mock_workers["worker-A"].stop_count == 1
    assert mock_workers["worker-B"].stop_count == 1


async def test_stop_all_no_workers(supervisor):
    """Test that stop_all with no workers does not raise."""
    await supervisor.stop_all()  # Should not raise


async def test_stop_all_sets_stopped_state(supervisor):
    """Test that stop_all sets worker state to STOPPED."""
    supervisor.create_worker("worker-A")
    await supervisor.start_all()
    await supervisor.stop_all()

    stats = supervisor.get_worker_stats()
    assert stats["worker-A"]["state"] == "stopped"


# --- Statistics Tests ---


async def test_get_worker_stats_empty(supervisor):
    """Test get_worker_stats with no workers."""
    stats = supervisor.get_worker_stats()
    assert stats == {}


async def test_get_worker_stats_with_agents(supervisor):
    """Test get_worker_stats returns correct agent counts."""
    await supervisor.assign_agent("agent-001", "shard-0")
    await supervisor.assign_agent("agent-002", "shard-0")

    stats = supervisor.get_worker_stats()

    assert len(stats) == 1
    worker_stats = next(iter(stats.values()))
    assert worker_stats["agent_count"] == 2
    assert worker_stats["capacity"] == 3
    assert sorted(worker_stats["agent_ids"]) == ["agent-001", "agent-002"]
    assert worker_stats["utilization"] == pytest.approx(2 / 3)


async def test_get_worker_stats_multiple_workers(supervisor):
    """Test get_worker_stats across multiple workers."""
    # Fill first worker (capacity=3)
    await supervisor.assign_agent("agent-001", "shard-0")
    await supervisor.assign_agent("agent-002", "shard-0")
    await supervisor.assign_agent("agent-003", "shard-0")
    # New worker gets this one
    await supervisor.assign_agent("agent-004", "shard-1")

    stats = supervisor.get_worker_stats()

    assert len(stats) == 2
    total_agents = sum(s["agent_count"] for s in stats.values())
    assert total_agents == 4


# --- Rebalance Tests ---


async def test_rebalance_no_workers(supervisor):
    """Test that rebalance with no workers returns empty dict."""
    result = await supervisor.rebalance()
    assert result == {}


async def test_rebalance_single_worker(supervisor):
    """Test that rebalance with single worker returns empty dict."""
    await supervisor.assign_agent("agent-001", "shard-0")
    result = await supervisor.rebalance()
    assert result == {}


async def test_rebalance_already_balanced(supervisor):
    """Test that rebalance on balanced workers returns empty dict."""
    # Create 2 workers manually, put 1 agent on each
    supervisor.create_worker("worker-A")
    supervisor.create_worker("worker-B")

    # Manually add 1 agent to each worker (via assign which picks worker-A first)
    # We need to directly manipulate to get specific distribution
    handle_a = supervisor._workers["worker-A"]
    await handle_a.add_agent("agent-001", "shard-0")
    supervisor._worker_info["worker-A"].agent_ids.add("agent-001")
    supervisor._worker_info["worker-A"].shard_ids["agent-001"] = "shard-0"
    supervisor._agent_to_worker["agent-001"] = "worker-A"

    handle_b = supervisor._workers["worker-B"]
    await handle_b.add_agent("agent-002", "shard-0")
    supervisor._worker_info["worker-B"].agent_ids.add("agent-002")
    supervisor._worker_info["worker-B"].shard_ids["agent-002"] = "shard-0"
    supervisor._agent_to_worker["agent-002"] = "worker-B"

    result = await supervisor.rebalance()
    assert result == {}


async def test_rebalance_moves_agents(supervisor, mock_workers):
    """Test that rebalance moves agents from overloaded to underloaded workers."""
    # Create 2 workers manually
    supervisor.create_worker("worker-A")
    supervisor.create_worker("worker-B")

    # Put 3 agents on worker-A, 0 on worker-B (imbalanced)
    for agent_id in ["agent-001", "agent-002", "agent-003"]:
        handle = supervisor._workers["worker-A"]
        await handle.add_agent(agent_id, "shard-0")
        supervisor._worker_info["worker-A"].agent_ids.add(agent_id)
        supervisor._worker_info["worker-A"].shard_ids[agent_id] = "shard-0"
        supervisor._agent_to_worker[agent_id] = "worker-A"

    # Rebalance: target = ceil(3/2) = 2 per worker
    # worker-A has 3 (over), worker-B has 0 (under)
    moves = await supervisor.rebalance()

    # At least one agent should have moved to worker-B
    assert "worker-B" in moves
    assert len(moves["worker-B"]) >= 1

    # Verify final distribution is more balanced
    count_a = len(supervisor._worker_info["worker-A"].agent_ids)
    count_b = len(supervisor._worker_info["worker-B"].agent_ids)
    assert abs(count_a - count_b) <= 1  # within 1 of balanced


async def test_rebalance_preserves_shard_ids(supervisor, mock_workers):
    """Test that rebalance preserves shard_ids when moving agents."""
    supervisor.create_worker("worker-A")
    supervisor.create_worker("worker-B")

    # Put 3 agents on worker-A with distinct shard IDs
    for i, agent_id in enumerate(["agent-001", "agent-002", "agent-003"]):
        shard_id = f"shard-{i}"
        handle = supervisor._workers["worker-A"]
        await handle.add_agent(agent_id, shard_id)
        supervisor._worker_info["worker-A"].agent_ids.add(agent_id)
        supervisor._worker_info["worker-A"].shard_ids[agent_id] = shard_id
        supervisor._agent_to_worker[agent_id] = "worker-A"

    await supervisor.rebalance()

    # All agents should still have their shard_ids regardless of which worker
    for _worker_id, info in supervisor._worker_info.items():
        for agent_id in info.agent_ids:
            assert agent_id in info.shard_ids
            # shard_id should match original assignment
            idx = int(agent_id.split("-")[1]) - 1
            assert info.shard_ids[agent_id] == f"shard-{idx}"


# --- Health Check Tests ---


async def test_health_check_all_healthy(supervisor, mock_workers):
    """Test health_check_all when all workers are healthy."""
    supervisor.create_worker("worker-A")
    supervisor.create_worker("worker-B")

    results = await supervisor.health_check_all()

    assert results == {"worker-A": True, "worker-B": True}


async def test_health_check_all_unhealthy_worker(supervisor, mock_workers):
    """Test health_check_all detects unhealthy worker."""
    supervisor.create_worker("worker-A")
    supervisor.create_worker("worker-B")

    # Make worker-B unhealthy
    mock_workers["worker-B"].set_unhealthy()

    results = await supervisor.health_check_all()

    assert results["worker-A"] is True
    assert results["worker-B"] is False

    # Worker state should be updated
    stats = supervisor.get_worker_stats()
    assert stats["worker-B"]["state"] == "unhealthy"


async def test_health_check_all_exception(config):
    """Test health_check_all handles workers that raise on health_check."""

    class ErrorWorker(MockWorker):
        async def health_check(self) -> bool:
            raise ConnectionError("Worker unreachable")

    workers: dict[str, MockWorker] = {}

    def error_factory(worker_id: str) -> MockWorker:
        w = ErrorWorker(worker_id)
        workers[worker_id] = w
        return w

    supervisor = AgentSupervisor(config, worker_factory=error_factory)
    supervisor.create_worker("worker-err")

    results = await supervisor.health_check_all()

    assert results["worker-err"] is False


# --- Properties Tests ---


async def test_properties_initial_state(supervisor):
    """Test supervisor properties in initial state."""
    assert supervisor.worker_count == 0
    assert supervisor.agent_count == 0
    assert supervisor.worker_ids == []
    assert supervisor.config.agents_per_worker == 3
    assert supervisor.config.max_workers == 4


async def test_worker_ids_returns_all_workers(supervisor):
    """Test that worker_ids returns all registered worker IDs."""
    supervisor.create_worker("worker-A")
    supervisor.create_worker("worker-B")

    ids = supervisor.worker_ids
    assert sorted(ids) == ["worker-A", "worker-B"]


# --- WorkerInfo / WorkerStats Model Tests ---


async def test_worker_info_model():
    """Test WorkerInfo Pydantic model."""
    info = WorkerInfo(worker_id="worker-0")

    assert info.worker_id == "worker-0"
    assert info.agent_ids == set()
    assert info.shard_ids == {}
    assert info.state == WorkerState.IDLE


async def test_worker_stats_model():
    """Test WorkerStats Pydantic model."""
    stats = WorkerStats(worker_id="worker-0", agent_count=5, state=WorkerState.RUNNING)

    assert stats.worker_id == "worker-0"
    assert stats.agent_count == 5
    assert stats.state == WorkerState.RUNNING
    assert stats.tick_count == 0
    assert stats.errors == 0


async def test_worker_state_values():
    """Test WorkerState enum values."""
    assert WorkerState.IDLE == "idle"
    assert WorkerState.RUNNING == "running"
    assert WorkerState.STOPPED == "stopped"
    assert WorkerState.UNHEALTHY == "unhealthy"


# --- Full Lifecycle Integration Test ---


async def test_full_lifecycle(supervisor, mock_workers):
    """Test complete supervisor lifecycle: create, assign, start, stats, stop, remove."""
    # 1. Assign agents (workers auto-created)
    w1 = await supervisor.assign_agent("agent-001", "shard-0")
    w2 = await supervisor.assign_agent("agent-002", "shard-0")
    w3 = await supervisor.assign_agent("agent-003", "shard-0")

    assert supervisor.worker_count == 1
    assert supervisor.agent_count == 3
    assert w1 == w2 == w3

    # 2. Add more to trigger second worker
    w4 = await supervisor.assign_agent("agent-004", "shard-1")

    assert supervisor.worker_count == 2
    assert w4 != w1

    # 3. Start all
    await supervisor.start_all()

    for worker in mock_workers.values():
        assert worker.start_count == 1

    stats = supervisor.get_worker_stats()
    for s in stats.values():
        assert s["state"] == "running"

    # 4. Check stats
    total = sum(s["agent_count"] for s in stats.values())
    assert total == 4

    # 5. Health check
    health = await supervisor.health_check_all()
    assert all(health.values())

    # 6. Remove an agent
    await supervisor.remove_agent("agent-002")
    assert supervisor.agent_count == 3

    # 7. Stop all
    await supervisor.stop_all()

    for worker in mock_workers.values():
        assert worker.stop_count == 1

    stats = supervisor.get_worker_stats()
    for s in stats.values():
        assert s["state"] == "stopped"


async def test_auto_scaling_1000_agents():
    """Test that supervisor can handle 1000 agents across multiple workers."""
    config = SupervisorConfig(agents_per_worker=250, max_workers=8)

    workers: dict[str, MockWorker] = {}

    def factory(worker_id: str) -> MockWorker:
        w = MockWorker(worker_id)
        workers[worker_id] = w
        return w

    supervisor = AgentSupervisor(config, worker_factory=factory)

    # Assign 1000 agents
    for i in range(1000):
        await supervisor.assign_agent(f"agent-{i:04d}", f"shard-{i % 4}")

    assert supervisor.agent_count == 1000
    assert supervisor.worker_count == 4  # 1000 / 250 = 4

    # Verify even distribution
    stats = supervisor.get_worker_stats()
    for s in stats.values():
        assert s["agent_count"] == 250
        assert s["utilization"] == pytest.approx(1.0)
