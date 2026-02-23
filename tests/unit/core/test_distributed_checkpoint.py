"""Tests for distributed checkpoint system with CC-cycle alignment."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from piano.core.distributed_checkpoint import (
    CCCycleAligner,
    CheckpointInfo,
    DistributedCheckpointManager,
    ShardCheckpointCoordinator,
)

# ---------------------------------------------------------------------------
# CheckpointInfo model
# ---------------------------------------------------------------------------


class TestCheckpointInfo:
    """Tests for the CheckpointInfo Pydantic model."""

    def test_creation_with_defaults(self):
        info = CheckpointInfo(shard_id=0)
        assert info.shard_id == 0
        assert info.agent_count == 0
        assert info.size_bytes == 0
        assert isinstance(info.timestamp, datetime)
        assert isinstance(info.checkpoint_id, str)
        assert len(info.checkpoint_id) > 0

    def test_creation_with_explicit_values(self):
        ts = datetime(2026, 2, 23, 12, 0, 0, tzinfo=UTC)
        info = CheckpointInfo(
            checkpoint_id="abc123",
            shard_id=3,
            timestamp=ts,
            agent_count=10,
            size_bytes=4096,
        )
        assert info.checkpoint_id == "abc123"
        assert info.shard_id == 3
        assert info.timestamp == ts
        assert info.agent_count == 10
        assert info.size_bytes == 4096

    def test_serialization_round_trip(self):
        info = CheckpointInfo(shard_id=1, agent_count=5, size_bytes=1024)
        data = info.model_dump(mode="json")
        restored = CheckpointInfo.model_validate(data)
        assert restored.shard_id == info.shard_id
        assert restored.agent_count == info.agent_count


# ---------------------------------------------------------------------------
# CCCycleAligner
# ---------------------------------------------------------------------------


class TestCCCycleAligner:
    """Tests for CC cycle boundary alignment."""

    def test_initial_state_not_safe(self):
        aligner = CCCycleAligner()
        assert aligner.is_safe_to_checkpoint("agent-001") is False

    def test_safe_after_cycle_completed(self):
        aligner = CCCycleAligner()
        aligner.mark_cycle_completed("agent-001")
        assert aligner.is_safe_to_checkpoint("agent-001") is True

    def test_not_safe_after_tick_started(self):
        aligner = CCCycleAligner()
        aligner.mark_cycle_completed("agent-001")
        aligner.mark_tick_started("agent-001")
        assert aligner.is_safe_to_checkpoint("agent-001") is False

    def test_agents_independent(self):
        aligner = CCCycleAligner()
        aligner.mark_cycle_completed("agent-001")
        assert aligner.is_safe_to_checkpoint("agent-001") is True
        assert aligner.is_safe_to_checkpoint("agent-002") is False

    async def test_wait_for_cc_boundary_already_safe(self):
        aligner = CCCycleAligner()
        aligner.mark_cycle_completed("agent-001")
        result = await aligner.wait_for_cc_boundary("agent-001", timeout=1.0)
        assert result is True

    async def test_wait_for_cc_boundary_timeout(self):
        aligner = CCCycleAligner()
        result = await aligner.wait_for_cc_boundary("agent-001", timeout=0.05)
        assert result is False

    async def test_wait_for_cc_boundary_signalled(self):
        aligner = CCCycleAligner()

        async def _signal_later():
            await asyncio.sleep(0.02)
            aligner.mark_cycle_completed("agent-001")

        task = asyncio.create_task(_signal_later())
        result = await aligner.wait_for_cc_boundary("agent-001", timeout=2.0)
        assert result is True
        await task

    def test_invalid_agent_id_rejected(self):
        aligner = CCCycleAligner()
        with pytest.raises(ValueError, match="Invalid agent_id"):
            aligner.mark_cycle_completed("../bad")
        with pytest.raises(ValueError, match="Invalid agent_id"):
            aligner.is_safe_to_checkpoint("")

    def test_cycle_lifecycle(self):
        """Full cycle: tick_started -> cycle_completed -> safe -> tick_started -> not safe."""
        aligner = CCCycleAligner()
        aligner.mark_tick_started("a1")
        assert aligner.is_safe_to_checkpoint("a1") is False
        aligner.mark_cycle_completed("a1")
        assert aligner.is_safe_to_checkpoint("a1") is True
        aligner.mark_tick_started("a1")
        assert aligner.is_safe_to_checkpoint("a1") is False


# ---------------------------------------------------------------------------
# DistributedCheckpointManager
# ---------------------------------------------------------------------------


class TestDistributedCheckpointManager:
    """Tests for the shard-based checkpoint manager."""

    async def test_save_and_restore_shard(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        states: dict[str, dict[str, Any]] = {
            "agent-001": {"goals": {"current_goal": "mine"}},
            "agent-002": {"goals": {"current_goal": "build"}},
        }
        info = await mgr.save_shard(0, states)

        assert info.shard_id == 0
        assert info.agent_count == 2
        assert info.size_bytes > 0

        restored = await mgr.restore_shard(0, info.checkpoint_id)
        assert restored == states

    async def test_restore_nonexistent_raises(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            await mgr.restore_shard(0, "nonexistent")

    async def test_list_checkpoints_sorted(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        for i in range(3):
            await mgr.save_shard(0, {f"agent-{i:03d}": {"tick": i}})

        cps = mgr.list_checkpoints(0)
        assert len(cps) == 3
        # Sorted newest first
        for i in range(len(cps) - 1):
            assert cps[i].timestamp >= cps[i + 1].timestamp

    async def test_list_checkpoints_empty_shard(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        assert mgr.list_checkpoints(999) == []

    async def test_get_latest_checkpoint(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        assert mgr.get_latest_checkpoint(0) is None

        await mgr.save_shard(0, {"a": {}})
        info2 = await mgr.save_shard(0, {"a": {}, "b": {}})

        latest = mgr.get_latest_checkpoint(0)
        assert latest is not None
        assert latest.checkpoint_id == info2.checkpoint_id

    async def test_cleanup_old_keeps_count(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path, max_checkpoints_per_shard=20)
        for i in range(7):
            await mgr.save_shard(0, {f"agent-{i}": {"v": i}})

        deleted = await mgr.cleanup_old(0, keep_count=3)
        assert deleted == 4
        assert len(mgr.list_checkpoints(0)) == 3

    async def test_auto_rotation_on_save(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path, max_checkpoints_per_shard=3)
        for i in range(5):
            await mgr.save_shard(0, {"a": {"v": i}})

        assert len(mgr.list_checkpoints(0)) == 3

    async def test_schedule_checkpoint(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        result = mgr.schedule_checkpoint([0, 1, 2])
        assert set(result.keys()) == {0, 1, 2}
        for ts in result.values():
            assert isinstance(ts, datetime)

    async def test_separate_shards(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        await mgr.save_shard(0, {"a": {"s": 0}})
        await mgr.save_shard(1, {"b": {"s": 1}})

        assert len(mgr.list_checkpoints(0)) == 1
        assert len(mgr.list_checkpoints(1)) == 1

        r0 = await mgr.restore_shard(0, mgr.list_checkpoints(0)[0].checkpoint_id)
        r1 = await mgr.restore_shard(1, mgr.list_checkpoints(1)[0].checkpoint_id)
        assert r0 == {"a": {"s": 0}}
        assert r1 == {"b": {"s": 1}}

    async def test_creates_directory(self, tmp_path: Path):
        deep = tmp_path / "a" / "b" / "c"
        assert not deep.exists()
        _mgr = DistributedCheckpointManager(checkpoint_dir=deep)
        assert deep.exists()

    async def test_size_bytes_grows(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        info_small = await mgr.save_shard(0, {"a": {"v": 1}})
        big_state = {f"agent-{i}": {"data": "x" * 500} for i in range(20)}
        info_big = await mgr.save_shard(1, big_state)
        assert info_big.size_bytes > info_small.size_bytes


# ---------------------------------------------------------------------------
# Path traversal security
# ---------------------------------------------------------------------------


class TestDistributedCheckpointSecurity:
    """Security tests for path traversal prevention."""

    async def test_negative_shard_id_rejected(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        with pytest.raises(ValueError, match="non-negative"):
            await mgr.save_shard(-1, {})

    async def test_traversal_checkpoint_id_rejected(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        await mgr.save_shard(0, {"a": {}})
        with pytest.raises(ValueError, match="Invalid checkpoint_id"):
            await mgr.restore_shard(0, "../etc/passwd")
        with pytest.raises(ValueError, match="Invalid checkpoint_id"):
            await mgr.restore_shard(0, "..\\windows")

    async def test_empty_checkpoint_id_rejected(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        with pytest.raises(ValueError, match="Invalid checkpoint_id"):
            await mgr.restore_shard(0, "")

    async def test_special_chars_checkpoint_id_rejected(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        bad_ids = ["a/b", "a\\b", "a b", "a:b", "a@b", "a.b"]
        for bad in bad_ids:
            with pytest.raises(ValueError, match="Invalid checkpoint_id"):
                await mgr.restore_shard(0, bad)


# ---------------------------------------------------------------------------
# ShardCheckpointCoordinator
# ---------------------------------------------------------------------------


class TestShardCheckpointCoordinator:
    """Tests for multi-shard coordinated checkpoints."""

    async def test_coordinate_checkpoint_basic(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        coord = ShardCheckpointCoordinator(mgr)

        shard_states: dict[int, dict[str, dict[str, Any]]] = {
            0: {"agent-001": {"v": 1}},
            1: {"agent-002": {"v": 2}},
        }
        infos = await coord.coordinate_checkpoint([0, 1], shard_states)
        assert len(infos) == 2
        assert infos[0].shard_id == 0
        assert infos[1].shard_id == 1

    async def test_coordinate_single_shard(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        coord = ShardCheckpointCoordinator(mgr)
        infos = await coord.coordinate_checkpoint([0], {0: {"a": {}}})
        assert len(infos) == 1

    async def test_skew_within_tolerance(self, tmp_path: Path):
        """Sequential saves should have minimal skew."""
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        coord = ShardCheckpointCoordinator(mgr, max_skew_seconds=10.0)
        shard_states = {i: {f"a-{i}": {"v": i}} for i in range(5)}
        infos = await coord.coordinate_checkpoint(list(range(5)), shard_states)
        timestamps = [i.timestamp for i in infos]
        skew = (max(timestamps) - min(timestamps)).total_seconds()
        assert skew <= 10.0

    async def test_restore_shards(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        coord = ShardCheckpointCoordinator(mgr)

        # Save
        shard_states: dict[int, dict[str, dict[str, Any]]] = {
            0: {"agent-001": {"val": "hello"}},
            1: {"agent-002": {"val": "world"}},
        }
        infos = await coord.coordinate_checkpoint([0, 1], shard_states)

        # Restore
        plan = {info.shard_id: info.checkpoint_id for info in infos}
        restored = await coord.restore_shards(plan)
        assert restored[0] == {"agent-001": {"val": "hello"}}
        assert restored[1] == {"agent-002": {"val": "world"}}

    async def test_convergence_tracking(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        coord = ShardCheckpointCoordinator(mgr, convergence_window=0.05)

        # Before any restore, considered converged
        assert coord.is_converged() is True
        assert coord.convergence_elapsed() == 0.0

        # After restore, not converged until window elapses
        await mgr.save_shard(0, {"a": {}})
        cp = mgr.get_latest_checkpoint(0)
        assert cp is not None
        await coord.restore_shards({0: cp.checkpoint_id})

        assert coord.is_converged() is False
        assert coord.convergence_elapsed() >= 0.0

        # Wait for convergence
        await asyncio.sleep(0.06)
        assert coord.is_converged() is True

    async def test_coordinator_exposes_aligner(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        aligner = CCCycleAligner()
        coord = ShardCheckpointCoordinator(mgr, aligner=aligner)
        assert coord.aligner is aligner

    async def test_coordinate_with_empty_shard(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path)
        coord = ShardCheckpointCoordinator(mgr)
        infos = await coord.coordinate_checkpoint([0], {0: {}})
        assert len(infos) == 1
        assert infos[0].agent_count == 0

    async def test_multiple_coordinated_checkpoints(self, tmp_path: Path):
        mgr = DistributedCheckpointManager(checkpoint_dir=tmp_path, max_checkpoints_per_shard=10)
        coord = ShardCheckpointCoordinator(mgr)

        for i in range(3):
            await coord.coordinate_checkpoint(
                [0, 1],
                {0: {"a": {"v": i}}, 1: {"b": {"v": i}}},
            )

        assert len(mgr.list_checkpoints(0)) == 3
        assert len(mgr.list_checkpoints(1)) == 3
