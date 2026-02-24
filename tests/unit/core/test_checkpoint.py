"""Tests for checkpoint/restore system."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from piano.core.checkpoint import Checkpoint, CheckpointManager, CheckpointMetadata
from piano.core.types import (
    ActionHistoryEntry,
    GoalData,
    MemoryEntry,
    PerceptData,
    SocialData,
)
from tests.helpers import InMemorySAS


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata model."""

    def test_metadata_creation(self):
        """Test creating checkpoint metadata."""
        metadata = CheckpointMetadata(
            agent_id="agent-001",
            tick_count=100,
        )

        assert metadata.agent_id == "agent-001"
        assert metadata.tick_count == 100
        assert metadata.version == "1.0"
        assert metadata.size_bytes == 0
        assert isinstance(metadata.timestamp, datetime)

    def test_metadata_with_custom_values(self):
        """Test metadata with custom timestamp and size."""
        timestamp = datetime(2026, 2, 23, 12, 0, 0, tzinfo=UTC)
        metadata = CheckpointMetadata(
            agent_id="agent-002",
            tick_count=500,
            timestamp=timestamp,
            size_bytes=1024,
        )

        assert metadata.agent_id == "agent-002"
        assert metadata.tick_count == 500
        assert metadata.timestamp == timestamp
        assert metadata.size_bytes == 1024


class TestCheckpoint:
    """Tests for Checkpoint model."""

    def test_checkpoint_creation(self):
        """Test creating a complete checkpoint."""
        metadata = CheckpointMetadata(agent_id="agent-001", tick_count=100)
        snapshot = {"percepts": {}, "goals": {}}

        checkpoint = Checkpoint(metadata=metadata, sas_snapshot=snapshot)

        assert checkpoint.metadata == metadata
        assert checkpoint.sas_snapshot == snapshot


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    async def test_save_creates_checkpoint_file(self, tmp_path: Path):
        """Test that save creates checkpoint file on disk."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas = InMemorySAS("agent-001")

        # Save checkpoint
        metadata = await manager.save("agent-001", sas, tick_count=10)

        # Verify metadata
        assert metadata.agent_id == "agent-001"
        assert metadata.tick_count == 10
        assert metadata.size_bytes > 0

        # Verify file exists
        checkpoint_path = manager.get_checkpoint_path("agent-001", metadata)
        assert checkpoint_path.exists()
        assert checkpoint_path.is_file()

    async def test_restore_loads_checkpoint(self, tmp_path: Path):
        """Test that restore loads checkpoint and populates SAS."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas_original = InMemorySAS("agent-001")

        # Populate original SAS
        await sas_original.update_percepts(
            PerceptData(health=15.0, hunger=10.0, nearby_players=["Bob"])
        )
        await sas_original.update_goals(GoalData(current_goal="Build house"))

        # Save checkpoint
        await manager.save("agent-001", sas_original, tick_count=42)

        # Create new SAS and restore
        sas_restored = InMemorySAS("agent-001")
        metadata = await manager.restore("agent-001", sas_restored)

        # Verify metadata
        assert metadata.agent_id == "agent-001"
        assert metadata.tick_count == 42

        # Verify restored data
        percepts = await sas_restored.get_percepts()
        assert percepts.health == 15.0
        assert percepts.hunger == 10.0
        assert percepts.nearby_players == ["Bob"]

        goals = await sas_restored.get_goals()
        assert goals.current_goal == "Build house"

    async def test_save_restore_round_trip(self, tmp_path: Path):
        """Test save → restore round trip preserves all data."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas_original = InMemorySAS("agent-001")

        # Populate all SAS sections
        await sas_original.update_percepts(PerceptData(health=18.5, nearby_players=["Alice"]))
        await sas_original.update_goals(GoalData(current_goal="Mine diamonds"))
        await sas_original.update_social(
            SocialData(relationships={"Alice": 0.8}, emotions={"joy": 7.0})
        )
        await sas_original.add_action(
            ActionHistoryEntry(action="mine", expected_result="get stone", success=True)
        )
        await sas_original.add_stm(MemoryEntry(content="Found cave", importance=0.9))
        await sas_original.set_cc_decision({"action": "mine", "reasoning": "Need resources"})

        # Save checkpoint
        await manager.save("agent-001", sas_original, tick_count=100)

        # Restore to new SAS
        sas_restored = InMemorySAS("agent-001")
        await manager.restore("agent-001", sas_restored)

        # Verify all sections preserved
        percepts = await sas_restored.get_percepts()
        assert percepts.health == 18.5
        assert percepts.nearby_players == ["Alice"]

        goals = await sas_restored.get_goals()
        assert goals.current_goal == "Mine diamonds"

        social = await sas_restored.get_social()
        assert social.relationships == {"Alice": 0.8}
        assert social.emotions == {"joy": 7.0}

        actions = await sas_restored.get_action_history()
        assert len(actions) == 1
        assert actions[0].action == "mine"

        stm = await sas_restored.get_stm()
        assert len(stm) == 1
        assert stm[0].content == "Found cave"

        cc_decision = await sas_restored.get_last_cc_decision()
        assert cc_decision is not None
        assert cc_decision["action"] == "mine"

    async def test_list_checkpoints_sorted_by_time(self, tmp_path: Path):
        """Test list_checkpoints returns sorted by time (newest first)."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas = InMemorySAS("agent-001")

        # Create multiple checkpoints with different tick counts
        await manager.save("agent-001", sas, tick_count=10)
        await manager.save("agent-001", sas, tick_count=20)
        await manager.save("agent-001", sas, tick_count=30)

        # List checkpoints
        checkpoints = manager.list_checkpoints("agent-001")

        assert len(checkpoints) == 3
        # Should be sorted newest first
        assert checkpoints[0].tick_count == 30
        assert checkpoints[1].tick_count == 20
        assert checkpoints[2].tick_count == 10

    async def test_delete_old_checkpoints_keeps_max(self, tmp_path: Path):
        """Test delete_old_checkpoints keeps only max_checkpoints most recent."""
        manager = CheckpointManager(checkpoint_dir=tmp_path, max_checkpoints=3)
        sas = InMemorySAS("agent-001")

        # Create 5 checkpoints
        for i in range(5):
            await manager.save("agent-001", sas, tick_count=i * 10)

        # Should only keep 3 most recent
        checkpoints = manager.list_checkpoints("agent-001")
        assert len(checkpoints) == 3
        assert checkpoints[0].tick_count == 40
        assert checkpoints[1].tick_count == 30
        assert checkpoints[2].tick_count == 20

    async def test_should_checkpoint_respects_interval(self, tmp_path: Path):
        """Test should_checkpoint respects interval."""
        manager = CheckpointManager(checkpoint_dir=tmp_path, interval_seconds=60.0)

        # No previous checkpoint
        assert manager.should_checkpoint(None) is True

        # Recent checkpoint (within interval)
        recent = datetime.now(UTC) - timedelta(seconds=30)
        assert manager.should_checkpoint(recent) is False

        # Old checkpoint (past interval)
        old = datetime.now(UTC) - timedelta(seconds=120)
        assert manager.should_checkpoint(old) is True

    async def test_metadata_has_correct_fields(self, tmp_path: Path):
        """Test metadata has all required fields."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas = InMemorySAS("agent-001")

        metadata = await manager.save("agent-001", sas, tick_count=42)

        assert metadata.agent_id == "agent-001"
        assert metadata.tick_count == 42
        assert isinstance(metadata.timestamp, datetime)
        assert metadata.version == "1.0"
        assert metadata.size_bytes > 0

    async def test_multiple_agents_separate_directories(self, tmp_path: Path):
        """Test multiple agents have separate checkpoint directories."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas1 = InMemorySAS("agent-001")
        sas2 = InMemorySAS("agent-002")

        # Save checkpoints for different agents
        meta1 = await manager.save("agent-001", sas1, tick_count=10)
        meta2 = await manager.save("agent-002", sas2, tick_count=20)

        # Verify separate directories
        path1 = manager.get_checkpoint_path("agent-001", meta1)
        path2 = manager.get_checkpoint_path("agent-002", meta2)

        assert path1.parent.name == "agent-001"
        assert path2.parent.name == "agent-002"
        assert path1.parent != path2.parent

        # Verify separate checkpoint lists
        checkpoints1 = manager.list_checkpoints("agent-001")
        checkpoints2 = manager.list_checkpoints("agent-002")

        assert len(checkpoints1) == 1
        assert len(checkpoints2) == 1
        assert checkpoints1[0].agent_id == "agent-001"
        assert checkpoints2[0].agent_id == "agent-002"

    async def test_restore_nonexistent_checkpoint_raises_error(self, tmp_path: Path):
        """Test restore non-existent checkpoint raises error."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas = InMemorySAS("agent-001")

        # Try to restore when no checkpoints exist
        with pytest.raises(FileNotFoundError, match="No checkpoints found"):
            await manager.restore("agent-001", sas)

    async def test_restore_specific_checkpoint_by_id(self, tmp_path: Path):
        """Test restore specific checkpoint by checkpoint_id."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas = InMemorySAS("agent-001")

        # Create multiple checkpoints
        await sas.update_goals(GoalData(current_goal="Goal 1"))
        meta1 = await manager.save("agent-001", sas, tick_count=10)

        await sas.update_goals(GoalData(current_goal="Goal 2"))
        await manager.save("agent-001", sas, tick_count=20)

        # Restore first checkpoint by ID
        checkpoint_id = meta1.timestamp.isoformat().replace(":", "-")
        sas_restored = InMemorySAS("agent-001")
        restored_meta = await manager.restore(
            "agent-001", sas_restored, checkpoint_id=checkpoint_id
        )

        assert restored_meta.tick_count == 10
        goals = await sas_restored.get_goals()
        assert goals.current_goal == "Goal 1"

    async def test_restore_invalid_checkpoint_id_raises_error(self, tmp_path: Path):
        """Test restore with invalid checkpoint_id raises error."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas = InMemorySAS("agent-001")

        # Create a checkpoint
        await manager.save("agent-001", sas, tick_count=10)

        # Try to restore with invalid ID
        sas_restored = InMemorySAS("agent-001")
        with pytest.raises(ValueError, match=r"Checkpoint .* not found"):
            await manager.restore("agent-001", sas_restored, checkpoint_id="invalid-id")

    async def test_checkpoint_file_is_valid_json(self, tmp_path: Path):
        """Test checkpoint file is valid JSON."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas = InMemorySAS("agent-001")

        # Populate SAS with complex data
        await sas.update_percepts(PerceptData(health=20.0, nearby_players=["Alice", "Bob"]))
        await sas.add_action(
            ActionHistoryEntry(action="mine", expected_result="stone", success=True)
        )

        # Save checkpoint
        metadata = await manager.save("agent-001", sas, tick_count=42)

        # Read and parse JSON
        checkpoint_path = manager.get_checkpoint_path("agent-001", metadata)
        checkpoint_json = checkpoint_path.read_text(encoding="utf-8")
        checkpoint_data = json.loads(checkpoint_json)

        # Verify structure
        assert "metadata" in checkpoint_data
        assert "sas_snapshot" in checkpoint_data
        assert checkpoint_data["metadata"]["agent_id"] == "agent-001"
        assert checkpoint_data["metadata"]["tick_count"] == 42

    async def test_checkpoint_with_empty_sas(self, tmp_path: Path):
        """Test checkpoint with empty SAS (default values)."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas = InMemorySAS("agent-001")

        # Save checkpoint with empty SAS
        await manager.save("agent-001", sas, tick_count=0)

        # Restore to new SAS
        sas_restored = InMemorySAS("agent-001")
        await manager.restore("agent-001", sas_restored)

        # Verify default values
        percepts = await sas_restored.get_percepts()
        assert percepts.health == 20.0
        assert percepts.nearby_players == []

        goals = await sas_restored.get_goals()
        assert goals.current_goal == ""

    async def test_checkpoint_manager_creates_directory(self, tmp_path: Path):
        """Test CheckpointManager creates checkpoint directory if it doesn't exist."""
        checkpoint_dir = tmp_path / "checkpoints" / "nested" / "dir"
        assert not checkpoint_dir.exists()

        _manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()

    async def test_list_checkpoints_empty_for_new_agent(self, tmp_path: Path):
        """Test list_checkpoints returns empty list for new agent."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)

        checkpoints = manager.list_checkpoints("nonexistent-agent")

        assert checkpoints == []

    async def test_checkpoint_size_bytes_calculation(self, tmp_path: Path):
        """Test size_bytes is correctly calculated."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas = InMemorySAS("agent-001")

        # Small checkpoint
        metadata_small = await manager.save("agent-001", sas, tick_count=1)

        # Larger checkpoint
        for i in range(10):
            await sas.add_stm(MemoryEntry(content=f"Memory {i}" * 100, importance=0.5))
        metadata_large = await manager.save("agent-001", sas, tick_count=2)

        # Large checkpoint should have bigger size
        assert metadata_large.size_bytes > metadata_small.size_bytes

    async def test_concurrent_checkpoints_for_different_agents(self, tmp_path: Path):
        """Test concurrent checkpoints for different agents don't interfere."""
        manager = CheckpointManager(checkpoint_dir=tmp_path, max_checkpoints=2)
        sas1 = InMemorySAS("agent-001")
        sas2 = InMemorySAS("agent-002")

        # Create checkpoints for both agents
        await manager.save("agent-001", sas1, tick_count=10)
        await manager.save("agent-002", sas2, tick_count=20)
        await manager.save("agent-001", sas1, tick_count=30)
        await manager.save("agent-002", sas2, tick_count=40)

        # Each agent should have 2 checkpoints
        checkpoints1 = manager.list_checkpoints("agent-001")
        checkpoints2 = manager.list_checkpoints("agent-002")

        assert len(checkpoints1) == 2
        assert len(checkpoints2) == 2

    async def test_checkpoint_preserves_list_order(self, tmp_path: Path):
        """Test checkpoint preserves order of list-based sections."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas = InMemorySAS("agent-001")

        # Add multiple items in specific order
        await sas.add_action(ActionHistoryEntry(action="action1"))
        await sas.add_action(ActionHistoryEntry(action="action2"))
        await sas.add_action(ActionHistoryEntry(action="action3"))

        await sas.add_stm(MemoryEntry(content="memory1"))
        await sas.add_stm(MemoryEntry(content="memory2"))

        # Save and restore
        await manager.save("agent-001", sas, tick_count=10)
        sas_restored = InMemorySAS("agent-001")
        await manager.restore("agent-001", sas_restored)

        # Verify order preserved (newest first for get_* methods)
        actions = await sas_restored.get_action_history()
        assert len(actions) == 3
        assert actions[0].action == "action3"  # Newest first
        assert actions[1].action == "action2"
        assert actions[2].action == "action1"

        stm = await sas_restored.get_stm()
        assert len(stm) == 2
        assert stm[0].content == "memory2"  # Newest first
        assert stm[1].content == "memory1"


class TestCheckpointPathTraversal:
    """Security tests for path traversal prevention in CheckpointManager."""

    async def test_path_traversal_rejected(self, tmp_path: Path):
        """Test that agent_id containing '../' is rejected with ValueError."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas = InMemorySAS("agent-001")

        traversal_ids = [
            "../etc/passwd",
            "..\\windows\\system32",
            "agent/../../../etc/shadow",
            "..",
            "../",
        ]
        for bad_id in traversal_ids:
            with pytest.raises(ValueError, match="Invalid agent_id"):
                await manager.save(bad_id, sas, tick_count=1)

            with pytest.raises(ValueError, match="Invalid agent_id"):
                await manager.restore(bad_id, sas)

            with pytest.raises(ValueError, match="Invalid agent_id"):
                manager.list_checkpoints(bad_id)

    async def test_invalid_agent_id_rejected(self, tmp_path: Path):
        """Test that agent_id with special characters is rejected."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas = InMemorySAS("agent-001")

        invalid_ids = [
            "agent 001",  # space
            "agent/001",  # forward slash
            "agent\\001",  # backslash
            "agent.001",  # dot
            "agent@001",  # at sign
            "agent:001",  # colon
            "",  # empty string
            "agent\x00id",  # null byte
        ]
        for bad_id in invalid_ids:
            with pytest.raises(ValueError, match="Invalid agent_id"):
                await manager.save(bad_id, sas, tick_count=1)

    async def test_valid_agent_id_accepted(self, tmp_path: Path):
        """Test that valid agent_ids are accepted without error."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        sas = InMemorySAS("agent-001")

        valid_ids = [
            "agent-001",
            "agent_test",
            "Agent123",
            "a",
            "ABC",
            "test-agent-with-long-name-123",
            "UPPER_lower_123-mix",
        ]
        for valid_id in valid_ids:
            # Should not raise - just verify save completes
            metadata = await manager.save(valid_id, sas, tick_count=1)
            assert metadata.agent_id == valid_id

            # list_checkpoints should also work
            checkpoints = manager.list_checkpoints(valid_id)
            assert len(checkpoints) >= 1
