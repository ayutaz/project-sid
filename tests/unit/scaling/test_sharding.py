"""Tests for agent sharding manager."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from piano.scaling.sharding import (
    ShardConfig,
    ShardingStrategy,
    ShardManager,
    ShardStats,
)

# ---------------------------------------------------------------------------
# ShardConfig / ShardStats model tests
# ---------------------------------------------------------------------------


class TestShardConfig:
    """Tests for ShardConfig model."""

    def test_defaults(self) -> None:
        """Default config has 4 shards and consistent_hash strategy."""
        cfg = ShardConfig()
        assert cfg.num_shards == 4
        assert cfg.strategy == "consistent_hash"

    def test_custom_values(self) -> None:
        """Custom shard count and strategy are accepted."""
        cfg = ShardConfig(num_shards=8, strategy="round_robin")
        assert cfg.num_shards == 8
        assert cfg.strategy == "round_robin"

    def test_num_shards_must_be_positive(self) -> None:
        """num_shards < 1 raises a validation error."""
        with pytest.raises(ValidationError):
            ShardConfig(num_shards=0)


class TestShardStats:
    """Tests for ShardStats model."""

    def test_fields(self) -> None:
        """ShardStats stores shard_id, agent_count, load_factor."""
        stats = ShardStats(shard_id=2, agent_count=5, load_factor=1.25)
        assert stats.shard_id == 2
        assert stats.agent_count == 5
        assert stats.load_factor == 1.25

    def test_defaults(self) -> None:
        """Default agent_count and load_factor are 0."""
        stats = ShardStats(shard_id=0)
        assert stats.agent_count == 0
        assert stats.load_factor == 0.0


class TestShardingStrategy:
    """Tests for ShardingStrategy enum."""

    def test_values(self) -> None:
        """All three strategies are defined."""
        assert ShardingStrategy.CONSISTENT_HASH == "consistent_hash"
        assert ShardingStrategy.ROUND_ROBIN == "round_robin"
        assert ShardingStrategy.SPATIAL == "spatial"


# ---------------------------------------------------------------------------
# ShardManager — consistent hash strategy
# ---------------------------------------------------------------------------


class TestShardManagerConsistentHash:
    """Tests for consistent_hash strategy."""

    def test_assign_returns_valid_shard(self) -> None:
        """assign_shard returns a shard in [0, num_shards)."""
        mgr = ShardManager(ShardConfig(num_shards=4))
        for i in range(20):
            shard = mgr.assign_shard(f"agent-{i:03d}")
            assert 0 <= shard < 4

    def test_assignment_is_deterministic(self) -> None:
        """Same agent_id always maps to the same shard."""
        mgr = ShardManager(ShardConfig(num_shards=8))
        first = mgr.assign_shard("agent-abc")
        # Re-calling should return the cached assignment
        assert mgr.assign_shard("agent-abc") == first

    def test_deterministic_across_instances(self) -> None:
        """Two independent managers yield the same hash assignment."""
        cfg = ShardConfig(num_shards=8, strategy="consistent_hash")
        mgr1 = ShardManager(cfg)
        mgr2 = ShardManager(cfg)
        for aid in ["a", "b", "c", "x-100", "z"]:
            # Both must compute the same shard (not yet cached)
            s1 = mgr1.assign_shard(aid)
            s2 = mgr2.assign_shard(aid)
            assert s1 == s2

    def test_distribution_is_reasonable(self) -> None:
        """100 agents across 4 shards should give each shard at least 1 agent."""
        mgr = ShardManager(ShardConfig(num_shards=4))
        for i in range(100):
            mgr.assign_shard(f"agent-{i}")
        stats = mgr.get_shard_stats()
        for shard_id in range(4):
            assert stats[shard_id].agent_count >= 1


# ---------------------------------------------------------------------------
# ShardManager — round robin strategy
# ---------------------------------------------------------------------------


class TestShardManagerRoundRobin:
    """Tests for round_robin strategy."""

    def test_round_robin_cycles(self) -> None:
        """Agents are assigned in cyclic order 0, 1, 2, 0, 1, 2, ..."""
        mgr = ShardManager(ShardConfig(num_shards=3, strategy="round_robin"))
        results = [mgr.assign_shard(f"agent-{i}") for i in range(9)]
        assert results == [0, 1, 2, 0, 1, 2, 0, 1, 2]

    def test_round_robin_even_distribution(self) -> None:
        """12 agents across 4 shards gives 3 per shard."""
        mgr = ShardManager(ShardConfig(num_shards=4, strategy="round_robin"))
        for i in range(12):
            mgr.assign_shard(f"agent-{i}")
        stats = mgr.get_shard_stats()
        for shard_id in range(4):
            assert stats[shard_id].agent_count == 3
            assert stats[shard_id].load_factor == pytest.approx(1.0)

    def test_existing_agent_returns_cached_shard(self) -> None:
        """Re-assigning an already-assigned agent returns the same shard."""
        mgr = ShardManager(ShardConfig(num_shards=3, strategy="round_robin"))
        first = mgr.assign_shard("agent-0")
        # Second call should not advance the counter
        second = mgr.assign_shard("agent-0")
        assert first == second
        # Next new agent should still get shard 1 (counter only advanced once)
        assert mgr.assign_shard("agent-1") == 1


# ---------------------------------------------------------------------------
# ShardManager — spatial strategy
# ---------------------------------------------------------------------------


class TestShardManagerSpatial:
    """Tests for spatial strategy."""

    def _make_spatial_manager(self) -> ShardManager:
        """Create a spatial manager with 3 shard centers."""
        mgr = ShardManager(ShardConfig(num_shards=3, strategy="spatial"))
        mgr.set_shard_centers(
            {
                0: {"x": 0.0, "y": 64.0, "z": 0.0},
                1: {"x": 1000.0, "y": 64.0, "z": 0.0},
                2: {"x": 0.0, "y": 64.0, "z": 1000.0},
            }
        )
        return mgr

    def test_nearest_center_wins(self) -> None:
        """Agent near center 0 is assigned to shard 0."""
        mgr = self._make_spatial_manager()
        mgr.set_agent_position("a1", {"x": 10.0, "y": 64.0, "z": 5.0})
        assert mgr.assign_shard("a1") == 0

    def test_agent_near_center_1(self) -> None:
        """Agent near center 1 is assigned to shard 1."""
        mgr = self._make_spatial_manager()
        mgr.set_agent_position("a2", {"x": 990.0, "y": 64.0, "z": 10.0})
        assert mgr.assign_shard("a2") == 1

    def test_agent_near_center_2(self) -> None:
        """Agent near center 2 is assigned to shard 2."""
        mgr = self._make_spatial_manager()
        mgr.set_agent_position("a3", {"x": 5.0, "y": 64.0, "z": 980.0})
        assert mgr.assign_shard("a3") == 2

    def test_fallback_to_hash_without_position(self) -> None:
        """Without position data, spatial falls back to consistent hash."""
        mgr = self._make_spatial_manager()
        # No position set for this agent
        shard = mgr.assign_shard("no-pos-agent")
        assert 0 <= shard < 3

    def test_fallback_to_hash_without_centers(self) -> None:
        """Without shard centers, spatial falls back to consistent hash."""
        mgr = ShardManager(ShardConfig(num_shards=3, strategy="spatial"))
        mgr.set_agent_position("a1", {"x": 10.0, "y": 64.0, "z": 5.0})
        shard = mgr.assign_shard("a1")
        assert 0 <= shard < 3

    def test_set_shard_centers_validates_range(self) -> None:
        """Out-of-range shard_id in centers raises ValueError."""
        mgr = ShardManager(ShardConfig(num_shards=2, strategy="spatial"))
        with pytest.raises(ValueError, match="out of range"):
            mgr.set_shard_centers({5: {"x": 0.0, "y": 0.0, "z": 0.0}})

    def test_set_shard_centers_validates_keys(self) -> None:
        """Position missing x/y/z keys raises ValueError."""
        mgr = ShardManager(ShardConfig(num_shards=2, strategy="spatial"))
        with pytest.raises(ValueError, match="missing keys"):
            mgr.set_shard_centers({0: {"x": 0.0, "y": 0.0}})  # no z

    def test_set_agent_position_validates_keys(self) -> None:
        """Agent position missing keys raises ValueError."""
        mgr = ShardManager(ShardConfig(num_shards=2, strategy="spatial"))
        with pytest.raises(ValueError, match="missing keys"):
            mgr.set_agent_position("a1", {"x": 0.0})  # missing y, z


# ---------------------------------------------------------------------------
# ShardManager — shared behaviour
# ---------------------------------------------------------------------------


class TestShardManagerCommon:
    """Tests for strategy-independent ShardManager behaviour."""

    def test_get_shard_for_agent(self) -> None:
        """get_shard_for_agent returns the assigned shard."""
        mgr = ShardManager(ShardConfig(num_shards=4))
        expected = mgr.assign_shard("agent-1")
        assert mgr.get_shard_for_agent("agent-1") == expected

    def test_get_shard_for_unknown_agent_raises(self) -> None:
        """get_shard_for_agent raises KeyError for unknown agent."""
        mgr = ShardManager()
        with pytest.raises(KeyError, match="has not been assigned"):
            mgr.get_shard_for_agent("unknown")

    def test_get_agents_in_shard(self) -> None:
        """get_agents_in_shard returns sorted list of agent ids."""
        mgr = ShardManager(ShardConfig(num_shards=2, strategy="round_robin"))
        mgr.assign_shard("b")  # shard 0
        mgr.assign_shard("c")  # shard 1
        mgr.assign_shard("a")  # shard 0
        assert mgr.get_agents_in_shard(0) == ["a", "b"]
        assert mgr.get_agents_in_shard(1) == ["c"]

    def test_get_agents_in_invalid_shard_raises(self) -> None:
        """get_agents_in_shard raises ValueError for out-of-range shard."""
        mgr = ShardManager(ShardConfig(num_shards=2))
        with pytest.raises(ValueError, match="out of range"):
            mgr.get_agents_in_shard(5)
        with pytest.raises(ValueError, match="out of range"):
            mgr.get_agents_in_shard(-1)

    def test_remove_agent(self) -> None:
        """remove_agent removes the agent from tracking."""
        mgr = ShardManager()
        mgr.assign_shard("agent-1")
        assert mgr.agent_count == 1
        shard = mgr.remove_agent("agent-1")
        assert 0 <= shard < mgr.num_shards
        assert mgr.agent_count == 0
        with pytest.raises(KeyError):
            mgr.get_shard_for_agent("agent-1")

    def test_remove_unknown_agent_raises(self) -> None:
        """remove_agent raises KeyError for unknown agent."""
        mgr = ShardManager()
        with pytest.raises(KeyError, match="is not assigned"):
            mgr.remove_agent("ghost")

    def test_rebalance_changes_num_shards(self) -> None:
        """After rebalance, num_shards matches the new value."""
        mgr = ShardManager(ShardConfig(num_shards=2))
        for i in range(10):
            mgr.assign_shard(f"agent-{i}")
        mgr.rebalance(8)
        assert mgr.num_shards == 8
        # All agents should still be assigned
        assert mgr.agent_count == 10

    def test_rebalance_returns_moves(self) -> None:
        """Rebalance returns a dict describing which agents moved."""
        mgr = ShardManager(ShardConfig(num_shards=2))
        for i in range(10):
            mgr.assign_shard(f"agent-{i}")
        moves = mgr.rebalance(16)
        # moves is a dict: agent_id -> (old_shard, new_shard)
        for _agent_id, (old, new) in moves.items():
            assert old != new
            assert 0 <= new < 16

    def test_rebalance_invalid_raises(self) -> None:
        """Rebalancing to 0 shards raises ValueError."""
        mgr = ShardManager()
        with pytest.raises(ValueError, match="new_num_shards must be >= 1"):
            mgr.rebalance(0)

    def test_shard_stats_load_factor(self) -> None:
        """Load factor is 1.0 for perfectly even distribution."""
        mgr = ShardManager(ShardConfig(num_shards=3, strategy="round_robin"))
        for i in range(9):
            mgr.assign_shard(f"agent-{i}")
        stats = mgr.get_shard_stats()
        for shard_id in range(3):
            assert stats[shard_id].load_factor == pytest.approx(1.0)

    def test_shard_stats_empty(self) -> None:
        """Stats for empty manager show 0 agents and 0 load."""
        mgr = ShardManager(ShardConfig(num_shards=2))
        stats = mgr.get_shard_stats()
        assert len(stats) == 2
        for s in stats.values():
            assert s.agent_count == 0
            assert s.load_factor == 0.0

    def test_default_config(self) -> None:
        """ShardManager with no config uses defaults."""
        mgr = ShardManager()
        assert mgr.num_shards == 4
        assert mgr.strategy == ShardingStrategy.CONSISTENT_HASH

    def test_properties(self) -> None:
        """Properties return expected values."""
        mgr = ShardManager(ShardConfig(num_shards=6, strategy="round_robin"))
        assert mgr.num_shards == 6
        assert mgr.strategy == ShardingStrategy.ROUND_ROBIN
        assert mgr.agent_count == 0
        mgr.assign_shard("x")
        assert mgr.agent_count == 1

    def test_reassign_agent(self) -> None:
        """reassign_agent recomputes shard for an agent."""
        mgr = ShardManager(ShardConfig(num_shards=3, strategy="spatial"))
        mgr.set_shard_centers(
            {
                0: {"x": 0.0, "y": 0.0, "z": 0.0},
                1: {"x": 1000.0, "y": 0.0, "z": 0.0},
                2: {"x": 0.0, "y": 0.0, "z": 1000.0},
            }
        )
        mgr.set_agent_position("a1", {"x": 5.0, "y": 0.0, "z": 0.0})
        mgr.assign_shard("a1")
        assert mgr.get_shard_for_agent("a1") == 0

        # Move the agent near center 1
        mgr.set_agent_position("a1", {"x": 999.0, "y": 0.0, "z": 0.0})
        old, new = mgr.reassign_agent("a1")
        assert old == 0
        assert new == 1

    def test_reassign_unknown_agent_raises(self) -> None:
        """reassign_agent raises KeyError for unknown agent."""
        mgr = ShardManager()
        with pytest.raises(KeyError, match="has not been assigned"):
            mgr.reassign_agent("ghost")
