"""Agent sharding for distributing agents across workers and MC servers.

Assigns agents to shards using pluggable strategies (consistent hashing,
round-robin, or spatial proximity). Supports dynamic rebalancing when the
number of shards changes at runtime.

Reference: docs/implementation/01-system-architecture.md
"""

from __future__ import annotations

__all__ = [
    "ShardConfig",
    "ShardManager",
    "ShardStats",
    "ShardingStrategy",
]

import hashlib
import math
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration & Data Models
# ---------------------------------------------------------------------------


class ShardingStrategy(StrEnum):
    """Available sharding strategies."""

    CONSISTENT_HASH = "consistent_hash"
    ROUND_ROBIN = "round_robin"
    SPATIAL = "spatial"


class ShardConfig(BaseModel):
    """Configuration for the shard manager.

    Attributes:
        num_shards: Number of shards to distribute agents across.
        strategy: Sharding strategy to use for agent assignment.
    """

    num_shards: int = Field(default=4, ge=1)
    strategy: ShardingStrategy = Field(default=ShardingStrategy.CONSISTENT_HASH)


class ShardStats(BaseModel):
    """Per-shard statistics.

    Attributes:
        shard_id: Identifier for this shard.
        agent_count: Number of agents currently assigned to this shard.
        load_factor: Ratio of actual agent count to ideal even distribution
            (1.0 = perfectly balanced).
    """

    shard_id: int
    agent_count: int = 0
    load_factor: float = 0.0


# ---------------------------------------------------------------------------
# Shard Manager
# ---------------------------------------------------------------------------


class ShardManager:
    """Manages assignment of agents to numbered shards.

    Supports three strategies:
    - **consistent_hash** (default): deterministic assignment based on a hash
      of the agent id.  Produces a stable mapping that only changes when
      ``rebalance`` is called.
    - **round_robin**: agents are assigned to shards in cyclic order of
      registration.
    - **spatial**: agents are assigned to the shard whose center is closest
      to the agent's Minecraft coordinates.

    Usage::

        config = ShardConfig(num_shards=4, strategy="consistent_hash")
        mgr = ShardManager(config)
        shard = mgr.assign_shard("agent-001")
        agents = mgr.get_agents_in_shard(0)
        stats = mgr.get_shard_stats()
    """

    def __init__(self, config: ShardConfig | None = None) -> None:
        """Initialize the shard manager.

        Args:
            config: Shard configuration.  Defaults to 4 shards with
                consistent hashing if not provided.
        """
        self._config = config or ShardConfig()
        self._num_shards = self._config.num_shards
        self._strategy = ShardingStrategy(self._config.strategy)

        # agent_id -> shard_id
        self._assignments: dict[str, int] = {}
        # Round-robin counter
        self._rr_counter: int = 0
        # Spatial: shard_id -> center position {x, y, z}
        self._shard_centers: dict[int, dict[str, float]] = {}
        # Spatial: agent_id -> position {x, y, z}
        self._agent_positions: dict[str, dict[str, float]] = {}
        # O(S) shard agent count tracking
        self._shard_agent_counts: dict[int, int] = {i: 0 for i in range(self._num_shards)}

        logger.info(
            "shard_manager_initialized",
            num_shards=self._num_shards,
            strategy=self._strategy.value,
        )

    # -- Properties ---------------------------------------------------------

    @property
    def num_shards(self) -> int:
        """Current number of shards."""
        return self._num_shards

    @property
    def strategy(self) -> ShardingStrategy:
        """Current sharding strategy."""
        return self._strategy

    @property
    def agent_count(self) -> int:
        """Total number of assigned agents."""
        return len(self._assignments)

    # -- Public API ---------------------------------------------------------

    def assign_shard(self, agent_id: str) -> int:
        """Assign an agent to a shard.

        If the agent is already assigned, the existing assignment is returned
        unchanged.

        Args:
            agent_id: Unique identifier of the agent.

        Returns:
            The shard id (0-based) the agent was assigned to.

        Raises:
            ValueError: If spatial strategy is used but no position was set
                for the agent and no shard centers are configured.
        """
        if agent_id in self._assignments:
            return self._assignments[agent_id]

        shard_id = self._compute_shard(agent_id)
        self._assignments[agent_id] = shard_id
        self._shard_agent_counts[shard_id] = self._shard_agent_counts.get(shard_id, 0) + 1

        logger.debug(
            "agent_assigned_to_shard",
            agent_id=agent_id,
            shard_id=shard_id,
            strategy=self._strategy.value,
        )
        return shard_id

    def get_shard_for_agent(self, agent_id: str) -> int:
        """Return the shard for an already-assigned agent.

        Args:
            agent_id: Unique identifier of the agent.

        Returns:
            The shard id.

        Raises:
            KeyError: If the agent has not been assigned yet.
        """
        if agent_id not in self._assignments:
            raise KeyError(f"Agent '{agent_id}' has not been assigned to a shard")
        return self._assignments[agent_id]

    def get_agents_in_shard(self, shard_id: int) -> list[str]:
        """List agents assigned to a particular shard.

        Args:
            shard_id: The shard identifier (0-based).

        Returns:
            Sorted list of agent ids in the shard.

        Raises:
            ValueError: If shard_id is out of range.
        """
        if shard_id < 0 or shard_id >= self._num_shards:
            raise ValueError(f"shard_id {shard_id} out of range [0, {self._num_shards})")
        return sorted(aid for aid, sid in self._assignments.items() if sid == shard_id)

    def get_shard_stats(self) -> dict[int, ShardStats]:
        """Compute per-shard statistics.

        Uses the maintained ``_shard_agent_counts`` dict for O(S) lookup
        instead of scanning all assignments.

        Returns:
            Mapping of shard_id to :class:`ShardStats`.
        """
        total = len(self._assignments)
        ideal = total / self._num_shards if self._num_shards > 0 else 0.0

        stats: dict[int, ShardStats] = {}
        for shard_id in range(self._num_shards):
            count = self._shard_agent_counts.get(shard_id, 0)
            load = count / ideal if ideal > 0 else 0.0
            stats[shard_id] = ShardStats(
                shard_id=shard_id,
                agent_count=count,
                load_factor=round(load, 4),
            )
        return stats

    def rebalance(self, new_num_shards: int) -> dict[str, tuple[int, int]]:
        """Rebalance agents across a new number of shards.

        All agents are reassigned according to the current strategy.

        Args:
            new_num_shards: The new number of shards (must be >= 1).

        Returns:
            Mapping of agent_id to (old_shard, new_shard) for agents that
            moved.

        Raises:
            ValueError: If *new_num_shards* < 1.
        """
        if new_num_shards < 1:
            raise ValueError("new_num_shards must be >= 1")

        old_assignments = dict(self._assignments)
        old_num_shards = self._num_shards

        self._num_shards = new_num_shards
        self._assignments.clear()
        self._rr_counter = 0
        self._shard_agent_counts = {i: 0 for i in range(new_num_shards)}

        # Reassign all agents in deterministic (sorted) order
        moves: dict[str, tuple[int, int]] = {}
        for agent_id in sorted(old_assignments):
            new_shard = self._compute_shard(agent_id)
            self._assignments[agent_id] = new_shard
            self._shard_agent_counts[new_shard] = self._shard_agent_counts.get(new_shard, 0) + 1
            old_shard = old_assignments[agent_id]
            if old_shard != new_shard:
                moves[agent_id] = (old_shard, new_shard)

        logger.info(
            "shard_rebalance_complete",
            old_num_shards=old_num_shards,
            new_num_shards=new_num_shards,
            agents_moved=len(moves),
            total_agents=len(self._assignments),
        )
        return moves

    def remove_agent(self, agent_id: str) -> int:
        """Remove an agent from shard tracking.

        Args:
            agent_id: The agent to remove.

        Returns:
            The shard the agent was previously assigned to.

        Raises:
            KeyError: If the agent was not assigned.
        """
        if agent_id not in self._assignments:
            raise KeyError(f"Agent '{agent_id}' is not assigned to any shard")

        shard_id = self._assignments.pop(agent_id)
        self._agent_positions.pop(agent_id, None)
        if shard_id in self._shard_agent_counts:
            self._shard_agent_counts[shard_id] = max(0, self._shard_agent_counts[shard_id] - 1)

        logger.debug("agent_removed_from_shard", agent_id=agent_id, shard_id=shard_id)
        return shard_id

    # -- Spatial helpers ----------------------------------------------------

    def set_shard_centers(self, centers: dict[int, dict[str, float]]) -> None:
        """Configure shard center positions for spatial strategy.

        Args:
            centers: Mapping of shard_id to ``{"x": float, "y": float, "z": float}``.

        Raises:
            ValueError: If any shard_id is out of range or position dict is
                missing required keys.
        """
        for shard_id, pos in centers.items():
            if shard_id < 0 or shard_id >= self._num_shards:
                raise ValueError(f"shard_id {shard_id} out of range [0, {self._num_shards})")
            missing = {"x", "y", "z"} - set(pos)
            if missing:
                raise ValueError(f"Position for shard {shard_id} missing keys: {missing}")
        self._shard_centers = dict(centers)

    def set_agent_position(self, agent_id: str, position: dict[str, float]) -> None:
        """Set or update an agent's position (used by spatial strategy).

        If the agent is already assigned, its shard is **not** automatically
        changed -- call :meth:`rebalance` or :meth:`reassign_agent` to move
        agents based on updated positions.

        Args:
            agent_id: The agent identifier.
            position: ``{"x": float, "y": float, "z": float}``

        Raises:
            ValueError: If the position dict is missing required keys.
        """
        missing = {"x", "y", "z"} - set(position)
        if missing:
            raise ValueError(f"Position missing keys: {missing}")
        self._agent_positions[agent_id] = dict(position)

    def reassign_agent(self, agent_id: str) -> tuple[int, int]:
        """Force-reassign a single agent based on current strategy/state.

        Useful after updating an agent's position in spatial mode.

        Args:
            agent_id: The agent to reassign.

        Returns:
            Tuple of (old_shard, new_shard).

        Raises:
            KeyError: If the agent was not previously assigned.
        """
        if agent_id not in self._assignments:
            raise KeyError(f"Agent '{agent_id}' has not been assigned to a shard")

        old_shard = self._assignments[agent_id]
        # Temporarily remove so _compute_shard doesn't short-circuit
        del self._assignments[agent_id]
        if old_shard in self._shard_agent_counts:
            self._shard_agent_counts[old_shard] = max(0, self._shard_agent_counts[old_shard] - 1)
        new_shard = self._compute_shard(agent_id)
        self._assignments[agent_id] = new_shard
        self._shard_agent_counts[new_shard] = self._shard_agent_counts.get(new_shard, 0) + 1
        return old_shard, new_shard

    # -- Private helpers ----------------------------------------------------

    def _compute_shard(self, agent_id: str) -> int:
        """Compute the shard for an agent based on the active strategy."""
        if self._strategy == ShardingStrategy.CONSISTENT_HASH:
            return self._hash_shard(agent_id)
        elif self._strategy == ShardingStrategy.ROUND_ROBIN:
            return self._round_robin_shard()
        elif self._strategy == ShardingStrategy.SPATIAL:
            return self._spatial_shard(agent_id)
        # Fallback (should never happen due to StrEnum validation)
        return self._hash_shard(agent_id)  # pragma: no cover

    def _hash_shard(self, agent_id: str) -> int:
        """Deterministic hash-based shard assignment."""
        digest = hashlib.sha256(agent_id.encode()).hexdigest()
        return int(digest, 16) % self._num_shards

    def _round_robin_shard(self) -> int:
        """Cyclic round-robin assignment."""
        shard_id = self._rr_counter % self._num_shards
        self._rr_counter += 1
        return shard_id

    def _spatial_shard(self, agent_id: str) -> int:
        """Assign agent to nearest shard center by Euclidean distance.

        Falls back to consistent hash if no position data is available.
        """
        pos = self._agent_positions.get(agent_id)
        if pos is None or not self._shard_centers:
            # Fallback to hash-based when position data is unavailable
            logger.debug(
                "spatial_shard_fallback",
                agent_id=agent_id,
                reason="no_position" if pos is None else "no_shard_centers",
            )
            return self._hash_shard(agent_id)

        best_shard = 0
        best_dist = float("inf")

        for shard_id, center in self._shard_centers.items():
            dist = math.sqrt(
                (pos["x"] - center["x"]) ** 2
                + (pos["y"] - center["y"]) ** 2
                + (pos["z"] - center["z"]) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_shard = shard_id

        return best_shard
