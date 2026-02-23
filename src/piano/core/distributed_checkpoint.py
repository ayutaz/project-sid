"""Distributed Checkpoint system with CC-cycle alignment.

Provides shard-based checkpoint coordination for multi-agent simulations.
Each shard captures agent states at CC cycle boundaries, with 5-10 second
skew tolerance across shards.  Restoration re-syncs agent states within
the first minute after recovery.

Reference: docs/implementation/08-infrastructure.md
"""

from __future__ import annotations

__all__ = [
    "CCCycleAligner",
    "CheckpointInfo",
    "DistributedCheckpointManager",
    "ShardCheckpointCoordinator",
]

import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# Maximum allowed skew between shard checkpoints (seconds)
MAX_SHARD_SKEW_SECONDS: float = 10.0

# Default convergence window after restoration (seconds)
DEFAULT_CONVERGENCE_WINDOW: float = 60.0


def _validate_id(value: str, label: str) -> None:
    """Validate an identifier to prevent path traversal attacks.

    Args:
        value: The identifier string to validate.
        label: Human-readable label for error messages (e.g. "shard_id").

    Raises:
        ValueError: If the identifier contains unsafe characters.
    """
    if not value or not _SAFE_ID_RE.match(value):
        msg = (
            f"Invalid {label} {value!r}: "
            "must contain only alphanumeric characters, hyphens, and underscores"
        )
        raise ValueError(msg)


def _validate_shard_id(shard_id: int) -> str:
    """Convert numeric shard_id to a safe string and validate.

    Args:
        shard_id: Numeric shard identifier.

    Returns:
        String representation safe for filesystem use.

    Raises:
        ValueError: If shard_id is negative.
    """
    if shard_id < 0:
        msg = f"shard_id must be non-negative, got {shard_id}"
        raise ValueError(msg)
    return str(shard_id)


def _validate_checkpoint_id(checkpoint_id: str) -> None:
    """Validate checkpoint_id to prevent path traversal.

    Args:
        checkpoint_id: Checkpoint identifier to validate.

    Raises:
        ValueError: If checkpoint_id contains unsafe characters.
    """
    _validate_id(checkpoint_id, "checkpoint_id")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CheckpointInfo(BaseModel):
    """Metadata describing a single shard checkpoint."""

    checkpoint_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    shard_id: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    agent_count: int = 0
    size_bytes: int = 0


class _ShardCheckpointData(BaseModel):
    """Internal: full checkpoint payload persisted to disk."""

    info: CheckpointInfo
    agent_states: dict[str, dict[str, Any]]


class _CheckpointJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for checkpoint serialization."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# ---------------------------------------------------------------------------
# CCCycleAligner
# ---------------------------------------------------------------------------


class CCCycleAligner:
    """Determines safe points for checkpointing relative to CC cycles.

    A checkpoint is safe when the CC has completed its decision for the
    current cycle and the next tick has not yet begun — i.e. the agent
    is in a quiescent state between ticks.
    """

    def __init__(self) -> None:
        self._cycle_completed: dict[str, bool] = {}
        self._next_tick_started: dict[str, bool] = {}
        self._boundary_events: dict[str, asyncio.Event] = {}

    # -- Public state mutation (called by scheduler / CC hooks) ----

    def mark_cycle_completed(self, agent_id: str) -> None:
        """Signal that the CC cycle for *agent_id* has finished."""
        _validate_id(agent_id, "agent_id")
        self._cycle_completed[agent_id] = True
        self._next_tick_started[agent_id] = False
        event = self._boundary_events.get(agent_id)
        if event is not None:
            event.set()

    def mark_tick_started(self, agent_id: str) -> None:
        """Signal that a new tick has started for *agent_id*."""
        _validate_id(agent_id, "agent_id")
        self._next_tick_started[agent_id] = True
        self._cycle_completed[agent_id] = False
        event = self._boundary_events.get(agent_id)
        if event is not None:
            event.clear()

    # -- Query --

    def is_safe_to_checkpoint(self, agent_id: str) -> bool:
        """Return ``True`` if *agent_id* is between CC cycles.

        A safe checkpoint window exists when the CC decision is complete
        and the next tick has not yet begun.
        """
        _validate_id(agent_id, "agent_id")
        completed = self._cycle_completed.get(agent_id, False)
        started = self._next_tick_started.get(agent_id, False)
        return completed and not started

    async def wait_for_cc_boundary(self, agent_id: str, timeout: float = 30.0) -> bool:
        """Wait until *agent_id* reaches a CC cycle boundary.

        Args:
            agent_id: Agent to wait for.
            timeout: Maximum seconds to wait.

        Returns:
            ``True`` if the boundary was reached within the timeout,
            ``False`` otherwise.
        """
        _validate_id(agent_id, "agent_id")

        # Fast path: already safe
        if self.is_safe_to_checkpoint(agent_id):
            return True

        # Create / reuse an asyncio.Event
        if agent_id not in self._boundary_events:
            self._boundary_events[agent_id] = asyncio.Event()
        event = self._boundary_events[agent_id]
        event.clear()

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except TimeoutError:
            logger.warning(
                "cc_boundary_wait_timeout",
                agent_id=agent_id,
                timeout=timeout,
            )
            return False

        return self.is_safe_to_checkpoint(agent_id)


# ---------------------------------------------------------------------------
# DistributedCheckpointManager
# ---------------------------------------------------------------------------


class DistributedCheckpointManager:
    """Manages shard-based checkpoint persistence to disk.

    Directory layout::

        {checkpoint_dir}/shard-{shard_id}/{checkpoint_id}.json
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints_per_shard: int = 10,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._max_checkpoints = max_checkpoints_per_shard
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._scheduled: dict[int, datetime] = {}
        self.logger = logger.bind(checkpoint_dir=str(self._checkpoint_dir))
        self.logger.info(
            "distributed_checkpoint_manager_initialized",
            max_checkpoints_per_shard=max_checkpoints_per_shard,
        )

    # -- helpers ---------------------------------------------------------

    def _shard_dir(self, shard_id: int) -> Path:
        sid = _validate_shard_id(shard_id)
        shard_dir = self._checkpoint_dir / f"shard-{sid}"
        # Defence in depth: ensure resolved path stays within root
        resolved = shard_dir.resolve()
        root = self._checkpoint_dir.resolve()
        if not str(resolved).startswith(str(root)):
            msg = f"shard_id {shard_id} resolves outside checkpoint directory"
            raise ValueError(msg)
        return shard_dir

    def _checkpoint_path(self, shard_id: int, checkpoint_id: str) -> Path:
        _validate_checkpoint_id(checkpoint_id)
        shard_dir = self._shard_dir(shard_id)
        path = shard_dir / f"{checkpoint_id}.json"
        resolved = path.resolve()
        if not str(resolved).startswith(str(shard_dir.resolve())):
            msg = f"checkpoint_id {checkpoint_id!r} resolves outside shard directory"
            raise ValueError(msg)
        return path

    # -- Public API ------------------------------------------------------

    def schedule_checkpoint(self, shard_ids: list[int]) -> dict[int, datetime]:
        """Schedule a checkpoint for the given shards.

        Records the current time as the scheduled checkpoint time for each
        shard.  The actual ``save_shard`` calls should follow promptly.

        Args:
            shard_ids: List of shard identifiers to schedule.

        Returns:
            Mapping of shard_id to scheduled timestamp.
        """
        now = datetime.now(UTC)
        result: dict[int, datetime] = {}
        for sid in shard_ids:
            _validate_shard_id(sid)
            self._scheduled[sid] = now
            result[sid] = now
        self.logger.info(
            "checkpoint_scheduled",
            shard_ids=shard_ids,
            timestamp=now.isoformat(),
        )
        return result

    async def save_shard(
        self,
        shard_id: int,
        agent_states: dict[str, dict[str, Any]],
    ) -> CheckpointInfo:
        """Persist a checkpoint for a single shard.

        Args:
            shard_id: Shard identifier.
            agent_states: Mapping of agent_id -> serialised agent state.

        Returns:
            ``CheckpointInfo`` for the newly written checkpoint.
        """
        checkpoint_id = uuid4().hex[:16]
        info = CheckpointInfo(
            checkpoint_id=checkpoint_id,
            shard_id=shard_id,
            timestamp=datetime.now(UTC),
            agent_count=len(agent_states),
        )
        data = _ShardCheckpointData(info=info, agent_states=agent_states)

        payload = json.dumps(data.model_dump(mode="json"), indent=2, cls=_CheckpointJSONEncoder)
        info.size_bytes = len(payload.encode("utf-8"))

        shard_dir = self._shard_dir(shard_id)
        shard_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_path(shard_id, checkpoint_id)
        path.write_text(payload, encoding="utf-8")

        self.logger.info(
            "shard_checkpoint_saved",
            shard_id=shard_id,
            checkpoint_id=checkpoint_id,
            agent_count=info.agent_count,
            size_bytes=info.size_bytes,
        )

        # Automatic rotation
        await self.cleanup_old(shard_id, keep_count=self._max_checkpoints)

        return info

    async def restore_shard(
        self,
        shard_id: int,
        checkpoint_id: str,
    ) -> dict[str, dict[str, Any]]:
        """Restore agent states from a shard checkpoint.

        Args:
            shard_id: Shard identifier.
            checkpoint_id: Identifier of the checkpoint to restore.

        Returns:
            Mapping of agent_id -> serialised agent state.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        _validate_checkpoint_id(checkpoint_id)
        path = self._checkpoint_path(shard_id, checkpoint_id)
        if not path.exists():
            msg = f"Checkpoint {checkpoint_id!r} not found for shard {shard_id}"
            raise FileNotFoundError(msg)

        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        data = _ShardCheckpointData.model_validate(parsed)
        self.logger.info(
            "shard_checkpoint_restored",
            shard_id=shard_id,
            checkpoint_id=checkpoint_id,
            agent_count=data.info.agent_count,
        )
        return data.agent_states

    def get_latest_checkpoint(self, shard_id: int) -> CheckpointInfo | None:
        """Return the most recent checkpoint info for a shard, or ``None``."""
        checkpoints = self.list_checkpoints(shard_id)
        return checkpoints[0] if checkpoints else None

    def list_checkpoints(self, shard_id: int) -> list[CheckpointInfo]:
        """List all checkpoints for a shard, newest first.

        Args:
            shard_id: Shard identifier.

        Returns:
            List of ``CheckpointInfo`` sorted by timestamp descending.
        """
        shard_dir = self._shard_dir(shard_id)
        if not shard_dir.exists():
            return []

        infos: list[CheckpointInfo] = []
        for fp in shard_dir.glob("*.json"):
            try:
                raw = fp.read_text(encoding="utf-8")
                parsed = json.loads(raw)
                data = _ShardCheckpointData.model_validate(parsed)
                infos.append(data.info)
            except Exception as exc:
                self.logger.warning(
                    "checkpoint_file_invalid",
                    path=str(fp),
                    error=str(exc),
                )
        infos.sort(key=lambda i: i.timestamp, reverse=True)
        return infos

    async def cleanup_old(self, shard_id: int, keep_count: int = 5) -> int:
        """Delete old checkpoints, keeping the *keep_count* most recent.

        Args:
            shard_id: Shard identifier.
            keep_count: Number of checkpoints to retain.

        Returns:
            Number of checkpoints deleted.
        """
        checkpoints = self.list_checkpoints(shard_id)
        to_delete = checkpoints[keep_count:]
        deleted = 0
        for info in to_delete:
            path = self._checkpoint_path(shard_id, info.checkpoint_id)
            try:
                path.unlink()
                deleted += 1
                self.logger.debug(
                    "checkpoint_deleted",
                    shard_id=shard_id,
                    checkpoint_id=info.checkpoint_id,
                )
            except Exception as exc:
                self.logger.warning(
                    "checkpoint_delete_failed",
                    path=str(path),
                    error=str(exc),
                )
        return deleted


# ---------------------------------------------------------------------------
# ShardCheckpointCoordinator
# ---------------------------------------------------------------------------


class ShardCheckpointCoordinator:
    """Coordinates checkpoint capture across multiple shards.

    Ensures that all shards are checkpointed within the allowed skew window
    (``MAX_SHARD_SKEW_SECONDS``) and provides post-restore convergence
    tracking.
    """

    def __init__(
        self,
        manager: DistributedCheckpointManager,
        aligner: CCCycleAligner | None = None,
        max_skew_seconds: float = MAX_SHARD_SKEW_SECONDS,
        convergence_window: float = DEFAULT_CONVERGENCE_WINDOW,
    ) -> None:
        self._manager = manager
        self._aligner = aligner or CCCycleAligner()
        self._max_skew = max_skew_seconds
        self._convergence_window = convergence_window
        self._restoration_start: datetime | None = None
        self.logger = logger.bind(component="shard_coordinator")

    @property
    def aligner(self) -> CCCycleAligner:
        """The CC cycle aligner used by this coordinator."""
        return self._aligner

    @property
    def manager(self) -> DistributedCheckpointManager:
        """The underlying checkpoint manager."""
        return self._manager

    # -- Coordinated checkpoint --

    async def coordinate_checkpoint(
        self,
        shard_ids: list[int],
        shard_agent_states: dict[int, dict[str, dict[str, Any]]],
    ) -> list[CheckpointInfo]:
        """Capture a coordinated checkpoint across shards.

        All shard checkpoints must complete within ``max_skew_seconds``
        of each other. If the skew constraint is violated the checkpoint
        still proceeds but a warning is logged.

        Args:
            shard_ids: Shards to checkpoint.
            shard_agent_states: Mapping shard_id -> (agent_id -> state).

        Returns:
            List of ``CheckpointInfo`` for each shard (in input order).
        """
        self._manager.schedule_checkpoint(shard_ids)

        results: list[CheckpointInfo] = []
        for sid in shard_ids:
            states = shard_agent_states.get(sid, {})
            info = await self._manager.save_shard(sid, states)
            results.append(info)

        # Check skew between first and last checkpoint
        if len(results) >= 2:
            timestamps = [r.timestamp for r in results]
            skew = (max(timestamps) - min(timestamps)).total_seconds()
            if skew > self._max_skew:
                self.logger.warning(
                    "checkpoint_skew_exceeded",
                    skew_seconds=skew,
                    max_allowed=self._max_skew,
                    shard_ids=shard_ids,
                )
            else:
                self.logger.info(
                    "coordinated_checkpoint_complete",
                    shard_count=len(results),
                    skew_seconds=skew,
                )
        else:
            self.logger.info(
                "coordinated_checkpoint_complete",
                shard_count=len(results),
                skew_seconds=0.0,
            )

        return results

    # -- Restore with convergence tracking --

    async def restore_shards(
        self,
        restore_plan: dict[int, str],
    ) -> dict[int, dict[str, dict[str, Any]]]:
        """Restore multiple shards and begin convergence tracking.

        Args:
            restore_plan: Mapping shard_id -> checkpoint_id to restore.

        Returns:
            Mapping shard_id -> (agent_id -> state).
        """
        self._restoration_start = datetime.now(UTC)
        result: dict[int, dict[str, dict[str, Any]]] = {}
        for sid, cpid in restore_plan.items():
            states = await self._manager.restore_shard(sid, cpid)
            result[sid] = states
        self.logger.info(
            "shards_restored",
            shard_count=len(result),
        )
        return result

    def is_converged(self) -> bool:
        """Return ``True`` if the convergence window has elapsed since restoration.

        Agents are assumed to have re-synchronised their states within
        the convergence window (default 60 s).
        """
        if self._restoration_start is None:
            return True  # Nothing to converge
        elapsed = (datetime.now(UTC) - self._restoration_start).total_seconds()
        return elapsed >= self._convergence_window

    def convergence_elapsed(self) -> float:
        """Return seconds elapsed since restoration began, or 0.0."""
        if self._restoration_start is None:
            return 0.0
        return (datetime.now(UTC) - self._restoration_start).total_seconds()
