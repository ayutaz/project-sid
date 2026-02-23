"""Checkpoint/Restore system for agent state persistence.

Provides periodic SAS snapshot capture and restoration for disaster recovery.
Implements the checkpoint requirements from docs/implementation/08-infrastructure.md.

Reference: docs/implementation/08-infrastructure.md Section 8.7
Reference: docs/implementation/roadmap.md (5min interval, 2min restore)
"""

from __future__ import annotations

__all__ = ["Checkpoint", "CheckpointManager", "CheckpointMetadata"]

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState

logger = structlog.get_logger(__name__)


class CheckpointMetadata(BaseModel):
    """Metadata for a checkpoint."""

    agent_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tick_count: int
    version: str = "1.0"
    size_bytes: int = 0


class Checkpoint(BaseModel):
    """Complete checkpoint including metadata and SAS snapshot."""

    metadata: CheckpointMetadata
    sas_snapshot: dict[str, Any]


class CheckpointJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for checkpoint serialization."""

    def default(self, obj: Any) -> Any:
        """Handle UUID and datetime serialization."""
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class CheckpointManager:
    """Manages checkpoint creation, restoration, and rotation.

    Implements periodic SAS snapshot capture to disk and restoration
    for disaster recovery scenarios.

    File format: {checkpoint_dir}/{agent_id}/{timestamp_iso}.json
    """

    _SAFE_AGENT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 10,
        interval_seconds: float = 300.0,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
            max_checkpoints: Maximum number of checkpoints to keep per agent
            interval_seconds: Minimum interval between checkpoints (default: 5 minutes)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.interval_seconds = interval_seconds
        self.logger = logger.bind(checkpoint_dir=str(self.checkpoint_dir))

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "checkpoint_manager_initialized",
            max_checkpoints=max_checkpoints,
            interval_seconds=interval_seconds,
        )

    def _validate_agent_id(self, agent_id: str) -> None:
        """Validate agent_id to prevent path traversal attacks.

        Args:
            agent_id: Agent ID to validate

        Raises:
            ValueError: If agent_id contains unsafe characters
        """
        if not self._SAFE_AGENT_ID_RE.match(agent_id):
            msg = (
                f"Invalid agent_id {agent_id!r}: "
                "must contain only alphanumeric characters, hyphens, and underscores"
            )
            raise ValueError(msg)

        # Defense in depth: verify resolved path stays within checkpoint_dir
        resolved = (self.checkpoint_dir / agent_id).resolve()
        checkpoint_root = self.checkpoint_dir.resolve()
        if not str(resolved).startswith(str(checkpoint_root)):
            msg = f"agent_id {agent_id!r} resolves outside checkpoint directory"
            raise ValueError(msg)

    def get_checkpoint_path(self, agent_id: str, metadata: CheckpointMetadata) -> Path:
        """Get the filesystem path for a checkpoint.

        Args:
            agent_id: Agent ID
            metadata: Checkpoint metadata

        Returns:
            Path to checkpoint file

        Raises:
            ValueError: If agent_id contains unsafe characters
        """
        self._validate_agent_id(agent_id)
        agent_dir = self.checkpoint_dir / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)
        # Use ISO format timestamp for filename (replace colons for Windows compatibility)
        timestamp_str = metadata.timestamp.isoformat().replace(":", "-")
        return agent_dir / f"{timestamp_str}.json"

    async def save(
        self, agent_id: str, sas: SharedAgentState, tick_count: int
    ) -> CheckpointMetadata:
        """Save a checkpoint of agent state to disk.

        Args:
            agent_id: Agent ID
            sas: Shared agent state to snapshot
            tick_count: Current simulation tick count

        Returns:
            CheckpointMetadata for the saved checkpoint
        """
        self._validate_agent_id(agent_id)
        self.logger.debug("checkpoint_save_start", agent_id=agent_id, tick_count=tick_count)

        # Take SAS snapshot
        snapshot = await sas.snapshot()

        # Create metadata
        metadata = CheckpointMetadata(
            agent_id=agent_id,
            tick_count=tick_count,
            timestamp=datetime.now(UTC),
        )

        # Create checkpoint
        checkpoint = Checkpoint(metadata=metadata, sas_snapshot=snapshot)

        # Serialize to JSON
        checkpoint_json = json.dumps(checkpoint.model_dump(), indent=2, cls=CheckpointJSONEncoder)

        # Update size in metadata
        metadata.size_bytes = len(checkpoint_json.encode("utf-8"))
        checkpoint.metadata.size_bytes = metadata.size_bytes

        # Write to disk
        checkpoint_path = self.get_checkpoint_path(agent_id, metadata)
        checkpoint_path.write_text(checkpoint_json, encoding="utf-8")

        self.logger.info(
            "checkpoint_saved",
            agent_id=agent_id,
            tick_count=tick_count,
            size_bytes=metadata.size_bytes,
            path=str(checkpoint_path),
        )

        # Clean up old checkpoints
        await self.delete_old_checkpoints(agent_id)

        return metadata

    async def restore(
        self,
        agent_id: str,
        sas: SharedAgentState,
        checkpoint_id: str | None = None,
    ) -> CheckpointMetadata:
        """Restore agent state from a checkpoint.

        Args:
            agent_id: Agent ID
            sas: Shared agent state to populate
            checkpoint_id: Specific checkpoint ID (timestamp), or None for latest

        Returns:
            CheckpointMetadata of the restored checkpoint

        Raises:
            FileNotFoundError: If no checkpoint exists for this agent
            ValueError: If checkpoint_id is specified but not found
        """
        self._validate_agent_id(agent_id)
        self.logger.debug(
            "checkpoint_restore_start", agent_id=agent_id, checkpoint_id=checkpoint_id
        )

        # List available checkpoints
        checkpoints = self.list_checkpoints(agent_id)
        if not checkpoints:
            msg = f"No checkpoints found for agent {agent_id}"
            raise FileNotFoundError(msg)

        # Find the checkpoint to restore
        if checkpoint_id is None:
            # Use latest checkpoint
            metadata = checkpoints[0]
        else:
            # Find specific checkpoint by timestamp
            matching = [
                cp
                for cp in checkpoints
                if cp.timestamp.isoformat().replace(":", "-") == checkpoint_id
            ]
            if not matching:
                msg = f"Checkpoint {checkpoint_id} not found for agent {agent_id}"
                raise ValueError(msg)
            metadata = matching[0]

        # Read checkpoint from disk
        checkpoint_path = self.get_checkpoint_path(agent_id, metadata)
        checkpoint_json = checkpoint_path.read_text(encoding="utf-8")
        checkpoint_data = json.loads(checkpoint_json)
        checkpoint = Checkpoint.model_validate(checkpoint_data)

        # Restore SAS state section by section
        snapshot = checkpoint.sas_snapshot

        # Import types for reconstruction
        from piano.core.types import (
            ActionHistoryEntry,
            GoalData,
            MemoryEntry,
            PerceptData,
            PlanData,
            SelfReflectionData,
            SocialData,
        )

        for section_name, section_data in snapshot.items():
            # Special handling for typed sections - reconstruct Pydantic models
            if section_name == "percepts":
                await sas.update_percepts(PerceptData.model_validate(section_data))
            elif section_name == "goals":
                await sas.update_goals(GoalData.model_validate(section_data))
            elif section_name == "social":
                await sas.update_social(SocialData.model_validate(section_data))
            elif section_name == "plans":
                await sas.update_plans(PlanData.model_validate(section_data))
            elif section_name == "self_reflection":
                await sas.update_self_reflection(SelfReflectionData.model_validate(section_data))
            elif section_name == "action_history":
                # Restore action history entries
                for entry_dict in section_data:
                    entry = ActionHistoryEntry.model_validate(entry_dict)
                    await sas.add_action(entry)
            elif section_name == "working_memory":
                # Restore working memory entries
                entries = [MemoryEntry.model_validate(e) for e in section_data]
                await sas.set_working_memory(entries)
            elif section_name == "stm":
                # Restore STM entries
                for entry_dict in section_data:
                    entry = MemoryEntry.model_validate(entry_dict)
                    await sas.add_stm(entry)
            elif section_name == "cc_decision":
                if section_data is not None:
                    await sas.set_cc_decision(section_data)
            else:
                # Generic section
                await sas.update_section(section_name, section_data)

        self.logger.info(
            "checkpoint_restored",
            agent_id=agent_id,
            tick_count=metadata.tick_count,
            timestamp=metadata.timestamp.isoformat(),
            path=str(checkpoint_path),
        )

        return metadata

    def list_checkpoints(self, agent_id: str) -> list[CheckpointMetadata]:
        """List all available checkpoints for an agent, sorted newest first.

        Args:
            agent_id: Agent ID

        Returns:
            List of checkpoint metadata, sorted by timestamp descending

        Raises:
            ValueError: If agent_id contains unsafe characters
        """
        self._validate_agent_id(agent_id)
        agent_dir = self.checkpoint_dir / agent_id
        if not agent_dir.exists():
            return []

        # Find all checkpoint files
        checkpoint_files = list(agent_dir.glob("*.json"))

        # Parse metadata from each checkpoint
        metadatas: list[CheckpointMetadata] = []
        for checkpoint_file in checkpoint_files:
            try:
                checkpoint_json = checkpoint_file.read_text(encoding="utf-8")
                checkpoint_data = json.loads(checkpoint_json)
                checkpoint = Checkpoint.model_validate(checkpoint_data)
                metadatas.append(checkpoint.metadata)
            except Exception as e:
                self.logger.warning(
                    "checkpoint_file_invalid",
                    path=str(checkpoint_file),
                    error=str(e),
                )

        # Sort by timestamp, newest first
        metadatas.sort(key=lambda m: m.timestamp, reverse=True)

        return metadatas

    async def delete_old_checkpoints(self, agent_id: str) -> None:
        """Delete old checkpoints, keeping only the most recent max_checkpoints.

        Args:
            agent_id: Agent ID

        Raises:
            ValueError: If agent_id contains unsafe characters
        """
        self._validate_agent_id(agent_id)
        checkpoints = self.list_checkpoints(agent_id)

        # Delete excess checkpoints
        if len(checkpoints) > self.max_checkpoints:
            to_delete = checkpoints[self.max_checkpoints :]
            for metadata in to_delete:
                checkpoint_path = self.get_checkpoint_path(agent_id, metadata)
                try:
                    checkpoint_path.unlink()
                    self.logger.debug(
                        "checkpoint_deleted",
                        agent_id=agent_id,
                        timestamp=metadata.timestamp.isoformat(),
                    )
                except Exception as e:
                    self.logger.warning(
                        "checkpoint_delete_failed",
                        path=str(checkpoint_path),
                        error=str(e),
                    )

    def should_checkpoint(self, last_checkpoint_time: datetime | None) -> bool:
        """Check if interval has elapsed since last checkpoint.

        Args:
            last_checkpoint_time: Timestamp of last checkpoint, or None if no checkpoint yet

        Returns:
            True if checkpoint should be created now
        """
        if last_checkpoint_time is None:
            return True

        elapsed = (datetime.now(UTC) - last_checkpoint_time).total_seconds()
        return elapsed >= self.interval_seconds
