"""Bridge protocol extensions for Phase 1 features.

Provides extended command types, event types, world query protocol,
batch command support, and validation/serialization helpers.
"""

from __future__ import annotations

__all__ = [
    "BatchCommand",
    "CommandType",
    "CommandValidator",
    "EventFilter",
    "EventType",
    "ProtocolSerializer",
    "WorldQuery",
]

import json
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from piano.core.types import BridgeCommand, BridgeEvent

# --- Command Types ---


class CommandType(StrEnum):
    """Extended command types for Phase 1 bridge protocol."""

    # Basic commands (Phase 0)
    MOVE = "move"
    MINE = "mine"
    PLACE = "place"
    CRAFT = "craft"
    EAT = "eat"
    CHAT = "chat"
    LOOK = "look"

    # Social interaction commands (Phase 1)
    TRADE = "trade"
    GIFT = "gift"
    VOTE = "vote"
    FOLLOW = "follow"

    # Combat commands (Phase 1)
    ATTACK = "attack"
    DEFEND = "defend"

    # Advanced crafting (Phase 1)
    SMELT = "smelt"
    FARM = "farm"

    # Exploration (Phase 1)
    EXPLORE = "explore"

    # Query protocol (Phase 1)
    QUERY = "query"

    # Batch command (Phase 1)
    BATCH = "batch"


# --- Event Types ---


class EventType(StrEnum):
    """Extended event types for Phase 1 bridge protocol."""

    # Basic events (Phase 0)
    PERCEPTION = "perception"
    CHAT = "chat"
    ERROR = "error"
    ACTION_COMPLETE = "action_complete"

    # Social events (Phase 1)
    AGENT_NEARBY = "agent_nearby"
    AGENT_LEFT = "agent_left"
    TRADE_REQUEST = "trade_request"
    TRADE_COMPLETE = "trade_complete"

    # Combat/survival events (Phase 1)
    DAMAGE = "damage"
    DEATH = "death"

    # Resource events (Phase 1)
    ITEM_COLLECTED = "item_collected"
    BLOCK_BROKEN = "block_broken"


# --- World Query Protocol ---


class WorldQuery(BaseModel):
    """World state query request for the bridge."""

    query_type: str  # "nearby_blocks", "nearby_entities", "inventory", "position", "time"
    params: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Validate query_type after initialization."""
        valid_types = {
            "nearby_blocks",
            "nearby_entities",
            "inventory",
            "position",
            "time",
        }
        if self.query_type not in valid_types:
            raise ValueError(f"Invalid query_type: {self.query_type}. Must be one of {valid_types}")


# --- Batch Command Protocol ---


class BatchCommand(BaseModel):
    """Batch command request for atomic multi-command execution."""

    commands: list[BridgeCommand]
    atomic: bool = False  # If True, all commands must succeed or all fail
    timeout_ms: int = 10000


# --- Command Validator ---


class CommandValidator:
    """Validates bridge commands and their required parameters."""

    # Mapping of action types to required parameter names
    REQUIRED_PARAMS: ClassVar[dict[str, list[str]]] = {
        "move": ["x", "y", "z"],
        "mine": ["x", "y", "z"],
        "place": ["x", "y", "z", "block_type"],
        "craft": ["item"],
        "eat": ["item"],
        "chat": ["message"],
        "look": ["x", "y", "z"],
        "trade": ["target_agent", "offer_items"],
        "gift": ["target_agent", "item"],
        "vote": ["proposal_id", "vote"],
        "follow": ["target_agent"],
        "attack": ["target"],
        "defend": [],
        "smelt": ["item", "fuel"],
        "farm": ["crop_type", "area"],
        "explore": ["radius"],
        "query": ["query_type"],
        "batch": ["commands"],
    }

    @classmethod
    def validate(cls, command: BridgeCommand) -> tuple[bool, str | None]:
        """Validate a bridge command has all required parameters.

        Args:
            command: The bridge command to validate.

        Returns:
            (is_valid, error_message) tuple. error_message is None if valid.
        """
        action = command.action
        params = command.params

        # Unknown actions are allowed (for extensibility)
        if action not in cls.REQUIRED_PARAMS:
            return True, None

        required = cls.REQUIRED_PARAMS[action]
        missing = [param for param in required if param not in params]

        if missing:
            return False, f"Missing required parameters for '{action}': {missing}"

        return True, None


# --- Protocol Serializer ---


class ProtocolSerializer:
    """Serializes and deserializes bridge protocol messages."""

    @staticmethod
    def serialize_command(command: BridgeCommand) -> bytes:
        """Serialize a bridge command to JSON bytes.

        Args:
            command: The bridge command to serialize.

        Returns:
            UTF-8 encoded JSON bytes.
        """
        return command.model_dump_json().encode("utf-8")

    @staticmethod
    def deserialize_event(data: bytes) -> BridgeEvent:
        """Deserialize a bridge event from JSON bytes.

        Args:
            data: UTF-8 encoded JSON bytes.

        Returns:
            Parsed BridgeEvent.

        Raises:
            ValueError: If data is not valid JSON or doesn't match BridgeEvent schema.
        """
        try:
            raw = json.loads(data.decode("utf-8"))
            return BridgeEvent.model_validate(raw)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to deserialize event: {e}") from e

    @staticmethod
    def serialize_batch(batch: BatchCommand) -> bytes:
        """Serialize a batch command to JSON bytes.

        Args:
            batch: The batch command to serialize.

        Returns:
            UTF-8 encoded JSON bytes.
        """
        return batch.model_dump_json().encode("utf-8")

    @staticmethod
    def deserialize_response(data: bytes) -> dict[str, Any]:
        """Deserialize a bridge response from JSON bytes.

        Args:
            data: UTF-8 encoded JSON bytes.

        Returns:
            Parsed response dictionary.

        Raises:
            ValueError: If data is not valid JSON.
        """
        try:
            return dict(json.loads(data.decode("utf-8")))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to deserialize response: {e}") from e


# --- Event Filter ---


class EventFilter:
    """Filters bridge events by event type."""

    def __init__(self, event_types: list[EventType]) -> None:
        """Initialize event filter with allowed event types.

        Args:
            event_types: List of event types to match.
        """
        self._event_types = set(event_types)

    def matches(self, event: BridgeEvent) -> bool:
        """Check if an event matches the filter.

        Args:
            event: The bridge event to check.

        Returns:
            True if event type is in the filter, False otherwise.
        """
        return event.event_type in self._event_types
