"""Tests for bridge protocol extensions."""

from __future__ import annotations

import json

import pytest

from piano.bridge.protocol import (
    BatchCommand,
    CommandType,
    CommandValidator,
    EventFilter,
    EventType,
    ProtocolSerializer,
    WorldQuery,
)
from piano.core.types import BridgeCommand, BridgeEvent

# ---------------------------------------------------------------------------
# Tests: CommandType
# ---------------------------------------------------------------------------


class TestCommandType:
    """Tests for CommandType enum."""

    def test_command_type_has_all_expected_values(self) -> None:
        """CommandType enum contains all expected command types."""
        expected = {
            "move",
            "mine",
            "place",
            "craft",
            "eat",
            "chat",
            "look",
            "trade",
            "gift",
            "vote",
            "follow",
            "attack",
            "defend",
            "smelt",
            "farm",
            "explore",
            "query",
            "batch",
        }
        actual = {ct.value for ct in CommandType}
        assert actual == expected

    def test_command_type_values_are_strings(self) -> None:
        """All CommandType values are strings."""
        for ct in CommandType:
            assert isinstance(ct.value, str)

    def test_command_type_values_are_unique(self) -> None:
        """All CommandType values are unique."""
        values = [ct.value for ct in CommandType]
        assert len(values) == len(set(values))

    def test_command_type_basic_commands(self) -> None:
        """Basic Phase 0 commands are present."""
        assert CommandType.MOVE == "move"
        assert CommandType.MINE == "mine"
        assert CommandType.CRAFT == "craft"
        assert CommandType.CHAT == "chat"

    def test_command_type_social_commands(self) -> None:
        """Phase 1 social commands are present."""
        assert CommandType.TRADE == "trade"
        assert CommandType.GIFT == "gift"
        assert CommandType.VOTE == "vote"
        assert CommandType.FOLLOW == "follow"

    def test_command_type_combat_commands(self) -> None:
        """Phase 1 combat commands are present."""
        assert CommandType.ATTACK == "attack"
        assert CommandType.DEFEND == "defend"

    def test_command_type_advanced_commands(self) -> None:
        """Phase 1 advanced commands are present."""
        assert CommandType.SMELT == "smelt"
        assert CommandType.FARM == "farm"
        assert CommandType.EXPLORE == "explore"

    def test_command_type_protocol_commands(self) -> None:
        """Phase 1 protocol commands are present."""
        assert CommandType.QUERY == "query"
        assert CommandType.BATCH == "batch"


# ---------------------------------------------------------------------------
# Tests: EventType
# ---------------------------------------------------------------------------


class TestEventType:
    """Tests for EventType enum."""

    def test_event_type_has_all_expected_values(self) -> None:
        """EventType enum contains all expected event types."""
        expected = {
            "perception",
            "chat",
            "error",
            "action_complete",
            "agent_nearby",
            "agent_left",
            "trade_request",
            "trade_complete",
            "damage",
            "death",
            "item_collected",
            "block_broken",
        }
        actual = {et.value for et in EventType}
        assert actual == expected

    def test_event_type_values_are_strings(self) -> None:
        """All EventType values are strings."""
        for et in EventType:
            assert isinstance(et.value, str)

    def test_event_type_values_are_unique(self) -> None:
        """All EventType values are unique."""
        values = [et.value for et in EventType]
        assert len(values) == len(set(values))

    def test_event_type_basic_events(self) -> None:
        """Basic Phase 0 events are present."""
        assert EventType.PERCEPTION == "perception"
        assert EventType.CHAT == "chat"
        assert EventType.ERROR == "error"
        assert EventType.ACTION_COMPLETE == "action_complete"

    def test_event_type_social_events(self) -> None:
        """Phase 1 social events are present."""
        assert EventType.AGENT_NEARBY == "agent_nearby"
        assert EventType.AGENT_LEFT == "agent_left"
        assert EventType.TRADE_REQUEST == "trade_request"
        assert EventType.TRADE_COMPLETE == "trade_complete"

    def test_event_type_combat_events(self) -> None:
        """Phase 1 combat/survival events are present."""
        assert EventType.DAMAGE == "damage"
        assert EventType.DEATH == "death"

    def test_event_type_resource_events(self) -> None:
        """Phase 1 resource events are present."""
        assert EventType.ITEM_COLLECTED == "item_collected"
        assert EventType.BLOCK_BROKEN == "block_broken"


# ---------------------------------------------------------------------------
# Tests: WorldQuery
# ---------------------------------------------------------------------------


class TestWorldQuery:
    """Tests for WorldQuery protocol."""

    def test_world_query_creation(self) -> None:
        """WorldQuery can be created with query_type and params."""
        query = WorldQuery(query_type="nearby_blocks", params={"radius": 10})
        assert query.query_type == "nearby_blocks"
        assert query.params["radius"] == 10

    def test_world_query_default_params(self) -> None:
        """WorldQuery params defaults to empty dict."""
        query = WorldQuery(query_type="position")
        assert query.params == {}

    def test_world_query_valid_types(self) -> None:
        """WorldQuery accepts all valid query types."""
        valid_types = [
            "nearby_blocks",
            "nearby_entities",
            "inventory",
            "position",
            "time",
        ]
        for qt in valid_types:
            query = WorldQuery(query_type=qt)
            assert query.query_type == qt

    def test_world_query_invalid_type_raises(self) -> None:
        """WorldQuery raises ValueError for invalid query_type."""
        with pytest.raises(ValueError, match="Input should be"):
            WorldQuery(query_type="invalid_type")

    def test_world_query_serialization(self) -> None:
        """WorldQuery serializes to JSON correctly."""
        query = WorldQuery(
            query_type="nearby_entities",
            params={"entity_type": "player", "range": 32},
        )
        data = json.loads(query.model_dump_json())
        assert data["query_type"] == "nearby_entities"
        assert data["params"]["entity_type"] == "player"
        assert data["params"]["range"] == 32

    def test_world_query_deserialization(self) -> None:
        """WorldQuery can be deserialized from dict."""
        raw = {
            "query_type": "inventory",
            "params": {"slot": 0},
        }
        query = WorldQuery.model_validate(raw)
        assert query.query_type == "inventory"
        assert query.params["slot"] == 0


# ---------------------------------------------------------------------------
# Tests: BatchCommand
# ---------------------------------------------------------------------------


class TestBatchCommand:
    """Tests for BatchCommand protocol."""

    def test_batch_command_creation(self) -> None:
        """BatchCommand can be created with list of commands."""
        cmd1 = BridgeCommand(action="move", params={"x": 1, "y": 2, "z": 3})
        cmd2 = BridgeCommand(action="mine", params={"x": 1, "y": 2, "z": 3})
        batch = BatchCommand(commands=[cmd1, cmd2])
        assert len(batch.commands) == 2
        assert batch.commands[0].action == "move"
        assert batch.commands[1].action == "mine"

    def test_batch_command_defaults(self) -> None:
        """BatchCommand has correct default values."""
        cmd = BridgeCommand(action="ping", params={})
        batch = BatchCommand(commands=[cmd])
        assert batch.atomic is False
        assert batch.timeout_ms == 10000

    def test_batch_command_atomic_flag(self) -> None:
        """BatchCommand atomic flag can be set."""
        cmd1 = BridgeCommand(action="move", params={"x": 1, "y": 2, "z": 3})
        cmd2 = BridgeCommand(action="chat", params={"message": "hi"})
        batch = BatchCommand(commands=[cmd1, cmd2], atomic=True)
        assert batch.atomic is True

    def test_batch_command_custom_timeout(self) -> None:
        """BatchCommand timeout_ms can be customized."""
        cmd = BridgeCommand(action="craft", params={"item": "planks"})
        batch = BatchCommand(commands=[cmd], timeout_ms=30000)
        assert batch.timeout_ms == 30000

    def test_batch_command_multiple_commands(self) -> None:
        """BatchCommand can contain multiple different commands."""
        commands = [
            BridgeCommand(action="move", params={"x": 10, "y": 64, "z": -5}),
            BridgeCommand(action="mine", params={"x": 10, "y": 64, "z": -5}),
            BridgeCommand(action="craft", params={"item": "stick"}),
            BridgeCommand(action="chat", params={"message": "Done!"}),
        ]
        batch = BatchCommand(commands=commands, atomic=True, timeout_ms=20000)
        assert len(batch.commands) == 4
        assert batch.atomic is True
        assert batch.timeout_ms == 20000

    def test_batch_command_empty_list(self) -> None:
        """BatchCommand can be created with empty commands list."""
        batch = BatchCommand(commands=[])
        assert len(batch.commands) == 0


# ---------------------------------------------------------------------------
# Tests: CommandValidator
# ---------------------------------------------------------------------------


class TestCommandValidator:
    """Tests for CommandValidator."""

    def test_validator_has_required_params_mapping(self) -> None:
        """CommandValidator.REQUIRED_PARAMS contains expected mappings."""
        assert "move" in CommandValidator.REQUIRED_PARAMS
        assert "mine" in CommandValidator.REQUIRED_PARAMS
        assert "craft" in CommandValidator.REQUIRED_PARAMS
        assert CommandValidator.REQUIRED_PARAMS["move"] == ["x", "y", "z"]
        assert CommandValidator.REQUIRED_PARAMS["mine"] == ["x", "y", "z"]

    def test_validator_validates_move_command(self) -> None:
        """Validator accepts valid move command."""
        cmd = BridgeCommand(action="move", params={"x": 1, "y": 2, "z": 3})
        valid, error = CommandValidator.validate(cmd)
        assert valid is True
        assert error is None

    def test_validator_rejects_move_missing_params(self) -> None:
        """Validator rejects move command missing required params."""
        cmd = BridgeCommand(action="move", params={"x": 1, "y": 2})  # missing z
        valid, error = CommandValidator.validate(cmd)
        assert valid is False
        assert error is not None
        assert "z" in error

    def test_validator_validates_craft_command(self) -> None:
        """Validator accepts valid craft command."""
        cmd = BridgeCommand(action="craft", params={"item": "planks", "count": 4})
        valid, error = CommandValidator.validate(cmd)
        assert valid is True
        assert error is None

    def test_validator_rejects_craft_missing_item(self) -> None:
        """Validator rejects craft command missing item param."""
        cmd = BridgeCommand(action="craft", params={"count": 4})
        valid, error = CommandValidator.validate(cmd)
        assert valid is False
        assert "item" in error

    def test_validator_validates_chat_command(self) -> None:
        """Validator accepts valid chat command."""
        cmd = BridgeCommand(action="chat", params={"message": "hello"})
        valid, error = CommandValidator.validate(cmd)
        assert valid is True
        assert error is None

    def test_validator_rejects_chat_missing_message(self) -> None:
        """Validator rejects chat command missing message param."""
        cmd = BridgeCommand(action="chat", params={})
        valid, error = CommandValidator.validate(cmd)
        assert valid is False
        assert "message" in error

    def test_validator_validates_trade_command(self) -> None:
        """Validator accepts valid trade command."""
        cmd = BridgeCommand(
            action="trade",
            params={"target_agent": "agent-002", "offer_items": ["diamond"]},
        )
        valid, error = CommandValidator.validate(cmd)
        assert valid is True
        assert error is None

    def test_validator_rejects_invalid_commands(self) -> None:
        """Validator rejects commands with missing required parameters."""
        invalid_commands = [
            BridgeCommand(action="place", params={"x": 1, "y": 2}),  # missing z, block_type
            BridgeCommand(action="smelt", params={"item": "iron_ore"}),  # missing fuel
            BridgeCommand(action="gift", params={"item": "diamond"}),  # missing target_agent
            BridgeCommand(action="vote", params={"proposal_id": "1"}),  # missing vote
        ]
        for cmd in invalid_commands:
            valid, error = CommandValidator.validate(cmd)
            assert valid is False
            assert error is not None

    def test_validator_allows_unknown_actions(self) -> None:
        """Validator accepts unknown action types (for extensibility)."""
        cmd = BridgeCommand(action="unknown_action", params={})
        valid, error = CommandValidator.validate(cmd)
        assert valid is True
        assert error is None

    def test_validator_allows_extra_params(self) -> None:
        """Validator accepts commands with extra parameters."""
        cmd = BridgeCommand(
            action="move",
            params={"x": 1, "y": 2, "z": 3, "speed": "fast", "extra": "data"},
        )
        valid, error = CommandValidator.validate(cmd)
        assert valid is True
        assert error is None

    def test_validator_defend_command_no_params(self) -> None:
        """Validator accepts defend command with no required params."""
        cmd = BridgeCommand(action="defend", params={})
        valid, error = CommandValidator.validate(cmd)
        assert valid is True
        assert error is None


# ---------------------------------------------------------------------------
# Tests: ProtocolSerializer
# ---------------------------------------------------------------------------


class TestProtocolSerializer:
    """Tests for ProtocolSerializer."""

    def test_serialize_command(self) -> None:
        """ProtocolSerializer serializes BridgeCommand to bytes."""
        cmd = BridgeCommand(action="move", params={"x": 10, "y": 64, "z": -5})
        data = ProtocolSerializer.serialize_command(cmd)
        assert isinstance(data, bytes)

        # Verify it's valid JSON
        parsed = json.loads(data.decode("utf-8"))
        assert parsed["action"] == "move"
        assert parsed["params"]["x"] == 10

    def test_deserialize_event(self) -> None:
        """ProtocolSerializer deserializes bytes to BridgeEvent."""
        raw = {
            "event_type": "chat",
            "data": {"username": "agent-001", "message": "hello"},
            "timestamp": "2026-01-01T00:00:00Z",
        }
        data = json.dumps(raw).encode("utf-8")
        event = ProtocolSerializer.deserialize_event(data)
        assert isinstance(event, BridgeEvent)
        assert event.event_type == "chat"
        assert event.data["message"] == "hello"

    def test_serialize_deserialize_roundtrip(self) -> None:
        """Command serialization and event deserialization round-trip correctly."""
        # Serialize command
        cmd = BridgeCommand(action="craft", params={"item": "stick", "count": 4})
        cmd_data = ProtocolSerializer.serialize_command(cmd)
        cmd_parsed = json.loads(cmd_data.decode("utf-8"))
        assert cmd_parsed["action"] == "craft"

        # Deserialize event
        event_raw = {
            "event_type": "action_complete",
            "data": {"action": "craft", "result": "success"},
            "timestamp": "2026-01-01T00:00:00Z",
        }
        event_data = json.dumps(event_raw).encode("utf-8")
        event = ProtocolSerializer.deserialize_event(event_data)
        assert event.event_type == "action_complete"

    def test_serialize_batch(self) -> None:
        """ProtocolSerializer serializes BatchCommand to bytes."""
        cmd1 = BridgeCommand(action="move", params={"x": 1, "y": 2, "z": 3})
        cmd2 = BridgeCommand(action="mine", params={"x": 1, "y": 2, "z": 3})
        batch = BatchCommand(commands=[cmd1, cmd2], atomic=True)
        data = ProtocolSerializer.serialize_batch(batch)
        assert isinstance(data, bytes)

        # Verify it's valid JSON
        parsed = json.loads(data.decode("utf-8"))
        assert len(parsed["commands"]) == 2
        assert parsed["atomic"] is True

    def test_deserialize_response(self) -> None:
        """ProtocolSerializer deserializes response bytes to dict."""
        raw = {
            "id": "test-id",
            "success": True,
            "data": {"moved": True, "position": {"x": 10, "y": 64, "z": -5}},
            "error": None,
        }
        data = json.dumps(raw).encode("utf-8")
        response = ProtocolSerializer.deserialize_response(data)
        assert isinstance(response, dict)
        assert response["success"] is True
        assert response["data"]["moved"] is True

    def test_deserialize_event_invalid_json_raises(self) -> None:
        """Deserializing invalid JSON raises ValueError."""
        invalid_data = b"not valid json"
        with pytest.raises(ValueError, match="Failed to deserialize event"):
            ProtocolSerializer.deserialize_event(invalid_data)

    def test_deserialize_response_invalid_json_raises(self) -> None:
        """Deserializing invalid response JSON raises ValueError."""
        invalid_data = b"{incomplete json"
        with pytest.raises(ValueError, match="Failed to deserialize response"):
            ProtocolSerializer.deserialize_response(invalid_data)

    def test_deserialize_event_invalid_schema_raises(self) -> None:
        """Deserializing event with invalid schema raises ValueError."""
        invalid_event = {"not_event_type": "foo"}
        data = json.dumps(invalid_event).encode("utf-8")
        with pytest.raises(ValueError, match="Failed to deserialize event"):
            ProtocolSerializer.deserialize_event(data)


# ---------------------------------------------------------------------------
# Tests: EventFilter
# ---------------------------------------------------------------------------


class TestEventFilter:
    """Tests for EventFilter."""

    def test_event_filter_creation(self) -> None:
        """EventFilter can be created with list of event types."""
        filter_ = EventFilter([EventType.CHAT, EventType.PERCEPTION])
        assert filter_ is not None

    def test_event_filter_matches_included_type(self) -> None:
        """EventFilter matches events of included types."""
        filter_ = EventFilter([EventType.CHAT, EventType.DAMAGE])
        event = BridgeEvent(event_type="chat", data={"message": "hi"})
        assert filter_.matches(event) is True

    def test_event_filter_rejects_excluded_type(self) -> None:
        """EventFilter rejects events of excluded types."""
        filter_ = EventFilter([EventType.CHAT, EventType.DAMAGE])
        event = BridgeEvent(event_type="perception", data={})
        assert filter_.matches(event) is False

    def test_event_filter_multiple_types(self) -> None:
        """EventFilter correctly handles multiple event types."""
        filter_ = EventFilter(
            [
                EventType.AGENT_NEARBY,
                EventType.AGENT_LEFT,
                EventType.TRADE_REQUEST,
            ]
        )
        assert filter_.matches(BridgeEvent(event_type="agent_nearby", data={})) is True
        assert filter_.matches(BridgeEvent(event_type="agent_left", data={})) is True
        assert filter_.matches(BridgeEvent(event_type="trade_request", data={})) is True
        assert filter_.matches(BridgeEvent(event_type="chat", data={})) is False
        assert filter_.matches(BridgeEvent(event_type="death", data={})) is False

    def test_event_filter_empty_list(self) -> None:
        """EventFilter with empty list rejects all events."""
        filter_ = EventFilter([])
        event = BridgeEvent(event_type="chat", data={"message": "hi"})
        assert filter_.matches(event) is False

    def test_event_filter_single_type(self) -> None:
        """EventFilter works with single event type."""
        filter_ = EventFilter([EventType.ERROR])
        assert filter_.matches(BridgeEvent(event_type="error", data={})) is True
        assert filter_.matches(BridgeEvent(event_type="chat", data={})) is False

    def test_event_filter_all_types(self) -> None:
        """EventFilter can be created with all event types."""
        all_types = list(EventType)
        filter_ = EventFilter(all_types)
        for et in EventType:
            event = BridgeEvent(event_type=et.value, data={})
            assert filter_.matches(event) is True
