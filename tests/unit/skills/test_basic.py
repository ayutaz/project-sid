"""Tests for basic skill functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from piano.skills.basic import (
    chat,
    craft_item,
    create_default_registry,
    get_inventory,
    get_position,
    look_at,
    mine_block,
    move_to,
)

if TYPE_CHECKING:
    from piano.core.types import BridgeCommand


class MockBridgeClient:
    """Mock bridge client that records commands and returns success."""

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.last_command: BridgeCommand | None = None
        self._response = response or {"success": True, "data": {}}

    async def send_command(self, cmd: BridgeCommand) -> dict[str, Any]:
        self.last_command = cmd
        return self._response


@pytest.fixture
def bridge() -> MockBridgeClient:
    return MockBridgeClient()


class TestMoveToSkill:
    async def test_move_to_sends_command(self, bridge: MockBridgeClient) -> None:
        result = await move_to(bridge, 10.0, 64.0, -20.0)
        assert result["success"] is True
        assert bridge.last_command is not None
        assert bridge.last_command.action == "move"
        assert bridge.last_command.params == {"x": 10.0, "y": 64.0, "z": -20.0}


class TestMineBlockSkill:
    async def test_mine_block_sends_command(self, bridge: MockBridgeClient) -> None:
        result = await mine_block(bridge, 5.0, 60.0, 5.0)
        assert result["success"] is True
        assert bridge.last_command is not None
        assert bridge.last_command.action == "mine"


class TestCraftItemSkill:
    async def test_craft_item_default_count(self, bridge: MockBridgeClient) -> None:
        result = await craft_item(bridge, "planks")
        assert result["success"] is True
        assert bridge.last_command is not None
        assert bridge.last_command.params["item"] == "planks"
        assert bridge.last_command.params["count"] == 1

    async def test_craft_item_custom_count(self, bridge: MockBridgeClient) -> None:
        await craft_item(bridge, "stick", count=4)
        assert bridge.last_command is not None
        assert bridge.last_command.params["count"] == 4


class TestChatSkill:
    async def test_chat_sends_message(self, bridge: MockBridgeClient) -> None:
        await chat(bridge, "Hello world")
        assert bridge.last_command is not None
        assert bridge.last_command.action == "chat"
        assert bridge.last_command.params["message"] == "Hello world"


class TestLookAtSkill:
    async def test_look_at_sends_command(self, bridge: MockBridgeClient) -> None:
        await look_at(bridge, 1.0, 2.0, 3.0)
        assert bridge.last_command is not None
        assert bridge.last_command.action == "look"


class TestGetPositionSkill:
    async def test_get_position_sends_command(self, bridge: MockBridgeClient) -> None:
        await get_position(bridge)
        assert bridge.last_command is not None
        assert bridge.last_command.action == "get_position"


class TestGetInventorySkill:
    async def test_get_inventory_sends_command(self, bridge: MockBridgeClient) -> None:
        await get_inventory(bridge)
        assert bridge.last_command is not None
        assert bridge.last_command.action == "get_inventory"


class TestCreateDefaultRegistry:
    def test_all_basic_skills_registered(self) -> None:
        registry = create_default_registry()
        expected = [
            "chat",
            "craft_item",
            "get_inventory",
            "get_position",
            "look_at",
            "mine_block",
            "move_to",
        ]
        assert registry.list_skills() == expected

    def test_skills_are_callable(self) -> None:
        registry = create_default_registry()
        for name in registry.list_skills():
            skill = registry.get(name)
            assert callable(skill.execute_fn)
