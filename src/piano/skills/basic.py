"""Basic Minecraft skill functions.

Each skill is an async function that takes a bridge client and
action-specific parameters, then returns a result dict.
These are the primitive building blocks for agent actions.

Reference: docs/implementation/05-minecraft-platform.md Section 3.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from piano.core.types import BridgeCommand
from piano.skills.registry import SkillRegistry

if TYPE_CHECKING:
    pass


@runtime_checkable
class BridgeClient(Protocol):
    """Protocol for the Python-Mineflayer bridge client.

    Implemented by the bridge module; skills depend only on this protocol.
    """

    async def send_command(self, cmd: BridgeCommand) -> dict[str, Any]: ...


# --- Skill Functions ---


async def move_to(bridge: BridgeClient, x: float, y: float, z: float) -> dict[str, Any]:
    """Move the agent to the specified coordinates.

    Args:
        bridge: Bridge client for Mineflayer communication.
        x: Target X coordinate.
        y: Target Y coordinate.
        z: Target Z coordinate.

    Returns:
        Result dict with success status and position data.
    """
    cmd = BridgeCommand(action="move", params={"x": x, "y": y, "z": z})
    return await bridge.send_command(cmd)


async def mine_block(bridge: BridgeClient, x: float, y: float, z: float) -> dict[str, Any]:
    """Mine a block at the specified coordinates.

    Args:
        bridge: Bridge client for Mineflayer communication.
        x: Block X coordinate.
        y: Block Y coordinate.
        z: Block Z coordinate.

    Returns:
        Result dict with success status and mined block data.
    """
    cmd = BridgeCommand(action="mine", params={"x": x, "y": y, "z": z})
    return await bridge.send_command(cmd)


async def craft_item(
    bridge: BridgeClient, item_name: str, count: int = 1
) -> dict[str, Any]:
    """Craft an item.

    Args:
        bridge: Bridge client for Mineflayer communication.
        item_name: Name of the item to craft.
        count: Number of items to craft.

    Returns:
        Result dict with success status and crafted item data.
    """
    cmd = BridgeCommand(action="craft", params={"item_name": item_name, "count": count})
    return await bridge.send_command(cmd)


async def chat(bridge: BridgeClient, message: str) -> dict[str, Any]:
    """Send a chat message.

    Args:
        bridge: Bridge client for Mineflayer communication.
        message: The message to send.

    Returns:
        Result dict with success status.
    """
    cmd = BridgeCommand(action="chat", params={"message": message})
    return await bridge.send_command(cmd)


async def look_at(bridge: BridgeClient, x: float, y: float, z: float) -> dict[str, Any]:
    """Look at the specified coordinates.

    Args:
        bridge: Bridge client for Mineflayer communication.
        x: Target X coordinate.
        y: Target Y coordinate.
        z: Target Z coordinate.

    Returns:
        Result dict with success status.
    """
    cmd = BridgeCommand(action="look", params={"x": x, "y": y, "z": z})
    return await bridge.send_command(cmd)


async def get_position(bridge: BridgeClient) -> dict[str, Any]:
    """Get the agent's current position.

    Args:
        bridge: Bridge client for Mineflayer communication.

    Returns:
        Result dict with position data (x, y, z).
    """
    cmd = BridgeCommand(action="get_position", params={})
    return await bridge.send_command(cmd)


async def get_inventory(bridge: BridgeClient) -> dict[str, Any]:
    """Get the agent's current inventory.

    Args:
        bridge: Bridge client for Mineflayer communication.

    Returns:
        Result dict with inventory items.
    """
    cmd = BridgeCommand(action="get_inventory", params={})
    return await bridge.send_command(cmd)


# --- Default Registry ---


def create_default_registry() -> SkillRegistry:
    """Create a SkillRegistry pre-populated with all basic skills.

    Returns:
        A SkillRegistry containing move_to, mine_block, craft_item,
        chat, look_at, get_position, and get_inventory.
    """
    registry = SkillRegistry()

    registry.register(
        "move_to",
        move_to,
        params_schema={"x": "float", "y": "float", "z": "float"},
        description="Move to coordinates (x, y, z)",
    )
    registry.register(
        "mine_block",
        mine_block,
        params_schema={"x": "float", "y": "float", "z": "float"},
        description="Mine a block at (x, y, z)",
    )
    registry.register(
        "craft_item",
        craft_item,
        params_schema={"item_name": "str", "count": "int"},
        description="Craft an item by name",
    )
    registry.register(
        "chat",
        chat,
        params_schema={"message": "str"},
        description="Send a chat message",
    )
    registry.register(
        "look_at",
        look_at,
        params_schema={"x": "float", "y": "float", "z": "float"},
        description="Look at coordinates (x, y, z)",
    )
    registry.register(
        "get_position",
        get_position,
        params_schema={},
        description="Get current position",
    )
    registry.register(
        "get_inventory",
        get_inventory,
        params_schema={},
        description="Get current inventory",
    )

    return registry
