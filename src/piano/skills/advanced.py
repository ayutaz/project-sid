"""Advanced Minecraft skill functions for complex multi-step behaviors.

These skills build on basic actions to provide higher-level capabilities
like crafting chains, building, farming, combat, and storage management.
Unlike basic skills, these return BridgeCommand objects or sequences
for planning and composition.

Reference: docs/implementation/05-minecraft-platform.md Section 3.3
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from piano.skills.registry import SkillRegistry

from piano.core.types import BridgeCommand

__all__ = [
    "ADVANCED_SKILLS",
    "CraftingRecipes",
    "attack_entity",
    "build_structure",
    "craft_chain",
    "defend",
    "deposit_items",
    "explore_direction",
    "farm_harvest",
    "farm_plant",
    "flee",
    "register_advanced_skills",
    "smelt_item",
    "withdraw_items",
]


# --- Crafting Recipe Database ---


class CraftingRecipes:
    """Common Minecraft crafting recipes for multi-step craft planning.

    Maps item names to their required ingredients and quantities.
    Used by craft_chain to determine crafting prerequisites.
    """

    RECIPES: ClassVar[dict[str, dict[str, int]]] = {
        # Wood tools
        "planks": {"log": 1},
        "stick": {"planks": 2},
        "wooden_pickaxe": {"planks": 3, "stick": 2},
        "wooden_axe": {"planks": 3, "stick": 2},
        "wooden_shovel": {"planks": 1, "stick": 2},
        "wooden_sword": {"planks": 2, "stick": 1},
        # Stone tools
        "stone_pickaxe": {"cobblestone": 3, "stick": 2},
        "stone_axe": {"cobblestone": 3, "stick": 2},
        "stone_shovel": {"cobblestone": 1, "stick": 2},
        "stone_sword": {"cobblestone": 2, "stick": 1},
        # Iron tools
        "iron_pickaxe": {"iron_ingot": 3, "stick": 2},
        "iron_axe": {"iron_ingot": 3, "stick": 2},
        "iron_shovel": {"iron_ingot": 1, "stick": 2},
        "iron_sword": {"iron_ingot": 2, "stick": 1},
        # Utility
        "crafting_table": {"planks": 4},
        "furnace": {"cobblestone": 8},
        "chest": {"planks": 8},
        "torch": {"stick": 1, "coal": 1},
        "bed": {"planks": 3, "wool": 3},
        # Smelting outputs (for planning)
        "iron_ingot": {"iron_ore": 1},
        "gold_ingot": {"gold_ore": 1},
        "glass": {"sand": 1},
    }

    @classmethod
    def get_recipe(cls, item: str) -> dict[str, int]:
        """Get the recipe for an item.

        Args:
            item: Item name to look up.

        Returns:
            Dict mapping ingredient names to required quantities.

        Raises:
            KeyError: If the recipe is not found.
        """
        return cls.RECIPES[item]

    @classmethod
    def has_recipe(cls, item: str) -> bool:
        """Check if a recipe exists for an item.

        Args:
            item: Item name to check.

        Returns:
            True if the recipe exists.
        """
        return item in cls.RECIPES


# --- Advanced Skill Functions ---


def craft_chain(
    target_item: str,
    inventory: dict[str, int],
    *,
    max_depth: int = 10,
    _seen: set[str] | None = None,
) -> list[BridgeCommand]:
    """Generate a sequence of craft commands to create a target item.

    Recursively resolves crafting dependencies and generates the minimum
    sequence of craft commands needed, accounting for current inventory.

    Args:
        target_item: The item to craft (e.g., "stone_pickaxe").
        inventory: Current inventory mapping item names to counts.
        max_depth: Maximum recursion depth to prevent infinite loops.
        _seen: Internal set for cycle detection.

    Returns:
        List of BridgeCommand objects representing the craft sequence.
        Returns empty list if recipe is not found, already have the item,
        a cycle is detected, or max_depth is exceeded.
    """
    if max_depth <= 0:
        return []

    if _seen is None:
        _seen = set()

    if target_item in _seen:
        return []

    if not CraftingRecipes.has_recipe(target_item):
        return []

    _seen = _seen | {target_item}

    recipe = CraftingRecipes.get_recipe(target_item)
    commands: list[BridgeCommand] = []
    current_inv = dict(inventory)

    for ingredient, needed in recipe.items():
        have = current_inv.get(ingredient, 0)
        if have < needed:
            shortage = needed - have
            sub_commands = craft_chain(
                ingredient,
                current_inv,
                max_depth=max_depth - 1,
                _seen=_seen,
            )
            commands.extend(sub_commands)
            current_inv[ingredient] = current_inv.get(ingredient, 0) + shortage

    commands.append(
        BridgeCommand(
            action="craft",
            params={"item": target_item, "count": 1},
        )
    )

    return commands


def build_structure(
    structure_type: str,
    position: dict[str, float],
    blocks: list[dict[str, Any]],
) -> list[BridgeCommand]:
    """Generate commands to build a structure at a location.

    Args:
        structure_type: Type of structure (e.g., "wall", "house", "tower").
        position: Base position dict with x, y, z keys.
        blocks: List of block placement dicts, each with:
            - offset_x, offset_y, offset_z: Offsets from base position
            - block_type: Type of block to place

    Returns:
        List of BridgeCommand objects for placing each block.
    """
    commands: list[BridgeCommand] = []
    base_x = position["x"]
    base_y = position["y"]
    base_z = position["z"]

    for block in blocks:
        x = base_x + block.get("offset_x", 0)
        y = base_y + block.get("offset_y", 0)
        z = base_z + block.get("offset_z", 0)
        block_type = block["block_type"]

        commands.append(
            BridgeCommand(
                action="place",
                params={
                    "x": x,
                    "y": y,
                    "z": z,
                    "block_type": block_type,
                },
            )
        )

    return commands


def farm_plant(crop: str, position: dict[str, float]) -> BridgeCommand:
    """Plant a crop at a position.

    Args:
        crop: Crop type to plant (e.g., "wheat_seeds", "carrot", "potato").
        position: Position dict with x, y, z keys.

    Returns:
        BridgeCommand for planting the crop.
    """
    return BridgeCommand(
        action="plant",
        params={
            "x": position["x"],
            "y": position["y"],
            "z": position["z"],
            "crop": crop,
        },
    )


def farm_harvest(position: dict[str, float]) -> BridgeCommand:
    """Harvest a crop at a position.

    Args:
        position: Position dict with x, y, z keys.

    Returns:
        BridgeCommand for harvesting the crop.
    """
    return BridgeCommand(
        action="harvest",
        params={
            "x": position["x"],
            "y": position["y"],
            "z": position["z"],
        },
    )


def smelt_item(input_item: str, fuel: str = "coal", count: int = 1) -> BridgeCommand:
    """Smelt an item in a furnace.

    Args:
        input_item: Item to smelt (e.g., "iron_ore", "sand").
        fuel: Fuel type to use (default: "coal").
        count: Number of items to smelt.

    Returns:
        BridgeCommand for smelting operation.
    """
    return BridgeCommand(
        action="smelt",
        params={
            "input_item": input_item,
            "fuel": fuel,
            "count": count,
        },
    )


def explore_direction(direction: str, distance: float = 50.0) -> BridgeCommand:
    """Explore in a cardinal direction.

    Args:
        direction: Cardinal direction ("north", "south", "east", "west").
        distance: Distance to travel in blocks.

    Returns:
        BridgeCommand for exploration.
    """
    return BridgeCommand(
        action="explore",
        params={
            "direction": direction,
            "distance": distance,
        },
    )


def attack_entity(target: str) -> BridgeCommand:
    """Attack a nearby entity.

    Args:
        target: Entity name or ID to attack.

    Returns:
        BridgeCommand for attacking.
    """
    return BridgeCommand(
        action="attack",
        params={"target": target},
    )


def defend(shield: bool = True) -> BridgeCommand:
    """Enter defensive stance.

    Args:
        shield: Whether to raise shield if available.

    Returns:
        BridgeCommand for defending.
    """
    return BridgeCommand(
        action="defend",
        params={"shield": shield},
    )


def flee(from_position: dict[str, float], distance: float = 20.0) -> BridgeCommand:
    """Flee from a position.

    Args:
        from_position: Position dict to flee from (x, y, z).
        distance: Distance to flee in blocks.

    Returns:
        BridgeCommand for fleeing.
    """
    return BridgeCommand(
        action="flee",
        params={
            "from_x": from_position["x"],
            "from_y": from_position["y"],
            "from_z": from_position["z"],
            "distance": distance,
        },
    )


def deposit_items(
    chest_position: dict[str, float],
    items: dict[str, int],
) -> BridgeCommand:
    """Deposit items into a chest.

    Args:
        chest_position: Position of the chest (x, y, z).
        items: Dict mapping item names to quantities to deposit.

    Returns:
        BridgeCommand for depositing items.
    """
    return BridgeCommand(
        action="deposit",
        params={
            "x": chest_position["x"],
            "y": chest_position["y"],
            "z": chest_position["z"],
            "items": items,
        },
    )


def withdraw_items(
    chest_position: dict[str, float],
    items: list[str],
) -> BridgeCommand:
    """Withdraw items from a chest.

    Args:
        chest_position: Position of the chest (x, y, z).
        items: List of item names to withdraw.

    Returns:
        BridgeCommand for withdrawing items.
    """
    return BridgeCommand(
        action="withdraw",
        params={
            "x": chest_position["x"],
            "y": chest_position["y"],
            "z": chest_position["z"],
            "items": items,
        },
    )


# --- Async Wrappers for Advanced Skills ---

_DEFAULT_POS: dict[str, float] = {"x": 0, "y": 0, "z": 0}


async def _async_craft_chain(
    bridge: Any,
    target_item: str = "",
    inventory: str | dict[str, int] = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Async wrapper for craft_chain that sends commands via bridge."""
    inv = inventory if isinstance(inventory, dict) else {}
    commands = craft_chain(target_item, inv)
    results = []
    for cmd in commands:
        result = await bridge.send_command(cmd)
        results.append(result)
    return {
        "success": True,
        "commands_sent": len(commands),
        "results": results,
    }


async def _async_build_structure(
    bridge: Any,
    structure_type: str = "",
    position: dict[str, float] | None = None,
    blocks: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Async wrapper for build_structure."""
    commands = build_structure(structure_type, position or _DEFAULT_POS, blocks or [])
    results = []
    for cmd in commands:
        result = await bridge.send_command(cmd)
        results.append(result)
    return {
        "success": True,
        "commands_sent": len(commands),
        "results": results,
    }


async def _async_farm_plant(
    bridge: Any,
    crop: str = "",
    position: dict[str, float] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Async wrapper for farm_plant."""
    cmd = farm_plant(crop, position or _DEFAULT_POS)
    return await bridge.send_command(cmd)


async def _async_farm_harvest(
    bridge: Any,
    position: dict[str, float] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Async wrapper for farm_harvest."""
    cmd = farm_harvest(position or _DEFAULT_POS)
    return await bridge.send_command(cmd)


async def _async_smelt_item(
    bridge: Any,
    input_item: str = "",
    fuel: str = "coal",
    count: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Async wrapper for smelt_item."""
    cmd = smelt_item(input_item, fuel=fuel, count=count)
    return await bridge.send_command(cmd)


async def _async_explore_direction(
    bridge: Any,
    direction: str = "north",
    distance: float = 50.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Async wrapper for explore_direction."""
    cmd = explore_direction(direction, distance=distance)
    return await bridge.send_command(cmd)


async def _async_attack_entity(bridge: Any, target: str = "", **kwargs: Any) -> dict[str, Any]:
    """Async wrapper for attack_entity."""
    cmd = attack_entity(target)
    return await bridge.send_command(cmd)


async def _async_defend(bridge: Any, shield: bool = True, **kwargs: Any) -> dict[str, Any]:
    """Async wrapper for defend."""
    cmd = defend(shield=shield)
    return await bridge.send_command(cmd)


async def _async_flee(
    bridge: Any,
    from_position: dict[str, float] | None = None,
    distance: float = 20.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Async wrapper for flee."""
    cmd = flee(from_position or _DEFAULT_POS, distance=distance)
    return await bridge.send_command(cmd)


async def _async_deposit_items(
    bridge: Any,
    chest_position: dict[str, float] | None = None,
    items: dict[str, int] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Async wrapper for deposit_items."""
    cmd = deposit_items(chest_position or _DEFAULT_POS, items or {})
    return await bridge.send_command(cmd)


async def _async_withdraw_items(
    bridge: Any,
    chest_position: dict[str, float] | None = None,
    items: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Async wrapper for withdraw_items."""
    cmd = withdraw_items(chest_position or _DEFAULT_POS, items or [])
    return await bridge.send_command(cmd)


async def _async_equip_item(
    bridge: Any,
    item: str = "",
    destination: str = "hand",
    **kwargs: Any,
) -> dict[str, Any]:
    """Async wrapper for equip_item that sends BridgeCommand."""
    cmd = BridgeCommand(
        action="equip",
        params={"item": item, "destination": destination},
    )
    return await bridge.send_command(cmd)


async def _async_use_item(bridge: Any, **kwargs: Any) -> dict[str, Any]:
    """Async wrapper for use_item that sends BridgeCommand."""
    cmd = BridgeCommand(action="use", params={})
    return await bridge.send_command(cmd)


async def _async_drop_item(
    bridge: Any,
    item: str = "",
    count: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Async wrapper for drop_item that sends BridgeCommand."""
    cmd = BridgeCommand(
        action="drop",
        params={"item": item, "count": count},
    )
    return await bridge.send_command(cmd)


async def _async_eat_food(
    bridge: Any,
    item: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Async wrapper for eat_food that sends BridgeCommand."""
    cmd = BridgeCommand(
        action="eat",
        params={"item": item},
    )
    return await bridge.send_command(cmd)


# --- Skill Registry ---


ADVANCED_SKILLS: dict[str, Callable[..., BridgeCommand | list[BridgeCommand]]] = {
    "craft_chain": craft_chain,
    "build_structure": build_structure,
    "farm_plant": farm_plant,
    "farm_harvest": farm_harvest,
    "smelt_item": smelt_item,
    "explore_direction": explore_direction,
    "attack_entity": attack_entity,
    "defend": defend,
    "flee": flee,
    "deposit_items": deposit_items,
    "withdraw_items": withdraw_items,
}


def register_advanced_skills(registry: SkillRegistry) -> None:
    """Register all advanced skills with a SkillRegistry.

    Advanced skills are wrapped in async executors so they can be
    dispatched by the SkillExecutor. Each wrapper sends the generated
    BridgeCommand(s) via the bridge client.

    Args:
        registry: SkillRegistry instance to register skills with.
    """
    registry.register(
        "attack_entity",
        _async_attack_entity,
        params_schema={"target": "str"},
        description="Attack a nearby entity",
    )
    registry.register(
        "build_structure",
        _async_build_structure,
        params_schema={
            "structure_type": "str",
            "position": "dict",
            "blocks": "list",
        },
        description="Build a structure at a location",
    )
    registry.register(
        "place_block",
        _async_build_structure,
        params_schema={
            "structure_type": "str",
            "position": "dict",
            "blocks": "list",
        },
        description="Place a block at a location",
    )
    registry.register(
        "flee",
        _async_flee,
        params_schema={"from_position": "dict", "distance": "float"},
        description="Flee from a position",
    )
    registry.register(
        "defend",
        _async_defend,
        params_schema={"shield": "bool"},
        description="Enter defensive stance",
    )
    registry.register(
        "defend_self",
        _async_defend,
        params_schema={"shield": "bool"},
        description="Enter defensive stance (alias)",
    )
    registry.register(
        "craft_chain_skill",
        _async_craft_chain,
        params_schema={"target_item": "str", "inventory": "dict"},
        description="Craft an item with full dependency chain",
    )
    registry.register(
        "farm_plant",
        _async_farm_plant,
        params_schema={"crop": "str", "position": "dict"},
        description="Plant a crop",
    )
    registry.register(
        "farm_harvest",
        _async_farm_harvest,
        params_schema={"position": "dict"},
        description="Harvest a crop",
    )
    registry.register(
        "smelt_item",
        _async_smelt_item,
        params_schema={
            "input_item": "str",
            "fuel": "str",
            "count": "int",
        },
        description="Smelt an item in a furnace",
    )
    registry.register(
        "explore_direction",
        _async_explore_direction,
        params_schema={"direction": "str", "distance": "float"},
        description="Explore in a cardinal direction",
    )
    registry.register(
        "deposit_items",
        _async_deposit_items,
        params_schema={"chest_position": "dict", "items": "dict"},
        description="Deposit items into a chest",
    )
    registry.register(
        "withdraw_items",
        _async_withdraw_items,
        params_schema={"chest_position": "dict", "items": "list"},
        description="Withdraw items from a chest",
    )
    registry.register(
        "equip_item",
        _async_equip_item,
        params_schema={"item": "str", "destination": "str"},
        description="Equip an item",
    )
    registry.register(
        "use_item",
        _async_use_item,
        params_schema={},
        description="Use an item",
    )
    registry.register(
        "drop_item",
        _async_drop_item,
        params_schema={"item": "str", "count": "int"},
        description="Drop an item",
    )
    registry.register(
        "eat_food",
        _async_eat_food,
        params_schema={"item": "str"},
        description="Eat food",
    )
