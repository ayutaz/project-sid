"""Action mapper - converts CC action names to skill registry names."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from piano.skills.basic import create_default_registry
from piano.skills.social import register_social_skills

if TYPE_CHECKING:
    from piano.skills.registry import SkillRegistry

logger = structlog.get_logger()

# CC action name -> SkillRegistry skill name
# None means the action should be ignored (no skill to execute)
ACTION_TO_SKILL: dict[str, str | None] = {
    # Basic
    "move": "move_to",
    "mine": "mine_block",
    "craft": "craft_item",
    "look": "look_at",
    "chat": "chat",
    "get_position": "get_position",
    "get_inventory": "get_inventory",
    # Advanced
    "explore": "explore_direction",
    "place": "place_block",
    "smelt": "smelt_item",
    "build": "build_structure",
    # Farming
    "plant": "farm_plant",
    "harvest": "farm_harvest",
    "farm": "farm_plant",
    # Item operations
    "equip": "equip_item",
    "use": "use_item",
    "drop": "drop_item",
    "eat": "eat_food",
    # Combat
    "attack": "attack_entity",
    "defend": "defend_self",
    "flee": "flee",
    # Social
    "follow": "follow_agent",
    "unfollow": "unfollow_agent",
    "trade": "trade_items",
    "gift": "gift_item",
    "vote": "vote",
    "send_message": "send_message",
    "request_help": "request_help",
    "form_group": "form_group",
    "leave_group": "leave_group",
    # Chest operations
    "deposit": "deposit_items",
    "withdraw": "withdraw_items",
    # Aliases
    "gather": "mine_block",
    "dig": "mine_block",
    # No-op actions
    "idle": None,
    "wait": None,
    "think": None,
    "observe": None,
}


def map_action(cc_action: str) -> str | None:
    """Map a CC action name to a skill registry name.

    Args:
        cc_action: Action name from CCDecision.action

    Returns:
        Skill name for SkillRegistry lookup, or None if action should be skipped.
        Returns None for unknown actions (logged as warning).
    """
    if cc_action in ACTION_TO_SKILL:
        return ACTION_TO_SKILL[cc_action]
    logger.warning("unmapped_cc_action", action=cc_action)
    return None


def create_full_registry() -> SkillRegistry:
    """Create a SkillRegistry with all basic + social + advanced skills registered.

    Returns:
        A SkillRegistry containing basic skills (move_to, mine_block, etc.),
        social skills (trade_items, gift_item, follow_agent, etc.),
        and advanced skills (attack_entity, build_structure, flee, etc.).
    """
    from piano.skills.advanced import register_advanced_skills

    registry = create_default_registry()
    register_social_skills(registry)
    register_advanced_skills(registry)
    return registry
