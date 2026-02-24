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
    "move": "move_to",
    "mine": "mine_block",
    "craft": "craft_item",
    "chat": "chat",
    "look": "look_at",
    "get_position": "get_position",
    "get_inventory": "get_inventory",
    "explore": "move_to",  # explore -> move_to with computed coords
    "attack": "attack_entity",
    "follow": "follow_agent",
    "trade": "trade_items",
    "gift": "gift_item",
    "gather": "mine_block",
    "dig": "mine_block",
    "place": "place_block",
    "equip": "equip_item",
    "use": "use_item",
    "drop": "drop_item",
    "flee": "flee",
    "build": "build_structure",
    "eat": "eat_food",
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
