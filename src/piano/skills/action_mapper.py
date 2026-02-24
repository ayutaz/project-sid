"""Action mapper - converts CC action names to skill registry names."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from piano.skills.basic import create_default_registry
from piano.skills.social import register_social_skills

if TYPE_CHECKING:
    from piano.skills.registry import SkillRegistry

logger = structlog.get_logger()

# Actions that intentionally map to None (no skill needed)
_NO_SKILL_ACTIONS: frozenset[str] = frozenset({"idle", "wait", "think", "observe"})

# CC action name -> SkillRegistry skill name
# None means the action should be ignored (no skill to execute)
ACTION_TO_SKILL: dict[str, str | None] = {
    "move": "move_to",
    "mine": "mine_block",
    "craft": "craft_item",
    "chat": "chat",
    "look": "look_at",
    "get_position": "look_at",  # reuse look_at for position queries
    "get_inventory": "look_at",  # reuse look_at for inventory queries
    "explore": "move_to",  # explore -> move_to with computed coords
    "attack": "chat",  # placeholder: announce attack in chat
    "follow": "follow_agent",
    "trade": "trade_items",
    "gift": "gift_item",
    "gather": "mine_block",
    "dig": "mine_block",
    "place": "chat",  # placeholder: no place skill yet
    "equip": "chat",  # placeholder: no equip skill yet
    "use": "chat",  # placeholder: no use skill yet
    "drop": "chat",  # placeholder: no drop skill yet
    "flee": "move_to",  # flee -> move_to away from threat
    "build": "craft_item",  # build -> closest match is craft_item
    "eat": "chat",  # placeholder
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
    """Create a SkillRegistry with all basic + social skills registered.

    Returns:
        A SkillRegistry containing basic skills (move_to, mine_block, etc.)
        and social skills (trade_items, gift_item, follow_agent, etc.).
    """
    registry = create_default_registry()
    register_social_skills(registry)
    return registry
