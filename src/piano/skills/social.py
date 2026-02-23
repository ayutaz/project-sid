"""Social interaction skill functions.

Skills for inter-agent communication and cooperation, including trading,
gifting, voting, following, requesting help, and group management.

Reference: docs/implementation/05-minecraft-platform.md Section 3.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from piano.core.types import BridgeCommand

if TYPE_CHECKING:
    from collections.abc import Callable

    from piano.skills.basic import BridgeClient
    from piano.skills.registry import SkillRegistry

__all__ = [
    "SOCIAL_SKILLS",
    "follow_agent",
    "form_group",
    "gift_item",
    "leave_group",
    "register_social_skills",
    "request_help",
    "send_message",
    "trade_items",
    "unfollow_agent",
    "vote",
]


# --- Social Skill Functions ---


async def trade_items(
    bridge: BridgeClient,
    target_agent: str,
    offer_items: dict[str, int],
    request_items: dict[str, int],
) -> dict[str, Any]:
    """Initiate a trade with another agent.

    Args:
        bridge: Bridge client for Mineflayer communication.
        target_agent: Name or ID of the target agent.
        offer_items: Items to offer (item_name -> count).
        request_items: Items to request (item_name -> count).

    Returns:
        Result dict with success status and trade data.
    """
    cmd = BridgeCommand(
        action="trade",
        params={
            "target_agent": target_agent,
            "offer_items": offer_items,
            "request_items": request_items,
        },
    )
    return await bridge.send_command(cmd)


async def gift_item(
    bridge: BridgeClient, target_agent: str, item: str, count: int = 1
) -> dict[str, Any]:
    """Gift an item to another agent without expecting anything in return.

    Args:
        bridge: Bridge client for Mineflayer communication.
        target_agent: Name or ID of the target agent.
        item: Item name to gift.
        count: Number of items to gift (must be > 0).

    Returns:
        Result dict with success status and gift data.

    Raises:
        ValueError: If count is not positive.
    """
    if count <= 0:
        raise ValueError(f"Gift count must be positive, got {count}")
    cmd = BridgeCommand(
        action="gift",
        params={"target_agent": target_agent, "item": item, "count": count},
    )
    return await bridge.send_command(cmd)


async def vote(bridge: BridgeClient, proposal_id: str, choice: str) -> dict[str, Any]:
    """Vote on a proposal.

    Args:
        bridge: Bridge client for Mineflayer communication.
        proposal_id: Unique identifier for the proposal.
        choice: Vote choice (e.g., "yes", "no", "abstain").

    Returns:
        Result dict with success status and vote data.
    """
    cmd = BridgeCommand(
        action="vote",
        params={"proposal_id": proposal_id, "choice": choice},
    )
    return await bridge.send_command(cmd)


async def follow_agent(bridge: BridgeClient, target_agent: str) -> dict[str, Any]:
    """Start following another agent.

    Args:
        bridge: Bridge client for Mineflayer communication.
        target_agent: Name or ID of the agent to follow.

    Returns:
        Result dict with success status and follow data.
    """
    cmd = BridgeCommand(action="follow", params={"target_agent": target_agent})
    return await bridge.send_command(cmd)


async def unfollow_agent(bridge: BridgeClient) -> dict[str, Any]:
    """Stop following the current agent.

    Args:
        bridge: Bridge client for Mineflayer communication.

    Returns:
        Result dict with success status.
    """
    cmd = BridgeCommand(action="unfollow", params={})
    return await bridge.send_command(cmd)


async def request_help(bridge: BridgeClient, message: str, radius: float = 16.0) -> dict[str, Any]:
    """Request help from nearby agents.

    Args:
        bridge: Bridge client for Mineflayer communication.
        message: Help request message.
        radius: Search radius for nearby agents (default: 16.0 blocks).

    Returns:
        Result dict with success status and response data.
    """
    cmd = BridgeCommand(
        action="request_help",
        params={"message": message, "radius": radius},
    )
    return await bridge.send_command(cmd)


async def form_group(
    bridge: BridgeClient, group_name: str, member_ids: list[str]
) -> dict[str, Any]:
    """Form a new group with specified members.

    Args:
        bridge: Bridge client for Mineflayer communication.
        group_name: Name of the group to form.
        member_ids: List of agent IDs to include in the group.

    Returns:
        Result dict with success status and group data.
    """
    cmd = BridgeCommand(
        action="form_group",
        params={"group_name": group_name, "member_ids": member_ids},
    )
    return await bridge.send_command(cmd)


async def leave_group(bridge: BridgeClient, group_name: str) -> dict[str, Any]:
    """Leave a group.

    Args:
        bridge: Bridge client for Mineflayer communication.
        group_name: Name of the group to leave.

    Returns:
        Result dict with success status.
    """
    cmd = BridgeCommand(action="leave_group", params={"group_name": group_name})
    return await bridge.send_command(cmd)


async def send_message(bridge: BridgeClient, target_agent: str, message: str) -> dict[str, Any]:
    """Send a direct message to another agent.

    Args:
        bridge: Bridge client for Mineflayer communication.
        target_agent: Name or ID of the target agent.
        message: Message content.

    Returns:
        Result dict with success status.
    """
    cmd = BridgeCommand(
        action="send_message",
        params={"target_agent": target_agent, "message": message},
    )
    return await bridge.send_command(cmd)


# --- Social Skills Registry ---

SOCIAL_SKILLS: dict[str, Callable] = {
    "trade_items": trade_items,
    "gift_item": gift_item,
    "vote": vote,
    "follow_agent": follow_agent,
    "unfollow_agent": unfollow_agent,
    "request_help": request_help,
    "form_group": form_group,
    "leave_group": leave_group,
    "send_message": send_message,
}


def register_social_skills(registry: SkillRegistry) -> None:
    """Register all social skills to the provided registry.

    Args:
        registry: SkillRegistry to register skills to.
    """
    registry.register(
        "trade_items",
        trade_items,
        params_schema={
            "target_agent": "str",
            "offer_items": "dict[str, int]",
            "request_items": "dict[str, int]",
        },
        description="Initiate a trade with another agent",
    )

    registry.register(
        "gift_item",
        gift_item,
        params_schema={
            "target_agent": "str",
            "item": "str",
            "count": "int",
        },
        description="Gift an item to another agent",
    )

    registry.register(
        "vote",
        vote,
        params_schema={"proposal_id": "str", "choice": "str"},
        description="Vote on a proposal",
    )

    registry.register(
        "follow_agent",
        follow_agent,
        params_schema={"target_agent": "str"},
        description="Start following another agent",
    )

    registry.register(
        "unfollow_agent",
        unfollow_agent,
        params_schema={},
        description="Stop following the current agent",
    )

    registry.register(
        "request_help",
        request_help,
        params_schema={"message": "str", "radius": "float"},
        description="Request help from nearby agents",
    )

    registry.register(
        "form_group",
        form_group,
        params_schema={"group_name": "str", "member_ids": "list[str]"},
        description="Form a new group with specified members",
    )

    registry.register(
        "leave_group",
        leave_group,
        params_schema={"group_name": "str"},
        description="Leave a group",
    )

    registry.register(
        "send_message",
        send_message,
        params_schema={"target_agent": "str", "message": "str"},
        description="Send a direct message to another agent",
    )
