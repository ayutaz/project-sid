"""Tests for social skill functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from piano.skills.social import (
    SOCIAL_SKILLS,
    follow_agent,
    form_group,
    gift_item,
    leave_group,
    register_social_skills,
    request_help,
    send_message,
    trade_items,
    unfollow_agent,
    vote,
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


class TestTradeItemsSkill:
    async def test_trade_items_sends_command(self, bridge: MockBridgeClient) -> None:
        offer = {"diamond": 2, "gold_ingot": 5}
        request = {"emerald": 10}
        result = await trade_items(bridge, "agent-002", offer, request)

        assert result["success"] is True
        assert bridge.last_command is not None
        assert bridge.last_command.action == "trade"
        assert bridge.last_command.params["target_agent"] == "agent-002"
        assert bridge.last_command.params["offer_items"] == offer
        assert bridge.last_command.params["request_items"] == request

    async def test_trade_items_empty_offer(self, bridge: MockBridgeClient) -> None:
        result = await trade_items(bridge, "agent-003", {}, {"iron": 5})
        assert result["success"] is True
        assert bridge.last_command.params["offer_items"] == {}


class TestGiftItemSkill:
    async def test_gift_item_default_count(self, bridge: MockBridgeClient) -> None:
        result = await gift_item(bridge, "agent-004", "diamond")

        assert result["success"] is True
        assert bridge.last_command is not None
        assert bridge.last_command.action == "gift"
        assert bridge.last_command.params["target_agent"] == "agent-004"
        assert bridge.last_command.params["item"] == "diamond"
        assert bridge.last_command.params["count"] == 1

    async def test_gift_item_custom_count(self, bridge: MockBridgeClient) -> None:
        await gift_item(bridge, "agent-005", "bread", count=10)
        assert bridge.last_command.params["count"] == 10

    async def test_gift_item_validates_positive_count(self, bridge: MockBridgeClient) -> None:
        with pytest.raises(ValueError, match="Gift count must be positive"):
            await gift_item(bridge, "agent-006", "stone", count=0)

        with pytest.raises(ValueError, match="Gift count must be positive"):
            await gift_item(bridge, "agent-006", "stone", count=-5)


class TestVoteSkill:
    async def test_vote_sends_command(self, bridge: MockBridgeClient) -> None:
        result = await vote(bridge, "proposal-123", "yes")

        assert result["success"] is True
        assert bridge.last_command is not None
        assert bridge.last_command.action == "vote"
        assert bridge.last_command.params["proposal_id"] == "proposal-123"
        assert bridge.last_command.params["choice"] == "yes"

    async def test_vote_different_choices(self, bridge: MockBridgeClient) -> None:
        await vote(bridge, "proposal-456", "no")
        assert bridge.last_command.params["choice"] == "no"

        await vote(bridge, "proposal-789", "abstain")
        assert bridge.last_command.params["choice"] == "abstain"


class TestFollowAgentSkill:
    async def test_follow_agent_sends_command(self, bridge: MockBridgeClient) -> None:
        result = await follow_agent(bridge, "agent-leader")

        assert result["success"] is True
        assert bridge.last_command is not None
        assert bridge.last_command.action == "follow"
        assert bridge.last_command.params["target_agent"] == "agent-leader"


class TestUnfollowAgentSkill:
    async def test_unfollow_agent_sends_command(self, bridge: MockBridgeClient) -> None:
        result = await unfollow_agent(bridge)

        assert result["success"] is True
        assert bridge.last_command is not None
        assert bridge.last_command.action == "unfollow"
        assert bridge.last_command.params == {}


class TestRequestHelpSkill:
    async def test_request_help_default_radius(self, bridge: MockBridgeClient) -> None:
        result = await request_help(bridge, "Need help mining!")

        assert result["success"] is True
        assert bridge.last_command is not None
        assert bridge.last_command.action == "request_help"
        assert bridge.last_command.params["message"] == "Need help mining!"
        assert bridge.last_command.params["radius"] == 16.0

    async def test_request_help_custom_radius(self, bridge: MockBridgeClient) -> None:
        await request_help(bridge, "Emergency!", radius=50.0)
        assert bridge.last_command.params["radius"] == 50.0


class TestFormGroupSkill:
    async def test_form_group_sends_command(self, bridge: MockBridgeClient) -> None:
        members = ["agent-001", "agent-002", "agent-003"]
        result = await form_group(bridge, "mining-team", members)

        assert result["success"] is True
        assert bridge.last_command is not None
        assert bridge.last_command.action == "form_group"
        assert bridge.last_command.params["group_name"] == "mining-team"
        assert bridge.last_command.params["member_ids"] == members

    async def test_form_group_empty_members(self, bridge: MockBridgeClient) -> None:
        result = await form_group(bridge, "solo-group", [])
        assert result["success"] is True
        assert bridge.last_command.params["member_ids"] == []


class TestLeaveGroupSkill:
    async def test_leave_group_sends_command(self, bridge: MockBridgeClient) -> None:
        result = await leave_group(bridge, "old-team")

        assert result["success"] is True
        assert bridge.last_command is not None
        assert bridge.last_command.action == "leave_group"
        assert bridge.last_command.params["group_name"] == "old-team"


class TestSendMessageSkill:
    async def test_send_message_sends_command(self, bridge: MockBridgeClient) -> None:
        result = await send_message(bridge, "agent-007", "Hello there!")

        assert result["success"] is True
        assert bridge.last_command is not None
        assert bridge.last_command.action == "send_message"
        assert bridge.last_command.params["target_agent"] == "agent-007"
        assert bridge.last_command.params["message"] == "Hello there!"


class TestSocialSkillsDict:
    def test_social_skills_dict_contains_all_skills(self) -> None:
        expected_skills = [
            "follow_agent",
            "form_group",
            "gift_item",
            "leave_group",
            "request_help",
            "send_message",
            "trade_items",
            "unfollow_agent",
            "vote",
        ]
        assert sorted(SOCIAL_SKILLS.keys()) == expected_skills

    def test_social_skills_are_callable(self) -> None:
        for skill_name, skill_fn in SOCIAL_SKILLS.items():
            assert callable(skill_fn), f"{skill_name} should be callable"


class TestRegisterSocialSkills:
    def test_register_social_skills_adds_all_to_registry(self) -> None:
        from piano.skills.registry import SkillRegistry

        registry = SkillRegistry()
        register_social_skills(registry)

        expected_skills = [
            "follow_agent",
            "form_group",
            "gift_item",
            "leave_group",
            "request_help",
            "send_message",
            "trade_items",
            "unfollow_agent",
            "vote",
        ]
        assert registry.list_skills() == expected_skills

    def test_registered_skills_are_callable(self) -> None:
        from piano.skills.registry import SkillRegistry

        registry = SkillRegistry()
        register_social_skills(registry)

        for skill_name in registry.list_skills():
            skill = registry.get(skill_name)
            assert callable(skill.execute_fn)
            assert skill.description != ""
