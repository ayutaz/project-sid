"""Tests for the rule-based ActionAwareness module."""

from __future__ import annotations

import json
from typing import Any

import pytest

from piano.awareness.action import ActionAwareness
from piano.core.sas import SharedAgentState
from piano.core.types import (
    ActionHistoryEntry,
    AgentId,
    GoalData,
    MemoryEntry,
    PerceptData,
    PlanData,
    SelfReflectionData,
    SocialData,
)

# ---------------------------------------------------------------------------
# InMemorySAS -- minimal test double
# ---------------------------------------------------------------------------


class InMemorySAS(SharedAgentState):
    """In-memory SAS implementation for unit tests."""

    def __init__(self, agent_id: str = "test-agent") -> None:
        self._agent_id = agent_id
        self._percepts = PerceptData()
        self._goals = GoalData()
        self._social = SocialData()
        self._plans = PlanData()
        self._action_history: list[ActionHistoryEntry] = []
        self._working_memory: list[MemoryEntry] = []
        self._stm: list[MemoryEntry] = []
        self._self_reflection = SelfReflectionData()
        self._cc_decision: dict[str, Any] | None = None
        self._sections: dict[str, dict[str, Any]] = {}

    @property
    def agent_id(self) -> AgentId:
        return self._agent_id

    async def get_percepts(self) -> PerceptData:
        return self._percepts

    async def update_percepts(self, percepts: PerceptData) -> None:
        self._percepts = percepts

    async def get_goals(self) -> GoalData:
        return self._goals

    async def update_goals(self, goals: GoalData) -> None:
        self._goals = goals

    async def get_social(self) -> SocialData:
        return self._social

    async def update_social(self, social: SocialData) -> None:
        self._social = social

    async def get_plans(self) -> PlanData:
        return self._plans

    async def update_plans(self, plans: PlanData) -> None:
        self._plans = plans

    async def get_action_history(self, limit: int = 50) -> list[ActionHistoryEntry]:
        return self._action_history[:limit]

    async def add_action(self, entry: ActionHistoryEntry) -> None:
        self._action_history.insert(0, entry)
        self._action_history = self._action_history[:50]

    async def get_working_memory(self) -> list[MemoryEntry]:
        return self._working_memory

    async def set_working_memory(self, entries: list[MemoryEntry]) -> None:
        self._working_memory = list(entries)

    async def get_stm(self, limit: int = 100) -> list[MemoryEntry]:
        return self._stm[:limit]

    async def add_stm(self, entry: MemoryEntry) -> None:
        self._stm.insert(0, entry)
        self._stm = self._stm[:100]

    async def get_self_reflection(self) -> SelfReflectionData:
        return self._self_reflection

    async def update_self_reflection(self, reflection: SelfReflectionData) -> None:
        self._self_reflection = reflection

    async def get_last_cc_decision(self) -> dict[str, Any] | None:
        return self._cc_decision

    async def set_cc_decision(self, decision: dict[str, Any]) -> None:
        self._cc_decision = decision

    async def get_section(self, section: str) -> dict[str, Any]:
        return self._sections.get(section, {})

    async def update_section(self, section: str, data: dict[str, Any]) -> None:
        self._sections[section] = data

    async def snapshot(self) -> dict[str, Any]:
        return {"agent_id": self._agent_id}

    async def initialize(self) -> None:
        pass

    async def clear(self) -> None:
        self._action_history.clear()
        self._percepts = PerceptData()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def module() -> ActionAwareness:
    return ActionAwareness()


@pytest.fixture
def sas() -> InMemorySAS:
    return InMemorySAS()


# ---------------------------------------------------------------------------
# Move tests
# ---------------------------------------------------------------------------


class TestCheckMove:
    """Tests for move action success/failure detection."""

    async def test_move_success_exact(self, module: ActionAwareness, sas: InMemorySAS) -> None:
        """Agent arrives exactly at the target position."""
        sas._action_history = [
            ActionHistoryEntry(
                action="move",
                expected_result=json.dumps({"x": 10, "y": 64, "z": 20}),
            )
        ]
        sas._percepts = PerceptData(position={"x": 10, "y": 64, "z": 20})

        result = await module.tick(sas)
        assert result.data["last_action_success"] is True
        assert result.data["discrepancies"] == []

    async def test_move_success_within_threshold(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Agent arrives within 5 blocks of the target."""
        sas._action_history = [
            ActionHistoryEntry(
                action="move",
                expected_result=json.dumps({"x": 10, "y": 64, "z": 20}),
            )
        ]
        # Distance = sqrt(3^2 + 0^2 + 4^2) = 5.0 -- exactly at threshold
        sas._percepts = PerceptData(position={"x": 13, "y": 64, "z": 24})

        result = await module.tick(sas)
        assert result.data["last_action_success"] is True

    async def test_move_failure_too_far(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Agent is more than 5 blocks from the target."""
        sas._action_history = [
            ActionHistoryEntry(
                action="move",
                expected_result=json.dumps({"x": 10, "y": 64, "z": 20}),
            )
        ]
        # Distance = sqrt(10^2 + 0^2 + 0^2) = 10.0
        sas._percepts = PerceptData(position={"x": 20, "y": 64, "z": 20})

        result = await module.tick(sas)
        assert result.data["last_action_success"] is False
        assert len(result.data["discrepancies"]) == 1
        assert result.data["discrepancies"][0]["type"] == "move_failure"


# ---------------------------------------------------------------------------
# Mine tests
# ---------------------------------------------------------------------------


class TestCheckMine:
    """Tests for mine action success/failure detection."""

    async def test_mine_success_block_removed(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Target block is no longer present at the position."""
        sas._action_history = [
            ActionHistoryEntry(
                action="mine",
                expected_result=json.dumps({"position": {"x": 5, "y": 60, "z": 5}}),
            )
        ]
        # No block at (5, 60, 5) in nearby_blocks
        sas._percepts = PerceptData(
            nearby_blocks=[
                {"type": "stone", "position": {"x": 6, "y": 60, "z": 5}},
            ]
        )

        result = await module.tick(sas)
        assert result.data["last_action_success"] is True

    async def test_mine_failure_block_still_present(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Target block is still at the position."""
        sas._action_history = [
            ActionHistoryEntry(
                action="mine",
                expected_result=json.dumps({"position": {"x": 5, "y": 60, "z": 5}}),
            )
        ]
        sas._percepts = PerceptData(
            nearby_blocks=[
                {"type": "stone", "position": {"x": 5, "y": 60, "z": 5}},
            ]
        )

        result = await module.tick(sas)
        assert result.data["last_action_success"] is False
        assert result.data["discrepancies"][0]["type"] == "mine_failure"


# ---------------------------------------------------------------------------
# Craft tests
# ---------------------------------------------------------------------------


class TestCheckCraft:
    """Tests for craft action success/failure detection."""

    async def test_craft_success_item_in_inventory(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Expected item is present in inventory with sufficient count."""
        sas._action_history = [
            ActionHistoryEntry(
                action="craft",
                expected_result=json.dumps({"item": "wooden_pickaxe", "count": 1}),
            )
        ]
        sas._percepts = PerceptData(inventory={"wooden_pickaxe": 1, "oak_log": 3})

        result = await module.tick(sas)
        assert result.data["last_action_success"] is True

    async def test_craft_failure_item_missing(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Expected item is not in inventory."""
        sas._action_history = [
            ActionHistoryEntry(
                action="craft",
                expected_result=json.dumps({"item": "iron_pickaxe", "count": 1}),
            )
        ]
        sas._percepts = PerceptData(inventory={"oak_log": 3})

        result = await module.tick(sas)
        assert result.data["last_action_success"] is False
        assert result.data["discrepancies"][0]["type"] == "craft_failure"

    async def test_craft_failure_insufficient_count(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Item exists but count is less than expected."""
        sas._action_history = [
            ActionHistoryEntry(
                action="craft",
                expected_result=json.dumps({"item": "stick", "count": 4}),
            )
        ]
        sas._percepts = PerceptData(inventory={"stick": 2})

        result = await module.tick(sas)
        assert result.data["last_action_success"] is False


# ---------------------------------------------------------------------------
# Chat tests
# ---------------------------------------------------------------------------


class TestCheckChat:
    """Tests for chat action success/failure detection."""

    async def test_chat_success_message_found(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Expected message appears in chat log."""
        sas._action_history = [
            ActionHistoryEntry(
                action="chat",
                expected_result=json.dumps({"message": "hello everyone"}),
            )
        ]
        sas._percepts = PerceptData(
            chat_messages=[
                {"sender": "test-agent", "text": "hello everyone"},
            ]
        )

        result = await module.tick(sas)
        assert result.data["last_action_success"] is True

    async def test_chat_failure_message_not_found(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Expected message is not in chat log."""
        sas._action_history = [
            ActionHistoryEntry(
                action="chat",
                expected_result=json.dumps({"message": "hello everyone"}),
            )
        ]
        sas._percepts = PerceptData(
            chat_messages=[
                {"sender": "other-agent", "text": "something else"},
            ]
        )

        result = await module.tick(sas)
        assert result.data["last_action_success"] is False
        assert result.data["discrepancies"][0]["type"] == "chat_failure"


# ---------------------------------------------------------------------------
# Consecutive failure tracking
# ---------------------------------------------------------------------------


class TestConsecutiveFailures:
    """Tests for consecutive failure counting and alert generation."""

    async def test_consecutive_failure_count_increments(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Consecutive failures increment the counter."""
        sas._percepts = PerceptData(position={"x": 100, "y": 100, "z": 100})

        for i in range(1, 3):
            sas._action_history = [
                ActionHistoryEntry(
                    action="move",
                    expected_result=json.dumps({"x": 0, "y": 0, "z": 0}),
                )
            ]
            result = await module.tick(sas)
            assert result.data["consecutive_failures"] == i
            assert "alert" not in result.data

    async def test_alert_on_third_consecutive_failure(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Alert is raised on the 3rd consecutive failure."""
        sas._percepts = PerceptData(position={"x": 100, "y": 100, "z": 100})

        for _ in range(3):
            sas._action_history = [
                ActionHistoryEntry(
                    action="move",
                    expected_result=json.dumps({"x": 0, "y": 0, "z": 0}),
                )
            ]
            result = await module.tick(sas)

        assert result.data["alert"] == "consecutive_failures"
        assert result.data["count"] == 3

    async def test_counter_resets_on_success(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Counter resets to 0 after a successful action."""
        sas._percepts = PerceptData(position={"x": 100, "y": 100, "z": 100})

        # Two failures
        for _ in range(2):
            sas._action_history = [
                ActionHistoryEntry(
                    action="move",
                    expected_result=json.dumps({"x": 0, "y": 0, "z": 0}),
                )
            ]
            await module.tick(sas)

        # Then a success
        sas._action_history = [
            ActionHistoryEntry(
                action="move",
                expected_result=json.dumps({"x": 100, "y": 100, "z": 100}),
            )
        ]
        result = await module.tick(sas)
        assert result.data["consecutive_failures"] == 0
        assert result.data["last_action_success"] is True

    async def test_alert_continues_past_threshold(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Alert persists for failures beyond the threshold."""
        sas._percepts = PerceptData(position={"x": 100, "y": 100, "z": 100})

        for _ in range(5):
            sas._action_history = [
                ActionHistoryEntry(
                    action="move",
                    expected_result=json.dumps({"x": 0, "y": 0, "z": 0}),
                )
            ]
            result = await module.tick(sas)

        assert result.data["alert"] == "consecutive_failures"
        assert result.data["count"] == 5


# ---------------------------------------------------------------------------
# Empty history
# ---------------------------------------------------------------------------


class TestEmptyHistory:
    """Tests for handling empty action history."""

    async def test_empty_action_history(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Returns success with no discrepancies when history is empty."""
        result = await module.tick(sas)
        assert result.data["last_action_success"] is True
        assert result.data["discrepancies"] == []
        assert result.data["consecutive_failures"] == 0


# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------


class TestModuleMetadata:
    """Tests for module properties and repr."""

    def test_name(self, module: ActionAwareness) -> None:
        assert module.name == "action_awareness"

    def test_tier(self, module: ActionAwareness) -> None:
        assert module.tier == "fast"

    def test_repr(self, module: ActionAwareness) -> None:
        assert "ActionAwareness" in repr(module)

    async def test_result_module_name(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """ModuleResult has correct module_name and tier."""
        result = await module.tick(sas)
        assert result.module_name == "action_awareness"
        assert result.tier == "fast"
        assert result.success is True


# ---------------------------------------------------------------------------
# Unknown action type
# ---------------------------------------------------------------------------


class TestUnknownAction:
    """Tests for unrecognized action types."""

    async def test_unknown_action_defaults_to_success(
        self, module: ActionAwareness, sas: InMemorySAS
    ) -> None:
        """Unknown action types are treated as successful."""
        sas._action_history = [
            ActionHistoryEntry(
                action="look",
                expected_result=json.dumps({"direction": "north"}),
            )
        ]

        result = await module.tick(sas)
        assert result.data["last_action_success"] is True
