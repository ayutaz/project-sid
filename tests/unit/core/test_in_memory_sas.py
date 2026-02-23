"""Unit tests for InMemorySAS test helper."""

from __future__ import annotations

import pytest

from piano.core.types import (
    ActionHistoryEntry,
    GoalData,
    MemoryEntry,
    PerceptData,
    SocialData,
)

from ...helpers import ACTION_HISTORY_CAPACITY, STM_CAPACITY, InMemorySAS


class TestInMemorySAS:
    """Test the in-memory SAS implementation."""

    async def test_agent_id(self) -> None:
        """SAS should return the configured agent_id."""
        sas = InMemorySAS(agent_id="sas-test-001")
        assert sas.agent_id == "sas-test-001"

    async def test_percepts_roundtrip(self, sample_percepts: PerceptData) -> None:
        """Percepts should survive a write/read roundtrip."""
        sas = InMemorySAS()
        await sas.update_percepts(sample_percepts)
        result = await sas.get_percepts()
        assert result.health == 18.0
        assert result.nearby_players == ["Alice", "Bob"]

    async def test_goals_roundtrip(self, sample_goals: GoalData) -> None:
        """Goals should survive a write/read roundtrip."""
        sas = InMemorySAS()
        await sas.update_goals(sample_goals)
        result = await sas.get_goals()
        assert result.current_goal == "Build a shelter"

    async def test_social_roundtrip(self, sample_social: SocialData) -> None:
        """Social data should survive a write/read roundtrip."""
        sas = InMemorySAS()
        await sas.update_social(sample_social)
        result = await sas.get_social()
        assert result.relationships["Alice"] == 0.8

    async def test_action_history_capacity(self) -> None:
        """Action history should be trimmed at ACTION_HISTORY_CAPACITY."""
        sas = InMemorySAS()
        for i in range(ACTION_HISTORY_CAPACITY + 20):
            await sas.add_action(ActionHistoryEntry(action=f"action-{i}"))

        history = await sas.get_action_history()
        assert len(history) == ACTION_HISTORY_CAPACITY

    async def test_stm_capacity(self) -> None:
        """STM should be trimmed at STM_CAPACITY."""
        sas = InMemorySAS()
        for i in range(STM_CAPACITY + 20):
            await sas.add_stm(MemoryEntry(content=f"memory-{i}"))

        stm = await sas.get_stm()
        assert len(stm) == STM_CAPACITY

    async def test_cc_decision_roundtrip(self) -> None:
        """CC decision should survive a write/read roundtrip."""
        sas = InMemorySAS()
        decision = {"action": "mine", "target": "stone"}
        await sas.set_cc_decision(decision)
        result = await sas.get_last_cc_decision()
        assert result == {"action": "mine", "target": "stone"}

    async def test_snapshot_has_all_sections(self) -> None:
        """snapshot() should include all SAS sections."""
        sas = InMemorySAS()
        snap = await sas.snapshot()
        expected_keys = {
            "percepts",
            "goals",
            "social",
            "plans",
            "action_history",
            "working_memory",
            "stm",
            "self_reflection",
            "cc_decision",
        }
        assert expected_keys == set(snap.keys())

    async def test_clear_resets_all(self) -> None:
        """clear() should reset all SAS data to defaults."""
        sas = InMemorySAS()
        await sas.update_goals(GoalData(current_goal="test"))
        await sas.add_stm(MemoryEntry(content="remember"))
        await sas.clear()

        goals = await sas.get_goals()
        assert goals.current_goal == ""
        stm = await sas.get_stm()
        assert len(stm) == 0

    async def test_assert_has_action(self) -> None:
        """assert_has_action should find the matching action."""
        sas = InMemorySAS()
        await sas.add_action(ActionHistoryEntry(action="mine"))
        entry = sas.assert_has_action("mine")
        assert entry.action == "mine"

    async def test_assert_has_action_missing(self) -> None:
        """assert_has_action should raise AssertionError if action is missing."""
        sas = InMemorySAS()
        with pytest.raises(AssertionError, match="No action 'missing'"):
            sas.assert_has_action("missing")

    async def test_generic_section_roundtrip(self) -> None:
        """Generic section access should store and retrieve data."""
        sas = InMemorySAS()
        await sas.update_section("custom", {"key": "value"})
        result = await sas.get_section("custom")
        assert result == {"key": "value"}
