"""Tests for the Redis-backed SharedAgentState implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

import fakeredis.aioredis
import pytest

from piano.core.sas_redis import _ACTION_HISTORY_LIMIT, _STM_LIMIT, RedisSAS
from piano.core.types import (
    ActionHistoryEntry,
    GoalData,
    MemoryEntry,
    PerceptData,
    PlanData,
    SelfReflectionData,
    SocialData,
)


@pytest.fixture
async def redis() -> fakeredis.aioredis.FakeRedis:
    """Create a fresh fakeredis async connection."""
    r = fakeredis.aioredis.FakeRedis()
    yield r  # type: ignore[misc]
    await r.flushall()
    await r.aclose()


@pytest.fixture
async def sas(redis: fakeredis.aioredis.FakeRedis, agent_id: str) -> RedisSAS:
    """Create an initialized RedisSAS instance."""
    instance = RedisSAS(redis=redis, agent_id=agent_id)
    await instance.initialize()
    return instance


# ---------------------------------------------------------------------------
# Agent ID
# ---------------------------------------------------------------------------


class TestAgentId:
    async def test_agent_id_property(self, sas: RedisSAS, agent_id: str) -> None:
        assert sas.agent_id == agent_id


# ---------------------------------------------------------------------------
# Percepts
# ---------------------------------------------------------------------------


class TestPercepts:
    async def test_get_default_percepts(self, sas: RedisSAS) -> None:
        percepts = await sas.get_percepts()
        assert isinstance(percepts, PerceptData)
        assert percepts.health == 20.0

    async def test_update_and_get_percepts(self, sas: RedisSAS) -> None:
        data = PerceptData(
            nearby_players=["alice", "bob"],
            health=15.0,
            position={"x": 10.0, "y": 64.0, "z": -5.0},
        )
        await sas.update_percepts(data)
        result = await sas.get_percepts()
        assert result.nearby_players == ["alice", "bob"]
        assert result.health == 15.0
        assert result.position["x"] == 10.0

    async def test_percepts_overwrite(self, sas: RedisSAS) -> None:
        await sas.update_percepts(PerceptData(health=10.0))
        await sas.update_percepts(PerceptData(health=5.0))
        result = await sas.get_percepts()
        assert result.health == 5.0


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


class TestGoals:
    async def test_get_default_goals(self, sas: RedisSAS) -> None:
        goals = await sas.get_goals()
        assert isinstance(goals, GoalData)
        assert goals.current_goal == ""

    async def test_update_and_get_goals(self, sas: RedisSAS) -> None:
        data = GoalData(
            current_goal="mine diamonds",
            goal_stack=["mine diamonds", "craft pickaxe"],
        )
        await sas.update_goals(data)
        result = await sas.get_goals()
        assert result.current_goal == "mine diamonds"
        assert len(result.goal_stack) == 2


# ---------------------------------------------------------------------------
# Social
# ---------------------------------------------------------------------------


class TestSocial:
    async def test_get_default_social(self, sas: RedisSAS) -> None:
        social = await sas.get_social()
        assert isinstance(social, SocialData)
        assert social.relationships == {}

    async def test_update_and_get_social(self, sas: RedisSAS) -> None:
        data = SocialData(
            relationships={"agent-002": 0.8},
            emotions={"happy": 7.5},
        )
        await sas.update_social(data)
        result = await sas.get_social()
        assert result.relationships["agent-002"] == 0.8
        assert result.emotions["happy"] == 7.5


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


class TestPlans:
    async def test_get_default_plans(self, sas: RedisSAS) -> None:
        plans = await sas.get_plans()
        assert isinstance(plans, PlanData)
        assert plans.plan_status == "idle"

    async def test_update_and_get_plans(self, sas: RedisSAS) -> None:
        data = PlanData(
            current_plan=["gather wood", "craft table", "craft pickaxe"],
            plan_status="executing",
            current_step=1,
        )
        await sas.update_plans(data)
        result = await sas.get_plans()
        assert result.plan_status == "executing"
        assert result.current_step == 1
        assert len(result.current_plan) == 3


# ---------------------------------------------------------------------------
# Action History
# ---------------------------------------------------------------------------


class TestActionHistory:
    async def test_empty_action_history(self, sas: RedisSAS) -> None:
        history = await sas.get_action_history()
        assert history == []

    async def test_add_and_get_action(self, sas: RedisSAS) -> None:
        entry = ActionHistoryEntry(action="mine", success=True)
        await sas.add_action(entry)
        history = await sas.get_action_history()
        assert len(history) == 1
        assert history[0].action == "mine"

    async def test_action_history_ordering(self, sas: RedisSAS) -> None:
        """Newest action should be first."""
        for i in range(5):
            await sas.add_action(ActionHistoryEntry(action=f"action-{i}"))
        history = await sas.get_action_history()
        assert history[0].action == "action-4"
        assert history[-1].action == "action-0"

    async def test_action_history_capacity_limit(self, sas: RedisSAS) -> None:
        """History should auto-trim to 50 entries."""
        for i in range(_ACTION_HISTORY_LIMIT + 20):
            await sas.add_action(ActionHistoryEntry(action=f"a-{i}"))
        history = await sas.get_action_history()
        assert len(history) == _ACTION_HISTORY_LIMIT
        # Newest entry should be the last one added
        assert history[0].action == f"a-{_ACTION_HISTORY_LIMIT + 19}"

    async def test_action_history_limit_parameter(self, sas: RedisSAS) -> None:
        """Requesting fewer entries than stored should work."""
        for i in range(10):
            await sas.add_action(ActionHistoryEntry(action=f"a-{i}"))
        history = await sas.get_action_history(limit=3)
        assert len(history) == 3


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------


class TestWorkingMemory:
    async def test_empty_working_memory(self, sas: RedisSAS) -> None:
        wm = await sas.get_working_memory()
        assert wm == []

    async def test_set_and_get_working_memory(self, sas: RedisSAS) -> None:
        entries = [
            MemoryEntry(content="I see a tree", category="perception"),
            MemoryEntry(content="I am hungry", category="reflection"),
        ]
        await sas.set_working_memory(entries)
        result = await sas.get_working_memory()
        assert len(result) == 2
        assert result[0].content == "I see a tree"
        assert result[1].content == "I am hungry"

    async def test_set_working_memory_replaces(self, sas: RedisSAS) -> None:
        """set_working_memory should fully replace previous contents."""
        await sas.set_working_memory([MemoryEntry(content="old")])
        await sas.set_working_memory([MemoryEntry(content="new")])
        result = await sas.get_working_memory()
        assert len(result) == 1
        assert result[0].content == "new"

    async def test_set_empty_working_memory(self, sas: RedisSAS) -> None:
        """Setting empty list should clear working memory."""
        await sas.set_working_memory([MemoryEntry(content="x")])
        await sas.set_working_memory([])
        result = await sas.get_working_memory()
        assert result == []


# ---------------------------------------------------------------------------
# Short-Term Memory (STM)
# ---------------------------------------------------------------------------


class TestSTM:
    async def test_empty_stm(self, sas: RedisSAS) -> None:
        stm = await sas.get_stm()
        assert stm == []

    async def test_add_and_get_stm(self, sas: RedisSAS) -> None:
        entry = MemoryEntry(content="saw a creeper", category="perception")
        await sas.add_stm(entry)
        stm = await sas.get_stm()
        assert len(stm) == 1
        assert stm[0].content == "saw a creeper"

    async def test_stm_ordering(self, sas: RedisSAS) -> None:
        """Newest entry should be first."""
        for i in range(5):
            await sas.add_stm(MemoryEntry(content=f"memory-{i}"))
        stm = await sas.get_stm()
        assert stm[0].content == "memory-4"
        assert stm[-1].content == "memory-0"

    async def test_stm_capacity_limit(self, sas: RedisSAS) -> None:
        """STM should auto-trim to 100 entries."""
        for i in range(_STM_LIMIT + 30):
            await sas.add_stm(MemoryEntry(content=f"m-{i}"))
        stm = await sas.get_stm()
        assert len(stm) == _STM_LIMIT
        assert stm[0].content == f"m-{_STM_LIMIT + 29}"

    async def test_stm_limit_parameter(self, sas: RedisSAS) -> None:
        """Requesting fewer entries should work."""
        for i in range(20):
            await sas.add_stm(MemoryEntry(content=f"m-{i}"))
        stm = await sas.get_stm(limit=5)
        assert len(stm) == 5


# ---------------------------------------------------------------------------
# Self Reflection
# ---------------------------------------------------------------------------


class TestSelfReflection:
    async def test_get_default_self_reflection(self, sas: RedisSAS) -> None:
        reflection = await sas.get_self_reflection()
        assert isinstance(reflection, SelfReflectionData)
        assert reflection.last_reflection == ""

    async def test_update_and_get_self_reflection(self, sas: RedisSAS) -> None:
        data = SelfReflectionData(
            last_reflection="I should be more careful",
            insights=["avoid creepers"],
            personality_traits={"openness": 0.8},
        )
        await sas.update_self_reflection(data)
        result = await sas.get_self_reflection()
        assert result.last_reflection == "I should be more careful"
        assert "avoid creepers" in result.insights


# ---------------------------------------------------------------------------
# CC Decision
# ---------------------------------------------------------------------------


class TestCCDecision:
    async def test_get_no_cc_decision(self, sas: RedisSAS) -> None:
        """After initialize, cc_decision key is 'null' -> returns None."""
        result = await sas.get_last_cc_decision()
        assert result is None

    async def test_set_and_get_cc_decision(self, sas: RedisSAS) -> None:
        decision: dict[str, Any] = {
            "action": "mine",
            "summary": "Mining diamonds",
            "reasoning": "Need resources",
        }
        await sas.set_cc_decision(decision)
        result = await sas.get_last_cc_decision()
        assert result is not None
        assert result["action"] == "mine"


# ---------------------------------------------------------------------------
# Generic Section Access
# ---------------------------------------------------------------------------


class TestGenericSection:
    async def test_get_missing_section(self, sas: RedisSAS) -> None:
        result = await sas.get_section("nonexistent")
        assert result == {}

    async def test_update_and_get_section(self, sas: RedisSAS) -> None:
        await sas.update_section("custom", {"foo": "bar", "count": 42})
        result = await sas.get_section("custom")
        assert result["foo"] == "bar"
        assert result["count"] == 42


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    async def test_snapshot_contains_all_sections(self, sas: RedisSAS) -> None:
        snap = await sas.snapshot()
        expected_keys = {
            "agent_id",
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
        assert set(snap.keys()) == expected_keys

    async def test_snapshot_reflects_current_state(
        self, sas: RedisSAS, agent_id: str
    ) -> None:
        await sas.update_goals(GoalData(current_goal="build house"))
        await sas.add_action(ActionHistoryEntry(action="collect wood"))
        snap = await sas.snapshot()
        assert snap["agent_id"] == agent_id
        assert snap["goals"]["current_goal"] == "build house"
        assert len(snap["action_history"]) == 1


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_initialize_sets_defaults(
        self, redis: fakeredis.aioredis.FakeRedis, agent_id: str
    ) -> None:
        sas = RedisSAS(redis=redis, agent_id=agent_id)
        await sas.initialize()
        percepts = await sas.get_percepts()
        assert percepts.health == 20.0

    async def test_initialize_preserves_existing(self, sas: RedisSAS) -> None:
        """Re-initializing should not overwrite existing data (nx=True)."""
        await sas.update_percepts(PerceptData(health=5.0))
        await sas.initialize()
        percepts = await sas.get_percepts()
        assert percepts.health == 5.0

    async def test_clear_removes_all_data(
        self, sas: RedisSAS, redis: fakeredis.aioredis.FakeRedis, agent_id: str
    ) -> None:
        await sas.update_goals(GoalData(current_goal="test"))
        await sas.add_action(ActionHistoryEntry(action="test"))
        await sas.clear()
        # After clear, all keys are gone - getting should return defaults
        goals = await sas.get_goals()
        assert goals.current_goal == ""
        history = await sas.get_action_history()
        assert history == []


# ---------------------------------------------------------------------------
# Key Partitioning
# ---------------------------------------------------------------------------


class TestKeyPartitioning:
    async def test_different_agents_isolated(
        self, redis: fakeredis.aioredis.FakeRedis
    ) -> None:
        """Two agents should not see each other's data."""
        sas_a = RedisSAS(redis=redis, agent_id="agent-a")
        sas_b = RedisSAS(redis=redis, agent_id="agent-b")
        await sas_a.initialize()
        await sas_b.initialize()

        await sas_a.update_goals(GoalData(current_goal="goal-a"))
        await sas_b.update_goals(GoalData(current_goal="goal-b"))

        assert (await sas_a.get_goals()).current_goal == "goal-a"
        assert (await sas_b.get_goals()).current_goal == "goal-b"

    async def test_clear_only_affects_own_agent(
        self, redis: fakeredis.aioredis.FakeRedis
    ) -> None:
        sas_a = RedisSAS(redis=redis, agent_id="agent-a")
        sas_b = RedisSAS(redis=redis, agent_id="agent-b")
        await sas_a.initialize()
        await sas_b.initialize()

        await sas_a.update_goals(GoalData(current_goal="goal-a"))
        await sas_b.update_goals(GoalData(current_goal="goal-b"))

        await sas_a.clear()

        # agent-a data is gone
        goals_a = await sas_a.get_goals()
        assert goals_a.current_goal == ""
        # agent-b data is untouched
        goals_b = await sas_b.get_goals()
        assert goals_b.current_goal == "goal-b"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_uninitialised_sas_returns_defaults(
        self, redis: fakeredis.aioredis.FakeRedis
    ) -> None:
        """Reading from a SAS that was never initialized should return defaults."""
        sas = RedisSAS(redis=redis, agent_id="brand-new")
        percepts = await sas.get_percepts()
        assert percepts == PerceptData()

    async def test_memory_entry_roundtrip_preserves_fields(self, sas: RedisSAS) -> None:
        """All fields of MemoryEntry should survive serialization round-trip."""
        entry = MemoryEntry(
            id=uuid4(),
            timestamp=datetime(2024, 6, 15, 12, 0, 0),
            content="test content",
            category="perception",
            importance=0.9,
            source_module="action_awareness",
            metadata={"key": "value"},
        )
        await sas.add_stm(entry)
        result = (await sas.get_stm())[0]
        assert result.content == entry.content
        assert result.importance == entry.importance
        assert result.source_module == entry.source_module
        assert result.metadata == entry.metadata

    async def test_action_entry_roundtrip_preserves_fields(self, sas: RedisSAS) -> None:
        """All fields of ActionHistoryEntry should survive serialization."""
        entry = ActionHistoryEntry(
            timestamp=datetime(2024, 6, 15, 12, 0, 0),
            action="craft",
            expected_result="wooden_pickaxe",
            actual_result="wooden_pickaxe",
            success=True,
        )
        await sas.add_action(entry)
        result = (await sas.get_action_history())[0]
        assert result.action == entry.action
        assert result.expected_result == entry.expected_result
        assert result.actual_result == entry.actual_result
        assert result.success is True
