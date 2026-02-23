"""Tests for the GoalGenerationModule."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from piano.core.types import GoalData, LLMResponse, ModuleTier
from piano.goals.generator import (
    GoalCategory,
    GoalGenerationModule,
    GoalGraph,
    GoalNode,
    GoalStatus,
)
from tests.helpers import InMemorySAS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Mock LLM provider that returns valid JSON goal responses."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps({
                "goals": [
                    {
                        "description": "Gather wood from nearby trees",
                        "category": "survival",
                        "priority": 0.8,
                        "estimated_difficulty": 0.3,
                    },
                    {
                        "description": "Talk to nearby players",
                        "category": "social",
                        "priority": 0.5,
                        "estimated_difficulty": 0.2,
                    },
                ]
            }),
            model="gpt-4o-mini",
        )
    )
    return llm


@pytest.fixture
def module(mock_llm: AsyncMock) -> GoalGenerationModule:
    """Goal generation module with mock LLM."""
    return GoalGenerationModule(llm_provider=mock_llm)


@pytest.fixture
def sas() -> InMemorySAS:
    """In-memory shared agent state."""
    return InMemorySAS()


# ---------------------------------------------------------------------------
# GoalNode Tests
# ---------------------------------------------------------------------------


class TestGoalNode:
    """Tests for GoalNode data model."""

    def test_default_values(self) -> None:
        """GoalNode has sensible defaults."""
        node = GoalNode(description="test goal")
        assert node.description == "test goal"
        assert node.status == GoalStatus.PENDING
        assert node.priority == 0.5
        assert node.category == GoalCategory.SURVIVAL
        assert node.prerequisites == []
        assert node.parent_goal_id is None
        assert 0.0 <= node.estimated_difficulty <= 1.0

    def test_custom_values(self) -> None:
        """GoalNode accepts custom values."""
        node = GoalNode(
            description="craft pickaxe",
            status=GoalStatus.ACTIVE,
            priority=0.9,
            category=GoalCategory.CRAFTING,
            prerequisites=["goal-1", "goal-2"],
            estimated_difficulty=0.7,
        )
        assert node.description == "craft pickaxe"
        assert node.status == GoalStatus.ACTIVE
        assert node.priority == 0.9
        assert node.category == GoalCategory.CRAFTING
        assert node.prerequisites == ["goal-1", "goal-2"]
        assert node.estimated_difficulty == 0.7

    def test_unique_id_generated(self) -> None:
        """Each GoalNode gets a unique ID."""
        node1 = GoalNode(description="goal 1")
        node2 = GoalNode(description="goal 2")
        assert node1.id != node2.id


# ---------------------------------------------------------------------------
# GoalGraph Tests
# ---------------------------------------------------------------------------


class TestGoalGraph:
    """Tests for GoalGraph operations."""

    @pytest.fixture
    def graph(self) -> GoalGraph:
        """Empty goal graph."""
        return GoalGraph()

    def test_add_and_get_goal(self, graph: GoalGraph) -> None:
        """Can add and retrieve goals."""
        goal = GoalNode(description="test goal")
        graph.add_goal(goal)
        retrieved = graph.get_goal(goal.id)
        assert retrieved is not None
        assert retrieved.description == "test goal"

    def test_remove_goal(self, graph: GoalGraph) -> None:
        """Can remove goals."""
        goal = GoalNode(description="test goal")
        graph.add_goal(goal)
        graph.remove_goal(goal.id)
        assert graph.get_goal(goal.id) is None

    def test_get_all_goals(self, graph: GoalGraph) -> None:
        """Can retrieve all goals."""
        goal1 = GoalNode(description="goal 1")
        goal2 = GoalNode(description="goal 2")
        graph.add_goal(goal1)
        graph.add_goal(goal2)
        all_goals = graph.get_all_goals()
        assert len(all_goals) == 2
        assert goal1 in all_goals
        assert goal2 in all_goals

    def test_get_executable_goals_no_prerequisites(self, graph: GoalGraph) -> None:
        """Goals with no prerequisites are executable if PENDING or ACTIVE."""
        goal1 = GoalNode(description="goal 1", status=GoalStatus.PENDING)
        goal2 = GoalNode(description="goal 2", status=GoalStatus.ACTIVE)
        goal3 = GoalNode(description="goal 3", status=GoalStatus.COMPLETED)
        graph.add_goal(goal1)
        graph.add_goal(goal2)
        graph.add_goal(goal3)

        executable = graph.get_executable_goals()
        assert len(executable) == 2
        assert goal1 in executable
        assert goal2 in executable
        assert goal3 not in executable

    def test_get_executable_goals_with_prerequisites(self, graph: GoalGraph) -> None:
        """Goals are executable only if prerequisites are completed."""
        goal1 = GoalNode(id="g1", description="goal 1", status=GoalStatus.COMPLETED)
        goal2 = GoalNode(
            id="g2", description="goal 2", status=GoalStatus.PENDING, prerequisites=["g1"]
        )
        goal3 = GoalNode(
            id="g3",
            description="goal 3",
            status=GoalStatus.PENDING,
            prerequisites=["g1", "g2"],
        )
        graph.add_goal(goal1)
        graph.add_goal(goal2)
        graph.add_goal(goal3)

        executable = graph.get_executable_goals()
        # goal2 is executable (g1 is completed)
        # goal3 is NOT executable (g2 is not completed)
        assert len(executable) == 1
        assert goal2 in executable
        assert goal3 not in executable

    def test_get_active_goals(self, graph: GoalGraph) -> None:
        """Returns only active goals."""
        goal1 = GoalNode(description="goal 1", status=GoalStatus.ACTIVE)
        goal2 = GoalNode(description="goal 2", status=GoalStatus.PENDING)
        goal3 = GoalNode(description="goal 3", status=GoalStatus.ACTIVE)
        graph.add_goal(goal1)
        graph.add_goal(goal2)
        graph.add_goal(goal3)

        active = graph.get_active_goals()
        assert len(active) == 2
        assert goal1 in active
        assert goal3 in active

    def test_recalculate_priorities(self, graph: GoalGraph) -> None:
        """Recalculates priorities based on prerequisites and difficulty."""
        # Goal with no prerequisites and low difficulty -> high priority
        goal1 = GoalNode(
            description="easy goal",
            prerequisites=[],
            estimated_difficulty=0.1,
        )
        # Goal with many prerequisites and high difficulty -> low priority
        goal2 = GoalNode(
            description="hard goal",
            prerequisites=["a", "b", "c"],
            estimated_difficulty=0.9,
        )
        graph.add_goal(goal1)
        graph.add_goal(goal2)

        graph.recalculate_priorities()

        # goal1 should have higher priority
        assert goal1.priority > goal2.priority
        # Priorities should be clamped to [0, 1]
        assert 0.0 <= goal1.priority <= 1.0
        assert 0.0 <= goal2.priority <= 1.0

    def test_clear(self, graph: GoalGraph) -> None:
        """Clear removes all goals."""
        goal1 = GoalNode(description="goal 1")
        goal2 = GoalNode(description="goal 2")
        graph.add_goal(goal1)
        graph.add_goal(goal2)
        graph.clear()
        assert len(graph.get_all_goals()) == 0


# ---------------------------------------------------------------------------
# Module Tests
# ---------------------------------------------------------------------------


class TestGoalGenerationModule:
    """Tests for GoalGenerationModule."""

    def test_name(self, module: GoalGenerationModule) -> None:
        """Module has correct name."""
        assert module.name == "goal_generation"

    def test_tier(self, module: GoalGenerationModule) -> None:
        """Module is SLOW tier."""
        assert module.tier == ModuleTier.SLOW

    def test_repr(self, module: GoalGenerationModule) -> None:
        """Module has a string representation."""
        assert "GoalGenerationModule" in repr(module)
        assert "goal_generation" in repr(module)

    async def test_tick_generates_goals(
        self, module: GoalGenerationModule, sas: InMemorySAS
    ) -> None:
        """Tick generates goals and writes them to SAS."""
        result = await module.tick(sas)

        assert result.success
        assert result.module_name == "goal_generation"
        assert result.tier == ModuleTier.SLOW
        assert result.data["goals_generated"] == 2
        assert result.data["total_goals"] >= 2

        # Check that goals were written to SAS
        goals = await sas.get_goals()
        assert len(goals.goal_stack) > 0

    async def test_tick_updates_sas_goals(
        self, module: GoalGenerationModule, sas: InMemorySAS
    ) -> None:
        """Tick updates the SAS goal data."""
        await module.tick(sas)

        goals = await sas.get_goals()
        # Should have at least one goal in the stack
        assert len(goals.goal_stack) > 0

    async def test_tick_calls_llm(
        self, module: GoalGenerationModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Tick calls the LLM with proper request."""
        await module.tick(sas)

        mock_llm.complete.assert_called_once()
        request = mock_llm.complete.call_args[0][0]
        assert request.tier == ModuleTier.SLOW
        assert request.json_mode is True
        assert "Minecraft agent" in request.system_prompt

    async def test_tick_with_empty_state(
        self, module: GoalGenerationModule, sas: InMemorySAS
    ) -> None:
        """Tick works with empty initial state."""
        result = await module.tick(sas)

        assert result.success
        assert result.data["goals_generated"] >= 0

    async def test_tick_with_existing_goals(
        self, module: GoalGenerationModule, sas: InMemorySAS
    ) -> None:
        """Tick works when there are existing goals in SAS."""
        await sas.update_goals(
            GoalData(
                current_goal="mine iron ore",
                goal_stack=["mine iron ore", "craft pickaxe"],
                completed_goals=["gather wood"],
            )
        )

        result = await module.tick(sas)
        assert result.success

    async def test_llm_failure_returns_error(
        self, module: GoalGenerationModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """If LLM fails, module returns error result but doesn't crash."""
        mock_llm.complete.side_effect = Exception("LLM timeout")

        result = await module.tick(sas)

        assert not result.success
        assert result.error is not None
        assert "Goal generation failed" in result.error
        assert result.data["goals_generated"] == 0

    async def test_invalid_json_response(
        self, module: GoalGenerationModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Invalid JSON from LLM is handled gracefully."""
        mock_llm.complete.return_value = LLMResponse(
            content="not valid json", model="gpt-4o-mini"
        )

        result = await module.tick(sas)

        # Should succeed but generate 0 goals
        assert result.success
        assert result.data["goals_generated"] == 0

    async def test_malformed_json_response(
        self, module: GoalGenerationModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Malformed JSON structure is handled gracefully."""
        mock_llm.complete.return_value = LLMResponse(
            content=json.dumps({"goals": "not a list"}), model="gpt-4o-mini"
        )

        result = await module.tick(sas)

        # Should not crash, but may generate 0 goals
        assert result.success

    async def test_goal_prioritization(
        self, module: GoalGenerationModule, sas: InMemorySAS
    ) -> None:
        """Generated goals are prioritized correctly."""
        result = await module.tick(sas)

        assert result.success
        # Goal graph should recalculate priorities
        goals = module._goal_graph.get_all_goals()
        for goal in goals:
            assert 0.0 <= goal.priority <= 1.0

    async def test_initialize_clears_goals(
        self, module: GoalGenerationModule, sas: InMemorySAS
    ) -> None:
        """Initialize clears the goal graph."""
        # Generate some goals
        await module.tick(sas)
        assert len(module._goal_graph.get_all_goals()) > 0

        # Initialize should clear
        await module.initialize()
        assert len(module._goal_graph.get_all_goals()) == 0

    async def test_shutdown_clears_goals(
        self, module: GoalGenerationModule, sas: InMemorySAS
    ) -> None:
        """Shutdown clears the goal graph."""
        await module.tick(sas)
        assert len(module._goal_graph.get_all_goals()) > 0

        await module.shutdown()
        assert len(module._goal_graph.get_all_goals()) == 0


# ---------------------------------------------------------------------------
# Goal Generation with Context Tests
# ---------------------------------------------------------------------------


class TestGoalGenerationWithContext:
    """Tests for goal generation with different contexts."""

    async def test_generation_with_nearby_players(
        self, module: GoalGenerationModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Goal generation considers nearby players."""
        from piano.core.types import PerceptData

        await sas.update_percepts(
            PerceptData(nearby_players=["player1", "player2"])
        )

        await module.tick(sas)

        # Check that LLM was called with nearby players in context
        mock_llm.complete.assert_called_once()
        request = mock_llm.complete.call_args[0][0]
        assert "player1" in request.prompt or "player2" in request.prompt

    async def test_generation_with_inventory(
        self, module: GoalGenerationModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Goal generation considers inventory state."""
        from piano.core.types import PerceptData

        await sas.update_percepts(
            PerceptData(inventory={"oak_log": 5, "iron_ore": 3})
        )

        await module.tick(sas)

        request = mock_llm.complete.call_args[0][0]
        assert "oak_log" in request.prompt or "iron_ore" in request.prompt

    async def test_generation_with_action_history(
        self, module: GoalGenerationModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Goal generation considers recent action history."""
        from piano.core.types import ActionHistoryEntry

        await sas.add_action(ActionHistoryEntry(action="mine", success=True))
        await sas.add_action(ActionHistoryEntry(action="craft", success=False))

        await module.tick(sas)

        request = mock_llm.complete.call_args[0][0]
        assert "mine" in request.prompt.lower()

    async def test_generation_with_low_health(
        self, module: GoalGenerationModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Goal generation considers health and hunger."""
        from piano.core.types import PerceptData

        await sas.update_percepts(PerceptData(health=5.0, hunger=3.0))

        await module.tick(sas)

        request = mock_llm.complete.call_args[0][0]
        assert "5" in request.prompt  # Health value should appear


# ---------------------------------------------------------------------------
# Social Goal Generation Tests
# ---------------------------------------------------------------------------


class TestSocialGoalGeneration:
    """Tests for social goal generation."""

    async def test_social_goals_when_nearby_players(
        self, module: GoalGenerationModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Social goals are generated when players are nearby."""
        from piano.core.types import PerceptData

        # Mock LLM to return a social goal
        mock_llm.complete.return_value = LLMResponse(
            content=json.dumps({
                "goals": [
                    {
                        "description": "Greet nearby players",
                        "category": "social",
                        "priority": 0.7,
                        "estimated_difficulty": 0.2,
                    }
                ]
            }),
            model="gpt-4o-mini",
        )

        await sas.update_percepts(PerceptData(nearby_players=["alice", "bob"]))

        result = await module.tick(sas)

        assert result.success
        goals = module._goal_graph.get_all_goals()
        assert any(g.category == GoalCategory.SOCIAL for g in goals)


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in goal generation."""

    async def test_keeps_previous_goals_on_llm_failure(
        self, module: GoalGenerationModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Previous goals are preserved when LLM fails."""
        # First successful generation
        await module.tick(sas)
        initial_goals = await sas.get_goals()

        # Second tick with LLM failure
        mock_llm.complete.side_effect = Exception("LLM failed")
        await module.tick(sas)

        # Goals should still be in SAS (unchanged)
        current_goals = await sas.get_goals()
        assert current_goals.current_goal == initial_goals.current_goal

    async def test_empty_goals_list_handled(
        self, module: GoalGenerationModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Empty goals list from LLM is handled gracefully."""
        mock_llm.complete.return_value = LLMResponse(
            content=json.dumps({"goals": []}), model="gpt-4o-mini"
        )

        result = await module.tick(sas)

        assert result.success
        assert result.data["goals_generated"] == 0
