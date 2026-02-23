"""Tests for the self-reflection module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from piano.core.types import (
    ActionHistoryEntry,
    LLMResponse,
    ModuleTier,
    SelfReflectionData,
)
from piano.reflection.module import (
    SelfReflectionModule,
)
from tests.helpers import InMemorySAS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_provider() -> AsyncMock:
    """Create a mock LLM provider."""
    provider = AsyncMock()
    provider.complete = AsyncMock()
    return provider


@pytest.fixture
def module(mock_llm_provider: AsyncMock) -> SelfReflectionModule:
    """Create a self-reflection module with mocked LLM."""
    return SelfReflectionModule(
        llm_provider=mock_llm_provider,
        reflection_interval_ticks=10,
        min_actions_for_reflection=3,
    )


@pytest.fixture
def sas() -> InMemorySAS:
    """Create an in-memory SAS for testing."""
    return InMemorySAS()


# ---------------------------------------------------------------------------
# Module metadata tests
# ---------------------------------------------------------------------------


class TestModuleMetadata:
    """Tests for module properties and metadata."""

    def test_name(self, module: SelfReflectionModule) -> None:
        """Module has correct name."""
        assert module.name == "self_reflection"

    def test_tier(self, module: SelfReflectionModule) -> None:
        """Module is SLOW tier."""
        assert module.tier == ModuleTier.SLOW

    def test_repr(self, module: SelfReflectionModule) -> None:
        """Module has meaningful repr."""
        assert "SelfReflectionModule" in repr(module)


# ---------------------------------------------------------------------------
# Reflection triggering tests
# ---------------------------------------------------------------------------


class TestReflectionTriggers:
    """Tests for when reflection should and should not occur."""

    async def test_no_reflection_when_interval_not_reached(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
    ) -> None:
        """No reflection when tick interval is not reached."""
        # Add sufficient actions
        for _i in range(5):
            await sas.add_action(
                ActionHistoryEntry(
                    action="move",
                    expected_result="{}",
                    actual_result="{}",
                    success=True,
                )
            )

        # Run only 5 ticks (interval is 10)
        for _ in range(5):
            result = await module.tick(sas)

        assert result.data["reflected"] is False
        assert result.data["ticks_since_reflection"] == 5

    async def test_no_reflection_when_insufficient_actions(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
    ) -> None:
        """No reflection when not enough actions have occurred."""
        # Add only 2 actions (min is 3)
        for _i in range(2):
            await sas.add_action(
                ActionHistoryEntry(
                    action="move",
                    expected_result="{}",
                    actual_result="{}",
                    success=True,
                )
            )

        # Run 10 ticks (interval reached)
        for _ in range(10):
            result = await module.tick(sas)

        assert result.data["reflected"] is False
        assert result.data["actions_since_last"] == 2

    async def test_reflection_triggers_after_interval_and_actions(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Reflection triggers when both interval and action thresholds are met."""
        # Mock LLM response
        mock_llm_provider.complete.return_value = LLMResponse(
            content='["Insight 1", "Insight 2"]',
            model="gpt-4o",
        )

        # Add sufficient actions (5 > min 3)
        for _i in range(5):
            await sas.add_action(
                ActionHistoryEntry(
                    action="move",
                    expected_result="{}",
                    actual_result="{}",
                    success=True,
                )
            )

        # Run 10 ticks (reaches interval)
        for _ in range(10):
            result = await module.tick(sas)

        # Last result should have reflection
        assert result.data["reflected"] is True
        assert result.success is True


# ---------------------------------------------------------------------------
# Three-stage process tests
# ---------------------------------------------------------------------------


class TestThreeStageProcess:
    """Tests for the 3-stage reflection process."""

    async def test_evaluate_stage_summarizes_actions(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Stage 1: Evaluation summarizes action success/failure rates."""
        mock_llm_provider.complete.return_value = LLMResponse(
            content='["Test insight"]',
            model="gpt-4o",
        )

        # Add mixed success/failure actions
        for i in range(10):
            await sas.add_action(
                ActionHistoryEntry(
                    action="move",
                    expected_result="{}",
                    actual_result="{}",
                    success=(i % 2 == 0),  # 50% success rate
                )
            )

        # Trigger reflection (interval is 10 ticks)
        for _ in range(9):
            result = await module.tick(sas)
            assert result.data["reflected"] is False

        # 10th tick should trigger reflection
        result = await module.tick(sas)

        assert result.data["reflected"] is True
        evaluation = result.data["evaluation"]
        assert "50.0%" in evaluation  # 50% success rate
        assert "10 recent actions" in evaluation

    async def test_generate_insights_calls_llm(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Stage 2: Insights are generated via LLM call."""
        expected_insights = ["Movement is reliable", "Need to improve mining"]
        mock_llm_provider.complete.return_value = LLMResponse(
            content=json.dumps(expected_insights),
            model="gpt-4o",
        )

        # Add actions
        for _i in range(5):
            await sas.add_action(
                ActionHistoryEntry(
                    action="move",
                    expected_result="{}",
                    actual_result="{}",
                    success=True,
                )
            )

        # Trigger reflection (10 ticks to reach interval)
        for _ in range(9):
            await module.tick(sas)
        result = await module.tick(sas)

        assert result.data["reflected"] is True
        assert result.data["insights"] == expected_insights
        # Verify LLM was called
        mock_llm_provider.complete.assert_called_once()

    async def test_update_behavior_merges_insights(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Stage 3: New insights are merged with existing reflection state."""
        # Set up existing reflection with old insights
        existing_reflection = SelfReflectionData(
            last_reflection="Previous reflection",
            insights=["Old insight 1", "Old insight 2"],
            personality_traits={},
        )
        await sas.update_self_reflection(existing_reflection)

        # Mock new insights
        new_insights = ["New insight 1", "New insight 2"]
        mock_llm_provider.complete.return_value = LLMResponse(
            content=json.dumps(new_insights),
            model="gpt-4o",
        )

        # Add actions and trigger reflection
        for _i in range(5):
            await sas.add_action(
                ActionHistoryEntry(
                    action="move",
                    expected_result="{}",
                    actual_result="{}",
                    success=True,
                )
            )

        for _ in range(9):
            await module.tick(sas)
        await module.tick(sas)

        # Check that reflection was updated in SAS
        updated_reflection = await sas.get_self_reflection()
        assert len(updated_reflection.insights) == 4  # 2 old + 2 new
        assert "Old insight 1" in updated_reflection.insights
        assert "New insight 1" in updated_reflection.insights


# ---------------------------------------------------------------------------
# Insight accumulation tests
# ---------------------------------------------------------------------------


class TestInsightAccumulation:
    """Tests for how insights accumulate across multiple reflections."""

    async def test_insights_accumulate_across_reflections(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Insights from multiple reflections are accumulated."""
        # First reflection
        mock_llm_provider.complete.return_value = LLMResponse(
            content='["First insight"]',
            model="gpt-4o",
        )
        for _i in range(5):
            await sas.add_action(
                ActionHistoryEntry(action="move", expected_result="{}", success=True)
            )
        for _ in range(10):
            await module.tick(sas)

        # Second reflection
        mock_llm_provider.complete.return_value = LLMResponse(
            content='["Second insight"]',
            model="gpt-4o",
        )
        for _i in range(5):
            await sas.add_action(
                ActionHistoryEntry(action="mine", expected_result="{}", success=True)
            )
        for _ in range(10):
            await module.tick(sas)

        # Check accumulated insights
        reflection = await sas.get_self_reflection()
        assert len(reflection.insights) == 2
        assert "First insight" in reflection.insights
        assert "Second insight" in reflection.insights

    async def test_insight_limit_prevents_unbounded_growth(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Insights are limited to prevent unbounded growth."""
        # Prepopulate with 20 insights (at limit)
        existing_insights = [f"Old insight {i}" for i in range(20)]
        await sas.update_self_reflection(
            SelfReflectionData(
                last_reflection="Old",
                insights=existing_insights,
            )
        )

        # Add new insights
        mock_llm_provider.complete.return_value = LLMResponse(
            content='["New insight"]',
            model="gpt-4o",
        )

        for _i in range(5):
            await sas.add_action(
                ActionHistoryEntry(action="move", expected_result="{}", success=True)
            )
        for _ in range(10):
            await module.tick(sas)

        # Check that old insights were pruned
        reflection = await sas.get_self_reflection()
        assert len(reflection.insights) == 20  # Still at limit
        assert "New insight" in reflection.insights
        assert "Old insight 0" not in reflection.insights  # Oldest pruned


# ---------------------------------------------------------------------------
# Goal revision tests
# ---------------------------------------------------------------------------


class TestGoalRevision:
    """Tests for goal revision flag based on reflection insights."""

    async def test_goal_revision_on_critical_failure(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Goal revision flag is set when critical failures are detected."""
        # All actions fail
        for _i in range(10):
            await sas.add_action(
                ActionHistoryEntry(
                    action="mine",
                    expected_result="{}",
                    actual_result="{}",
                    success=False,
                )
            )

        # Mock LLM to return critical insight
        mock_llm_provider.complete.return_value = LLMResponse(
            content='["Critical failure in mining actions"]',
            model="gpt-4o",
        )

        for _ in range(9):
            await module.tick(sas)
        result = await module.tick(sas)

        assert result.data["reflected"] is True
        assert result.data["should_revise_goals"] is True

    async def test_no_goal_revision_on_success(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Goal revision flag is not set when actions are successful."""
        # All actions succeed
        for _i in range(10):
            await sas.add_action(
                ActionHistoryEntry(
                    action="move",
                    expected_result="{}",
                    actual_result="{}",
                    success=True,
                )
            )

        mock_llm_provider.complete.return_value = LLMResponse(
            content='["Movement actions are reliable"]',
            model="gpt-4o",
        )

        for _ in range(9):
            await module.tick(sas)
        result = await module.tick(sas)

        assert result.data["reflected"] is True
        assert result.data["should_revise_goals"] is False


# ---------------------------------------------------------------------------
# Reflection interval tests
# ---------------------------------------------------------------------------


class TestReflectionInterval:
    """Tests for reflection interval timing."""

    async def test_reflection_interval_respected(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Reflection does not occur before interval is reached."""
        mock_llm_provider.complete.return_value = LLMResponse(
            content='["Insight"]',
            model="gpt-4o",
        )

        # Add sufficient actions
        for _i in range(10):
            await sas.add_action(
                ActionHistoryEntry(action="move", expected_result="{}", success=True)
            )

        # Run 9 ticks (just before interval of 10)
        for _ in range(9):
            result = await module.tick(sas)
            assert result.data["reflected"] is False

        # 10th tick should trigger reflection
        result = await module.tick(sas)
        assert result.data["reflected"] is True

    async def test_counters_reset_after_reflection(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Tick and action counters reset after reflection."""
        mock_llm_provider.complete.return_value = LLMResponse(
            content='["Insight"]',
            model="gpt-4o",
        )

        # First reflection cycle
        for _i in range(5):
            await sas.add_action(
                ActionHistoryEntry(action="move", expected_result="{}", success=True)
            )
        for _ in range(9):
            await module.tick(sas)
        result = await module.tick(sas)  # 10th tick triggers reflection
        assert result.data["reflected"] is True

        # Immediately after, counters should be reset
        result = await module.tick(sas)
        assert result.data["ticks_since_reflection"] == 1
        assert result.data["actions_since_last"] == 0


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling during reflection."""

    async def test_llm_failure_preserves_previous_reflection(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """LLM failure keeps previous reflection state unchanged."""
        # Set up existing reflection
        existing_reflection = SelfReflectionData(
            last_reflection="Previous reflection",
            insights=["Old insight"],
        )
        await sas.update_self_reflection(existing_reflection)

        # Mock LLM to raise error
        mock_llm_provider.complete.side_effect = Exception("LLM API error")

        # Add actions and trigger reflection
        for _i in range(5):
            await sas.add_action(
                ActionHistoryEntry(action="move", expected_result="{}", success=True)
            )
        for _ in range(9):
            await module.tick(sas)
        result = await module.tick(sas)

        # Reflection should fail gracefully
        assert result.data["reflected"] is False
        assert "error" in result.data
        assert result.error is not None

        # Previous reflection should be preserved
        reflection = await sas.get_self_reflection()
        assert reflection.last_reflection == "Previous reflection"
        assert reflection.insights == ["Old insight"]

    async def test_invalid_json_response_fallback(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Invalid JSON response uses evaluation as fallback insight."""
        # Mock LLM to return invalid JSON
        mock_llm_provider.complete.return_value = LLMResponse(
            content="This is not JSON",
            model="gpt-4o",
        )

        for _i in range(5):
            await sas.add_action(
                ActionHistoryEntry(action="move", expected_result="{}", success=True)
            )
        for _ in range(9):
            await module.tick(sas)
        result = await module.tick(sas)

        # Should still reflect, using fallback
        assert result.data["reflected"] is True
        assert len(result.data["insights"]) > 0


# ---------------------------------------------------------------------------
# Action type breakdown tests
# ---------------------------------------------------------------------------


class TestActionTypeBreakdown:
    """Tests for per-action-type evaluation."""

    async def test_evaluation_breaks_down_by_action_type(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Evaluation includes breakdown by action type."""
        mock_llm_provider.complete.return_value = LLMResponse(
            content='["Insight"]',
            model="gpt-4o",
        )

        # Add different action types with different success rates
        # Move: 100% success
        for _i in range(5):
            await sas.add_action(
                ActionHistoryEntry(action="move", expected_result="{}", success=True)
            )
        # Mine: 0% success
        for _i in range(5):
            await sas.add_action(
                ActionHistoryEntry(action="mine", expected_result="{}", success=False)
            )

        for _ in range(9):
            await module.tick(sas)
        result = await module.tick(sas)

        evaluation = result.data["evaluation"]
        assert "move: 100%" in evaluation
        assert "mine: 0%" in evaluation


# ---------------------------------------------------------------------------
# Empty action history tests
# ---------------------------------------------------------------------------


class TestEmptyActionHistory:
    """Tests for handling empty action history."""

    async def test_no_reflection_with_empty_history(
        self,
        module: SelfReflectionModule,
        sas: InMemorySAS,
    ) -> None:
        """No reflection occurs when action history is empty."""
        # Run ticks without any actions
        for _ in range(10):
            result = await module.tick(sas)

        assert result.data["reflected"] is False
        assert result.data["actions_since_last"] == 0
