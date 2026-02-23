"""Tests for the PlanningModule."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from piano.core.types import (
    ActionHistoryEntry,
    CCDecision,
    GoalData,
    LLMResponse,
    PerceptData,
    PlanData,
)
from piano.planning.planner import (
    Plan,
    PlanningModule,
    PlanStatus,
    PlanStep,
    StepStatus,
)
from tests.helpers import InMemorySAS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Mock LLM provider that returns valid JSON plans."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(
                {
                    "steps": [
                        {
                            "action": "mine",
                            "parameters": {"block": "oak_log"},
                            "preconditions": ["have_axe"],
                            "expected_outcome": "oak_log in inventory",
                        },
                        {
                            "action": "craft",
                            "parameters": {"item": "planks"},
                            "preconditions": ["have_oak_log"],
                            "expected_outcome": "planks in inventory",
                        },
                    ]
                }
            ),
            model="gpt-4o",
        )
    )
    return llm


@pytest.fixture
def module(mock_llm: AsyncMock) -> PlanningModule:
    """Planning module with mock LLM."""
    return PlanningModule(llm=mock_llm)


@pytest.fixture
def sas() -> InMemorySAS:
    """In-memory SAS for testing."""
    return InMemorySAS()


# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------


class TestModuleMetadata:
    """Tests for module properties."""

    def test_name(self, module: PlanningModule) -> None:
        assert module.name == "planning"

    def test_tier(self, module: PlanningModule) -> None:
        assert module.tier == "slow"

    def test_repr(self, module: PlanningModule) -> None:
        assert "PlanningModule" in repr(module)


# ---------------------------------------------------------------------------
# Plan generation
# ---------------------------------------------------------------------------


class TestPlanGeneration:
    """Tests for generating plans from goals."""

    async def test_generate_plan_from_goal(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """When a goal exists and no plan, generate a new plan."""
        await sas.update_goals(GoalData(current_goal="collect wood"))

        result = await module.tick(sas)

        assert result.success
        assert result.data["event"] == "plan_generated"
        assert result.data["step_count"] == 2
        assert result.data["current_goal"] == "collect wood"

        # Verify LLM was called
        mock_llm.complete.assert_called_once()
        request = mock_llm.complete.call_args[0][0]
        assert "collect wood" in request.prompt
        assert request.json_mode is True

        # Verify plan written to SAS
        plan_data = await sas.get_plans()
        assert plan_data.plan_status == "executing"
        assert len(plan_data.current_plan) == 2
        assert plan_data.current_plan[0] == "mine"

    async def test_no_plan_generation_without_goal(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """When no goal exists, remain idle."""
        result = await module.tick(sas)

        assert result.success
        assert result.data["event"] == "idle"
        assert result.data["plan_exists"] is False

        # LLM should not be called
        mock_llm.complete.assert_not_called()

    async def test_plan_generation_with_context(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Plan prompt includes percepts and action history."""
        await sas.update_goals(GoalData(current_goal="build shelter"))
        await sas.update_percepts(
            PerceptData(
                position={"x": 10, "y": 64, "z": 20},
                inventory={"stone": 5},
                health=15.0,
            )
        )
        await sas.add_action(ActionHistoryEntry(action="mine", success=True))

        await module.tick(sas)

        request = mock_llm.complete.call_args[0][0]
        assert "10" in request.prompt  # position
        assert "stone" in request.prompt  # inventory
        assert "15.0" in request.prompt  # health
        assert "mine" in request.prompt  # recent action

    async def test_plan_generation_failure_handling(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """LLM failure is caught and reported."""
        mock_llm.complete.side_effect = Exception("LLM API error")
        await sas.update_goals(GoalData(current_goal="test goal"))

        result = await module.tick(sas)

        assert not result.success
        assert result.data["event"] == "plan_generation_failed"
        assert "LLM API error" in result.error


# ---------------------------------------------------------------------------
# Plan execution tracking
# ---------------------------------------------------------------------------


class TestPlanExecution:
    """Tests for tracking plan progress."""

    async def test_plan_executing_status(
        self, module: PlanningModule, sas: InMemorySAS
    ) -> None:
        """When plan is executing, report current step."""
        # Manually set an executing plan in SAS
        await sas.update_plans(
            PlanData(
                current_plan=["mine", "craft", "build"],
                plan_status="executing",
                current_step=1,
            )
        )
        await sas.update_goals(GoalData(current_goal="build shelter"))

        result = await module.tick(sas)

        assert result.success
        assert result.data["event"] == "plan_executing"
        assert result.data["current_step"] == "craft"
        assert result.data["step_index"] == 1
        assert result.data["total_steps"] == 3

    async def test_plan_completion_detection(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """When all steps complete, mark plan as completed."""
        # Generate a plan first
        await sas.update_goals(GoalData(current_goal="test"))
        await module.tick(sas)

        # Simulate step completion by advancing to end
        plan_data = await sas.get_plans()
        await sas.update_plans(
            PlanData(
                current_plan=plan_data.current_plan,
                plan_status="executing",
                current_step=len(plan_data.current_plan),  # all steps done
            )
        )

        result = await module.tick(sas)

        assert result.success
        assert result.data["event"] == "plan_completed"

        # Plan should be cleared from SAS
        plan_data = await sas.get_plans()
        assert plan_data.plan_status == "idle"
        assert plan_data.current_plan == []


# ---------------------------------------------------------------------------
# Replanning
# ---------------------------------------------------------------------------


class TestReplanning:
    """Tests for replanning after failures."""

    async def test_replan_on_failure(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """When plan status is failed, trigger replanning."""
        # Set up a failed plan
        await sas.update_plans(
            PlanData(
                current_plan=["mine", "craft"],
                plan_status="failed",
                current_step=1,
            )
        )
        await sas.update_goals(GoalData(current_goal="original goal"))

        # Mock LLM to return different plan for replanning
        mock_llm.complete.return_value = LLMResponse(
            content=json.dumps(
                {
                    "steps": [
                        {
                            "action": "retry_mine",
                            "parameters": {},
                            "preconditions": [],
                            "expected_outcome": "success",
                        }
                    ]
                }
            ),
            model="gpt-4o",
        )

        result = await module.tick(sas)

        assert result.success
        assert result.data["event"] == "replanned"
        assert result.data["step_count"] == 1

        # Verify new plan written to SAS
        plan_data = await sas.get_plans()
        assert plan_data.plan_status == "executing"
        assert plan_data.current_plan[0] == "retry_mine"

    async def test_replan_prompt_includes_failures(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Replanning prompt includes failure context."""
        # Set up a failed plan in SAS (matching internal state)
        await sas.update_plans(
            PlanData(
                current_plan=["mine", "craft"],
                plan_status="failed",
                current_step=1,
            )
        )
        await sas.update_goals(GoalData(current_goal="test goal"))

        # Also set internal plan with detailed failure info
        failed_plan = Plan(
            goal_description="test goal",
            steps=[
                PlanStep(step_id=0, action="mine", status=StepStatus.FAILED),
                PlanStep(step_id=1, action="craft", status=StepStatus.PENDING),
            ],
            status=PlanStatus.FAILED,
        )
        module._current_plan = failed_plan

        await module.tick(sas)

        request = mock_llm.complete.call_args[0][0]
        assert "Previous Failures" in request.prompt
        assert "mine" in request.prompt
        assert "failed" in request.prompt

    async def test_replan_failure_handling(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """If replanning fails, clear plan and report error."""
        await sas.update_plans(
            PlanData(current_plan=["mine"], plan_status="failed", current_step=0)
        )
        await sas.update_goals(GoalData(current_goal="test"))

        mock_llm.complete.side_effect = Exception("Replan error")

        result = await module.tick(sas)

        assert not result.success
        assert result.data["event"] == "replan_failed"
        assert "Replan error" in result.error

        # Plan should be cleared
        plan_data = await sas.get_plans()
        assert plan_data.plan_status == "idle"


# ---------------------------------------------------------------------------
# Plan interruption via CC broadcast
# ---------------------------------------------------------------------------


class TestPlanInterruption:
    """Tests for plan interruption via CC decisions."""

    async def test_plan_interrupted_by_broadcast(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """CC broadcast with different action interrupts the plan."""
        # Generate a plan
        await sas.update_goals(GoalData(current_goal="test"))
        await module.tick(sas)

        # Create a CC decision that contradicts current plan step
        decision = CCDecision(action="flee", action_params={})
        await module.on_broadcast(decision)

        # Next tick should clear the plan
        result = await module.tick(sas)

        assert result.success
        assert result.data["event"] == "plan_interrupted"

        # Plan should be cleared
        plan_data = await sas.get_plans()
        assert plan_data.plan_status == "idle"

    async def test_no_interruption_when_actions_match(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """No interruption if CC action matches current step."""
        await sas.update_goals(GoalData(current_goal="test"))
        await module.tick(sas)

        # CC decision matches the current plan step (mine)
        decision = CCDecision(action="mine", action_params={})
        await module.on_broadcast(decision)

        result = await module.tick(sas)

        # Plan should continue executing
        assert result.data["event"] == "plan_executing"

    async def test_no_interruption_when_no_plan(
        self, module: PlanningModule, sas: InMemorySAS
    ) -> None:
        """Broadcast is ignored when no plan is active."""
        decision = CCDecision(action="test", action_params={})
        await module.on_broadcast(decision)

        result = await module.tick(sas)

        # Should remain idle
        assert result.data["event"] == "idle"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    async def test_empty_goal_state(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """No plan generated when goal is empty string."""
        await sas.update_goals(GoalData(current_goal=""))

        result = await module.tick(sas)

        assert result.success
        assert result.data["event"] == "idle"
        mock_llm.complete.assert_not_called()

    async def test_plan_with_zero_steps(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """LLM returns empty steps list - plan generates and completes."""
        mock_llm.complete.return_value = LLMResponse(
            content=json.dumps({"steps": []}), model="gpt-4o"
        )
        await sas.update_goals(GoalData(current_goal="impossible task"))

        result = await module.tick(sas)

        assert result.success
        assert result.data["step_count"] == 0
        # Zero-step plan is written to SAS as executing with empty steps

        plan_data = await sas.get_plans()
        assert len(plan_data.current_plan) == 0
        assert plan_data.plan_status == "executing"

        # On next tick, empty plan should be detected as complete
        result = await module.tick(sas)
        assert result.data["event"] == "plan_completed"

    async def test_malformed_llm_response(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """LLM returns invalid JSON."""
        mock_llm.complete.return_value = LLMResponse(
            content="not valid json", model="gpt-4o"
        )
        await sas.update_goals(GoalData(current_goal="test"))

        result = await module.tick(sas)

        assert not result.success
        assert result.data["event"] == "plan_generation_failed"
        # JSON decode error should be captured
        assert "expecting value" in result.error.lower() or "json" in result.error.lower()

    async def test_plan_with_multiple_steps(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Plan with many steps is handled correctly."""
        mock_llm.complete.return_value = LLMResponse(
            content=json.dumps(
                {
                    "steps": [
                        {
                            "action": f"step_{i}",
                            "parameters": {},
                            "preconditions": [],
                            "expected_outcome": "",
                        }
                        for i in range(10)
                    ]
                }
            ),
            model="gpt-4o",
        )
        await sas.update_goals(GoalData(current_goal="complex task"))

        result = await module.tick(sas)

        assert result.success
        assert result.data["step_count"] == 10

        plan_data = await sas.get_plans()
        assert len(plan_data.current_plan) == 10

    async def test_step_precondition_parsing(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Step preconditions and expected outcomes are parsed."""
        mock_llm.complete.return_value = LLMResponse(
            content=json.dumps(
                {
                    "steps": [
                        {
                            "action": "craft",
                            "parameters": {"item": "pickaxe"},
                            "preconditions": ["have_wood", "have_cobblestone"],
                            "expected_outcome": "pickaxe in inventory",
                        }
                    ]
                }
            ),
            model="gpt-4o",
        )
        await sas.update_goals(GoalData(current_goal="make pickaxe"))

        await module.tick(sas)

        # Verify internal plan structure
        assert module._current_plan is not None
        step = module._current_plan.steps[0]
        assert step.action == "craft"
        assert step.params == {"item": "pickaxe"}
        assert step.preconditions == ["have_wood", "have_cobblestone"]
        assert step.expected_outcome == "pickaxe in inventory"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestPlanLifecycle:
    """End-to-end tests for full plan lifecycle."""

    async def test_full_lifecycle_generate_execute_complete(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Complete lifecycle: generate -> execute -> complete."""
        # Step 1: Generate plan
        await sas.update_goals(GoalData(current_goal="build house"))
        result1 = await module.tick(sas)
        assert result1.data["event"] == "plan_generated"

        # Step 2: Execute (simulate step progress)
        plan_data = await sas.get_plans()
        await sas.update_plans(
            PlanData(
                current_plan=plan_data.current_plan,
                plan_status="executing",
                current_step=0,
            )
        )
        result2 = await module.tick(sas)
        assert result2.data["event"] == "plan_executing"
        assert result2.data["step_index"] == 0

        # Step 3: Advance to completion
        await sas.update_plans(
            PlanData(
                current_plan=plan_data.current_plan,
                plan_status="executing",
                current_step=len(plan_data.current_plan),
            )
        )
        result3 = await module.tick(sas)
        assert result3.data["event"] == "plan_completed"

    async def test_lifecycle_generate_fail_replan(
        self, module: PlanningModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Lifecycle with failure: generate -> fail -> replan."""
        # Generate initial plan
        await sas.update_goals(GoalData(current_goal="test"))
        await module.tick(sas)

        # Mark plan as failed
        plan_data = await sas.get_plans()
        await sas.update_plans(
            PlanData(
                current_plan=plan_data.current_plan,
                plan_status="failed",
                current_step=1,
            )
        )

        # Trigger replanning
        result = await module.tick(sas)
        assert result.data["event"] == "replanned"

        # New plan should be executing
        plan_data = await sas.get_plans()
        assert plan_data.plan_status == "executing"
