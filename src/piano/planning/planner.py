"""Planning module for the PIANO architecture.

Converts goals into actionable step-by-step plans using HTN-style decomposition
with LLM assistance. Tracks plan execution progress, handles replanning on
failures, and supports interruption via CC broadcasts.

This is a SLOW module that performs full LLM calls to generate plans.

Reference: docs/implementation/07-goal-planning.md Section 2
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

from piano.core.module import Module
from piano.core.types import LLMRequest, ModuleResult, ModuleTier, PlanData

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState
    from piano.llm.provider import LLMProvider

logger = structlog.get_logger(__name__)

__all__ = ["Plan", "PlanStatus", "PlanStep", "PlanningModule", "StepStatus"]


# --- Internal types ---


class StepStatus(StrEnum):
    """Status of a single plan step."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanStatus(StrEnum):
    """Status of an entire plan."""

    IDLE = "idle"
    EXECUTING = "executing"
    REPLANNING = "replanning"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanStep(BaseModel):
    """A single step in a plan."""

    step_id: int = Field(ge=0)
    action: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    expected_outcome: str = ""
    status: StepStatus = StepStatus.PENDING


class Plan(BaseModel):
    """A complete plan with multiple steps."""

    goal_description: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    current_step_index: int = Field(default=0, ge=0)
    status: PlanStatus = PlanStatus.IDLE


# --- Planning module ---


class PlanningModule(Module):
    """Planning module that converts goals into executable action sequences.

    Reads goal state from SAS, uses LLM to generate step-by-step plans,
    tracks execution progress, and handles replanning on failures.
    """

    @property
    def name(self) -> str:
        """Unique module name."""
        return "planning"

    @property
    def tier(self) -> ModuleTier:
        """Execution tier -- SLOW (full LLM calls)."""
        return ModuleTier.SLOW

    def __init__(self, llm: LLMProvider) -> None:
        """Initialize the planning module.

        Args:
            llm: LLM provider for plan generation.
        """
        self._llm = llm
        self._current_plan: Plan | None = None
        self._plan_interrupted: bool = False
        self.log = logger.bind(module=self.name)

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        """Execute one planning tick.

        1. Read current plan state and goals from SAS.
        2. If no active plan and goal exists, generate a new plan.
        3. If plan is executing, check step progress.
        4. If plan failed, trigger replanning.
        5. Write updated plan state back to SAS.

        Returns:
            ModuleResult with plan status and metadata.
        """
        self.log.debug("tick.start")

        goals = await sas.get_goals()
        plan_data = await sas.get_plans()

        # Sync internal plan state from SAS, but preserve detailed step info
        # if we already have a plan with the same status
        if plan_data.plan_status != "idle":
            # If we don't have a plan, or the status changed, deserialize from SAS
            if (
                self._current_plan is None
                or self._current_plan.status.value != plan_data.plan_status
            ):
                synced_plan = self._deserialize_plan(plan_data)
                if synced_plan:
                    self._current_plan = synced_plan
            else:
                # Update current step index from SAS (external can advance it)
                self._current_plan.current_step_index = plan_data.current_step
        else:
            # plan_status is "idle" - clear internal plan
            self._current_plan = None

        data: dict[str, Any] = {
            "plan_exists": self._current_plan is not None,
            "current_goal": goals.current_goal,
        }

        # Case 1: Plan interrupted by CC broadcast
        if self._plan_interrupted:
            self.log.info("plan.interrupted")
            self._current_plan = None
            self._plan_interrupted = False
            await self._write_plan_to_sas(sas, None)
            data["event"] = "plan_interrupted"
            return ModuleResult(module_name=self.name, tier=self.tier, data=data)

        # Case 2: No active plan and no goal -> idle
        if not self._current_plan and not goals.current_goal:
            self.log.debug("idle", reason="no_plan_no_goal")
            await self._write_plan_to_sas(sas, None)
            data["event"] = "idle"
            return ModuleResult(module_name=self.name, tier=self.tier, data=data)

        # Case 3: No active plan but goal exists -> generate new plan
        if not self._current_plan and goals.current_goal:
            self.log.info("plan.generate", goal=goals.current_goal)
            try:
                self._current_plan = await self._generate_plan(goals.current_goal, sas)
                self._current_plan.status = PlanStatus.EXECUTING
                await self._write_plan_to_sas(sas, self._current_plan)
                data["event"] = "plan_generated"
                data["step_count"] = len(self._current_plan.steps)
            except Exception as e:
                self.log.error("plan.generate.failed", error=str(e))
                data["event"] = "plan_generation_failed"
                data["error"] = str(e)
                return ModuleResult(
                    module_name=self.name, tier=self.tier, data=data, error=str(e)
                )
            return ModuleResult(module_name=self.name, tier=self.tier, data=data)

        # Case 4: Plan executing -> check progress
        if self._current_plan and self._current_plan.status == PlanStatus.EXECUTING:
            plan_complete = self._current_plan.current_step_index >= len(
                self._current_plan.steps
            )
            if plan_complete:
                self.log.info("plan.completed")
                # Clear the plan from both memory and SAS
                self._current_plan = None
                await self._write_plan_to_sas(sas, None)
                data["event"] = "plan_completed"
            else:
                current_step = self._current_plan.steps[
                    self._current_plan.current_step_index
                ]
                data["event"] = "plan_executing"
                data["current_step"] = current_step.action
                data["step_index"] = self._current_plan.current_step_index
                data["total_steps"] = len(self._current_plan.steps)
                self.log.debug(
                    "plan.executing",
                    step=current_step.action,
                    index=self._current_plan.current_step_index,
                )

            return ModuleResult(module_name=self.name, tier=self.tier, data=data)

        # Case 5: Plan failed -> trigger replanning
        if self._current_plan and self._current_plan.status == PlanStatus.FAILED:
            self.log.warning("plan.failed", goal=self._current_plan.goal_description)
            try:
                self._current_plan = await self._replan(self._current_plan, sas)
                self._current_plan.status = PlanStatus.EXECUTING
                await self._write_plan_to_sas(sas, self._current_plan)
                data["event"] = "replanned"
                data["step_count"] = len(self._current_plan.steps)
            except Exception as e:
                self.log.error("replan.failed", error=str(e))
                self._current_plan = None
                await self._write_plan_to_sas(sas, None)
                data["event"] = "replan_failed"
                data["error"] = str(e)
                return ModuleResult(
                    module_name=self.name, tier=self.tier, data=data, error=str(e)
                )
            return ModuleResult(module_name=self.name, tier=self.tier, data=data)

        # Default: unknown state
        self.log.debug("plan.state.unknown")
        data["event"] = "unknown_state"
        return ModuleResult(module_name=self.name, tier=self.tier, data=data)

    async def on_broadcast(self, decision: Any) -> None:
        """Handle CC broadcast decision.

        If CC decides a different action than the current plan step,
        mark the plan as interrupted (will be cleared on next tick).

        Args:
            decision: CCDecision from the cognitive controller.
        """
        if not self._current_plan or self._current_plan.status != PlanStatus.EXECUTING:
            return

        current_step = self._current_plan.steps[self._current_plan.current_step_index]
        decision_action = getattr(decision, "action", "")

        if decision_action and decision_action != current_step.action:
            self.log.info(
                "plan.interrupt",
                plan_action=current_step.action,
                cc_action=decision_action,
            )
            self._plan_interrupted = True

    # --- Plan generation ---

    async def _generate_plan(self, goal: str, sas: SharedAgentState) -> Plan:
        """Generate a plan for the given goal using LLM.

        Args:
            goal: The goal description to plan for.
            sas: Shared agent state for context.

        Returns:
            A new Plan with steps.

        Raises:
            Exception: If LLM call fails or response parsing fails.
        """
        percepts = await sas.get_percepts()
        action_history = await sas.get_action_history(limit=5)

        # Build prompt
        prompt = self._build_plan_prompt(goal, percepts, action_history)

        # Call LLM
        request = LLMRequest(
            prompt=prompt,
            system_prompt=(
                "You are a planning assistant for a Minecraft agent. "
                "Generate step-by-step action plans in JSON format."
            ),
            tier=ModuleTier.SLOW,
            temperature=0.7,
            max_tokens=1024,
            json_mode=True,
        )

        response = await self._llm.complete(request)
        self.log.debug("llm.response", content=response.content[:100])

        # Parse response
        plan_data = json.loads(response.content)
        steps = [
            PlanStep(
                step_id=i,
                action=step.get("action", ""),
                params=step.get("parameters", {}),
                preconditions=step.get("preconditions", []),
                expected_outcome=step.get("expected_outcome", ""),
                status=StepStatus.PENDING,
            )
            for i, step in enumerate(plan_data.get("steps", []))
        ]

        return Plan(
            goal_description=goal,
            steps=steps,
            current_step_index=0,
            status=PlanStatus.IDLE,
        )

    async def _replan(self, failed_plan: Plan, sas: SharedAgentState) -> Plan:
        """Generate a new plan after a failure.

        Args:
            failed_plan: The plan that failed.
            sas: Shared agent state.

        Returns:
            A new plan attempting to achieve the same goal.

        Raises:
            Exception: If replanning fails.
        """
        self.log.info("replan.start", goal=failed_plan.goal_description)

        percepts = await sas.get_percepts()
        action_history = await sas.get_action_history(limit=5)

        # Build replanning prompt with failure context
        failed_steps = [s for s in failed_plan.steps if s.status == StepStatus.FAILED]
        failure_context = "\n".join(
            f"- Step {s.step_id}: {s.action} (failed)" for s in failed_steps
        )

        prompt = self._build_plan_prompt(
            failed_plan.goal_description,
            percepts,
            action_history,
            failure_context=failure_context,
        )

        request = LLMRequest(
            prompt=prompt,
            system_prompt=(
                "You are a planning assistant. The previous plan failed. "
                "Generate a new plan that addresses the failures."
            ),
            tier=ModuleTier.SLOW,
            temperature=0.8,
            max_tokens=1024,
            json_mode=True,
        )

        response = await self._llm.complete(request)
        plan_data = json.loads(response.content)

        steps = [
            PlanStep(
                step_id=i,
                action=step.get("action", ""),
                params=step.get("parameters", {}),
                preconditions=step.get("preconditions", []),
                expected_outcome=step.get("expected_outcome", ""),
                status=StepStatus.PENDING,
            )
            for i, step in enumerate(plan_data.get("steps", []))
        ]

        return Plan(
            goal_description=failed_plan.goal_description,
            steps=steps,
            current_step_index=0,
            status=PlanStatus.IDLE,
        )

    def _build_plan_prompt(
        self,
        goal: str,
        percepts: Any,
        action_history: list[Any],
        failure_context: str = "",
    ) -> str:
        """Build the LLM prompt for plan generation."""
        history_summary = "\n".join(
            f"- {entry.action}" for entry in action_history[:3]
        )

        failure_section = ""
        if failure_context:
            failure_section = f"\n## Previous Failures\n{failure_context}\n"

        return f"""Create a step-by-step plan to achieve the following goal.

## Goal
{goal}

## Current State
- Position: {percepts.position}
- Inventory: {percepts.inventory}
- Health: {percepts.health}
- Nearby Players: {percepts.nearby_players}

## Recent Actions
{history_summary or "None"}
{failure_section}
Generate a plan with concrete, executable steps. Each step should be a single action.

Respond in JSON format:
{{
  "steps": [
    {{
      "action": "<action_name>",
      "parameters": {{}},
      "preconditions": ["<condition1>"],
      "expected_outcome": "<what should happen>"
    }}
  ]
}}"""

    # --- SAS serialization ---

    def _deserialize_plan(self, plan_data: PlanData) -> Plan | None:
        """Convert PlanData from SAS into internal Plan object."""
        if plan_data.plan_status == "idle":
            return None

        # Allow empty plan list (zero-step plans are valid)
        steps = [
            PlanStep(
                step_id=i,
                action=step,
                params={},
                preconditions=[],
                expected_outcome="",
                status=StepStatus.PENDING,
            )
            for i, step in enumerate(plan_data.current_plan)
        ]

        return Plan(
            goal_description="",
            steps=steps,
            current_step_index=plan_data.current_step,
            status=PlanStatus(plan_data.plan_status),
        )

    async def _write_plan_to_sas(
        self, sas: SharedAgentState, plan: Plan | None
    ) -> None:
        """Write current plan to SAS."""
        if plan is None:
            await sas.update_plans(
                PlanData(current_plan=[], plan_status="idle", current_step=0)
            )
            return

        await sas.update_plans(
            PlanData(
                current_plan=[step.action for step in plan.steps],
                plan_status=plan.status.value,
                current_step=plan.current_step_index,
            )
        )
