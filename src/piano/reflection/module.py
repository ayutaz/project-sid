"""Self-reflection module based on the Reflexion paper.

Implements a 3-stage reflection process:
1. EVALUATE: Analyze recent actions and outcomes
2. GENERATE_INSIGHTS: Generate learnings from evaluation
3. UPDATE_BEHAVIOR: Update self-reflection state with new insights

This module periodically reflects on the agent's action history,
compares expected vs actual outcomes, and generates insights that
can trigger goal revision and behavior updates.

Reference: docs/implementation/07-goal-planning.md Section 4
"""

from __future__ import annotations

__all__ = ["ReflectionResult", "ReflectionStage", "SelfReflectionModule"]

from enum import StrEnum
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from piano.core.module import Module
from piano.core.types import (
    LLMRequest,
    ModuleResult,
    ModuleTier,
    SelfReflectionData,
)

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState
    from piano.llm.provider import LLMProvider

logger = structlog.get_logger(__name__)


class ReflectionStage(StrEnum):
    """Stages of the Reflexion reflection process."""

    EVALUATE = "evaluate"
    GENERATE_INSIGHTS = "generate_insights"
    UPDATE_BEHAVIOR = "update_behavior"


class ReflectionResult(BaseModel):
    """Result of a reflection cycle."""

    stage: ReflectionStage
    evaluation: str = ""
    insights: list[str] = Field(default_factory=list)
    behavior_updates: list[str] = Field(default_factory=list)
    should_revise_goals: bool = False


class SelfReflectionModule(Module):
    """Self-reflection module implementing the Reflexion 3-stage process.

    Periodically analyzes recent actions and outcomes to:
    - Detect patterns of success and failure
    - Generate insights about behavior effectiveness
    - Update self-reflection state with learnings
    - Trigger goal revision when critical failures are detected
    """

    @property
    def name(self) -> str:
        """Unique module name."""
        return "self_reflection"

    @property
    def tier(self) -> ModuleTier:
        """Execution tier -- SLOW (full LLM calls, 1-10s)."""
        return ModuleTier.SLOW

    def __init__(
        self,
        llm_provider: LLMProvider,
        *,
        reflection_interval_ticks: int = 30,
        min_actions_for_reflection: int = 5,
    ) -> None:
        """Initialize the self-reflection module.

        Args:
            llm_provider: LLM provider for generating insights.
            reflection_interval_ticks: Number of ticks between reflections (default: 30).
            min_actions_for_reflection: Minimum actions required before reflection (default: 5).
        """
        self.llm_provider = llm_provider
        self.reflection_interval_ticks = reflection_interval_ticks
        self.min_actions_for_reflection = min_actions_for_reflection

        self._ticks_since_reflection = 0
        self._actions_at_last_reflection = 0

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        """Execute one tick of self-reflection.

        Checks if enough actions have occurred since the last reflection,
        and if so, runs the 3-stage reflection process.

        Args:
            sas: Shared agent state.

        Returns:
            ModuleResult with reflection data.
        """
        self._ticks_since_reflection += 1

        # Check if we should reflect
        action_history = await sas.get_action_history(limit=50)
        current_action_count = len(action_history)
        actions_since_last = current_action_count - self._actions_at_last_reflection

        should_reflect = (
            self._ticks_since_reflection >= self.reflection_interval_ticks
            and actions_since_last >= self.min_actions_for_reflection
        )

        if not should_reflect:
            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                data={
                    "reflected": False,
                    "ticks_since_reflection": self._ticks_since_reflection,
                    "actions_since_last": actions_since_last,
                },
            )

        # Run the 3-stage reflection process
        try:
            # Stage 1: Evaluate
            evaluation = await self._evaluate_actions(action_history)

            # Stage 2: Generate insights
            memory_context = await self._get_memory_context(sas)
            insights = await self._generate_insights(evaluation, memory_context)

            # Stage 3: Update behavior
            current_reflection = await sas.get_self_reflection()
            updated_reflection = await self._update_behavior(insights, current_reflection)

            # Write updated reflection back to SAS
            await sas.update_self_reflection(updated_reflection)

            # Reset counters
            self._ticks_since_reflection = 0
            self._actions_at_last_reflection = current_action_count

            result = ReflectionResult(
                stage=ReflectionStage.UPDATE_BEHAVIOR,
                evaluation=evaluation,
                insights=insights,
                behavior_updates=[],
                should_revise_goals=self._should_revise_goals(evaluation, insights),
            )

            logger.info(
                "reflection_complete",
                insights_count=len(insights),
                should_revise_goals=result.should_revise_goals,
            )

            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                data={
                    "reflected": True,
                    "evaluation": evaluation,
                    "insights": insights,
                    "total_insights": len(updated_reflection.insights),
                    "should_revise_goals": result.should_revise_goals,
                },
            )

        except Exception as exc:
            logger.error("reflection_failed", error=str(exc))
            # On error, keep previous reflection state and return error
            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                error=f"Reflection failed: {exc}",
                data={
                    "reflected": False,
                    "error": str(exc),
                },
            )

    async def _evaluate_actions(self, action_history: list) -> str:
        """Stage 1: Evaluate recent actions and summarize success/failure rates.

        Args:
            action_history: List of recent action history entries.

        Returns:
            Evaluation summary string.
        """
        if not action_history:
            return "No actions to evaluate."

        total_actions = len(action_history)
        successful_actions = sum(1 for entry in action_history if entry.success)
        failed_actions = total_actions - successful_actions

        success_rate = (successful_actions / total_actions * 100) if total_actions > 0 else 0

        # Group by action type
        action_types: dict[str, dict[str, int]] = {}
        for entry in action_history:
            action = entry.action
            if action not in action_types:
                action_types[action] = {"total": 0, "success": 0, "failure": 0}

            action_types[action]["total"] += 1
            if entry.success:
                action_types[action]["success"] += 1
            else:
                action_types[action]["failure"] += 1

        evaluation = f"Evaluated {total_actions} recent actions. "
        evaluation += (
            f"Success rate: {success_rate:.1f}% "
            f"({successful_actions} successful, {failed_actions} failed). "
        )

        # Add per-action breakdown
        if action_types:
            evaluation += "By action type: "
            breakdowns = []
            for action, stats in action_types.items():
                rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
                breakdowns.append(f"{action}: {rate:.0f}% ({stats['success']}/{stats['total']})")
            evaluation += ", ".join(breakdowns) + "."

        return evaluation

    async def _get_memory_context(self, sas: SharedAgentState) -> str:
        """Get relevant memory context for insight generation.

        Args:
            sas: Shared agent state.

        Returns:
            Memory context string.
        """
        goals = await sas.get_goals()
        plans = await sas.get_plans()

        context = f"Current goal: {goals.current_goal}. "
        if goals.completed_goals:
            context += f"Recently completed: {', '.join(goals.completed_goals[-3:])}. "
        if plans.current_plan:
            context += f"Plan status: {plans.plan_status}."

        return context

    async def _generate_insights(self, evaluation: str, memory_context: str) -> list[str]:
        """Stage 2: Generate insights from evaluation using LLM.

        Args:
            evaluation: Evaluation summary from stage 1.
            memory_context: Relevant memory context.

        Returns:
            List of insight strings.
        """
        prompt = f"""You are reflecting on your recent actions and performance.

Evaluation:
{evaluation}

Context:
{memory_context}

Based on this evaluation, generate 2-3 key insights about:
1. What patterns of success or failure do you observe?
2. What could be improved in future actions?
3. Are there any critical issues that require immediate attention?

Respond with a JSON array of insight strings.
Example: ["I am failing at mining actions, may need better tools",
         "Movement to distant locations is reliable"]
"""

        request = LLMRequest(
            prompt=prompt,
            system_prompt=(
                "You are an AI agent reflecting on your recent performance. "
                "Be concise and actionable."
            ),
            tier=ModuleTier.SLOW,
            temperature=0.7,
            max_tokens=256,
            json_mode=True,
        )

        response = await self.llm_provider.complete(request)
        content = response.content.strip()

        # Parse JSON response
        import json

        try:
            insights = json.loads(content)
            if isinstance(insights, list):
                return [str(insight) for insight in insights[:5]]  # Limit to 5 insights
            return [content]
        except json.JSONDecodeError:
            logger.warning("failed_to_parse_insights", content=content)
            # Fallback: use evaluation as single insight
            return [evaluation]

    async def _update_behavior(
        self,
        insights: list[str],
        current_reflection: SelfReflectionData,
    ) -> SelfReflectionData:
        """Stage 3: Update self-reflection state with new insights.

        Args:
            insights: New insights from stage 2.
            current_reflection: Current self-reflection state.

        Returns:
            Updated self-reflection state.
        """
        # Merge new insights with existing ones
        all_insights = current_reflection.insights + insights

        # Keep most recent insights (limit to last 20)
        if len(all_insights) > 20:
            all_insights = all_insights[-20:]

        # Create summary of latest insights
        latest_reflection = " ".join(insights) if insights else current_reflection.last_reflection

        return SelfReflectionData(
            last_reflection=latest_reflection,
            insights=all_insights,
            personality_traits=current_reflection.personality_traits,  # Preserve traits
        )

    def _should_revise_goals(self, evaluation: str, insights: list[str]) -> bool:
        """Determine if goals should be revised based on reflection.

        Args:
            evaluation: Evaluation summary.
            insights: Generated insights.

        Returns:
            True if goals should be revised.
        """
        # Check for critical failure patterns in evaluation
        evaluation_lower = evaluation.lower()

        # Look for very low success rates (0-20%)
        import re
        success_rate_match = re.search(r"success rate: (\d+\.?\d*)%", evaluation_lower)
        if success_rate_match:
            success_rate = float(success_rate_match.group(1))
            if success_rate < 20.0:  # Less than 20% success is critical
                return True

        # Check for critical keywords that indicate severe issues
        critical_keywords = ["critical", "failing", ": 0%"]  # More specific patterns

        # Check if evaluation mentions critical failures
        if any(keyword in evaluation_lower for keyword in critical_keywords):
            return True

        # Check if insights mention critical issues
        for insight in insights:
            insight_lower = insight.lower()
            if any(keyword in insight_lower for keyword in critical_keywords):
                return True

        return False
