"""Goal generation module for the PIANO architecture.

Generates goals based on current state using LLM-based graph-structured reasoning.
Maintains a goal hierarchy: long-term → mid-term → short-term → immediate actions.

This is a SLOW module that runs periodically (aligned with CC cycles, 5-10s).
Goals are influenced by percepts, social context, memory, and action history.

Reference: docs/implementation/07-goal-planning.md Section 1
"""

from __future__ import annotations

__all__ = ["GoalCategory", "GoalGenerationModule", "GoalGraph", "GoalNode", "GoalStatus"]

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from piano.core.module import Module
from piano.core.types import GoalData, ModuleResult, ModuleTier

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState
    from piano.llm.provider import LLMProvider

logger = structlog.get_logger()


# --- Enums ---


class GoalCategory(StrEnum):
    """Goal categories for role specialization."""

    SURVIVAL = "survival"
    SOCIAL = "social"
    EXPLORATION = "exploration"
    CRAFTING = "crafting"
    BUILDING = "building"
    MINING = "mining"
    FARMING = "farming"


class GoalStatus(StrEnum):
    """Status of a goal in the goal graph."""

    PENDING = "pending"  # Not yet started
    ACTIVE = "active"  # Currently being pursued
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed
    BLOCKED = "blocked"  # Prerequisites not met


# --- Data Models ---


class GoalNode(BaseModel):
    """A single goal in the goal hierarchy."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING
    priority: float = 0.5  # 0.0 - 1.0
    category: GoalCategory = GoalCategory.SURVIVAL
    prerequisites: list[str] = Field(default_factory=list)  # List of goal IDs
    parent_goal_id: str | None = None  # Parent goal in hierarchy
    estimated_difficulty: float = 0.5  # 0.0 - 1.0


# --- Goal Graph ---


class GoalGraph:
    """Manages goal hierarchy and dependency tracking."""

    def __init__(self) -> None:
        self._goals: dict[str, GoalNode] = {}

    def add_goal(self, goal: GoalNode) -> None:
        """Add a goal to the graph."""
        self._goals[goal.id] = goal

    def remove_goal(self, goal_id: str) -> None:
        """Remove a goal from the graph."""
        self._goals.pop(goal_id, None)

    def get_goal(self, goal_id: str) -> GoalNode | None:
        """Get a goal by ID."""
        return self._goals.get(goal_id)

    def get_all_goals(self) -> list[GoalNode]:
        """Get all goals."""
        return list(self._goals.values())

    def get_executable_goals(self) -> list[GoalNode]:
        """Get goals that can be executed (prerequisites met, status=PENDING or ACTIVE)."""
        executable = []
        for goal in self._goals.values():
            if goal.status not in (GoalStatus.PENDING, GoalStatus.ACTIVE):
                continue

            # Check prerequisites
            prereqs_met = all(
                self._goals.get(prereq_id, GoalNode()).status == GoalStatus.COMPLETED
                for prereq_id in goal.prerequisites
            )

            if prereqs_met:
                executable.append(goal)

        return executable

    def get_active_goals(self) -> list[GoalNode]:
        """Get all active goals."""
        return [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]

    def recalculate_priorities(self) -> None:
        """Recalculate goal priorities based on dependencies and category weights.

        Goals with fewer prerequisites and lower difficulty get higher priority.
        """
        for goal in self._goals.values():
            # Base priority
            priority = 0.5

            # Increase priority if fewer prerequisites
            prereq_penalty = len(goal.prerequisites) * 0.1
            priority -= prereq_penalty

            # Decrease priority based on difficulty
            difficulty_penalty = goal.estimated_difficulty * 0.2
            priority -= difficulty_penalty

            # Clamp to [0, 1]
            goal.priority = max(0.0, min(1.0, priority))

    def clear(self) -> None:
        """Clear all goals."""
        self._goals.clear()


# --- Module ---


class GoalGenerationModule(Module):
    """Goal generation module using LLM-based reasoning.

    Reads current state from SAS (percepts, social, memory, action_history)
    and generates new goals using an LLM. Updates the goal hierarchy in SAS.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        """Initialize the goal generation module.

        Args:
            llm_provider: LLM provider for goal generation.
        """
        self._llm = llm_provider
        self._goal_graph = GoalGraph()

    @property
    def name(self) -> str:
        """Unique module name."""
        return "goal_generation"

    @property
    def tier(self) -> ModuleTier:
        """Execution tier -- SLOW (LLM-based, 1-10s)."""
        return ModuleTier.SLOW

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        """Execute one tick of goal generation.

        1. Read current state from SAS (percepts, goals, social, memory, action_history).
        2. Build context for LLM prompt.
        3. Call LLM to generate new goals.
        4. Parse response and update goal graph.
        5. Write updated goals back to SAS.
        6. Return ModuleResult with generation summary.
        """
        try:
            # 1. Read current state
            percepts = await sas.get_percepts()
            current_goals = await sas.get_goals()
            social = await sas.get_social()
            action_history = await sas.get_action_history(limit=10)

            # 2. Build context
            context = self._build_context(percepts, current_goals, social, action_history)

            # 3. Generate goals using LLM
            new_goals = await self._generate_goals_llm(context)

            # 4. Update goal graph
            for goal in new_goals:
                self._goal_graph.add_goal(goal)

            self._goal_graph.recalculate_priorities()

            # 5. Write back to SAS
            updated_goal_data = self._build_goal_data()
            await sas.update_goals(updated_goal_data)

            # 6. Return result
            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                data={
                    "goals_generated": len(new_goals),
                    "total_goals": len(self._goal_graph.get_all_goals()),
                    "active_goals": len(self._goal_graph.get_active_goals()),
                    "executable_goals": len(self._goal_graph.get_executable_goals()),
                    "new_goal_descriptions": [g.description for g in new_goals],
                },
            )

        except Exception as exc:
            logger.error("goal_generation_failed", error=str(exc))
            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                error=f"Goal generation failed: {exc}",
                data={"goals_generated": 0},
            )

    async def initialize(self) -> None:
        """Initialize the module (clear goal graph)."""
        self._goal_graph.clear()

    async def shutdown(self) -> None:
        """Shutdown the module (clear goal graph)."""
        self._goal_graph.clear()

    # --- Private Methods ---

    def _build_context(
        self,
        percepts: Any,
        current_goals: GoalData,
        social: Any,
        action_history: list[Any],
    ) -> dict[str, Any]:
        """Build context dictionary for LLM prompt."""
        return {
            "inventory": percepts.inventory,
            "nearby_players": percepts.nearby_players,
            "health": percepts.health,
            "hunger": percepts.hunger,
            "current_goal": current_goals.current_goal,
            "goal_stack": current_goals.goal_stack,
            "completed_goals": current_goals.completed_goals,
            "recent_actions": [
                {"action": entry.action, "success": entry.success}
                for entry in action_history[:5]
            ],
            "relationships": social.relationships,
            "recent_interactions": social.recent_interactions[:3],
        }

    async def _generate_goals_llm(self, context: dict[str, Any]) -> list[GoalNode]:
        """Generate goals using LLM.

        Args:
            context: Context dictionary with current state.

        Returns:
            List of generated GoalNode objects.
        """
        from piano.core.types import LLMRequest

        prompt = self._build_prompt(context)

        request = LLMRequest(
            prompt=prompt,
            system_prompt="You are a helpful AI that generates goals for a Minecraft agent. "
            "Respond in valid JSON format only.",
            tier=ModuleTier.SLOW,
            temperature=0.7,
            max_tokens=512,
            json_mode=True,
        )

        response = await self._llm.complete(request)

        return self._parse_llm_response(response.content)

    def _build_prompt(self, context: dict[str, Any]) -> str:
        """Build the LLM prompt for goal generation."""
        nearby_players = ", ".join(context["nearby_players"]) or "none"
        inventory_summary = ", ".join(
            f"{item}:{count}" for item, count in list(context["inventory"].items())[:5]
        ) or "empty"
        recent_actions_summary = ", ".join(
            f"{a['action']}({'success' if a['success'] else 'fail'})"
            for a in context["recent_actions"]
        ) or "none"
        completed = ", ".join(context["completed_goals"][:3]) or "none"

        return f"""You are an autonomous Minecraft agent.
Based on your current state, generate 1-3 new goals.

## Current State
- Inventory: {inventory_summary}
- Nearby Players: {nearby_players}
- Health: {context['health']}/20, Hunger: {context['hunger']}/20
- Current Goal: {context['current_goal'] or 'none'}
- Completed Goals: {completed}
- Recent Actions: {recent_actions_summary}

## Task
Generate 1-3 new goals that are:
1. Achievable with your current resources or nearby environment
2. Consistent with your recent behavior
3. Varied in category (survival, social, exploration, crafting, building, mining, farming)

Respond in JSON format:
{{
  "goals": [
    {{
      "description": "concise goal description",
      "category": "survival|social|exploration|crafting|building|mining|farming",
      "priority": 0.0-1.0,
      "estimated_difficulty": 0.0-1.0
    }}
  ]
}}"""

    def _parse_llm_response(self, content: str) -> list[GoalNode]:
        """Parse LLM response JSON into GoalNode objects."""
        try:
            data = json.loads(content)
            goals = data.get("goals", [])

            # Ensure goals is a list
            if not isinstance(goals, list):
                logger.warning("goals_not_list", goals_type=type(goals).__name__)
                return []

            result = []
            for g in goals:
                # Skip if g is not a dict
                if not isinstance(g, dict):
                    continue

                goal = GoalNode(
                    description=g.get("description", ""),
                    category=GoalCategory(g.get("category", "survival")),
                    priority=float(g.get("priority", 0.5)),
                    estimated_difficulty=float(g.get("estimated_difficulty", 0.5)),
                    status=GoalStatus.ACTIVE,  # New goals start as ACTIVE
                )
                result.append(goal)

            return result

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.warning("goal_parse_failed", error=str(exc), content=content[:200])
            return []

    def _build_goal_data(self) -> GoalData:
        """Build GoalData from the goal graph to write back to SAS."""
        active_goals = self._goal_graph.get_active_goals()
        all_goals = self._goal_graph.get_all_goals()
        completed = [g.description for g in all_goals if g.status == GoalStatus.COMPLETED]

        # Set current_goal to highest priority active goal
        current_goal = ""
        if active_goals:
            active_goals.sort(key=lambda g: g.priority, reverse=True)
            current_goal = active_goals[0].description

        # Build goal_stack from active goals
        goal_stack = [g.description for g in active_goals[:5]]

        return GoalData(
            current_goal=current_goal,
            goal_stack=goal_stack,
            completed_goals=completed[:10],
        )
