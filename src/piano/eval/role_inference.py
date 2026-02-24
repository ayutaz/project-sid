"""LLM-based role inference pipeline for PIANO architecture.

Infers agent roles from their recent social goals using LLM analysis.
Used for Section 5 specialization evaluation in the paper.

Reference: docs/implementation/00-overview.md (Specialization Metrics)
"""

from __future__ import annotations

__all__ = [
    "AgentRole",
    "RoleHistory",
    "RoleInferencePipeline",
    "RoleInferenceRequest",
    "RoleInferenceResult",
]

import asyncio
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from piano.core.types import LLMRequest, ModuleTier

if TYPE_CHECKING:
    from piano.llm.provider import LLMProvider

logger = structlog.get_logger(__name__)


class AgentRole(StrEnum):
    """Possible agent roles in the Minecraft simulation."""

    FARMER = "farmer"
    MINER = "miner"
    ENGINEER = "engineer"
    GUARD = "guard"
    EXPLORER = "explorer"
    BLACKSMITH = "blacksmith"
    SCOUT = "scout"
    STRATEGIST = "strategist"
    CURATOR = "curator"
    COLLECTOR = "collector"
    OTHER = "other"


class RoleInferenceRequest(BaseModel):
    """Request to infer an agent's role from recent goals."""

    agent_id: str = Field(description="ID of the agent to analyze")
    recent_goals: list[str] = Field(
        description="Most recent social goals (up to 5)",
        max_length=5,
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the inference was requested",
    )


class RoleInferenceResult(BaseModel):
    """Result of a role inference for a single agent."""

    agent_id: str = Field(description="ID of the agent")
    inferred_role: AgentRole = Field(description="The inferred role")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the inference"
    )
    reasoning: str = Field(description="LLM reasoning for the role assignment")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the inference was produced",
    )


# --- Prompt template ---

_ROLE_INFERENCE_SYSTEM_PROMPT = """\
You are an expert at analyzing agent behavior in Minecraft simulations.
Given an agent's recent social goals, infer the most likely role they are fulfilling.

Available roles:
- farmer: Focuses on farming, food production, crop management
- miner: Focuses on mining ores, digging, resource extraction
- engineer: Focuses on building structures, redstone, infrastructure
- guard: Focuses on defense, security, protecting the settlement
- explorer: Focuses on exploring new areas, mapping, scouting distant regions
- blacksmith: Focuses on crafting tools, weapons, armor
- scout: Focuses on reconnaissance, tracking, gathering intel on surroundings
- strategist: Focuses on planning, coordination, leadership
- curator: Focuses on collecting, organizing, maintaining resources or knowledge
- collector: Focuses on gathering diverse items, stockpiling materials
- other: Does not fit any of the above roles

Respond ONLY with a JSON object in this exact format:
{"role": "<role_name>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}
"""

_ROLE_INFERENCE_USER_PROMPT = """\
Agent ID: {agent_id}
Recent social goals (most recent first):
{goals_text}

Based on these goals, what role is this agent most likely fulfilling?
"""


class RoleInferencePipeline:
    """Pipeline for inferring agent roles from their social goals using LLM."""

    def __init__(self, llm: LLMProvider) -> None:
        """Initialize the pipeline.

        Args:
            llm: LLM provider for role inference calls.
        """
        self._llm = llm

    def _build_prompt(self, request: RoleInferenceRequest) -> str:
        """Build the user prompt from a request.

        Args:
            request: The role inference request.

        Returns:
            Formatted user prompt string.
        """
        if not request.recent_goals:
            goals_text = "- (no goals recorded)"
        else:
            goals_text = "\n".join(
                f"- {goal}" for goal in request.recent_goals
            )

        return _ROLE_INFERENCE_USER_PROMPT.format(
            agent_id=request.agent_id,
            goals_text=goals_text,
        )

    def _parse_response(self, content: str, agent_id: str) -> RoleInferenceResult:
        """Parse LLM response JSON into a RoleInferenceResult.

        Falls back to OTHER with low confidence if parsing fails.

        Args:
            content: Raw LLM response content.
            agent_id: The agent ID for the result.

        Returns:
            Parsed RoleInferenceResult.
        """
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "role_inference_parse_error",
                agent_id=agent_id,
                content=content[:200],
            )
            return RoleInferenceResult(
                agent_id=agent_id,
                inferred_role=AgentRole.OTHER,
                confidence=0.0,
                reasoning=f"Failed to parse LLM response: {content[:200]}",
            )

        # Extract role
        raw_role = str(data.get("role", "other")).lower().strip()
        try:
            role = AgentRole(raw_role)
        except ValueError:
            logger.warning(
                "role_inference_unknown_role",
                agent_id=agent_id,
                raw_role=raw_role,
            )
            role = AgentRole.OTHER

        # Extract confidence
        try:
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        reasoning = str(data.get("reasoning", ""))

        return RoleInferenceResult(
            agent_id=agent_id,
            inferred_role=role,
            confidence=confidence,
            reasoning=reasoning,
        )

    async def infer_role(
        self, request: RoleInferenceRequest
    ) -> RoleInferenceResult:
        """Infer the role of a single agent from their recent goals.

        Args:
            request: The role inference request.

        Returns:
            RoleInferenceResult with the inferred role, confidence, and reasoning.
        """
        prompt = self._build_prompt(request)

        llm_request = LLMRequest(
            prompt=prompt,
            system_prompt=_ROLE_INFERENCE_SYSTEM_PROMPT,
            tier=ModuleTier.SLOW,
            temperature=0.0,
            max_tokens=256,
            json_mode=True,
        )

        try:
            response = await self._llm.complete(llm_request)
        except (RuntimeError, ValueError, KeyError, OSError):
            logger.exception(
                "role_inference_llm_error",
                agent_id=request.agent_id,
            )
            return RoleInferenceResult(
                agent_id=request.agent_id,
                inferred_role=AgentRole.OTHER,
                confidence=0.0,
                reasoning="LLM call failed",
            )

        return self._parse_response(response.content, request.agent_id)

    async def infer_roles_batch(
        self, requests: list[RoleInferenceRequest]
    ) -> list[RoleInferenceResult]:
        """Infer roles for multiple agents with bounded concurrency.

        Uses a semaphore to limit concurrent LLM calls to 5.

        Args:
            requests: List of role inference requests.

        Returns:
            List of RoleInferenceResult in the same order as requests.
        """
        if not requests:
            return []

        sem = asyncio.Semaphore(5)

        async def _limited(req: RoleInferenceRequest) -> RoleInferenceResult:
            async with sem:
                return await self.infer_role(req)

        results = await asyncio.gather(*[_limited(r) for r in requests])
        return list(results)

    @staticmethod
    def get_role_distribution(
        results: list[RoleInferenceResult],
    ) -> dict[str, float]:
        """Compute the distribution of roles across inference results.

        Args:
            results: List of role inference results.

        Returns:
            Dict mapping role name to fraction of agents (sums to 1.0).
            Empty dict if no results.
        """
        if not results:
            return {}

        from collections import Counter

        counts = Counter(r.inferred_role.value for r in results)
        total = len(results)
        return {role: count / total for role, count in counts.items()}


class RoleHistory:
    """Tracks role inference results over time for role transition analysis."""

    def __init__(self) -> None:
        """Initialize empty role history."""
        self._results: list[RoleInferenceResult] = []

    def add_result(self, result: RoleInferenceResult) -> None:
        """Record a role inference result.

        Args:
            result: The inference result to record.
        """
        self._results.append(result)

    @property
    def results(self) -> list[RoleInferenceResult]:
        """Return all recorded results (read-only copy)."""
        return list(self._results)

    def get_role_transitions(
        self, agent_id: str
    ) -> list[tuple[datetime, AgentRole]]:
        """Get the chronological role history for a specific agent.

        Args:
            agent_id: The agent to look up.

        Returns:
            List of (timestamp, role) tuples sorted by time.
        """
        agent_results = [r for r in self._results if r.agent_id == agent_id]
        agent_results.sort(key=lambda r: r.timestamp)
        return [(r.timestamp, r.inferred_role) for r in agent_results]

    def get_role_persistence(self, agent_id: str) -> float:
        """Calculate role persistence (stability) for a specific agent.

        Persistence is the fraction of consecutive time steps where
        the agent maintained the same role.

        Args:
            agent_id: The agent to evaluate.

        Returns:
            Persistence score in [0, 1]. 1.0 = perfectly stable.
            Returns 1.0 if fewer than 2 observations.
        """
        transitions = self.get_role_transitions(agent_id)

        if len(transitions) < 2:
            return 1.0

        maintained = sum(
            1
            for i in range(len(transitions) - 1)
            if transitions[i][1] == transitions[i + 1][1]
        )
        return maintained / (len(transitions) - 1)
