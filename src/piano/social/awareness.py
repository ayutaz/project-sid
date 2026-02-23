"""Social Awareness module for the PIANO architecture.

This module implements selective activation based on social context:
- Activates during conversations
- Activates when nearby agents are detected
- Activates on a periodic social goal cycle (5-10s)

Uses LLM to infer other agents' intents, emotions, and recommend
interaction strategies. Updates social data (sentiment/affinity) in SAS.

Reference: docs/implementation/06-social-cognition.md Section 1
"""

from __future__ import annotations

__all__ = [
    "ActivationTrigger",
    "SocialAwarenessModule",
    "SocialAwarenessOutput",
    "SocialSignal",
]

import json
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

from piano.cc.compression import sanitize_text
from piano.core.module import Module
from piano.core.types import LLMRequest, ModuleResult, ModuleTier

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState
    from piano.llm.provider import LLMProvider

logger = structlog.get_logger(__name__)


class ActivationTrigger(StrEnum):
    """Social awareness module activation triggers."""

    CONVERSATION_START = "conversation_start"
    CONVERSATION_ONGOING = "conversation_ongoing"
    NEARBY_AGENT = "nearby_agent"
    SOCIAL_GOAL_CYCLE = "social_goal_cycle"


class SocialSignal(BaseModel):
    """A social signal detected by the awareness module."""

    signal_type: str  # e.g., "help_needed", "opportunity", "conflict"
    description: str
    target_agent_id: str | None = None
    urgency: float = 0.5  # 0.0 - 1.0


class SocialAwarenessOutput(BaseModel):
    """Output from the social awareness module."""

    sentiment_updates: dict[str, float] = Field(default_factory=dict)  # {agent_id: sentiment}
    inferred_intents: dict[str, str] = Field(default_factory=dict)  # {agent_id: intent}
    social_signals: list[SocialSignal] = Field(default_factory=list)
    interaction_strategy: str = ""


class SocialAwarenessModule(Module):
    """Social Awareness module with selective activation.

    Only activates when:
    1. In conversation
    2. Nearby agents detected within proximity threshold
    3. Social goal cycle timer elapsed (5-10s)

    When activated, uses LLM to infer social context and update SAS.
    """

    @property
    def name(self) -> str:
        """Unique module name."""
        return "social_awareness"

    @property
    def tier(self) -> ModuleTier:
        """Execution tier -- MID (lightweight LLM)."""
        return ModuleTier.MID

    def __init__(
        self,
        llm_provider: LLMProvider,
        *,
        proximity_threshold: float = 16.0,
        social_goal_interval: float = 7.5,
    ) -> None:
        """Initialize the social awareness module.

        Args:
            llm_provider: LLM provider for intent/emotion inference.
            proximity_threshold: Distance threshold for nearby agent detection (blocks).
            social_goal_interval: Interval for periodic social goal cycle (seconds).
        """
        self.llm_provider = llm_provider
        self.proximity_threshold = proximity_threshold
        self.social_goal_interval = social_goal_interval
        self._last_activation_time: float = 0.0

    def should_activate(
        self,
        sas: SharedAgentState,
        elapsed_since_last: float,
    ) -> tuple[bool, ActivationTrigger | None]:
        """Determine if the module should activate on this tick.

        Args:
            sas: Shared agent state (not used in async context, passed for checks).
            elapsed_since_last: Time elapsed since last activation (seconds).

        Returns:
            (should_activate, trigger_type) tuple.
        """
        # Will be checked asynchronously in tick() - this is a placeholder
        # for synchronous pre-check if needed
        return False, None

    async def tick(self, sas: SharedAgentState) -> Any:
        """Execute one tick of social awareness.

        1. Check activation conditions.
        2. If activated, gather social context from SAS.
        3. Use LLM to infer intents, emotions, and strategy.
        4. Update SAS social data.
        5. Return ModuleResult with social signals.

        Args:
            sas: Shared agent state.

        Returns:
            ModuleResult with social awareness data.
        """
        # Calculate elapsed time since last activation
        current_time = time.time()
        if self._last_activation_time > 0:
            elapsed = current_time - self._last_activation_time
        else:
            elapsed = 999.0

        # Check activation conditions
        should_activate, trigger = await self._check_activation(sas, elapsed)

        if not should_activate:
            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                data={
                    "activated": False,
                    "reason": "no_trigger",
                },
            )

        # Mark activation time
        self._last_activation_time = current_time

        logger.info(
            "social_awareness_activated",
            agent_id=sas.agent_id,
            trigger=trigger,
            elapsed_since_last=elapsed,
        )

        # Gather context for LLM inference
        try:
            context = await self._gather_context(sas)
            output = await self._infer_social_context(context)

            # Update SAS with new social data
            await self._update_social_data(sas, output)

            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                data={
                    "activated": True,
                    "trigger": trigger,
                    "sentiment_updates": output.sentiment_updates,
                    "inferred_intents": output.inferred_intents,
                    "social_signals": [s.model_dump() for s in output.social_signals],
                    "interaction_strategy": output.interaction_strategy,
                },
            )
        except Exception as e:
            logger.error(
                "social_awareness_error",
                agent_id=sas.agent_id,
                error=str(e),
            )
            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                error=str(e),
            )

    async def _check_activation(
        self,
        sas: SharedAgentState,
        elapsed: float,
    ) -> tuple[bool, ActivationTrigger | None]:
        """Check activation conditions asynchronously.

        Args:
            sas: Shared agent state.
            elapsed: Time elapsed since last activation.

        Returns:
            (should_activate, trigger_type) tuple.
        """
        percepts = await sas.get_percepts()

        # Condition 1: Conversation ongoing (chat messages present)
        if percepts.chat_messages:
            return True, ActivationTrigger.CONVERSATION_ONGOING

        # Condition 2: Nearby agents detected
        if percepts.nearby_players:
            return True, ActivationTrigger.NEARBY_AGENT

        # Condition 3: Social goal cycle timer
        if elapsed >= self.social_goal_interval:
            return True, ActivationTrigger.SOCIAL_GOAL_CYCLE

        return False, None

    async def _gather_context(self, sas: SharedAgentState) -> dict[str, Any]:
        """Gather social context from SAS for LLM inference.

        Args:
            sas: Shared agent state.

        Returns:
            Context dictionary with relevant social information.
        """
        percepts = await sas.get_percepts()
        social = await sas.get_social()
        goals = await sas.get_goals()
        reflection = await sas.get_self_reflection()
        stm = await sas.get_stm(limit=10)

        return {
            "agent_id": sas.agent_id,
            "nearby_players": percepts.nearby_players,
            "chat_messages": percepts.chat_messages[-5:] if percepts.chat_messages else [],
            "current_relationships": social.relationships,
            "current_emotions": social.emotions,
            "recent_interactions": (
                social.recent_interactions[-3:] if social.recent_interactions else []
            ),
            "current_goal": goals.current_goal,
            "personality_traits": reflection.personality_traits,
            "recent_memories": [m.content for m in stm[:5]],
        }

    async def _infer_social_context(self, context: dict[str, Any]) -> SocialAwarenessOutput:
        """Use LLM to infer social context, intents, and strategy.

        Args:
            context: Gathered social context.

        Returns:
            SocialAwarenessOutput with inferences.
        """
        # Build LLM prompt
        prompt = self._build_prompt(context)

        request = LLMRequest(
            prompt=prompt,
            system_prompt="You are a social awareness system for an AI agent in Minecraft. "
            "Analyze the social context and provide insights about other agents' intents, "
            "emotions, and recommend interaction strategies. Respond in valid JSON format.",
            tier=ModuleTier.MID,
            temperature=0.7,
            max_tokens=512,
            json_mode=True,
        )

        response = await self.llm_provider.complete(request)

        # Parse LLM response
        try:
            data = json.loads(response.content)
            return SocialAwarenessOutput(
                sentiment_updates=data.get("sentiment_updates", {}),
                inferred_intents=data.get("inferred_intents", {}),
                social_signals=[SocialSignal(**s) for s in data.get("social_signals", [])],
                interaction_strategy=data.get("interaction_strategy", ""),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(
                "social_awareness_parse_error",
                error=str(e),
                response=response.content,
            )
            return SocialAwarenessOutput()

    def _build_prompt(self, context: dict[str, Any]) -> str:
        """Build the LLM prompt from context.

        Args:
            context: Social context dictionary.

        Returns:
            Formatted prompt string.
        """
        nearby = ", ".join(context["nearby_players"]) if context["nearby_players"] else "none"
        chat_summary = (
            "\n".join(
                f"- {sanitize_text(msg.get('username', 'unknown'), max_length=50)}: "
                f"{sanitize_text(msg.get('message', ''), max_length=200)}"
                for msg in context["chat_messages"]
            )
            or "No recent chat"
        )

        relationships_summary = (
            ", ".join(
                f"{agent}: {score:.2f}"
                for agent, score in list(context["current_relationships"].items())[:3]
            )
            or "No established relationships"
        )

        prompt = f"""Analyze the current social situation for agent {context["agent_id"]}.

Nearby players: {nearby}
Current goal: {context["current_goal"]}

Recent chat messages:
{chat_summary}

Current relationships: {relationships_summary}
Current emotions: {context["current_emotions"]}

Provide your analysis in JSON format with these fields:
- sentiment_updates: dict mapping agent IDs to sentiment scores (-1.0 to 1.0)
- inferred_intents: dict mapping agent IDs to their inferred intentions
- social_signals: list of objects with signal_type, description, target_agent_id, urgency
- interaction_strategy: recommended approach for social interaction

Example:
{{
  "sentiment_updates": {{"player1": 0.3}},
  "inferred_intents": {{"player1": "seeking collaboration"}},
  "social_signals": [
    {{
      "signal_type": "opportunity",
      "description": "Player1 is looking for help with building",
      "target_agent_id": "player1",
      "urgency": 0.6
    }}
  ],
  "interaction_strategy": "Approach player1 with friendly greeting and offer assistance"
}}"""
        return prompt

    async def _update_social_data(
        self,
        sas: SharedAgentState,
        output: SocialAwarenessOutput,
    ) -> None:
        """Update SAS social data with new inferences.

        Args:
            sas: Shared agent state.
            output: Social awareness output with updates.
        """
        social = await sas.get_social()

        # Update relationships with new sentiment scores
        for agent_id, sentiment in output.sentiment_updates.items():
            # Blend new sentiment with existing (60% new, 40% old)
            current = social.relationships.get(agent_id, 0.0)
            social.relationships[agent_id] = 0.6 * sentiment + 0.4 * current

        # Store inferred intents in recent_interactions
        if output.inferred_intents or output.social_signals:
            interaction_record = {
                "timestamp": time.time(),
                "inferred_intents": output.inferred_intents,
                "social_signals": [s.model_dump() for s in output.social_signals],
                "interaction_strategy": output.interaction_strategy,
            }
            social.recent_interactions.append(interaction_record)

            # Keep only recent interactions (last 10)
            if len(social.recent_interactions) > 10:
                social.recent_interactions = social.recent_interactions[-10:]

        await sas.update_social(social)
