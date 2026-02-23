"""Talking/utterance generation module for the PIANO architecture.

This module generates contextually appropriate speech when the Cognitive Controller
decides the agent should talk. It reacts to CC broadcast decisions and uses LLM
to produce personality-consistent, context-aware utterances.

Reference: docs/implementation/03-cognitive-controller.md Section 3.3
"""

from __future__ import annotations

__all__ = ["ConversationContext", "TalkingModule", "Utterance"]

from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from piano.cc.compression import sanitize_text
from piano.core.module import Module
from piano.core.types import CCDecision, LLMRequest, ModuleResult, ModuleTier

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState
    from piano.llm.provider import LLMProvider

logger = structlog.get_logger(__name__)

# Big Five personality trait defaults (neutral personality)
DEFAULT_PERSONALITY = {
    "openness": 0.5,
    "conscientiousness": 0.5,
    "extraversion": 0.5,
    "agreeableness": 0.5,
    "neuroticism": 0.5,
}


class Utterance(BaseModel):
    """A generated utterance with metadata."""

    content: str = ""
    target_agent_id: str | None = None  # None = broadcast to all
    tone: str = "neutral"  # friendly, formal, casual, urgent, etc.
    is_response: bool = False  # True if replying to someone


class ConversationContext(BaseModel):
    """Context for generating contextually appropriate utterances."""

    recent_messages: list[dict[str, str]] = Field(default_factory=list)
    current_topic: str = ""
    participants: list[str] = Field(default_factory=list)


class TalkingModule(Module):
    """Talking module that generates speech based on CC decisions.

    This is primarily a reactive module driven by on_broadcast. When the CC
    decides the agent should speak (decision.speaking is set), this module
    generates an appropriate utterance using LLM, considering personality,
    conversation history, emotions, and social relationships.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        personality_traits: dict[str, float] | None = None,
        sas: SharedAgentState | None = None,
    ) -> None:
        """Initialize the Talking module.

        Args:
            llm_provider: LLM provider for generating utterances.
            personality_traits: Big Five personality traits (0.0-1.0 each).
                               Defaults to neutral (0.5) if not provided.
            sas: Optional SAS reference for on_broadcast access.
                 Module ABC's on_broadcast(decision) does not receive SAS,
                 so we store a reference here to enable utterance generation
                 and storage from broadcast handlers.
        """
        self._llm = llm_provider
        self._personality = personality_traits or dict(DEFAULT_PERSONALITY)
        self._sas = sas

    @property
    def name(self) -> str:
        """Unique module name."""
        return "talking"

    @property
    def tier(self) -> ModuleTier:
        """Execution tier -- MID (lightweight LLM calls)."""
        return ModuleTier.MID

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        """Execute one tick of the Talking module.

        Talking is primarily driven by on_broadcast, so tick is mostly passive.
        It checks for any pending utterances that need processing but does not
        generate new utterances on its own.

        Returns:
            ModuleResult with minimal status info.
        """
        # Check if there's a recent utterance stored in SAS
        section = await sas.get_section("talking")
        utterance_count = len(section.get("recent_utterances", []))

        return ModuleResult(
            module_name=self.name,
            tier=self.tier,
            data={
                "utterance_count": utterance_count,
                "status": "passive",
            },
        )

    async def on_broadcast(self, decision: CCDecision) -> None:
        """Handle CC broadcast decision.

        When CC decision includes a 'speaking' field, generate an appropriate
        utterance and store it in SAS for the bridge to send as a chat message.

        Args:
            decision: The CC decision broadcast.
        """
        # Only act if CC has decided the agent should speak
        if not decision.speaking:
            logger.debug("on_broadcast: no speaking directive, skipping")
            return

        if self._sas is None:
            logger.warning(
                "on_broadcast: speaking directive received but no SAS reference; "
                "pass sas= to TalkingModule constructor to enable utterance generation",
                speaking=decision.speaking,
            )
            return

        logger.info("on_broadcast: generating utterance", speaking=decision.speaking)

        utterance = await self._generate_utterance(decision, self._sas)
        await self._store_utterance(self._sas, utterance)

    async def _generate_utterance(
        self,
        decision: CCDecision,
        sas: SharedAgentState,
    ) -> Utterance:
        """Generate an utterance based on CC decision and context.

        Args:
            decision: The CC decision containing speaking directive.
            sas: Shared agent state for accessing context.

        Returns:
            Generated utterance with metadata.
        """
        # Build conversation context from SAS
        context = await self._build_conversation_context(sas)

        # Build LLM prompt
        prompt = self._build_utterance_prompt(decision, context)

        # Call LLM
        try:
            request = LLMRequest(
                prompt=prompt,
                system_prompt=self._build_system_prompt(),
                tier=ModuleTier.MID,
                temperature=0.7,  # Allow some creativity
                max_tokens=150,  # Utterances should be concise
            )
            response = await self._llm.complete(request)

            # Parse response to extract utterance
            content = response.content.strip()

            # Determine if this is a response
            is_response = len(context.participants) > 0

            # Extract target if present (for directed messages)
            target = context.participants[0] if context.participants else None

            # Determine tone from personality and context
            tone = self._infer_tone(decision, context)

            return Utterance(
                content=content,
                target_agent_id=target,
                tone=tone,
                is_response=is_response,
            )

        except Exception as exc:
            logger.error("Failed to generate utterance", error=str(exc))
            # Return empty utterance on error
            return Utterance(content="", tone="neutral")

    async def _build_conversation_context(self, sas: SharedAgentState) -> ConversationContext:
        """Build conversation context from SAS.

        Args:
            sas: Shared agent state.

        Returns:
            Conversation context for utterance generation.
        """
        percepts = await sas.get_percepts()
        # Note: social data could be used for relationship-aware tone in future
        # social = await sas.get_social()

        # Extract recent chat messages (last 5)
        recent_messages = []
        for msg in percepts.chat_messages[-5:]:
            recent_messages.append(
                {
                    "speaker": msg.get("username", "unknown"),
                    "message": msg.get("message", ""),
                }
            )

        # Determine current topic (simplified: last message content)
        current_topic = ""
        if recent_messages:
            current_topic = recent_messages[-1]["message"]

        # Extract participants (nearby players who recently spoke)
        participants = []
        for msg in recent_messages:
            speaker = msg["speaker"]
            if speaker not in participants and speaker != sas.agent_id:
                participants.append(speaker)

        return ConversationContext(
            recent_messages=recent_messages,
            current_topic=current_topic,
            participants=participants,
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt with personality traits."""
        traits_desc = []
        for trait, value in self._personality.items():
            level = "high" if value > 0.7 else "low" if value < 0.3 else "moderate"
            traits_desc.append(f"{trait}: {level}")

        return f"""You are generating speech for an AI agent in Minecraft.

Personality traits:
{chr(10).join(f"- {t}" for t in traits_desc)}

Generate natural, contextually appropriate dialogue that:
- Matches the agent's personality
- Fits the conversation context
- Is concise (1-2 sentences)
- Sounds natural for Minecraft chat

Output only the speech content, no extra formatting."""

    def _build_utterance_prompt(
        self,
        decision: CCDecision,
        context: ConversationContext,
    ) -> str:
        """Build the LLM prompt for utterance generation.

        Args:
            decision: CC decision with speaking directive.
            context: Conversation context.

        Returns:
            Prompt string for LLM.
        """
        sections = []

        # CC speaking directive (sanitized to prevent prompt injection)
        safe_speaking = sanitize_text(decision.speaking or "", max_length=500)
        sections.append(f"## Speaking Directive\n{safe_speaking}")

        # CC reasoning (sanitized to prevent prompt injection)
        if decision.reasoning:
            safe_reasoning = sanitize_text(decision.reasoning, max_length=500)
            sections.append(f"## Context\n{safe_reasoning}")

        # Conversation history
        if context.recent_messages:
            history = "\n".join(
                f"- {msg['speaker']}: {msg['message']}" for msg in context.recent_messages
            )
            sections.append(f"## Recent Conversation\n{history}")

        # Current topic
        if context.current_topic:
            sections.append(f"## Current Topic\n{context.current_topic}")

        sections.append("\nGenerate an appropriate utterance based on the above context.")

        return "\n\n".join(sections)

    def _infer_tone(
        self,
        decision: CCDecision,
        context: ConversationContext,
    ) -> str:
        """Infer the appropriate tone from decision and context.

        Args:
            decision: CC decision.
            context: Conversation context.

        Returns:
            Tone string (e.g., "friendly", "formal", "urgent").
        """
        # Simple heuristic-based tone inference
        speaking_lower = decision.speaking.lower() if decision.speaking else ""

        # Check friendly first (more specific matches)
        if any(word in speaking_lower for word in ["thanks", "appreciate", "glad"]):
            return "friendly"
        elif any(word in speaking_lower for word in ["please", "request", "could"]):
            return "polite"
        elif any(word in speaking_lower for word in ["urgent", "quick", "hurry", "help!"]):
            return "urgent"
        elif context.participants:
            # Talking to someone = default friendly
            return "friendly"
        else:
            return "neutral"

    async def _store_utterance(
        self,
        sas: SharedAgentState,
        utterance: Utterance,
    ) -> None:
        """Store generated utterance in SAS for bridge to send.

        Args:
            sas: Shared agent state.
            utterance: Generated utterance.
        """
        section = await sas.get_section("talking")

        # Initialize recent_utterances if not present
        if "recent_utterances" not in section:
            section["recent_utterances"] = []

        # Add new utterance
        section["recent_utterances"].append(utterance.model_dump())

        # Keep only last 10 utterances
        section["recent_utterances"] = section["recent_utterances"][-10:]

        # Store the latest utterance separately for easy access
        section["latest_utterance"] = utterance.model_dump()

        await sas.update_section("talking", section)
        logger.info("Stored utterance in SAS", content=utterance.content)
