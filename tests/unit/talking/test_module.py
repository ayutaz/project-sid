"""Tests for the Talking module."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from piano.core.types import CCDecision, LLMResponse, ModuleTier, PerceptData
from piano.talking.module import (
    DEFAULT_PERSONALITY,
    ConversationContext,
    TalkingModule,
    Utterance,
)
from tests.helpers import InMemorySAS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Create a mock LLM provider."""
    llm = AsyncMock()
    llm.complete = AsyncMock()
    return llm


@pytest.fixture
def module(mock_llm: AsyncMock) -> TalkingModule:
    """Create a TalkingModule with mock LLM."""
    return TalkingModule(llm_provider=mock_llm)


@pytest.fixture
def sas() -> InMemorySAS:
    """Create an in-memory SAS for testing."""
    return InMemorySAS()


# ---------------------------------------------------------------------------
# Basic Module Tests
# ---------------------------------------------------------------------------


class TestTalkingModuleBasics:
    """Test basic module properties and initialization."""

    def test_module_name(self, module: TalkingModule) -> None:
        """Module name should be 'talking'."""
        assert module.name == "talking"

    def test_module_tier(self, module: TalkingModule) -> None:
        """Module tier should be MID."""
        assert module.tier == ModuleTier.MID

    def test_default_personality(self, mock_llm: AsyncMock) -> None:
        """Module should use default personality if not provided."""
        module = TalkingModule(llm_provider=mock_llm)
        assert module._personality == DEFAULT_PERSONALITY

    def test_custom_personality(self, mock_llm: AsyncMock) -> None:
        """Module should accept custom personality traits."""
        custom = {
            "openness": 0.8,
            "conscientiousness": 0.6,
            "extraversion": 0.9,
            "agreeableness": 0.7,
            "neuroticism": 0.3,
        }
        module = TalkingModule(llm_provider=mock_llm, personality_traits=custom)
        assert module._personality == custom


# ---------------------------------------------------------------------------
# Tick Tests
# ---------------------------------------------------------------------------


class TestTick:
    """Test the tick method."""

    async def test_tick_no_utterances(
        self, module: TalkingModule, sas: InMemorySAS
    ) -> None:
        """Tick with no utterances should return passive status."""
        result = await module.tick(sas)

        assert result.module_name == "talking"
        assert result.tier == ModuleTier.MID
        assert result.data["status"] == "passive"
        assert result.data["utterance_count"] == 0

    async def test_tick_with_stored_utterances(
        self, module: TalkingModule, sas: InMemorySAS
    ) -> None:
        """Tick should report count of stored utterances."""
        # Store some utterances in SAS
        await sas.update_section(
            "talking",
            {
                "recent_utterances": [
                    {"content": "Hello", "tone": "friendly"},
                    {"content": "How are you?", "tone": "friendly"},
                ]
            },
        )

        result = await module.tick(sas)

        assert result.data["utterance_count"] == 2
        assert result.data["status"] == "passive"


# ---------------------------------------------------------------------------
# on_broadcast Tests
# ---------------------------------------------------------------------------


class TestOnBroadcast:
    """Test the on_broadcast reaction to CC decisions."""

    async def test_on_broadcast_no_speaking(self, module: TalkingModule) -> None:
        """on_broadcast with no speaking directive should do nothing."""
        decision = CCDecision(
            summary="Exploring the area",
            action="move",
            speaking=None,  # No speaking directive
        )

        # Should not raise any errors
        await module.on_broadcast(decision)

    async def test_on_broadcast_with_speaking(self, module: TalkingModule) -> None:
        """on_broadcast with speaking directive should log intent."""
        decision = CCDecision(
            summary="Greeting nearby player",
            action="idle",
            speaking="Say hello to the nearby player",
        )

        # Should not raise any errors (just logs in current implementation)
        await module.on_broadcast(decision)

    async def test_on_broadcast_empty_speaking(self, module: TalkingModule) -> None:
        """Empty speaking string should be treated as no directive."""
        decision = CCDecision(
            summary="Thinking",
            action="idle",
            speaking="",  # Empty string
        )

        # Should not raise any errors
        await module.on_broadcast(decision)


# ---------------------------------------------------------------------------
# Utterance Generation Tests
# ---------------------------------------------------------------------------


class TestUtteranceGeneration:
    """Test _generate_utterance method."""

    async def test_generate_simple_utterance(
        self, module: TalkingModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Generate a simple utterance from CC decision."""
        # Setup mock LLM response
        mock_llm.complete.return_value = LLMResponse(
            content="Hello there!",
            model="gpt-4o-mini",
        )

        decision = CCDecision(
            summary="Greeting",
            action="idle",
            speaking="Greet the nearby player",
            reasoning="Player just joined the server",
        )

        utterance = await module._generate_utterance(decision, sas)

        assert utterance.content == "Hello there!"
        assert utterance.tone in ["friendly", "neutral", "polite"]
        assert not utterance.is_response  # No conversation context

    async def test_generate_utterance_with_context(
        self, module: TalkingModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Generate utterance considering conversation context."""
        # Setup conversation in SAS
        await sas.update_percepts(
            PerceptData(
                chat_messages=[
                    {"username": "player1", "message": "Hi, how are you?"},
                ]
            )
        )

        mock_llm.complete.return_value = LLMResponse(
            content="I'm doing great, thanks!",
            model="gpt-4o-mini",
        )

        decision = CCDecision(
            summary="Responding to greeting",
            action="idle",
            speaking="Respond positively to greeting",
        )

        utterance = await module._generate_utterance(decision, sas)

        assert utterance.content == "I'm doing great, thanks!"
        assert utterance.is_response  # Should detect this is a response
        assert utterance.target_agent_id == "player1"

    async def test_generate_utterance_llm_error(
        self, module: TalkingModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """LLM error should return empty utterance."""
        mock_llm.complete.side_effect = Exception("LLM API error")

        decision = CCDecision(
            summary="Speaking",
            action="idle",
            speaking="Say something",
        )

        utterance = await module._generate_utterance(decision, sas)

        assert utterance.content == ""
        assert utterance.tone == "neutral"

    async def test_utterance_uses_personality(
        self, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Generated utterance should consider personality traits."""
        custom_personality = {
            "openness": 0.9,
            "conscientiousness": 0.8,
            "extraversion": 0.9,
            "agreeableness": 0.8,
            "neuroticism": 0.2,
        }

        module = TalkingModule(
            llm_provider=mock_llm,
            personality_traits=custom_personality,
        )

        mock_llm.complete.return_value = LLMResponse(
            content="I'd love to help you build!",
            model="gpt-4o-mini",
        )

        decision = CCDecision(
            summary="Offering help",
            action="idle",
            speaking="Offer to help build",
        )

        await module._generate_utterance(decision, sas)

        # Verify LLM was called with system prompt containing personality
        assert mock_llm.complete.called
        call_args = mock_llm.complete.call_args
        request = call_args[0][0]
        assert "openness: high" in request.system_prompt
        assert "extraversion: high" in request.system_prompt


# ---------------------------------------------------------------------------
# Conversation Context Tests
# ---------------------------------------------------------------------------


class TestConversationContext:
    """Test conversation context building."""

    async def test_build_context_empty(
        self, module: TalkingModule, sas: InMemorySAS
    ) -> None:
        """Build context with no conversation history."""
        context = await module._build_conversation_context(sas)

        assert context.recent_messages == []
        assert context.current_topic == ""
        assert context.participants == []

    async def test_build_context_with_messages(
        self, module: TalkingModule, sas: InMemorySAS
    ) -> None:
        """Build context from chat messages."""
        await sas.update_percepts(
            PerceptData(
                chat_messages=[
                    {"username": "player1", "message": "Hello"},
                    {"username": "player2", "message": "Hi there"},
                    {"username": "player1", "message": "How's it going?"},
                ]
            )
        )

        context = await module._build_conversation_context(sas)

        assert len(context.recent_messages) == 3
        assert context.current_topic == "How's it going?"
        assert "player1" in context.participants
        assert "player2" in context.participants

    async def test_build_context_limits_messages(
        self, module: TalkingModule, sas: InMemorySAS
    ) -> None:
        """Context should only include last 5 messages."""
        messages = [
            {"username": f"player{i}", "message": f"Message {i}"}
            for i in range(10)
        ]
        await sas.update_percepts(PerceptData(chat_messages=messages))

        context = await module._build_conversation_context(sas)

        # Should only keep last 5
        assert len(context.recent_messages) == 5
        assert context.recent_messages[0]["message"] == "Message 5"
        assert context.recent_messages[-1]["message"] == "Message 9"

    async def test_build_context_excludes_self(
        self, module: TalkingModule, sas: InMemorySAS
    ) -> None:
        """Participants should not include the agent itself."""
        await sas.update_percepts(
            PerceptData(
                chat_messages=[
                    {"username": "test-agent-001", "message": "I said something"},
                    {"username": "player1", "message": "Nice!"},
                ]
            )
        )

        context = await module._build_conversation_context(sas)

        # Should only include player1, not test-agent-001
        assert context.participants == ["player1"]


# ---------------------------------------------------------------------------
# Tone Inference Tests
# ---------------------------------------------------------------------------


class TestToneInference:
    """Test tone inference logic."""

    def test_urgent_tone(self, module: TalkingModule) -> None:
        """Detect urgent tone from keywords."""
        decision = CCDecision(
            summary="Emergency",
            action="flee",
            speaking="Quick, we need help!",
        )
        context = ConversationContext()

        tone = module._infer_tone(decision, context)
        assert tone == "urgent"

    def test_friendly_tone(self, module: TalkingModule) -> None:
        """Detect friendly tone from keywords."""
        decision = CCDecision(
            summary="Thanking",
            action="idle",
            speaking="Thanks for your help!",
        )
        context = ConversationContext()

        tone = module._infer_tone(decision, context)
        assert tone == "friendly"

    def test_polite_tone(self, module: TalkingModule) -> None:
        """Detect polite tone from keywords."""
        decision = CCDecision(
            summary="Requesting",
            action="idle",
            speaking="Could you please help me?",
        )
        context = ConversationContext()

        tone = module._infer_tone(decision, context)
        assert tone == "polite"

    def test_friendly_tone_with_participants(self, module: TalkingModule) -> None:
        """Default to friendly when talking to someone."""
        decision = CCDecision(
            summary="Chatting",
            action="idle",
            speaking="Let's work together",
        )
        context = ConversationContext(participants=["player1"])

        tone = module._infer_tone(decision, context)
        assert tone == "friendly"

    def test_neutral_tone_default(self, module: TalkingModule) -> None:
        """Default to neutral when no specific tone detected."""
        decision = CCDecision(
            summary="Stating fact",
            action="idle",
            speaking="The sun is setting",
        )
        context = ConversationContext()

        tone = module._infer_tone(decision, context)
        assert tone == "neutral"


# ---------------------------------------------------------------------------
# Storage Tests
# ---------------------------------------------------------------------------


class TestUtteranceStorage:
    """Test utterance storage in SAS."""

    async def test_store_utterance(
        self, module: TalkingModule, sas: InMemorySAS
    ) -> None:
        """Store utterance in SAS."""
        utterance = Utterance(
            content="Hello world",
            tone="friendly",
            is_response=False,
        )

        await module._store_utterance(sas, utterance)

        section = await sas.get_section("talking")
        assert "latest_utterance" in section
        assert section["latest_utterance"]["content"] == "Hello world"
        assert len(section["recent_utterances"]) == 1

    async def test_store_multiple_utterances(
        self, module: TalkingModule, sas: InMemorySAS
    ) -> None:
        """Store multiple utterances."""
        for i in range(3):
            utterance = Utterance(content=f"Message {i}", tone="neutral")
            await module._store_utterance(sas, utterance)

        section = await sas.get_section("talking")
        assert len(section["recent_utterances"]) == 3
        assert section["latest_utterance"]["content"] == "Message 2"

    async def test_store_limits_history(
        self, module: TalkingModule, sas: InMemorySAS
    ) -> None:
        """Storage should limit to 10 recent utterances."""
        for i in range(15):
            utterance = Utterance(content=f"Message {i}", tone="neutral")
            await module._store_utterance(sas, utterance)

        section = await sas.get_section("talking")
        # Should only keep last 10
        assert len(section["recent_utterances"]) == 10
        # First should be message 5 (0-4 dropped)
        assert section["recent_utterances"][0]["content"] == "Message 5"
        assert section["recent_utterances"][-1]["content"] == "Message 14"


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestTalkingIntegration:
    """Integration tests combining multiple components."""

    async def test_full_utterance_generation_flow(
        self, module: TalkingModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Test full flow: CC decision -> utterance -> storage."""
        # Setup context
        await sas.update_percepts(
            PerceptData(
                chat_messages=[
                    {"username": "player1", "message": "Want to build a house?"},
                ]
            )
        )

        mock_llm.complete.return_value = LLMResponse(
            content="Sure, I'd love to help!",
            model="gpt-4o-mini",
        )

        decision = CCDecision(
            summary="Accepting invitation",
            action="idle",
            speaking="Accept the building invitation enthusiastically",
            reasoning="Player asked for help, and we have time",
        )

        # Generate and store
        utterance = await module._generate_utterance(decision, sas)
        await module._store_utterance(sas, utterance)

        # Verify stored correctly
        section = await sas.get_section("talking")
        assert section["latest_utterance"]["content"] == "Sure, I'd love to help!"
        assert section["latest_utterance"]["is_response"] is True

    async def test_error_handling_flow(
        self, module: TalkingModule, sas: InMemorySAS, mock_llm: AsyncMock
    ) -> None:
        """Test error handling doesn't crash the module."""
        mock_llm.complete.side_effect = Exception("API error")

        decision = CCDecision(
            summary="Speaking",
            action="idle",
            speaking="Say something",
        )

        # Should handle error gracefully
        utterance = await module._generate_utterance(decision, sas)
        await module._store_utterance(sas, utterance)

        # Should store empty utterance
        section = await sas.get_section("talking")
        assert section["latest_utterance"]["content"] == ""
