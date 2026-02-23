"""Tests for the Social Awareness module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from piano.core.types import LLMResponse
from piano.social.awareness import (
    ActivationTrigger,
    SocialAwarenessModule,
    SocialAwarenessOutput,
    SocialSignal,
)
from tests.helpers import InMemorySAS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Create a mock LLM provider."""
    llm = AsyncMock()
    # Default response
    llm.complete.return_value = LLMResponse(
        content=json.dumps(
            {
                "sentiment_updates": {"player1": 0.5},
                "inferred_intents": {"player1": "seeking collaboration"},
                "social_signals": [
                    {
                        "signal_type": "opportunity",
                        "description": "Player wants to collaborate",
                        "target_agent_id": "player1",
                        "urgency": 0.6,
                    }
                ],
                "interaction_strategy": "Friendly approach recommended",
            }
        ),
        model="gpt-4o-mini",
    )
    return llm


@pytest.fixture
def social_module(mock_llm: AsyncMock) -> SocialAwarenessModule:
    """Create a social awareness module with mock LLM."""
    return SocialAwarenessModule(
        llm_provider=mock_llm,
        proximity_threshold=16.0,
        social_goal_interval=7.5,
    )


@pytest.fixture
def sas() -> InMemorySAS:
    """Create an in-memory SAS for testing."""
    return InMemorySAS(agent_id="test-agent")


# ---------------------------------------------------------------------------
# Module Metadata Tests
# ---------------------------------------------------------------------------


async def test_module_name(social_module: SocialAwarenessModule) -> None:
    """Module has correct name."""
    assert social_module.name == "social_awareness"


async def test_module_tier(social_module: SocialAwarenessModule) -> None:
    """Module is in MID tier."""
    from piano.core.types import ModuleTier

    assert social_module.tier == ModuleTier.MID


# ---------------------------------------------------------------------------
# Activation Condition Tests
# ---------------------------------------------------------------------------


async def test_activation_on_conversation(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
) -> None:
    """Module activates when chat messages are present."""
    # Set up conversation
    percepts = await sas.get_percepts()
    percepts.chat_messages = [
        {"username": "player1", "message": "Hello!"},
    ]
    await sas.update_percepts(percepts)

    result = await social_module.tick(sas)

    assert result.success
    assert result.data["activated"] is True
    assert result.data["trigger"] == ActivationTrigger.CONVERSATION_ONGOING


async def test_activation_on_nearby_agents(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
) -> None:
    """Module activates when nearby players are detected."""
    # Set up nearby players
    percepts = await sas.get_percepts()
    percepts.nearby_players = ["player1", "player2"]
    await sas.update_percepts(percepts)

    result = await social_module.tick(sas)

    assert result.success
    assert result.data["activated"] is True
    assert result.data["trigger"] == ActivationTrigger.NEARBY_AGENT


async def test_activation_on_social_goal_cycle(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
) -> None:
    """Module activates after social goal interval elapsed."""
    import time

    # Set last activation time to past
    social_module._last_activation_time = time.time() - 10.0  # 10 seconds ago

    # No conversation, no nearby players
    result = await social_module.tick(sas)

    assert result.success
    assert result.data["activated"] is True
    assert result.data["trigger"] == ActivationTrigger.SOCIAL_GOAL_CYCLE


async def test_no_activation_when_alone_and_timer_not_elapsed(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
) -> None:
    """Module does not activate when no triggers are present."""
    import time

    # Set last activation to recent
    social_module._last_activation_time = time.time() - 2.0  # 2 seconds ago (< 7.5s interval)

    # No conversation, no nearby players
    result = await social_module.tick(sas)

    assert result.success
    assert result.data["activated"] is False
    assert result.data["reason"] == "no_trigger"


# ---------------------------------------------------------------------------
# LLM Inference Tests
# ---------------------------------------------------------------------------


async def test_sentiment_update_from_conversation(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
    mock_llm: AsyncMock,
) -> None:
    """Module updates sentiment based on conversation."""
    # Set up conversation
    percepts = await sas.get_percepts()
    percepts.chat_messages = [
        {"username": "player1", "message": "Thanks for helping!"},
    ]
    await sas.update_percepts(percepts)

    # Configure mock to return positive sentiment
    mock_llm.complete.return_value = LLMResponse(
        content=json.dumps(
            {
                "sentiment_updates": {"player1": 0.8},
                "inferred_intents": {},
                "social_signals": [],
                "interaction_strategy": "",
            }
        ),
        model="gpt-4o-mini",
    )

    result = await social_module.tick(sas)

    assert result.success
    assert result.data["activated"] is True
    assert "player1" in result.data["sentiment_updates"]
    assert result.data["sentiment_updates"]["player1"] == 0.8

    # Check SAS was updated
    social = await sas.get_social()
    assert "player1" in social.relationships
    # New sentiment is blended: 0.6 * 0.8 + 0.4 * 0.0 = 0.48
    assert abs(social.relationships["player1"] - 0.48) < 0.01


async def test_intent_inference(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
    mock_llm: AsyncMock,
) -> None:
    """Module infers other agents' intents."""
    # Set up nearby player
    percepts = await sas.get_percepts()
    percepts.nearby_players = ["player1"]
    await sas.update_percepts(percepts)

    # Configure mock to return intent
    mock_llm.complete.return_value = LLMResponse(
        content=json.dumps(
            {
                "sentiment_updates": {},
                "inferred_intents": {"player1": "gathering resources"},
                "social_signals": [],
                "interaction_strategy": "",
            }
        ),
        model="gpt-4o-mini",
    )

    result = await social_module.tick(sas)

    assert result.success
    assert result.data["inferred_intents"]["player1"] == "gathering resources"


async def test_social_signal_generation(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
    mock_llm: AsyncMock,
) -> None:
    """Module generates social signals for goal generation."""
    # Set up conversation
    percepts = await sas.get_percepts()
    percepts.chat_messages = [
        {"username": "player1", "message": "Can anyone help me build?"},
    ]
    await sas.update_percepts(percepts)

    # Configure mock to return help signal
    mock_llm.complete.return_value = LLMResponse(
        content=json.dumps(
            {
                "sentiment_updates": {},
                "inferred_intents": {},
                "social_signals": [
                    {
                        "signal_type": "help_needed",
                        "description": "Player1 needs building assistance",
                        "target_agent_id": "player1",
                        "urgency": 0.8,
                    }
                ],
                "interaction_strategy": "",
            }
        ),
        model="gpt-4o-mini",
    )

    result = await social_module.tick(sas)

    assert result.success
    assert len(result.data["social_signals"]) == 1
    signal = result.data["social_signals"][0]
    assert signal["signal_type"] == "help_needed"
    assert signal["target_agent_id"] == "player1"
    assert signal["urgency"] == 0.8


async def test_interaction_strategy_recommendation(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
    mock_llm: AsyncMock,
) -> None:
    """Module recommends interaction strategies."""
    # Set up nearby player
    percepts = await sas.get_percepts()
    percepts.nearby_players = ["player1"]
    await sas.update_percepts(percepts)

    # Configure mock to return strategy
    mock_llm.complete.return_value = LLMResponse(
        content=json.dumps(
            {
                "sentiment_updates": {},
                "inferred_intents": {},
                "social_signals": [],
                "interaction_strategy": "Greet player1 and offer to trade",
            }
        ),
        model="gpt-4o-mini",
    )

    result = await social_module.tick(sas)

    assert result.success
    assert result.data["interaction_strategy"] == "Greet player1 and offer to trade"


# ---------------------------------------------------------------------------
# SAS Update Tests
# ---------------------------------------------------------------------------


async def test_sentiment_blending_with_existing(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
    mock_llm: AsyncMock,
) -> None:
    """Module blends new sentiment with existing relationships."""
    # Set up existing relationship
    social = await sas.get_social()
    social.relationships["player1"] = 0.2
    await sas.update_social(social)

    # Set up trigger
    percepts = await sas.get_percepts()
    percepts.nearby_players = ["player1"]
    await sas.update_percepts(percepts)

    # New sentiment: 0.8
    mock_llm.complete.return_value = LLMResponse(
        content=json.dumps(
            {
                "sentiment_updates": {"player1": 0.8},
                "inferred_intents": {},
                "social_signals": [],
                "interaction_strategy": "",
            }
        ),
        model="gpt-4o-mini",
    )

    await social_module.tick(sas)

    # Check blended value: 0.6 * 0.8 + 0.4 * 0.2 = 0.56
    social = await sas.get_social()
    assert abs(social.relationships["player1"] - 0.56) < 0.01


async def test_interaction_history_storage(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
) -> None:
    """Module stores interaction records in SAS."""
    # Set up trigger
    percepts = await sas.get_percepts()
    percepts.nearby_players = ["player1"]
    await sas.update_percepts(percepts)

    await social_module.tick(sas)

    # Check interaction was recorded
    social = await sas.get_social()
    assert len(social.recent_interactions) == 1
    interaction = social.recent_interactions[0]
    assert "inferred_intents" in interaction
    assert "social_signals" in interaction
    assert "interaction_strategy" in interaction


async def test_interaction_history_limit(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
) -> None:
    """Module limits interaction history to 10 entries."""
    # Pre-fill with 10 interactions
    social = await sas.get_social()
    social.recent_interactions = [{"id": i} for i in range(10)]
    await sas.update_social(social)

    # Trigger new interaction
    percepts = await sas.get_percepts()
    percepts.nearby_players = ["player1"]
    await sas.update_percepts(percepts)

    await social_module.tick(sas)

    # Should still have 10 entries (oldest dropped)
    social = await sas.get_social()
    assert len(social.recent_interactions) == 10


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


async def test_error_handling_on_llm_failure(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
    mock_llm: AsyncMock,
) -> None:
    """Module handles LLM failures gracefully."""
    # Set up trigger
    percepts = await sas.get_percepts()
    percepts.nearby_players = ["player1"]
    await sas.update_percepts(percepts)

    # Configure mock to raise error
    mock_llm.complete.side_effect = Exception("LLM service unavailable")

    result = await social_module.tick(sas)

    assert not result.success
    assert result.error == "LLM service unavailable"


async def test_error_handling_on_invalid_json(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
    mock_llm: AsyncMock,
) -> None:
    """Module handles invalid JSON responses gracefully."""
    # Set up trigger
    percepts = await sas.get_percepts()
    percepts.nearby_players = ["player1"]
    await sas.update_percepts(percepts)

    # Configure mock to return invalid JSON
    mock_llm.complete.return_value = LLMResponse(
        content="This is not valid JSON",
        model="gpt-4o-mini",
    )

    result = await social_module.tick(sas)

    # Should succeed but with empty output
    assert result.success
    assert result.data["activated"] is True
    assert result.data["sentiment_updates"] == {}
    assert result.data["inferred_intents"] == {}


# ---------------------------------------------------------------------------
# Pydantic Model Tests
# ---------------------------------------------------------------------------


def test_social_signal_creation() -> None:
    """SocialSignal model can be created."""
    signal = SocialSignal(
        signal_type="opportunity",
        description="Test signal",
        target_agent_id="player1",
        urgency=0.7,
    )
    assert signal.signal_type == "opportunity"
    assert signal.urgency == 0.7


def test_social_signal_defaults() -> None:
    """SocialSignal has correct defaults."""
    signal = SocialSignal(
        signal_type="test",
        description="Test",
    )
    assert signal.target_agent_id is None
    assert signal.urgency == 0.5


def test_social_awareness_output_creation() -> None:
    """SocialAwarenessOutput model can be created."""
    output = SocialAwarenessOutput(
        sentiment_updates={"player1": 0.5},
        inferred_intents={"player1": "test"},
        social_signals=[SocialSignal(signal_type="test", description="Test signal")],
        interaction_strategy="Test strategy",
    )
    assert len(output.sentiment_updates) == 1
    assert len(output.social_signals) == 1


def test_social_awareness_output_defaults() -> None:
    """SocialAwarenessOutput has correct defaults."""
    output = SocialAwarenessOutput()
    assert output.sentiment_updates == {}
    assert output.inferred_intents == {}
    assert output.social_signals == []
    assert output.interaction_strategy == ""


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


async def test_first_activation_with_no_last_time(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
) -> None:
    """Module handles first activation correctly."""
    # No last activation time set
    assert social_module._last_activation_time == 0.0

    # Should activate on social goal cycle (elapsed > interval)
    result = await social_module.tick(sas)

    assert result.success
    assert result.data["activated"] is True
    assert social_module._last_activation_time > 0.0


async def test_multiple_activation_triggers_prefer_conversation(
    social_module: SocialAwarenessModule,
    sas: InMemorySAS,
) -> None:
    """Module prefers conversation trigger when multiple triggers present."""
    import time

    # Set all triggers
    percepts = await sas.get_percepts()
    percepts.chat_messages = [{"username": "player1", "message": "Hi"}]
    percepts.nearby_players = ["player1"]
    await sas.update_percepts(percepts)

    social_module._last_activation_time = time.time() - 10.0  # Timer also elapsed

    result = await social_module.tick(sas)

    # Should trigger on conversation (checked first)
    assert result.data["trigger"] == ActivationTrigger.CONVERSATION_ONGOING
