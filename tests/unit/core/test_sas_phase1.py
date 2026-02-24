"""Tests for Phase 1 SAS extension mixin.

Tests the SASPhase1Mixin helper that provides typed access to Phase 1
sections (personality, emotions, social graph, LTM stats, talking state,
checkpoint metadata) using the existing get_section/update_section interface.
"""

from __future__ import annotations

import pytest

from piano.core.sas_phase1 import SASPhase1Mixin
from tests.helpers import InMemorySAS


@pytest.fixture
def sas() -> InMemorySAS:
    """Create a test SAS instance."""
    return InMemorySAS(agent_id="test-agent-phase1")


@pytest.fixture
def phase1(sas: InMemorySAS) -> SASPhase1Mixin:
    """Create a Phase 1 mixin wrapping the test SAS."""
    return SASPhase1Mixin(sas)


class TestPersonality:
    """Tests for personality trait access."""

    async def test_get_personality_default_empty(self, phase1: SASPhase1Mixin) -> None:
        """Unset personality returns empty dict."""
        traits = await phase1.get_personality()
        assert traits == {}

    async def test_update_and_get_personality_round_trip(self, phase1: SASPhase1Mixin) -> None:
        """Personality write and read preserves data."""
        original = {
            "openness": 0.8,
            "conscientiousness": 0.6,
            "extraversion": 0.7,
            "agreeableness": 0.5,
            "neuroticism": 0.3,
        }
        await phase1.update_personality(original)
        result = await phase1.get_personality()
        assert result == original

    async def test_update_personality_overwrites(self, phase1: SASPhase1Mixin) -> None:
        """Second update replaces previous personality data."""
        await phase1.update_personality({"openness": 0.5})
        await phase1.update_personality({"extraversion": 0.9})
        result = await phase1.get_personality()
        assert result == {"extraversion": 0.9}


class TestEmotions:
    """Tests for emotion state access."""

    async def test_get_emotion_state_default_empty(self, phase1: SASPhase1Mixin) -> None:
        """Unset emotion state returns empty dict."""
        state = await phase1.get_emotion_state()
        assert state == {}

    async def test_update_and_get_emotion_state_round_trip(self, phase1: SASPhase1Mixin) -> None:
        """Emotion state write and read preserves data."""
        original = {
            "valence": 0.5,
            "arousal": 0.3,
            "dominance": 0.7,
            "metadata": {"source": "event_trigger", "timestamp": 12345},
        }
        await phase1.update_emotion_state(original)
        result = await phase1.get_emotion_state()
        assert result == original


class TestSocialGraph:
    """Tests for social graph snapshot access."""

    async def test_get_social_graph_snapshot_default_empty(self, phase1: SASPhase1Mixin) -> None:
        """Unset social graph returns empty dict."""
        graph = await phase1.get_social_graph_snapshot()
        assert graph == {}

    async def test_update_and_get_social_graph_round_trip(self, phase1: SASPhase1Mixin) -> None:
        """Social graph write and read preserves data."""
        original = {
            "nodes": ["agent-001", "agent-002", "agent-003"],
            "edges": [
                {"from": "agent-001", "to": "agent-002", "trust": 0.8},
                {"from": "agent-001", "to": "agent-003", "trust": 0.5},
            ],
            "timestamp": 67890,
        }
        await phase1.update_social_graph_snapshot(original)
        result = await phase1.get_social_graph_snapshot()
        assert result == original


class TestLTMStats:
    """Tests for LTM statistics access."""

    async def test_get_ltm_stats_default_empty(self, phase1: SASPhase1Mixin) -> None:
        """Unset LTM stats returns empty dict."""
        stats = await phase1.get_ltm_stats()
        assert stats == {}

    async def test_update_and_get_ltm_stats_round_trip(self, phase1: SASPhase1Mixin) -> None:
        """LTM stats write and read preserves data."""
        original = {
            "vector_count": 1234,
            "last_sync_time": 9876543210,
            "avg_similarity": 0.72,
            "collections": {"episodic": 800, "semantic": 434},
        }
        await phase1.update_ltm_stats(original)
        result = await phase1.get_ltm_stats()
        assert result == original


class TestTalkingState:
    """Tests for talking/conversation state access."""

    async def test_get_talking_state_default_empty(self, phase1: SASPhase1Mixin) -> None:
        """Unset talking state returns empty dict."""
        state = await phase1.get_talking_state()
        assert state == {}

    async def test_update_and_get_talking_state_round_trip(self, phase1: SASPhase1Mixin) -> None:
        """Talking state write and read preserves data."""
        original = {
            "pending_utterances": [
                {"text": "Hello!", "target": "agent-002", "priority": 1},
                {"text": "How are you?", "target": "agent-002", "priority": 2},
            ],
            "conversation_context": {
                "active_topic": "mining",
                "last_speaker": "agent-002",
                "turn_count": 5,
            },
        }
        await phase1.update_talking_state(original)
        result = await phase1.get_talking_state()
        assert result == original


class TestCheckpointMetadata:
    """Tests for checkpoint metadata access."""

    async def test_get_checkpoint_metadata_default_empty(self, phase1: SASPhase1Mixin) -> None:
        """Unset checkpoint metadata returns empty dict."""
        meta = await phase1.get_checkpoint_metadata()
        assert meta == {}

    async def test_update_and_get_checkpoint_metadata_round_trip(
        self, phase1: SASPhase1Mixin
    ) -> None:
        """Checkpoint metadata write and read preserves data."""
        original = {
            "version": "1.0.0",
            "timestamp": 1234567890,
            "agent_id": "test-agent-phase1",
            "total_ticks": 5000,
        }
        await phase1.update_checkpoint_metadata(original)
        result = await phase1.get_checkpoint_metadata()
        assert result == original


class TestSectionIndependence:
    """Tests that different sections don't interfere with each other."""

    async def test_multiple_sections_are_independent(self, phase1: SASPhase1Mixin) -> None:
        """Writing to one section doesn't affect others."""
        # Write to multiple sections
        await phase1.update_personality({"openness": 0.9})
        await phase1.update_emotion_state({"valence": 0.2})
        await phase1.update_social_graph_snapshot({"nodes": ["a", "b"]})
        await phase1.update_ltm_stats({"count": 100})
        await phase1.update_talking_state({"pending": []})
        await phase1.update_checkpoint_metadata({"version": "1.0"})

        # Verify each section retained its data
        assert await phase1.get_personality() == {"openness": 0.9}
        assert await phase1.get_emotion_state() == {"valence": 0.2}
        assert await phase1.get_social_graph_snapshot() == {"nodes": ["a", "b"]}
        assert await phase1.get_ltm_stats() == {"count": 100}
        assert await phase1.get_talking_state() == {"pending": []}
        assert await phase1.get_checkpoint_metadata() == {"version": "1.0"}


class TestMixinWrapsAnySAS:
    """Tests that mixin works with any SharedAgentState implementation."""

    async def test_mixin_wraps_inmemory_sas(self, sas: InMemorySAS) -> None:
        """Mixin can wrap InMemorySAS."""
        mixin = SASPhase1Mixin(sas)
        assert mixin.sas is sas

        # Verify it works
        await mixin.update_personality({"test": 0.5})
        result = await mixin.get_personality()
        assert result == {"test": 0.5}

    async def test_mixin_provides_sas_access(self, phase1: SASPhase1Mixin) -> None:
        """Mixin provides access to underlying SAS."""
        underlying_sas = phase1.sas
        assert isinstance(underlying_sas, InMemorySAS)
        assert underlying_sas.agent_id == "test-agent-phase1"


class TestDataTypes:
    """Tests for data type handling."""

    async def test_personality_converts_to_float(self, phase1: SASPhase1Mixin) -> None:
        """Personality trait values are converted to float."""
        # Store with mixed numeric types
        await phase1.update_personality({"a": 1, "b": 0.5, "c": 0})
        result = await phase1.get_personality()

        # All values should be float
        assert isinstance(result["a"], float)
        assert isinstance(result["b"], float)
        assert isinstance(result["c"], float)
        assert result == {"a": 1.0, "b": 0.5, "c": 0.0}

    async def test_personality_skips_invalid_float_values(self, phase1: SASPhase1Mixin) -> None:
        """Personality values that cannot be converted to float should be skipped."""
        await phase1.update_personality(
            {"openness": 0.8, "bad_value": "not_a_number", "valid": 0.5}
        )
        result = await phase1.get_personality()

        # Non-numeric values should be silently skipped
        assert result == {"openness": 0.8, "valid": 0.5}
        assert "bad_value" not in result

    async def test_personality_handles_none_values(self, phase1: SASPhase1Mixin) -> None:
        """Personality values that are None should be skipped."""
        await phase1.update_personality({"openness": 0.8, "bad": None})
        result = await phase1.get_personality()
        assert result == {"openness": 0.8}
        assert "bad" not in result

    async def test_emotion_state_preserves_nested_structure(self, phase1: SASPhase1Mixin) -> None:
        """Emotion state preserves nested dictionaries and lists."""
        original = {
            "emotions": [
                {"name": "joy", "intensity": 0.8},
                {"name": "curiosity", "intensity": 0.6},
            ],
            "metadata": {"trigger": "event_123", "decay_rate": 0.1},
        }
        await phase1.update_emotion_state(original)
        result = await phase1.get_emotion_state()
        assert result == original


class TestIntegrationWithCoreSAS:
    """Tests integration with core SAS sections."""

    async def test_phase1_sections_separate_from_core_sections(
        self, sas: InMemorySAS, phase1: SASPhase1Mixin
    ) -> None:
        """Phase 1 sections are independent from core SAS sections."""
        # Write to core SAS sections
        from piano.core.types import GoalData, PerceptData

        await sas.update_percepts(PerceptData(position={"x": 10.0, "y": 20.0, "z": 30.0}))
        await sas.update_goals(GoalData(current_goal="explore"))

        # Write to Phase 1 sections
        await phase1.update_personality({"openness": 0.8})
        await phase1.update_emotion_state({"valence": 0.5})

        # Verify core sections unchanged
        percepts = await sas.get_percepts()
        goals = await sas.get_goals()
        assert percepts.position == {"x": 10.0, "y": 20.0, "z": 30.0}
        assert goals.current_goal == "explore"

        # Verify Phase 1 sections correct
        personality = await phase1.get_personality()
        emotions = await phase1.get_emotion_state()
        assert personality == {"openness": 0.8}
        assert emotions == {"valence": 0.5}
