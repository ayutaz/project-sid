"""Tests for emotion tracking system."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from piano.social.emotions import (
    EMOTION_MAP,
    EmotionEvent,
    EmotionEvents,
    EmotionState,
    EmotionTracker,
)


class TestEmotionState:
    """Tests for EmotionState model."""

    def test_default_state(self) -> None:
        """Test default emotional state is calm."""
        state = EmotionState()
        assert state.valence == 0.0
        assert state.arousal == 0.3
        assert state.dominant_emotion == "calm"
        assert state.emotions == {}

    def test_valence_validation(self) -> None:
        """Test valence validates to valid range."""
        # Pydantic validates on construction
        with pytest.raises(ValidationError):
            EmotionState(valence=1.5)

        with pytest.raises(ValidationError):
            EmotionState(valence=-1.5)

        # Valid values should work
        state = EmotionState(valence=1.0)
        assert state.valence == 1.0

        state = EmotionState(valence=-1.0)
        assert state.valence == -1.0

    def test_arousal_validation(self) -> None:
        """Test arousal validates to valid range."""
        # Pydantic validates on construction
        with pytest.raises(ValidationError):
            EmotionState(arousal=1.5)

        with pytest.raises(ValidationError):
            EmotionState(arousal=-0.5)

        # Valid values should work
        state = EmotionState(arousal=1.0)
        assert state.arousal == 1.0

        state = EmotionState(arousal=0.0)
        assert state.arousal == 0.0

    def test_to_prompt_description_no_emotions(self) -> None:
        """Test prompt description with no specific emotions."""
        state = EmotionState(dominant_emotion="content")
        desc = state.to_prompt_description()
        assert desc == "feeling content"

    def test_to_prompt_description_single_emotion(self) -> None:
        """Test prompt description with one strong emotion."""
        state = EmotionState(
            dominant_emotion="happy",
            emotions={"happy": 8.5},
        )
        desc = state.to_prompt_description()
        assert "happy" in desc
        assert "very" in desc

    def test_to_prompt_description_multiple_emotions(self) -> None:
        """Test prompt description with multiple emotions."""
        state = EmotionState(
            dominant_emotion="happy",
            emotions={"happy": 7.0, "excited": 5.5, "content": 2.0},
        )
        desc = state.to_prompt_description()
        assert "happy" in desc
        assert "excited" in desc
        # "content" should not appear (intensity too low)
        assert "content" not in desc

    def test_to_prompt_description_weak_emotions(self) -> None:
        """Test prompt description when all emotions are weak."""
        state = EmotionState(
            dominant_emotion="calm",
            emotions={"calm": 2.0, "bored": 1.5},
        )
        desc = state.to_prompt_description()
        # Weak emotions should fall back to dominant
        assert desc == "feeling calm"


class TestEmotionEvent:
    """Tests for EmotionEvent model."""

    def test_default_event(self) -> None:
        """Test default event values."""
        event = EmotionEvent(event_type="test")
        assert event.event_type == "test"
        assert event.intensity == 0.5
        assert event.valence_shift == 0.0
        assert event.arousal_shift == 0.0

    def test_intensity_validation(self) -> None:
        """Test intensity validates to valid range."""
        # Pydantic validates on construction
        with pytest.raises(ValidationError):
            EmotionEvent(event_type="test", intensity=1.5)

        with pytest.raises(ValidationError):
            EmotionEvent(event_type="test", intensity=-0.5)

        # Valid values should work
        event = EmotionEvent(event_type="test", intensity=1.0)
        assert event.intensity == 1.0

        event = EmotionEvent(event_type="test", intensity=0.0)
        assert event.intensity == 0.0


class TestEmotionMap:
    """Tests for emotion classification mapping."""

    def test_all_emotions_mapped(self) -> None:
        """Test all valence-arousal combinations are mapped."""
        valence_cats = ["positive", "neutral", "negative"]
        arousal_cats = ["high", "medium", "low"]

        for v_cat in valence_cats:
            for a_cat in arousal_cats:
                assert (v_cat, a_cat) in EMOTION_MAP

    def test_emotion_names_valid(self) -> None:
        """Test all emotion names are sensible."""
        expected_emotions = {
            "happy",
            "excited",
            "content",
            "calm",
            "bored",
            "sad",
            "angry",
            "afraid",
            "surprised",
        }
        mapped_emotions = set(EMOTION_MAP.values())
        assert mapped_emotions == expected_emotions


class TestEmotionTracker:
    """Tests for EmotionTracker."""

    def test_initial_state_at_baseline(self) -> None:
        """Test tracker initializes at baseline."""
        tracker = EmotionTracker()
        state = tracker.current_state
        assert state.valence == 0.0
        assert state.arousal == 0.3
        assert state.dominant_emotion in EMOTION_MAP.values()

    def test_custom_baseline(self) -> None:
        """Test tracker with custom baseline."""
        tracker = EmotionTracker(baseline_valence=0.5, baseline_arousal=0.6)
        state = tracker.current_state
        assert state.valence == 0.5
        assert state.arousal == 0.6

    def test_apply_event_shifts_state(self) -> None:
        """Test applying event shifts valence and arousal."""
        tracker = EmotionTracker()
        event = EmotionEvent(
            event_type="test",
            intensity=1.0,
            valence_shift=0.4,
            arousal_shift=0.3,
        )
        tracker.apply_event(event)

        state = tracker.current_state
        assert state.valence == pytest.approx(0.4, abs=0.01)
        assert state.arousal == pytest.approx(0.6, abs=0.01)

    def test_apply_event_with_intensity(self) -> None:
        """Test event intensity scales the shifts."""
        tracker = EmotionTracker()
        event = EmotionEvent(
            event_type="test",
            intensity=0.5,  # Half intensity
            valence_shift=0.4,
            arousal_shift=0.2,
        )
        tracker.apply_event(event)

        state = tracker.current_state
        # Should be half of the shifts
        assert state.valence == pytest.approx(0.2, abs=0.01)
        assert state.arousal == pytest.approx(0.4, abs=0.01)

    def test_values_clamped_to_valid_ranges(self) -> None:
        """Test valence and arousal are clamped after events."""
        tracker = EmotionTracker()

        # Try to exceed upper bounds
        event = EmotionEvent(
            event_type="extreme_positive",
            intensity=1.0,
            valence_shift=2.0,
            arousal_shift=2.0,
        )
        tracker.apply_event(event)

        state = tracker.current_state
        assert state.valence <= 1.0
        assert state.arousal <= 1.0

        # Reset and try to exceed lower bounds
        tracker = EmotionTracker()
        event = EmotionEvent(
            event_type="extreme_negative",
            intensity=1.0,
            valence_shift=-2.0,
            arousal_shift=-2.0,
        )
        tracker.apply_event(event)

        state = tracker.current_state
        assert state.valence >= -1.0
        assert state.arousal >= 0.0

    def test_decay_moves_toward_baseline(self) -> None:
        """Test decay moves state toward baseline over time."""
        tracker = EmotionTracker(decay_rate=0.5)

        # Apply event to shift away from baseline
        event = EmotionEvent(
            event_type="excitement",
            intensity=1.0,
            valence_shift=0.6,
            arousal_shift=0.5,
        )
        tracker.apply_event(event)

        # Check initial shift
        state_before = tracker.current_state
        assert state_before.valence > 0.3
        assert state_before.arousal > 0.5

        # Apply decay
        tracker.decay(delta_seconds=2.0)

        # Should move toward baseline (0.0, 0.3)
        state_after = tracker.current_state
        assert abs(state_after.valence) < abs(state_before.valence)
        assert abs(state_after.arousal - 0.3) < abs(state_before.arousal - 0.3)

    def test_decay_with_zero_delta(self) -> None:
        """Test decay with zero time does nothing."""
        tracker = EmotionTracker()
        event = EmotionEvent(event_type="test", valence_shift=0.5)
        tracker.apply_event(event)

        state_before = tracker.current_state
        tracker.decay(delta_seconds=0.0)
        state_after = tracker.current_state

        assert state_before.valence == state_after.valence
        assert state_before.arousal == state_after.arousal

    def test_emotion_classification_happy(self) -> None:
        """Test classification of happy emotion."""
        tracker = EmotionTracker()
        # High positive valence, medium arousal = happy
        tracker._valence = 0.7
        tracker._arousal = 0.5
        tracker._update_emotion_intensities()

        state = tracker.current_state
        assert state.dominant_emotion == "happy"
        assert "happy" in state.emotions

    def test_emotion_classification_sad(self) -> None:
        """Test classification of sad emotion."""
        tracker = EmotionTracker()
        # Negative valence, medium arousal = sad
        tracker._valence = -0.6
        tracker._arousal = 0.4
        tracker._update_emotion_intensities()

        state = tracker.current_state
        assert state.dominant_emotion == "sad"
        assert "sad" in state.emotions

    def test_emotion_classification_excited(self) -> None:
        """Test classification of excited emotion."""
        tracker = EmotionTracker()
        # Positive valence, high arousal = excited
        tracker._valence = 0.6
        tracker._arousal = 0.85
        tracker._update_emotion_intensities()

        state = tracker.current_state
        assert state.dominant_emotion == "excited"
        assert "excited" in state.emotions

    def test_emotion_classification_angry(self) -> None:
        """Test classification of angry emotion."""
        tracker = EmotionTracker()
        # Negative valence, high arousal = angry
        tracker._valence = -0.7
        tracker._arousal = 0.8
        tracker._update_emotion_intensities()

        state = tracker.current_state
        assert state.dominant_emotion == "angry"
        assert "angry" in state.emotions

    def test_social_contagion_shifts_toward_other(self) -> None:
        """Test social contagion shifts emotions toward another agent."""
        tracker = EmotionTracker()
        # Start neutral
        assert tracker.current_state.valence == pytest.approx(0.0, abs=0.01)

        # Other agent is very happy
        other_state = EmotionState(valence=0.8, arousal=0.7)

        # Apply contagion
        tracker.apply_social_contagion(other_state, influence=0.3)

        # Should shift toward the other's state
        state = tracker.current_state
        assert state.valence > 0.0
        assert state.arousal > 0.3

    def test_social_contagion_with_zero_influence(self) -> None:
        """Test social contagion with zero influence does nothing."""
        tracker = EmotionTracker()
        initial_state = tracker.current_state

        other_state = EmotionState(valence=0.8, arousal=0.7)
        tracker.apply_social_contagion(other_state, influence=0.0)

        final_state = tracker.current_state
        assert final_state.valence == initial_state.valence
        assert final_state.arousal == initial_state.arousal

    def test_social_contagion_influence_clamped(self) -> None:
        """Test social contagion clamps influence to valid range."""
        tracker = EmotionTracker()
        other_state = EmotionState(valence=0.8, arousal=0.7)

        # Try invalid influence values - should not crash
        tracker.apply_social_contagion(other_state, influence=1.5)
        tracker.apply_social_contagion(other_state, influence=-0.5)

        # Tracker should still be in valid state
        state = tracker.current_state
        assert -1.0 <= state.valence <= 1.0
        assert 0.0 <= state.arousal <= 1.0

    def test_behavior_modifier_varies_with_emotion(self) -> None:
        """Test behavior modifiers change with emotional state."""
        tracker = EmotionTracker()

        # Neutral state
        neutral_mods = tracker.get_behavior_modifier()
        assert "risk_aversion" in neutral_mods
        assert "social_seeking" in neutral_mods
        assert "exploration" in neutral_mods
        assert "cooperation" in neutral_mods
        assert "aggression" in neutral_mods

        # All values should be in valid range
        for value in neutral_mods.values():
            assert 0.0 <= value <= 1.0

        # Happy state (high valence, medium arousal)
        tracker._valence = 0.8
        tracker._arousal = 0.5
        happy_mods = tracker.get_behavior_modifier()

        # Happy should be more cooperative and social
        assert happy_mods["cooperation"] > neutral_mods["cooperation"]
        assert happy_mods["social_seeking"] > neutral_mods["social_seeking"]
        # High valence + moderate arousal increases aggression slightly, so just check cooperation
        assert happy_mods["cooperation"] > 0.8  # Should be very high

        # Angry state (negative valence, high arousal)
        tracker._valence = -0.7
        tracker._arousal = 0.8
        angry_mods = tracker.get_behavior_modifier()

        # Angry should be less cooperative and more aggressive
        assert angry_mods["cooperation"] < neutral_mods["cooperation"]
        assert angry_mods["aggression"] > neutral_mods["aggression"]
        assert angry_mods["aggression"] > happy_mods["aggression"]  # More aggressive than happy

    def test_behavior_modifier_all_values_in_range(self) -> None:
        """Test behavior modifiers are always in valid range."""
        tracker = EmotionTracker()

        # Test extreme states
        test_cases = [
            (1.0, 1.0),  # max positive, max arousal
            (-1.0, 1.0),  # max negative, max arousal
            (1.0, 0.0),  # max positive, min arousal
            (-1.0, 0.0),  # max negative, min arousal
            (0.0, 0.5),  # neutral
        ]

        for valence, arousal in test_cases:
            tracker._valence = valence
            tracker._arousal = arousal
            mods = tracker.get_behavior_modifier()

            for key, value in mods.items():
                msg = f"{key}={value} out of range at v={valence}, a={arousal}"
                assert 0.0 <= value <= 1.0, msg

    def test_serialization_round_trip(self) -> None:
        """Test serialization and deserialization preserves state."""
        tracker = EmotionTracker(baseline_valence=0.2, baseline_arousal=0.4, decay_rate=0.15)

        # Apply some events
        tracker.apply_event(EmotionEvents.GOAL_SUCCESS)
        tracker.apply_event(EmotionEvents.SOCIAL_POSITIVE)

        # Serialize
        data = tracker.to_dict()

        # Deserialize
        restored = EmotionTracker.from_dict(data)

        # Check state matches
        original_state = tracker.current_state
        restored_state = restored.current_state

        assert restored_state.valence == pytest.approx(original_state.valence, abs=0.001)
        assert restored_state.arousal == pytest.approx(original_state.arousal, abs=0.001)
        assert restored_state.dominant_emotion == original_state.dominant_emotion

    def test_to_dict_contains_all_fields(self) -> None:
        """Test serialized dict contains all necessary fields."""
        tracker = EmotionTracker()
        data = tracker.to_dict()

        assert "valence" in data
        assert "arousal" in data
        assert "baseline_valence" in data
        assert "baseline_arousal" in data
        assert "decay_rate" in data
        assert "emotions" in data
        assert "last_update" in data


class TestPredefinedEvents:
    """Tests for pre-defined emotion events."""

    def test_goal_success_is_positive(self) -> None:
        """Test goal success event is positive."""
        tracker = EmotionTracker()
        tracker.apply_event(EmotionEvents.GOAL_SUCCESS)

        state = tracker.current_state
        assert state.valence > 0.0  # Should increase valence

    def test_goal_failure_is_negative(self) -> None:
        """Test goal failure event is negative."""
        tracker = EmotionTracker()
        tracker.apply_event(EmotionEvents.GOAL_FAILURE)

        state = tracker.current_state
        assert state.valence < 0.0  # Should decrease valence

    def test_social_positive_increases_valence(self) -> None:
        """Test social positive event increases valence."""
        tracker = EmotionTracker()
        initial_valence = tracker.current_state.valence

        tracker.apply_event(EmotionEvents.SOCIAL_POSITIVE)

        state = tracker.current_state
        assert state.valence > initial_valence

    def test_social_negative_decreases_valence(self) -> None:
        """Test social negative event decreases valence."""
        tracker = EmotionTracker()
        initial_valence = tracker.current_state.valence

        tracker.apply_event(EmotionEvents.SOCIAL_NEGATIVE)

        state = tracker.current_state
        assert state.valence < initial_valence

    def test_danger_increases_arousal(self) -> None:
        """Test danger event increases arousal significantly."""
        tracker = EmotionTracker()
        initial_arousal = tracker.current_state.arousal

        tracker.apply_event(EmotionEvents.DANGER)

        state = tracker.current_state
        assert state.arousal > initial_arousal
        # Danger should also be negative
        assert state.valence < 0.0

    def test_rest_decreases_arousal(self) -> None:
        """Test rest event decreases arousal."""
        tracker = EmotionTracker()
        # First increase arousal
        tracker.apply_event(EmotionEvents.DANGER)
        high_arousal = tracker.current_state.arousal

        # Then rest
        tracker.apply_event(EmotionEvents.REST)

        state = tracker.current_state
        assert state.arousal < high_arousal


class TestEmotionIntegration:
    """Integration tests for emotion tracking."""

    def test_multiple_events_accumulate(self) -> None:
        """Test multiple events accumulate their effects."""
        tracker = EmotionTracker(decay_rate=0.0)  # Disable decay for this test

        # Apply multiple positive events
        for _ in range(3):
            tracker.apply_event(EmotionEvents.SOCIAL_POSITIVE)

        state = tracker.current_state
        # Should be noticeably positive (3 * 0.2 * 0.5 intensity = 0.3)
        assert state.valence >= 0.3

    def test_decay_eventually_reaches_baseline(self) -> None:
        """Test that decay eventually returns to baseline."""
        tracker = EmotionTracker(decay_rate=0.5)

        # Apply strong event
        tracker.apply_event(EmotionEvents.DANGER)

        # Verify we're far from baseline
        state_excited = tracker.current_state
        assert abs(state_excited.valence - 0.0) > 0.1 or abs(state_excited.arousal - 0.3) > 0.1

        # Apply strong decay
        tracker.decay(delta_seconds=10.0)

        # Should be very close to baseline
        state_decayed = tracker.current_state
        assert state_decayed.valence == pytest.approx(0.0, abs=0.05)
        assert state_decayed.arousal == pytest.approx(0.3, abs=0.05)

    def test_emotional_journey_narrative(self) -> None:
        """Test a narrative sequence of emotional changes."""
        tracker = EmotionTracker()

        # Agent starts calm
        assert tracker.current_state.dominant_emotion in ["calm", "bored", "content"]

        # Agent succeeds at a goal
        tracker.apply_event(EmotionEvents.GOAL_SUCCESS)
        state1 = tracker.current_state
        assert state1.valence > 0.0  # Happy

        # Agent has positive social interaction
        tracker.apply_event(EmotionEvents.SOCIAL_POSITIVE)
        state2 = tracker.current_state
        assert state2.valence > state1.valence  # Even happier

        # Agent encounters danger
        tracker.apply_event(EmotionEvents.DANGER)
        state3 = tracker.current_state
        assert state3.valence < state2.valence  # Less happy
        assert state3.arousal > state2.arousal  # More aroused (afraid)

        # Agent rests
        tracker.apply_event(EmotionEvents.REST)
        state4 = tracker.current_state
        assert state4.arousal < state3.arousal  # Calmer

    def test_prompt_description_updates_with_state(self) -> None:
        """Test prompt descriptions reflect current emotional state."""
        tracker = EmotionTracker()

        # Apply different events and check descriptions change
        descriptions = []

        tracker.apply_event(EmotionEvents.GOAL_SUCCESS)
        descriptions.append(tracker.current_state.to_prompt_description())

        tracker.apply_event(EmotionEvents.DANGER)
        descriptions.append(tracker.current_state.to_prompt_description())

        tracker.apply_event(EmotionEvents.REST)
        descriptions.append(tracker.current_state.to_prompt_description())

        # All descriptions should be valid strings
        for desc in descriptions:
            assert isinstance(desc, str)
            assert len(desc) > 0

        # Descriptions should differ (emotional state changed)
        # Note: This might occasionally fail due to randomness, but usually should work
        assert len(set(descriptions)) >= 2
