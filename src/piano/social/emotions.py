"""Emotion tracking using valence-arousal model for PIANO agents.

This module implements emotion tracking based on the valence-arousal model,
where emotions are represented as points in a 2D space:
- Valence: pleasant (positive) to unpleasant (negative), range [-1, 1]
- Arousal: activation level, range [0, 1]

Emotions influence agent behavior and can spread between agents through
social contagion.

Reference: docs/implementation/06-social-cognition.md Section 2.3
"""

from __future__ import annotations

__all__ = [
    "EMOTION_MAP",
    "EmotionEvent",
    "EmotionEvents",
    "EmotionState",
    "EmotionTracker",
]

import math
import time
from typing import Any

from pydantic import BaseModel, Field


class EmotionState(BaseModel):
    """Current emotional state of an agent.

    Attributes:
        valence: Emotional valence from -1 (negative) to 1 (positive)
        arousal: Arousal level from 0 (calm) to 1 (excited)
        dominant_emotion: Name of the dominant emotion
        emotions: Intensity of specific emotions (0-10 scale)
    """

    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.3, ge=0.0, le=1.0)
    dominant_emotion: str = "calm"
    emotions: dict[str, float] = Field(default_factory=dict)

    def to_prompt_description(self) -> str:
        """Generate natural language description of emotional state for LLM prompts."""
        if not self.emotions:
            return f"feeling {self.dominant_emotion}"

        # Get top 2 emotions
        sorted_emotions = sorted(self.emotions.items(), key=lambda x: x[1], reverse=True)[:2]

        if not sorted_emotions:
            return f"feeling {self.dominant_emotion}"

        # Format intensities as adjectives
        def intensity_adjective(value: float) -> str:
            if value >= 8.0:
                return "very"
            elif value >= 5.0:
                return "moderately"
            else:
                return "slightly"

        parts = []
        for emotion_name, intensity in sorted_emotions:
            if intensity >= 3.0:  # Only include noticeable emotions
                parts.append(f"{intensity_adjective(intensity)} {emotion_name}")

        if not parts:
            return f"feeling {self.dominant_emotion}"

        if len(parts) == 1:
            return f"feeling {parts[0]}"
        else:
            return f"feeling {parts[0]} and {parts[1]}"


class EmotionEvent(BaseModel):
    """An event that affects emotional state.

    Attributes:
        event_type: Type of event (e.g., "goal_success", "social_positive")
        intensity: Strength of the event (0.0-1.0)
        valence_shift: Change to valence dimension
        arousal_shift: Change to arousal dimension
    """

    event_type: str
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)
    valence_shift: float = Field(default=0.0)
    arousal_shift: float = Field(default=0.0)


# Mapping from (valence_range, arousal_range) to emotion names
# valence: negative (-1 to -0.33), neutral (-0.33 to 0.33), positive (0.33 to 1)
# arousal: low (0 to 0.33), medium (0.33 to 0.66), high (0.66 to 1)
EMOTION_MAP: dict[tuple[str, str], str] = {
    ("positive", "high"): "excited",
    ("positive", "medium"): "happy",
    ("positive", "low"): "content",
    ("neutral", "high"): "surprised",
    ("neutral", "medium"): "calm",
    ("neutral", "low"): "bored",
    ("negative", "high"): "angry",
    ("negative", "medium"): "sad",
    ("negative", "low"): "depressed",
}


class EmotionTracker:
    """Tracks agent emotional state over time with decay toward baseline.

    Implements valence-arousal emotion model with:
    - Event-driven emotion updates
    - Social contagion from nearby agents
    - Decay toward baseline over time
    - Emotion-to-behavior mapping

    Attributes:
        baseline_valence: Neutral valence point (default 0.0)
        baseline_arousal: Neutral arousal point (default 0.3)
        decay_rate: Rate of decay toward baseline per second (default 0.1)
    """

    def __init__(
        self,
        baseline_valence: float = 0.0,
        baseline_arousal: float = 0.3,
        decay_rate: float = 0.1,
    ) -> None:
        """Initialize emotion tracker.

        Args:
            baseline_valence: Neutral valence point (-1 to 1)
            baseline_arousal: Neutral arousal point (0 to 1)
            decay_rate: Decay rate toward baseline per second
        """
        self._baseline_valence = max(-1.0, min(1.0, baseline_valence))
        self._baseline_arousal = max(0.0, min(1.0, baseline_arousal))
        self._decay_rate = decay_rate

        self._valence = self._baseline_valence
        self._arousal = self._baseline_arousal
        self._last_update = time.time()

        # Track specific emotion intensities
        self._emotions: dict[str, float] = {}

    @property
    def current_state(self) -> EmotionState:
        """Get current emotional state."""
        return EmotionState(
            valence=self._valence,
            arousal=self._arousal,
            dominant_emotion=self._classify_emotion(),
            emotions=dict(self._emotions),
        )

    def apply_event(self, event: EmotionEvent, *, timestamp: float | None = None) -> None:
        """Apply an emotional event, shifting valence and arousal.

        Args:
            event: The emotion event to apply
            timestamp: Optional explicit timestamp (defaults to time.time())
        """
        # Apply decay before the new event
        now = timestamp if timestamp is not None else time.time()
        delta = now - self._last_update
        self.decay(delta)

        # Apply shifts with intensity scaling
        self._valence += event.valence_shift * event.intensity
        self._arousal += event.arousal_shift * event.intensity

        # Clamp to valid ranges
        self._valence = max(-1.0, min(1.0, self._valence))
        self._arousal = max(0.0, min(1.0, self._arousal))

        # Update specific emotion intensities
        self._update_emotion_intensities()

        self._last_update = now

    def apply_social_contagion(
        self,
        other_state: EmotionState,
        influence: float = 0.1,
        *,
        timestamp: float | None = None,
    ) -> None:
        """Apply social contagion from another agent's emotional state.

        Args:
            other_state: Emotional state of the other agent
            influence: Strength of influence (0.0-1.0)
            timestamp: Optional explicit timestamp (defaults to time.time())
        """
        # Apply decay first
        now = timestamp if timestamp is not None else time.time()
        delta = now - self._last_update
        self.decay(delta)

        # Shift toward other agent's state
        influence = max(0.0, min(1.0, influence))

        valence_diff = other_state.valence - self._valence
        arousal_diff = other_state.arousal - self._arousal

        self._valence += valence_diff * influence
        self._arousal += arousal_diff * influence

        # Clamp to valid ranges
        self._valence = max(-1.0, min(1.0, self._valence))
        self._arousal = max(0.0, min(1.0, self._arousal))

        # Update emotion intensities
        self._update_emotion_intensities()

        self._last_update = now

    def decay(self, delta_seconds: float) -> None:
        """Apply decay toward baseline emotional state.

        Args:
            delta_seconds: Time elapsed since last update
        """
        if delta_seconds <= 0:
            return

        # Exponential decay toward baseline
        decay_factor = 1.0 - math.exp(-self._decay_rate * delta_seconds)

        self._valence += (self._baseline_valence - self._valence) * decay_factor
        self._arousal += (self._baseline_arousal - self._arousal) * decay_factor

        # Update emotion intensities
        self._update_emotion_intensities()

    def _classify_emotion(self) -> str:
        """Determine dominant emotion from current valence and arousal."""
        # Categorize valence
        if self._valence > 0.33:
            valence_cat = "positive"
        elif self._valence < -0.33:
            valence_cat = "negative"
        else:
            valence_cat = "neutral"

        # Categorize arousal
        if self._arousal > 0.66:
            arousal_cat = "high"
        elif self._arousal > 0.33:
            arousal_cat = "medium"
        else:
            arousal_cat = "low"

        return EMOTION_MAP.get((valence_cat, arousal_cat), "calm")

    def _update_emotion_intensities(self) -> None:
        """Update specific emotion intensities based on valence/arousal."""
        # Calculate distance-based intensities for each emotion category
        # Use inverse distance to emotion "centers" in valence-arousal space

        emotion_centers = {
            "happy": (0.7, 0.5),
            "excited": (0.6, 0.85),
            "content": (0.5, 0.2),
            "calm": (0.0, 0.15),
            "bored": (0.0, 0.1),
            "surprised": (0.0, 0.85),
            "sad": (-0.6, 0.4),
            "angry": (-0.7, 0.8),
            "afraid": (-0.5, 0.7),
            "depressed": (-0.5, 0.15),
        }

        self._emotions.clear()

        for emotion_name, (v_center, a_center) in emotion_centers.items():
            # Calculate Euclidean distance in normalized space
            v_dist = (self._valence - v_center) ** 2
            a_dist = (self._arousal - a_center) ** 2
            distance = math.sqrt(v_dist + a_dist)

            # Convert distance to intensity (0-10 scale)
            # Closer = higher intensity
            if distance < 0.1:
                intensity = 10.0
            elif distance < 0.3:
                intensity = 8.0 - (distance - 0.1) * 20.0
            elif distance < 0.6:
                intensity = 4.0 - (distance - 0.3) * 10.0
            else:
                intensity = max(0.0, 1.0 - (distance - 0.6) * 2.0)

            if intensity > 0.5:  # Only track noticeable emotions
                self._emotions[emotion_name] = intensity

    def get_behavior_modifier(self) -> dict[str, float]:
        """Get behavioral parameters influenced by current emotional state.

        Returns:
            Dictionary of behavior modifiers:
            - risk_aversion: How risk-averse (0.0-1.0, higher = more cautious)
            - social_seeking: Desire for social interaction (0.0-1.0)
            - exploration: Tendency to explore (0.0-1.0)
            - cooperation: Willingness to cooperate (0.0-1.0)
            - aggression: Aggressive tendency (0.0-1.0)
        """
        # High positive valence + high arousal = social seeking, exploration
        # High negative valence + high arousal = aggression, low cooperation
        # High negative valence + low arousal = risk aversion, low social
        # Low arousal overall = less exploration

        risk_aversion = 0.5 - self._valence * 0.3 + (1.0 - self._arousal) * 0.2
        social_seeking = 0.5 + self._valence * 0.3 + self._arousal * 0.2
        exploration = 0.5 + self._arousal * 0.4 + self._valence * 0.1
        cooperation = 0.5 + self._valence * 0.5
        aggression = 0.2 + max(0.0, -self._valence) * 0.5 + self._arousal * 0.3

        return {
            "risk_aversion": max(0.0, min(1.0, risk_aversion)),
            "social_seeking": max(0.0, min(1.0, social_seeking)),
            "exploration": max(0.0, min(1.0, exploration)),
            "cooperation": max(0.0, min(1.0, cooperation)),
            "aggression": max(0.0, min(1.0, aggression)),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker state to dictionary."""
        return {
            "valence": self._valence,
            "arousal": self._arousal,
            "baseline_valence": self._baseline_valence,
            "baseline_arousal": self._baseline_arousal,
            "decay_rate": self._decay_rate,
            "emotions": dict(self._emotions),
            "last_update": self._last_update,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> EmotionTracker:
        """Deserialize tracker from dictionary."""
        tracker = EmotionTracker(
            baseline_valence=data.get("baseline_valence", 0.0),
            baseline_arousal=data.get("baseline_arousal", 0.3),
            decay_rate=data.get("decay_rate", 0.1),
        )
        tracker._valence = data.get("valence", 0.0)
        tracker._arousal = data.get("arousal", 0.3)
        tracker._emotions = data.get("emotions", {})
        tracker._last_update = data.get("last_update", time.time())
        return tracker


class EmotionEvents:
    """Pre-defined emotion events for common situations."""

    GOAL_SUCCESS = EmotionEvent(
        event_type="goal_success",
        intensity=0.7,
        valence_shift=0.3,
        arousal_shift=0.2,
    )

    GOAL_FAILURE = EmotionEvent(
        event_type="goal_failure",
        intensity=0.6,
        valence_shift=-0.3,
        arousal_shift=0.15,
    )

    SOCIAL_POSITIVE = EmotionEvent(
        event_type="social_positive",
        intensity=0.5,
        valence_shift=0.2,
        arousal_shift=0.1,
    )

    SOCIAL_NEGATIVE = EmotionEvent(
        event_type="social_negative",
        intensity=0.5,
        valence_shift=-0.25,
        arousal_shift=0.2,
    )

    DANGER = EmotionEvent(
        event_type="danger",
        intensity=0.8,
        valence_shift=-0.4,
        arousal_shift=0.5,
    )

    REST = EmotionEvent(
        event_type="rest",
        intensity=0.3,
        valence_shift=0.1,
        arousal_shift=-0.3,
    )
