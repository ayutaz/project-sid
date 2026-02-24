"""Big Five personality modeling for PIANO agents.

This module implements the personality system based on the OCEAN (Big Five) model
that influences agent behavior in social interactions, decision-making, and
speech patterns. Each agent has a unique personality profile that affects:
- Social initiative and interaction patterns
- Risk tolerance in action selection
- Speech style and tone
- Goal priorities and preferences

Reference: docs/implementation/06-social-cognition.md Section 4
"""

from __future__ import annotations

__all__ = [
    "PersonalityArchetypes",
    "PersonalityInfluencer",
    "PersonalityProfile",
]

import random
from typing import ClassVar

import structlog
from pydantic import BaseModel, field_validator

logger = structlog.get_logger(__name__)


class PersonalityProfile(BaseModel):
    """Big Five personality trait profile (OCEAN model).

    Each trait is a float in the range [0.0, 1.0]:
    - 0.0-0.3: Low expression of the trait
    - 0.3-0.7: Moderate expression
    - 0.7-1.0: High expression

    Traits:
        openness: Openness to experience (curiosity, creativity)
        conscientiousness: Conscientiousness (organization, discipline)
        extraversion: Extraversion (sociability, outgoingness)
        agreeableness: Agreeableness (cooperation, trust)
        neuroticism: Neuroticism (emotional sensitivity, anxiety)
    """

    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    @field_validator(
        "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"
    )
    @classmethod
    def validate_trait_range(cls, v: float) -> float:
        """Ensure all traits are in [0.0, 1.0] range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Trait value must be between 0.0 and 1.0, got {v}")
        return v

    def to_prompt_modifier(self) -> str:
        """Generate personality description for LLM prompts.

        Returns a natural language description of the agent's personality
        that can be embedded in LLM system prompts to influence behavior.

        Returns:
            Natural language personality description
        """
        traits = []

        # Openness
        if self.openness > 0.7:
            traits.append("highly curious and creative")
        elif self.openness < 0.3:
            traits.append("practical and conventional")
        else:
            traits.append("has moderate openness")

        # Conscientiousness
        if self.conscientiousness > 0.7:
            traits.append("organized and disciplined")
        elif self.conscientiousness < 0.3:
            traits.append("flexible and spontaneous")
        else:
            traits.append("has moderate conscientiousness")

        # Extraversion
        if self.extraversion > 0.7:
            traits.append("outgoing and sociable")
        elif self.extraversion < 0.3:
            traits.append("introverted and reserved")
        else:
            traits.append("has moderate extraversion")

        # Agreeableness
        if self.agreeableness > 0.7:
            traits.append("cooperative and trusting")
        elif self.agreeableness < 0.3:
            traits.append("competitive and skeptical")
        else:
            traits.append("has moderate agreeableness")

        # Neuroticism
        if self.neuroticism > 0.7:
            traits.append("emotionally sensitive")
        elif self.neuroticism < 0.3:
            traits.append("emotionally stable and calm")
        else:
            traits.append("has moderate neuroticism")

        return f"You are {', '.join(traits)}."

    def get_behavior_weights(self) -> dict[str, float]:
        """Calculate behavior weights based on personality traits.

        Returns weights that can be used to bias action selection, goal priorities,
        and decision-making based on the agent's personality.

        Returns:
            Dictionary mapping behavior categories to weight multipliers
        """
        return {
            "exploration": 0.5 + (self.openness * 0.5),  # 0.5-1.0
            "risk_taking": 0.5 + ((self.openness + (1.0 - self.neuroticism)) / 4.0),  # 0.5-1.0
            "social_seeking": 0.3 + (self.extraversion * 0.7),  # 0.3-1.0
            "cooperation": 0.3 + (self.agreeableness * 0.7),  # 0.3-1.0
            "planning": 0.3 + (self.conscientiousness * 0.7),  # 0.3-1.0
            "routine": (
                0.3 + (self.conscientiousness * 0.5) + ((1.0 - self.openness) * 0.2)
            ),  # 0.3-1.0
            "emotional_reactivity": 0.2 + (self.neuroticism * 0.8),  # 0.2-1.0
            "stability": 0.2 + ((1.0 - self.neuroticism) * 0.8),  # 0.2-1.0
        }

    def social_initiative_score(self) -> float:
        """Calculate likelihood of initiating social interaction.

        Higher scores indicate greater tendency to start conversations and
        seek out social contact. Based primarily on extraversion.

        Returns:
            Score in range [0.0, 1.0]
        """
        # Extraverts are more likely to initiate
        # Also influenced slightly by agreeableness (friendly approach)
        # and negatively by neuroticism (social anxiety)
        return self.extraversion * 0.7 + self.agreeableness * 0.2 + (1.0 - self.neuroticism) * 0.1

    def risk_tolerance(self) -> float:
        """Calculate willingness to take risky actions.

        Higher scores indicate greater tolerance for uncertain outcomes.
        Based on openness (willingness to try new things) and neuroticism
        (anxiety about negative outcomes).

        Returns:
            Score in range [0.0, 1.0]
        """
        # High openness increases risk tolerance
        # High neuroticism decreases risk tolerance
        # Conscientiousness slightly decreases (prefer planning)
        return (
            self.openness * 0.5
            + (1.0 - self.neuroticism) * 0.4
            + (1.0 - self.conscientiousness) * 0.1
        )

class PersonalityArchetypes:
    """Predefined personality archetypes for common agent roles.

    These archetypes are designed to create diverse and realistic agent populations
    with distinct behavioral patterns. Each archetype is tuned to match expected
    behaviors for that role.
    """

    @staticmethod
    def farmer() -> PersonalityProfile:
        """Hardworking, routine-oriented farmer.

        High conscientiousness for disciplined work.
        Moderate agreeableness for community cooperation.
        Lower openness (prefers traditional methods).
        """
        return PersonalityProfile(
            openness=0.3,
            conscientiousness=0.8,
            extraversion=0.4,
            agreeableness=0.6,
            neuroticism=0.4,
        )

    @staticmethod
    def explorer() -> PersonalityProfile:
        """Adventurous explorer seeking new experiences.

        High openness for curiosity and discovery.
        Low neuroticism for emotional stability in unknown situations.
        Moderate extraversion (comfortable alone or with others).
        """
        return PersonalityProfile(
            openness=0.9,
            conscientiousness=0.4,
            extraversion=0.6,
            agreeableness=0.5,
            neuroticism=0.2,
        )

    @staticmethod
    def socialite() -> PersonalityProfile:
        """Social butterfly who thrives on interaction.

        High extraversion for constant social seeking.
        High agreeableness for friendly, cooperative nature.
        Moderate openness (enjoys variety in social settings).
        """
        return PersonalityProfile(
            openness=0.6,
            conscientiousness=0.4,
            extraversion=0.9,
            agreeableness=0.8,
            neuroticism=0.3,
        )

    @staticmethod
    def guard() -> PersonalityProfile:
        """Vigilant guard maintaining order and security.

        High conscientiousness for duty and discipline.
        Low openness (prefers established protocols).
        Moderate extraversion (observant but not overly social).
        Low neuroticism for emotional stability under pressure.
        """
        return PersonalityProfile(
            openness=0.2,
            conscientiousness=0.9,
            extraversion=0.4,
            agreeableness=0.5,
            neuroticism=0.3,
        )

    @staticmethod
    def scholar() -> PersonalityProfile:
        """Intellectual scholar focused on learning and knowledge.

        High openness for intellectual curiosity.
        High conscientiousness for systematic study.
        Lower extraversion (prefers quiet contemplation).
        """
        return PersonalityProfile(
            openness=0.9,
            conscientiousness=0.8,
            extraversion=0.3,
            agreeableness=0.6,
            neuroticism=0.4,
        )

    @staticmethod
    def random(rng: random.Random | None = None) -> PersonalityProfile:
        """Generate a random personality with Gaussian distribution.

        Uses normal distribution centered at 0.5 with controlled variance
        to create realistic personality variation. Clamps values to [0.0, 1.0].

        Args:
            rng: Random number generator for reproducibility. If None, uses default.

        Returns:
            Random personality profile
        """
        if rng is None:
            rng = random.Random()

        def gaussian_trait(mean: float = 0.5, stddev: float = 0.2) -> float:
            """Generate a trait value from Gaussian distribution."""
            value = rng.gauss(mean, stddev)
            return max(0.0, min(1.0, value))

        # Extraversion often has bimodal distribution (introverts vs extraverts)
        # 40% chance of being introvert-leaning, 60% extravert-leaning
        if rng.random() < 0.4:
            extraversion = gaussian_trait(mean=0.25, stddev=0.15)
        else:
            extraversion = gaussian_trait(mean=0.7, stddev=0.15)

        return PersonalityProfile(
            openness=gaussian_trait(mean=0.5, stddev=0.2),
            conscientiousness=gaussian_trait(mean=0.5, stddev=0.2),
            extraversion=extraversion,
            agreeableness=gaussian_trait(mean=0.6, stddev=0.2),  # Slight positive bias
            neuroticism=gaussian_trait(mean=0.4, stddev=0.2),  # Slight stability bias
        )


class PersonalityInfluencer:
    """Modifies decision parameters based on personality traits.

    This class provides methods to adjust goal priorities, speech styles,
    and other behavioral parameters based on an agent's personality profile.
    """

    # Goal category modifiers
    GOAL_PERSONALITY_WEIGHTS: ClassVar[dict[str, dict[str, float]]] = {
        "exploration": {"openness": 0.6, "extraversion": 0.2, "neuroticism": -0.2},
        "social": {"extraversion": 0.7, "agreeableness": 0.3},
        "building": {"conscientiousness": 0.6, "openness": 0.2},
        "survival": {"neuroticism": 0.4, "conscientiousness": 0.3},
        "combat": {"neuroticism": -0.5, "agreeableness": -0.3, "openness": 0.2},
        "resource_gathering": {"conscientiousness": 0.5, "openness": -0.2},
        "creative": {"openness": 0.7, "conscientiousness": 0.2},
        "routine": {"conscientiousness": 0.6, "openness": -0.3},
    }

    @staticmethod
    def modify_goal_priority(goal_category: str, personality: PersonalityProfile) -> float:
        """Calculate personality-based priority adjustment for a goal category.

        Args:
            goal_category: Category of goal (e.g., "exploration", "social")
            personality: Agent's personality profile

        Returns:
            Priority multiplier in range [0.5, 1.5] (1.0 = neutral)
        """
        weights = PersonalityInfluencer.GOAL_PERSONALITY_WEIGHTS.get(goal_category, {})
        if not weights:
            return 1.0

        # Calculate weighted sum of trait influences
        adjustment = 0.0
        for trait, weight in weights.items():
            trait_value = getattr(personality, trait, 0.5)
            adjustment += (trait_value - 0.5) * weight

        # Convert to multiplier: [-0.5, 0.5] -> [0.5, 1.5]
        multiplier = 1.0 + max(-0.5, min(0.5, adjustment))

        logger.debug(
            "goal_priority_adjusted",
            category=goal_category,
            adjustment=adjustment,
            multiplier=multiplier,
        )

        return multiplier

    @staticmethod
    def modify_speech_style(personality: PersonalityProfile) -> dict[str, str]:
        """Generate speech style parameters based on personality.

        Args:
            personality: Agent's personality profile

        Returns:
            Dictionary with tone, verbosity, and formality parameters
        """
        # Determine tone
        if personality.agreeableness > 0.7:
            tone = "friendly and warm"
        elif personality.agreeableness < 0.3:
            tone = "direct and blunt"
        elif personality.neuroticism > 0.7:
            tone = "cautious and hesitant"
        else:
            tone = "neutral and balanced"

        # Determine verbosity
        if personality.extraversion > 0.7:
            verbosity = "verbose and expressive"
        elif personality.extraversion < 0.3:
            verbosity = "brief and concise"
        else:
            verbosity = "moderate"

        # Determine formality
        if personality.conscientiousness > 0.7:
            formality = "formal and proper"
        elif personality.conscientiousness < 0.3:
            formality = "casual and relaxed"
        else:
            formality = "moderately formal"

        return {
            "tone": tone,
            "verbosity": verbosity,
            "formality": formality,
        }
