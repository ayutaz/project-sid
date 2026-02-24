"""Tests for Big Five personality modeling system."""

import random

import pytest

from piano.social.personality import (
    PersonalityArchetypes,
    PersonalityInfluencer,
    PersonalityProfile,
)


class TestPersonalityProfile:
    """Test PersonalityProfile creation and validation."""

    def test_default_personality_creation(self):
        """Test creating a personality with default values."""
        profile = PersonalityProfile()
        assert profile.openness == 0.5
        assert profile.conscientiousness == 0.5
        assert profile.extraversion == 0.5
        assert profile.agreeableness == 0.5
        assert profile.neuroticism == 0.5

    def test_custom_personality_creation(self):
        """Test creating a personality with custom values."""
        profile = PersonalityProfile(
            openness=0.8,
            conscientiousness=0.3,
            extraversion=0.9,
            agreeableness=0.4,
            neuroticism=0.2,
        )
        assert profile.openness == 0.8
        assert profile.conscientiousness == 0.3
        assert profile.extraversion == 0.9
        assert profile.agreeableness == 0.4
        assert profile.neuroticism == 0.2

    def test_trait_validation_below_range(self):
        """Test that traits below 0.0 raise ValueError."""
        with pytest.raises(ValueError, match=r"must be between 0\.0 and 1\.0"):
            PersonalityProfile(openness=-0.1)

    def test_trait_validation_above_range(self):
        """Test that traits above 1.0 raise ValueError."""
        with pytest.raises(ValueError, match=r"must be between 0\.0 and 1\.0"):
            PersonalityProfile(conscientiousness=1.5)

    def test_trait_validation_all_traits(self):
        """Test validation on all traits."""
        with pytest.raises(ValueError):
            PersonalityProfile(openness=-0.1)
        with pytest.raises(ValueError):
            PersonalityProfile(conscientiousness=1.1)
        with pytest.raises(ValueError):
            PersonalityProfile(extraversion=-0.5)
        with pytest.raises(ValueError):
            PersonalityProfile(agreeableness=2.0)
        with pytest.raises(ValueError):
            PersonalityProfile(neuroticism=-1.0)

    def test_trait_validation_boundary_values(self):
        """Test that boundary values (0.0 and 1.0) are valid."""
        profile = PersonalityProfile(
            openness=0.0,
            conscientiousness=1.0,
            extraversion=0.0,
            agreeableness=1.0,
            neuroticism=0.5,
        )
        assert profile.openness == 0.0
        assert profile.conscientiousness == 1.0
        assert profile.extraversion == 0.0
        assert profile.agreeableness == 1.0


class TestToPromptModifier:
    """Test personality description generation for LLM prompts."""

    def test_balanced_personality_prompt(self):
        """Test prompt for balanced personality (all 0.5) includes moderate text."""
        profile = PersonalityProfile()
        prompt = profile.to_prompt_modifier()
        assert "moderate" in prompt.lower()

    def test_moderate_trait_text(self):
        """Test that traits in 0.3-0.7 range include 'moderate' descriptions."""
        profile = PersonalityProfile(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
        )
        prompt = profile.to_prompt_modifier()
        assert "moderate openness" in prompt.lower()
        assert "moderate conscientiousness" in prompt.lower()
        assert "moderate extraversion" in prompt.lower()
        assert "moderate agreeableness" in prompt.lower()
        assert "moderate neuroticism" in prompt.lower()

    def test_high_extraversion_prompt(self):
        """Test prompt includes extraversion description."""
        profile = PersonalityProfile(extraversion=0.9)
        prompt = profile.to_prompt_modifier()
        assert "outgoing" in prompt.lower() or "sociable" in prompt.lower()

    def test_low_extraversion_prompt(self):
        """Test prompt includes introversion description."""
        profile = PersonalityProfile(extraversion=0.1)
        prompt = profile.to_prompt_modifier()
        assert "introverted" in prompt.lower() or "reserved" in prompt.lower()

    def test_high_openness_prompt(self):
        """Test prompt includes high openness description."""
        profile = PersonalityProfile(openness=0.9)
        prompt = profile.to_prompt_modifier()
        assert "curious" in prompt.lower() or "creative" in prompt.lower()

    def test_low_openness_prompt(self):
        """Test prompt includes low openness description."""
        profile = PersonalityProfile(openness=0.1)
        prompt = profile.to_prompt_modifier()
        assert "practical" in prompt.lower() or "conventional" in prompt.lower()

    def test_high_agreeableness_prompt(self):
        """Test prompt includes high agreeableness description."""
        profile = PersonalityProfile(agreeableness=0.9)
        prompt = profile.to_prompt_modifier()
        assert "cooperative" in prompt.lower() or "trusting" in prompt.lower()

    def test_high_conscientiousness_prompt(self):
        """Test prompt includes conscientiousness description."""
        profile = PersonalityProfile(conscientiousness=0.9)
        prompt = profile.to_prompt_modifier()
        assert "organized" in prompt.lower() or "disciplined" in prompt.lower()

    def test_high_neuroticism_prompt(self):
        """Test prompt includes neuroticism description."""
        profile = PersonalityProfile(neuroticism=0.9)
        prompt = profile.to_prompt_modifier()
        assert "sensitive" in prompt.lower()

    def test_low_neuroticism_prompt(self):
        """Test prompt includes emotional stability description."""
        profile = PersonalityProfile(neuroticism=0.1)
        prompt = profile.to_prompt_modifier()
        assert "stable" in prompt.lower() or "calm" in prompt.lower()

    def test_prompt_coherence(self):
        """Test that prompt produces coherent natural language."""
        profile = PersonalityProfile(
            openness=0.9,
            conscientiousness=0.2,
            extraversion=0.8,
            agreeableness=0.1,
            neuroticism=0.3,
        )
        prompt = profile.to_prompt_modifier()
        # Should be a proper sentence starting with "You"
        assert prompt.startswith("You")
        assert len(prompt) > 20  # Should be substantial description


class TestBehaviorWeights:
    """Test behavior weight calculation."""

    def test_behavior_weights_structure(self):
        """Test that behavior weights return expected keys."""
        profile = PersonalityProfile()
        weights = profile.get_behavior_weights()

        expected_keys = {
            "exploration",
            "risk_taking",
            "social_seeking",
            "cooperation",
            "planning",
            "routine",
            "emotional_reactivity",
            "stability",
        }
        assert set(weights.keys()) == expected_keys

    def test_behavior_weights_range(self):
        """Test that all behavior weights are in reasonable range."""
        profile = PersonalityProfile(
            openness=0.0,
            conscientiousness=0.0,
            extraversion=0.0,
            agreeableness=0.0,
            neuroticism=0.0,
        )
        weights = profile.get_behavior_weights()
        for key, value in weights.items():
            assert 0.0 <= value <= 1.5, f"{key} weight {value} out of range"

    def test_high_openness_increases_exploration(self):
        """Test that high openness increases exploration weight."""
        low_openness = PersonalityProfile(openness=0.1)
        high_openness = PersonalityProfile(openness=0.9)

        high_exploration = high_openness.get_behavior_weights()["exploration"]
        low_exploration = low_openness.get_behavior_weights()["exploration"]
        assert high_exploration > low_exploration

    def test_high_extraversion_increases_social_seeking(self):
        """Test that high extraversion increases social seeking."""
        introvert = PersonalityProfile(extraversion=0.1)
        extravert = PersonalityProfile(extraversion=0.9)

        extravert_social = extravert.get_behavior_weights()["social_seeking"]
        introvert_social = introvert.get_behavior_weights()["social_seeking"]
        assert extravert_social > introvert_social

    def test_high_conscientiousness_increases_planning(self):
        """Test that high conscientiousness increases planning."""
        low_c = PersonalityProfile(conscientiousness=0.1)
        high_c = PersonalityProfile(conscientiousness=0.9)

        assert high_c.get_behavior_weights()["planning"] > low_c.get_behavior_weights()["planning"]

    def test_high_neuroticism_increases_emotional_reactivity(self):
        """Test that high neuroticism increases emotional reactivity."""
        stable = PersonalityProfile(neuroticism=0.1)
        anxious = PersonalityProfile(neuroticism=0.9)

        anxious_reactivity = anxious.get_behavior_weights()["emotional_reactivity"]
        stable_reactivity = stable.get_behavior_weights()["emotional_reactivity"]
        assert anxious_reactivity > stable_reactivity


class TestSocialInitiativeScore:
    """Test social initiative score calculation."""

    def test_high_extraversion_high_initiative(self):
        """Test that extraverts have high social initiative."""
        extravert = PersonalityProfile(extraversion=0.9)
        score = extravert.social_initiative_score()
        assert score > 0.6

    def test_low_extraversion_low_initiative(self):
        """Test that introverts have lower social initiative."""
        introvert = PersonalityProfile(extraversion=0.1)
        score = introvert.social_initiative_score()
        assert score < 0.5

    def test_initiative_in_valid_range(self):
        """Test that social initiative is in [0.0, 1.0] range."""
        for _ in range(20):
            profile = PersonalityProfile(
                openness=random.random(),
                conscientiousness=random.random(),
                extraversion=random.random(),
                agreeableness=random.random(),
                neuroticism=random.random(),
            )
            score = profile.social_initiative_score()
            assert 0.0 <= score <= 1.0


class TestRiskTolerance:
    """Test risk tolerance calculation."""

    def test_high_openness_high_risk_tolerance(self):
        """Test that high openness increases risk tolerance."""
        cautious = PersonalityProfile(openness=0.1)
        adventurous = PersonalityProfile(openness=0.9)
        assert adventurous.risk_tolerance() > cautious.risk_tolerance()

    def test_low_neuroticism_high_risk_tolerance(self):
        """Test that low neuroticism increases risk tolerance."""
        anxious = PersonalityProfile(neuroticism=0.9)
        stable = PersonalityProfile(neuroticism=0.1)
        assert stable.risk_tolerance() > anxious.risk_tolerance()

    def test_combined_high_openness_low_neuroticism(self):
        """Test that high openness + low neuroticism = high risk tolerance."""
        risk_averse = PersonalityProfile(openness=0.1, neuroticism=0.9)
        risk_tolerant = PersonalityProfile(openness=0.9, neuroticism=0.1)
        assert risk_tolerant.risk_tolerance() > risk_averse.risk_tolerance()
        assert risk_tolerant.risk_tolerance() > 0.6

    def test_risk_tolerance_in_valid_range(self):
        """Test that risk tolerance is in [0.0, 1.0] range."""
        for _ in range(20):
            profile = PersonalityProfile(
                openness=random.random(),
                conscientiousness=random.random(),
                extraversion=random.random(),
                agreeableness=random.random(),
                neuroticism=random.random(),
            )
            tolerance = profile.risk_tolerance()
            assert 0.0 <= tolerance <= 1.0


class TestPersonalityArchetypes:
    """Test predefined personality archetypes."""

    def test_farmer_archetype(self):
        """Test farmer archetype has expected traits."""
        farmer = PersonalityArchetypes.farmer()
        assert farmer.conscientiousness > 0.7  # High discipline
        assert farmer.openness < 0.4  # Traditional methods

    def test_explorer_archetype(self):
        """Test explorer archetype has expected traits."""
        explorer = PersonalityArchetypes.explorer()
        assert explorer.openness > 0.7  # High curiosity
        assert explorer.neuroticism < 0.4  # Low anxiety

    def test_socialite_archetype(self):
        """Test socialite archetype has expected traits."""
        socialite = PersonalityArchetypes.socialite()
        assert socialite.extraversion > 0.7  # High sociability
        assert socialite.agreeableness > 0.7  # High cooperation

    def test_guard_archetype(self):
        """Test guard archetype has expected traits."""
        guard = PersonalityArchetypes.guard()
        assert guard.conscientiousness > 0.7  # High discipline
        assert guard.openness < 0.4  # Follows protocol

    def test_scholar_archetype(self):
        """Test scholar archetype has expected traits."""
        scholar = PersonalityArchetypes.scholar()
        assert scholar.openness > 0.7  # High intellectual curiosity
        assert scholar.conscientiousness > 0.7  # Systematic study

    def test_all_archetypes_valid(self):
        """Test that all archetypes have valid trait ranges."""
        archetypes = [
            PersonalityArchetypes.farmer(),
            PersonalityArchetypes.explorer(),
            PersonalityArchetypes.socialite(),
            PersonalityArchetypes.guard(),
            PersonalityArchetypes.scholar(),
        ]

        for archetype in archetypes:
            assert 0.0 <= archetype.openness <= 1.0
            assert 0.0 <= archetype.conscientiousness <= 1.0
            assert 0.0 <= archetype.extraversion <= 1.0
            assert 0.0 <= archetype.agreeableness <= 1.0
            assert 0.0 <= archetype.neuroticism <= 1.0


class TestRandomPersonality:
    """Test random personality generation."""

    def test_random_personality_generates_valid_values(self):
        """Test that random personalities have valid trait values."""
        for _ in range(50):
            profile = PersonalityArchetypes.random()
            assert 0.0 <= profile.openness <= 1.0
            assert 0.0 <= profile.conscientiousness <= 1.0
            assert 0.0 <= profile.extraversion <= 1.0
            assert 0.0 <= profile.agreeableness <= 1.0
            assert 0.0 <= profile.neuroticism <= 1.0

    def test_random_personality_with_rng(self):
        """Test random personality with custom RNG for reproducibility."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        profile1 = PersonalityArchetypes.random(rng1)
        profile2 = PersonalityArchetypes.random(rng2)

        assert profile1.openness == profile2.openness
        assert profile1.conscientiousness == profile2.conscientiousness
        assert profile1.extraversion == profile2.extraversion
        assert profile1.agreeableness == profile2.agreeableness
        assert profile1.neuroticism == profile2.neuroticism

    def test_random_personality_variation(self):
        """Test that random personalities produce varied results."""
        profiles = [PersonalityArchetypes.random() for _ in range(20)]

        # Check that we get variation (not all the same)
        openness_values = [p.openness for p in profiles]
        assert len(set(openness_values)) > 10  # Should have variety

    def test_random_personality_gaussian_distribution(self):
        """Test that random personalities approximate Gaussian distribution."""
        rng = random.Random(12345)
        profiles = [PersonalityArchetypes.random(rng) for _ in range(100)]

        # Check that mean is roughly centered (allowing for randomness)
        openness_mean = sum(p.openness for p in profiles) / len(profiles)
        assert 0.3 < openness_mean < 0.7  # Should cluster around 0.5


class TestPersonalityInfluencer:
    """Test personality-based decision modification."""

    def test_modify_goal_priority_exploration(self):
        """Test that high openness increases exploration goal priority."""
        high_openness = PersonalityProfile(openness=0.9)
        low_openness = PersonalityProfile(openness=0.1)

        high_priority = PersonalityInfluencer.modify_goal_priority("exploration", high_openness)
        low_priority = PersonalityInfluencer.modify_goal_priority("exploration", low_openness)

        assert high_priority > 1.0  # Above neutral
        assert low_priority < 1.0  # Below neutral
        assert high_priority > low_priority

    def test_modify_goal_priority_social(self):
        """Test that high extraversion increases social goal priority."""
        extravert = PersonalityProfile(extraversion=0.9, agreeableness=0.8)
        introvert = PersonalityProfile(extraversion=0.1, agreeableness=0.3)

        extravert_priority = PersonalityInfluencer.modify_goal_priority("social", extravert)
        introvert_priority = PersonalityInfluencer.modify_goal_priority("social", introvert)

        assert extravert_priority > introvert_priority

    def test_modify_goal_priority_unknown_category(self):
        """Test that unknown categories return neutral priority."""
        profile = PersonalityProfile()
        priority = PersonalityInfluencer.modify_goal_priority("unknown_category", profile)
        assert priority == 1.0

    def test_modify_goal_priority_range(self):
        """Test that goal priorities stay in reasonable range."""
        for _ in range(20):
            profile = PersonalityArchetypes.random()
            categories = ["exploration", "social", "building", "survival", "combat"]
            for category in categories:
                priority = PersonalityInfluencer.modify_goal_priority(category, profile)
                assert 0.5 <= priority <= 1.5


class TestSpeechStyleModification:
    """Test speech style modification based on personality."""

    def test_speech_style_structure(self):
        """Test that speech style returns expected keys."""
        profile = PersonalityProfile()
        style = PersonalityInfluencer.modify_speech_style(profile)

        assert "tone" in style
        assert "verbosity" in style
        assert "formality" in style

    def test_high_agreeableness_friendly_tone(self):
        """Test that high agreeableness produces friendly tone."""
        profile = PersonalityProfile(agreeableness=0.9)
        style = PersonalityInfluencer.modify_speech_style(profile)
        assert "friendly" in style["tone"].lower() or "warm" in style["tone"].lower()

    def test_low_agreeableness_direct_tone(self):
        """Test that low agreeableness produces direct tone."""
        profile = PersonalityProfile(agreeableness=0.1)
        style = PersonalityInfluencer.modify_speech_style(profile)
        assert "direct" in style["tone"].lower() or "blunt" in style["tone"].lower()

    def test_high_extraversion_verbose(self):
        """Test that high extraversion produces verbose speech."""
        profile = PersonalityProfile(extraversion=0.9)
        style = PersonalityInfluencer.modify_speech_style(profile)
        assert "verbose" in style["verbosity"].lower() or "expressive" in style["verbosity"].lower()

    def test_low_extraversion_concise(self):
        """Test that low extraversion produces concise speech."""
        profile = PersonalityProfile(extraversion=0.1)
        style = PersonalityInfluencer.modify_speech_style(profile)
        assert "brief" in style["verbosity"].lower() or "concise" in style["verbosity"].lower()

    def test_high_conscientiousness_formal(self):
        """Test that high conscientiousness produces formal speech."""
        profile = PersonalityProfile(conscientiousness=0.9)
        style = PersonalityInfluencer.modify_speech_style(profile)
        assert "formal" in style["formality"].lower()

    def test_low_conscientiousness_casual(self):
        """Test that low conscientiousness produces casual speech."""
        profile = PersonalityProfile(conscientiousness=0.1)
        style = PersonalityInfluencer.modify_speech_style(profile)
        assert "casual" in style["formality"].lower() or "relaxed" in style["formality"].lower()

    def test_speech_style_varies_by_personality(self):
        """Test that speech style varies across different personalities."""
        socialite = PersonalityArchetypes.socialite()
        scholar = PersonalityArchetypes.scholar()

        socialite_style = PersonalityInfluencer.modify_speech_style(socialite)
        scholar_style = PersonalityInfluencer.modify_speech_style(scholar)

        # Socialite should be more verbose than farmer or scholar
        assert "verbose" in socialite_style["verbosity"].lower()
        # Scholar should be formal
        assert "formal" in scholar_style["formality"].lower()
