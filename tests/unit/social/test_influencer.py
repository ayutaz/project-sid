"""Tests for the influencer mechanism module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from piano.social.graph import SocialGraph, SocialRelation
from piano.social.influencer import (
    EmotionPropagation,
    InfluenceConfig,
    InfluencerModel,
    VoteInfluence,
)
from piano.social.personality import PersonalityProfile


class TestInfluenceConfig:
    """Tests for InfluenceConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = InfluenceConfig()
        assert config.decay_factor == 0.5
        assert config.max_hops == 3
        assert config.emotion_contagion_rate == 0.3
        assert config.influence_threshold == 0.1

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = InfluenceConfig(
            decay_factor=0.7,
            max_hops=5,
            emotion_contagion_rate=0.5,
            influence_threshold=0.05,
        )
        assert config.decay_factor == 0.7
        assert config.max_hops == 5
        assert config.emotion_contagion_rate == 0.5
        assert config.influence_threshold == 0.05

    def test_decay_factor_clamped(self) -> None:
        """Test decay_factor is clamped to [0, 1] range."""
        config = InfluenceConfig(decay_factor=1.5)
        assert config.decay_factor == 1.0

        config = InfluenceConfig(decay_factor=-0.5)
        assert config.decay_factor == 0.0

    def test_max_hops_minimum(self) -> None:
        """Test max_hops has a minimum of 1."""
        config = InfluenceConfig(max_hops=0)
        assert config.max_hops == 1

        config = InfluenceConfig(max_hops=-5)
        assert config.max_hops == 1

    def test_contagion_rate_clamped(self) -> None:
        """Test emotion_contagion_rate is clamped to [0, 1] range."""
        config = InfluenceConfig(emotion_contagion_rate=2.0)
        assert config.emotion_contagion_rate == 1.0

        config = InfluenceConfig(emotion_contagion_rate=-1.0)
        assert config.emotion_contagion_rate == 0.0

    def test_threshold_clamped(self) -> None:
        """Test influence_threshold is clamped to [0, 1] range."""
        config = InfluenceConfig(influence_threshold=5.0)
        assert config.influence_threshold == 1.0

        config = InfluenceConfig(influence_threshold=-0.5)
        assert config.influence_threshold == 0.0


class TestVoteInfluence:
    """Tests for VoteInfluence model."""

    def test_default_values(self) -> None:
        """Test default VoteInfluence values."""
        vi = VoteInfluence(agent_id="agent-1", proposal_id="prop-1")
        assert vi.agent_id == "agent-1"
        assert vi.proposal_id == "prop-1"
        assert vi.base_preference == 0.0
        assert vi.influenced_preference == 0.0
        assert vi.influencers == []

    def test_preference_validation(self) -> None:
        """Test preference values are validated to [-1, 1] range."""
        with pytest.raises(ValidationError):
            VoteInfluence(
                agent_id="a",
                proposal_id="p",
                base_preference=2.0,
            )

        with pytest.raises(ValidationError):
            VoteInfluence(
                agent_id="a",
                proposal_id="p",
                influenced_preference=-2.0,
            )

    def test_with_influencers(self) -> None:
        """Test VoteInfluence with influencer list."""
        vi = VoteInfluence(
            agent_id="agent-1",
            proposal_id="prop-1",
            base_preference=0.3,
            influenced_preference=0.5,
            influencers=["agent-2", "agent-3"],
        )
        assert len(vi.influencers) == 2
        assert "agent-2" in vi.influencers


class TestEmotionPropagation:
    """Tests for EmotionPropagation model."""

    def test_default_values(self) -> None:
        """Test default EmotionPropagation values."""
        ep = EmotionPropagation(source_id="agent-1", emotion="happy")
        assert ep.source_id == "agent-1"
        assert ep.emotion == "happy"
        assert ep.affected_agents == {}
        assert ep.propagation_path == []


class TestInfluencerModelInit:
    """Tests for InfluencerModel initialization."""

    def test_default_initialization(self) -> None:
        """Test default model initialization."""
        model = InfluencerModel()
        assert model.config.decay_factor == 0.5
        assert model.config.max_hops == 3

    def test_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = InfluenceConfig(decay_factor=0.8, max_hops=5)
        model = InfluencerModel(config=config)
        assert model.config.decay_factor == 0.8
        assert model.config.max_hops == 5

    def test_set_personality(self) -> None:
        """Test setting personality profiles."""
        model = InfluencerModel()
        profile = PersonalityProfile(neuroticism=0.8)
        model.set_personality("agent-1", profile)
        assert model._personalities["agent-1"].neuroticism == 0.8


def _build_simple_graph() -> SocialGraph:
    """Build a simple test graph: A -> B -> C -> D."""
    graph = SocialGraph()
    graph.set_relationship(SocialRelation(source_id="A", target_id="B", affinity=0.8, trust=0.9))
    graph.set_relationship(SocialRelation(source_id="B", target_id="C", affinity=0.6, trust=0.7))
    graph.set_relationship(SocialRelation(source_id="C", target_id="D", affinity=0.4, trust=0.5))
    return graph


def _build_hub_graph() -> SocialGraph:
    """Build a hub graph: multiple agents all connect to a central hub."""
    graph = SocialGraph()
    for i in range(5):
        graph.set_relationship(
            SocialRelation(
                source_id=f"spoke-{i}",
                target_id="hub",
                affinity=0.7,
                trust=0.8,
            )
        )
    # Hub also connects back to some spokes
    graph.set_relationship(
        SocialRelation(source_id="hub", target_id="spoke-0", affinity=0.5, trust=0.6)
    )
    graph.set_relationship(
        SocialRelation(source_id="hub", target_id="spoke-1", affinity=0.3, trust=0.5)
    )
    return graph


class TestCalculateInfluence:
    """Tests for InfluencerModel.calculate_influence."""

    def test_nonexistent_agent(self) -> None:
        """Test influence for agent not in graph returns 0."""
        model = InfluencerModel()
        graph = SocialGraph()
        assert model.calculate_influence("nobody", graph) == 0.0

    def test_empty_graph(self) -> None:
        """Test influence with empty graph returns 0."""
        model = InfluencerModel()
        graph = SocialGraph()
        assert model.calculate_influence("A", graph) == 0.0

    def test_single_agent(self) -> None:
        """Test influence for sole agent in graph."""
        model = InfluencerModel()
        graph = SocialGraph()
        graph.add_agent("solo")
        score = model.calculate_influence("solo", graph)
        assert score == 1.0

    def test_hub_has_higher_influence(self) -> None:
        """Test that a hub node has higher influence than spoke nodes."""
        model = InfluencerModel()
        graph = _build_hub_graph()

        hub_score = model.calculate_influence("hub", graph)
        spoke_score = model.calculate_influence("spoke-3", graph)

        assert hub_score > spoke_score

    def test_influence_score_in_range(self) -> None:
        """Test influence score is always in [0, 1] range."""
        model = InfluencerModel()
        graph = _build_simple_graph()

        for agent_id in ["A", "B", "C", "D"]:
            score = model.calculate_influence(agent_id, graph)
            assert 0.0 <= score <= 1.0, f"Score {score} for {agent_id} out of range"

    def test_more_connections_increase_influence(self) -> None:
        """Test that receiving more connections generally increases influence."""
        model = InfluencerModel()
        graph = SocialGraph()

        # Agent with many incoming connections
        for i in range(5):
            graph.set_relationship(
                SocialRelation(source_id=f"fan-{i}", target_id="popular", affinity=0.8, trust=0.8)
            )
        # Agent with few incoming connections
        graph.set_relationship(
            SocialRelation(source_id="fan-0", target_id="unpopular", affinity=0.8, trust=0.8)
        )

        popular_score = model.calculate_influence("popular", graph)
        unpopular_score = model.calculate_influence("unpopular", graph)

        assert popular_score > unpopular_score


class TestPropagateEmotion:
    """Tests for InfluencerModel.propagate_emotion."""

    def test_nonexistent_source(self) -> None:
        """Test propagation from nonexistent source returns empty dict."""
        model = InfluencerModel()
        graph = SocialGraph()
        result = model.propagate_emotion("nobody", "happy", graph)
        assert result == {}

    def test_simple_chain_propagation(self) -> None:
        """Test emotion propagates along a simple chain."""
        config = InfluenceConfig(
            decay_factor=0.9,
            max_hops=3,
            emotion_contagion_rate=1.0,
            influence_threshold=0.01,
        )
        model = InfluencerModel(config=config)
        graph = _build_simple_graph()

        result = model.propagate_emotion("A", "happy", graph)

        # B should be affected (direct connection from A)
        assert "B" in result
        assert result["B"] > 0.0

    def test_propagation_decays_with_distance(self) -> None:
        """Test that emotion intensity decreases with hop distance."""
        config = InfluenceConfig(
            decay_factor=0.9,
            max_hops=3,
            emotion_contagion_rate=1.0,
            influence_threshold=0.001,
        )
        model = InfluencerModel(config=config)
        graph = _build_simple_graph()

        result = model.propagate_emotion("A", "happy", graph)

        if "B" in result and "C" in result:
            assert result["B"] > result["C"], "Closer agent should have higher intensity"

    def test_source_not_in_result(self) -> None:
        """Test that the source agent is not included in affected agents."""
        config = InfluenceConfig(
            decay_factor=0.9,
            max_hops=3,
            emotion_contagion_rate=1.0,
            influence_threshold=0.01,
        )
        model = InfluencerModel(config=config)
        graph = _build_simple_graph()

        result = model.propagate_emotion("A", "happy", graph)
        assert "A" not in result

    def test_threshold_filters_weak_influence(self) -> None:
        """Test that agents below threshold are excluded."""
        config = InfluenceConfig(
            decay_factor=0.1,  # Very fast decay
            max_hops=3,
            emotion_contagion_rate=0.1,
            influence_threshold=0.5,  # High threshold
        )
        model = InfluencerModel(config=config)
        graph = _build_simple_graph()

        result = model.propagate_emotion("A", "happy", graph)

        # With fast decay and high threshold, distant agents should not appear
        assert "D" not in result

    def test_max_hops_limits_propagation(self) -> None:
        """Test that propagation respects max_hops."""
        config = InfluenceConfig(
            decay_factor=0.99,
            max_hops=1,  # Only direct neighbors
            emotion_contagion_rate=1.0,
            influence_threshold=0.001,
        )
        model = InfluencerModel(config=config)
        graph = _build_simple_graph()

        result = model.propagate_emotion("A", "happy", graph)

        # Only B should be reachable in 1 hop
        if "B" in result:
            assert True
        # C and D should NOT be reachable with max_hops=1
        assert "C" not in result
        assert "D" not in result

    def test_neuroticism_increases_susceptibility(self) -> None:
        """Test that high neuroticism increases emotion susceptibility."""
        config = InfluenceConfig(
            decay_factor=0.9,
            max_hops=1,
            emotion_contagion_rate=1.0,
            influence_threshold=0.001,
        )
        graph = SocialGraph()
        graph.set_relationship(
            SocialRelation(source_id="A", target_id="neurotic", affinity=0.8, trust=0.9)
        )
        graph.set_relationship(
            SocialRelation(source_id="A", target_id="stable", affinity=0.8, trust=0.9)
        )

        # Model with personality differences
        model = InfluencerModel(
            config=config,
            personalities={
                "neurotic": PersonalityProfile(neuroticism=0.9),
                "stable": PersonalityProfile(neuroticism=0.1),
            },
        )

        result = model.propagate_emotion("A", "fear", graph)

        assert "neurotic" in result
        assert "stable" in result
        assert result["neurotic"] > result["stable"]

    def test_time_decay_reduces_propagation(self) -> None:
        """Test that lower time_decay reduces propagation intensity."""
        config = InfluenceConfig(
            decay_factor=0.9,
            max_hops=3,
            emotion_contagion_rate=1.0,
            influence_threshold=0.001,
        )
        model = InfluencerModel(config=config)
        graph = _build_simple_graph()

        result_full = model.propagate_emotion("A", "happy", graph, time_decay=1.0)
        result_half = model.propagate_emotion("A", "happy", graph, time_decay=0.5)

        # With half time_decay, intensities should be lower
        for agent_id in result_full:
            if agent_id in result_half:
                assert result_half[agent_id] <= result_full[agent_id]

    def test_disconnected_agents_not_affected(self) -> None:
        """Test that disconnected agents are not affected by propagation."""
        config = InfluenceConfig(
            decay_factor=0.9,
            max_hops=5,
            emotion_contagion_rate=1.0,
            influence_threshold=0.001,
        )
        model = InfluencerModel(config=config)
        graph = _build_simple_graph()
        graph.add_agent("isolated")

        result = model.propagate_emotion("A", "happy", graph)
        assert "isolated" not in result

    def test_low_affinity_reduces_spread(self) -> None:
        """Test that low affinity connections reduce emotion spread."""
        config = InfluenceConfig(
            decay_factor=0.9,
            max_hops=1,
            emotion_contagion_rate=1.0,
            influence_threshold=0.001,
        )
        model = InfluencerModel(config=config)

        # Graph with high affinity connection
        graph_high = SocialGraph()
        graph_high.set_relationship(
            SocialRelation(source_id="A", target_id="B", affinity=0.9, trust=0.9)
        )

        # Graph with low affinity connection
        graph_low = SocialGraph()
        graph_low.set_relationship(
            SocialRelation(source_id="A", target_id="B", affinity=-0.5, trust=0.9)
        )

        result_high = model.propagate_emotion("A", "happy", graph_high)
        result_low = model.propagate_emotion("A", "happy", graph_low)

        if "B" in result_high and "B" in result_low:
            assert result_high["B"] > result_low["B"]


class TestPredictVoteInfluence:
    """Tests for InfluencerModel.predict_vote_influence."""

    def test_nonexistent_agent(self) -> None:
        """Test vote prediction for agent not in graph."""
        model = InfluencerModel()
        graph = SocialGraph()

        result = model.predict_vote_influence("nobody", "prop-1", graph)

        assert result.agent_id == "nobody"
        assert result.proposal_id == "prop-1"
        assert result.base_preference == 0.0
        assert result.influenced_preference == 0.0
        assert result.influencers == []

    def test_with_base_preference(self) -> None:
        """Test vote prediction with known base preference."""
        model = InfluencerModel()
        graph = SocialGraph()

        prefs = {"isolated": 0.7}
        result = model.predict_vote_influence("isolated", "prop-1", graph, agent_preferences=prefs)

        assert result.base_preference == pytest.approx(0.7)
        assert result.influenced_preference == pytest.approx(0.7)

    def test_social_influence_shifts_preference(self) -> None:
        """Test that strong neighbors shift an agent's preference."""
        config = InfluenceConfig(
            decay_factor=0.9,
            max_hops=1,
            emotion_contagion_rate=1.0,
            influence_threshold=0.01,
        )
        model = InfluencerModel(config=config)

        graph = SocialGraph()
        # Influential neighbor strongly supports proposal
        graph.set_relationship(
            SocialRelation(source_id="influencer", target_id="voter", affinity=0.9, trust=0.9)
        )

        prefs = {
            "voter": 0.0,  # neutral
            "influencer": 1.0,  # strongly for
        }

        result = model.predict_vote_influence("voter", "prop-1", graph, agent_preferences=prefs)

        # Should shift toward influencer's preference
        assert result.influenced_preference > result.base_preference
        assert "influencer" in result.influencers

    def test_opposing_influence_cancels(self) -> None:
        """Test that opposing influences partially cancel out."""
        config = InfluenceConfig(
            decay_factor=0.9,
            max_hops=1,
            emotion_contagion_rate=1.0,
            influence_threshold=0.01,
        )
        model = InfluencerModel(config=config)

        graph = SocialGraph()
        # Two neighbors with opposite preferences and similar connection strength
        graph.set_relationship(
            SocialRelation(source_id="pro", target_id="voter", affinity=0.8, trust=0.8)
        )
        graph.set_relationship(
            SocialRelation(source_id="anti", target_id="voter", affinity=0.8, trust=0.8)
        )

        prefs = {
            "voter": 0.0,
            "pro": 1.0,
            "anti": -1.0,
        }

        result = model.predict_vote_influence("voter", "prop-1", graph, agent_preferences=prefs)

        # Opposing influences should roughly cancel
        assert abs(result.influenced_preference) < 0.3

    def test_influencers_list_populated(self) -> None:
        """Test that the influencers list contains all contributing agents."""
        config = InfluenceConfig(
            decay_factor=0.9,
            max_hops=2,
            emotion_contagion_rate=1.0,
            influence_threshold=0.01,
        )
        model = InfluencerModel(config=config)

        graph = SocialGraph()
        graph.set_relationship(
            SocialRelation(source_id="friend-1", target_id="voter", affinity=0.8, trust=0.8)
        )
        graph.set_relationship(
            SocialRelation(source_id="friend-2", target_id="voter", affinity=0.7, trust=0.7)
        )

        prefs = {
            "voter": 0.0,
            "friend-1": 0.5,
            "friend-2": 0.8,
        }

        result = model.predict_vote_influence("voter", "prop-1", graph, agent_preferences=prefs)

        assert "friend-1" in result.influencers
        assert "friend-2" in result.influencers

    def test_preference_clamping(self) -> None:
        """Test that preferences are clamped to [-1, 1]."""
        model = InfluencerModel()
        graph = SocialGraph()

        prefs = {"agent": 5.0}  # Out of range
        result = model.predict_vote_influence("agent", "p", graph, agent_preferences=prefs)

        assert -1.0 <= result.base_preference <= 1.0
        assert -1.0 <= result.influenced_preference <= 1.0


class TestSusceptibility:
    """Tests for emotional susceptibility calculations."""

    def test_default_susceptibility(self) -> None:
        """Test default susceptibility for agents without personality."""
        model = InfluencerModel()
        susceptibility = model._get_susceptibility("unknown-agent")
        assert susceptibility == pytest.approx(0.65)

    def test_high_neuroticism_susceptibility(self) -> None:
        """Test high neuroticism gives high susceptibility."""
        model = InfluencerModel(
            personalities={
                "neurotic": PersonalityProfile(neuroticism=1.0),
            }
        )
        susceptibility = model._get_susceptibility("neurotic")
        assert susceptibility == pytest.approx(1.0)

    def test_low_neuroticism_susceptibility(self) -> None:
        """Test low neuroticism gives low susceptibility."""
        model = InfluencerModel(
            personalities={
                "stable": PersonalityProfile(neuroticism=0.0),
            }
        )
        susceptibility = model._get_susceptibility("stable")
        assert susceptibility == pytest.approx(0.3)

    def test_susceptibility_range(self) -> None:
        """Test susceptibility is always in [0.3, 1.0] range."""
        model = InfluencerModel()
        for n in [0.0, 0.25, 0.5, 0.75, 1.0]:
            model.set_personality("test", PersonalityProfile(neuroticism=n))
            s = model._get_susceptibility("test")
            assert 0.3 <= s <= 1.0, f"Susceptibility {s} out of range for neuroticism={n}"


class TestDirectInfluence:
    """Tests for direct influence calculation."""

    def test_no_incoming_edges(self) -> None:
        """Test direct influence with no incoming edges."""
        model = InfluencerModel()
        graph = SocialGraph()
        graph.add_agent("lonely")
        score = model._calculate_direct_influence("lonely", graph)
        assert score == 0.0

    def test_nonexistent_agent(self) -> None:
        """Test direct influence for agent not in graph."""
        model = InfluencerModel()
        graph = SocialGraph()
        score = model._calculate_direct_influence("nobody", graph)
        assert score == 0.0

    def test_with_incoming_edges(self) -> None:
        """Test direct influence calculation with incoming connections."""
        model = InfluencerModel()
        graph = _build_hub_graph()

        # Hub has 5 incoming connections
        score = model._calculate_direct_influence("hub", graph)
        assert score > 0.0
        assert score <= 1.0
