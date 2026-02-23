"""Tests for the social graph module."""

from __future__ import annotations

from datetime import UTC, datetime

from piano.social.graph import SocialGraph, SocialRelation


class TestSocialRelation:
    """Tests for SocialRelation model."""

    def test_default_values(self):
        """Test default values are set correctly."""
        relation = SocialRelation(source_id="agent-1", target_id="agent-2")
        assert relation.source_id == "agent-1"
        assert relation.target_id == "agent-2"
        assert relation.affinity == 0.0
        assert relation.trust == 0.5
        assert relation.interaction_count == 0
        assert relation.last_interaction is None
        assert relation.tags == []

    def test_affinity_clamping(self):
        """Test affinity is clamped to [-1, 1] range."""
        # Below minimum
        relation = SocialRelation(source_id="a", target_id="b", affinity=-2.0)
        assert relation.affinity == -1.0

        # Above maximum
        relation = SocialRelation(source_id="a", target_id="b", affinity=2.0)
        assert relation.affinity == 1.0

        # Within range
        relation = SocialRelation(source_id="a", target_id="b", affinity=0.5)
        assert relation.affinity == 0.5

    def test_trust_clamping(self):
        """Test trust is clamped to [0, 1] range."""
        # Below minimum
        relation = SocialRelation(source_id="a", target_id="b", trust=-0.5)
        assert relation.trust == 0.0

        # Above maximum
        relation = SocialRelation(source_id="a", target_id="b", trust=1.5)
        assert relation.trust == 1.0

        # Within range
        relation = SocialRelation(source_id="a", target_id="b", trust=0.7)
        assert relation.trust == 0.7


class TestSocialGraph:
    """Tests for SocialGraph."""

    def test_initialization(self):
        """Test graph initializes empty."""
        graph = SocialGraph()
        assert graph.agent_count == 0
        assert graph.edge_count == 0

    def test_add_agent(self):
        """Test adding agents to the graph."""
        graph = SocialGraph()
        graph.add_agent("agent-1")
        assert graph.agent_count == 1

        # Adding same agent again should not increase count
        graph.add_agent("agent-1")
        assert graph.agent_count == 1

        # Adding different agent should increase count
        graph.add_agent("agent-2")
        assert graph.agent_count == 2

    def test_remove_agent(self):
        """Test removing agents from the graph."""
        graph = SocialGraph()
        graph.add_agent("agent-1")
        graph.add_agent("agent-2")
        assert graph.agent_count == 2

        graph.remove_agent("agent-1")
        assert graph.agent_count == 1

        # Removing non-existent agent should not error
        graph.remove_agent("non-existent")
        assert graph.agent_count == 1

    def test_remove_agent_removes_edges(self):
        """Test removing an agent also removes all its edges."""
        graph = SocialGraph()
        graph.update_relationship("agent-1", "agent-2", affinity_delta=0.5)
        graph.update_relationship("agent-2", "agent-1", affinity_delta=0.3)
        assert graph.edge_count == 2

        graph.remove_agent("agent-1")
        assert graph.edge_count == 0

    def test_update_relationship_creates_edge(self):
        """Test updating a non-existent relationship creates it."""
        graph = SocialGraph()
        relation = graph.update_relationship(
            "agent-1", "agent-2", affinity_delta=0.5, trust_delta=0.1
        )

        assert graph.edge_count == 1
        assert relation.affinity == 0.5
        assert relation.trust == 0.6  # 0.5 default + 0.1 delta
        assert relation.interaction_count == 1
        assert relation.last_interaction is not None

    def test_update_relationship_updates_existing(self):
        """Test updating an existing relationship."""
        graph = SocialGraph()
        # Create initial relationship
        graph.update_relationship("agent-1", "agent-2", affinity_delta=0.3, trust_delta=0.1)

        # Update it
        relation = graph.update_relationship(
            "agent-1", "agent-2", affinity_delta=0.2, trust_delta=-0.1
        )

        assert graph.edge_count == 1
        assert relation.affinity == 0.5  # 0.3 + 0.2
        assert relation.trust == 0.5  # 0.6 - 0.1
        assert relation.interaction_count == 2

    def test_update_relationship_auto_adds_agents(self):
        """Test updating relationship automatically adds agents if they don't exist."""
        graph = SocialGraph()
        graph.update_relationship("agent-1", "agent-2", affinity_delta=0.5)

        assert graph.agent_count == 2
        assert "agent-1" in graph._graph
        assert "agent-2" in graph._graph

    def test_set_relationship(self):
        """Test setting a relationship directly."""
        graph = SocialGraph()
        relation = SocialRelation(
            source_id="agent-1",
            target_id="agent-2",
            affinity=0.8,
            trust=0.9,
            interaction_count=5,
            last_interaction=datetime.now(UTC),
            tags=["friend", "ally"],
        )

        graph.set_relationship(relation)
        assert graph.edge_count == 1

        retrieved = graph.get_relationship("agent-1", "agent-2")
        assert retrieved is not None
        assert retrieved.affinity == 0.8
        assert retrieved.trust == 0.9
        assert retrieved.interaction_count == 5
        assert len(retrieved.tags) == 2

    def test_get_relationship(self):
        """Test retrieving relationships."""
        graph = SocialGraph()
        graph.update_relationship("agent-1", "agent-2", affinity_delta=0.5)

        # Existing relationship
        relation = graph.get_relationship("agent-1", "agent-2")
        assert relation is not None
        assert relation.affinity == 0.5

        # Non-existent relationship
        relation = graph.get_relationship("agent-2", "agent-1")
        assert relation is None

    def test_get_friends(self):
        """Test getting friends with positive affinity."""
        graph = SocialGraph()
        graph.update_relationship("agent-1", "agent-2", affinity_delta=0.5)
        graph.update_relationship("agent-1", "agent-3", affinity_delta=0.8)
        graph.update_relationship("agent-1", "agent-4", affinity_delta=0.2)  # Below threshold
        graph.update_relationship("agent-1", "agent-5", affinity_delta=-0.5)  # Negative

        friends = graph.get_friends("agent-1", min_affinity=0.3)
        assert len(friends) == 2
        assert "agent-2" in friends
        assert "agent-3" in friends

    def test_get_friends_empty_graph(self):
        """Test getting friends for non-existent agent."""
        graph = SocialGraph()
        friends = graph.get_friends("non-existent")
        assert friends == []

    def test_get_enemies(self):
        """Test getting enemies with negative affinity."""
        graph = SocialGraph()
        graph.update_relationship("agent-1", "agent-2", affinity_delta=-0.5)
        graph.update_relationship("agent-1", "agent-3", affinity_delta=-0.8)
        graph.update_relationship("agent-1", "agent-4", affinity_delta=-0.2)  # Above threshold
        graph.update_relationship("agent-1", "agent-5", affinity_delta=0.5)  # Positive

        enemies = graph.get_enemies("agent-1", max_affinity=-0.3)
        assert len(enemies) == 2
        assert "agent-2" in enemies
        assert "agent-3" in enemies

    def test_get_strangers(self):
        """Test getting agents with no relationship."""
        graph = SocialGraph()
        graph.add_agent("agent-1")
        graph.add_agent("agent-2")
        graph.add_agent("agent-3")
        graph.add_agent("agent-4")

        # agent-1 has relationship with agent-2
        graph.update_relationship("agent-1", "agent-2", affinity_delta=0.5)

        strangers = graph.get_strangers("agent-1")
        assert len(strangers) == 2
        assert "agent-3" in strangers
        assert "agent-4" in strangers
        assert "agent-1" not in strangers  # Should not include self
        assert "agent-2" not in strangers  # Has relationship

    def test_get_strangers_empty_graph(self):
        """Test getting strangers for non-existent agent."""
        graph = SocialGraph()
        strangers = graph.get_strangers("non-existent")
        assert strangers == []

    def test_influence_score_empty_graph(self):
        """Test influence score in empty graph."""
        graph = SocialGraph()
        graph.add_agent("agent-1")
        score = graph.get_influence_score("agent-1")
        assert score == 1.0  # Only agent has 100% influence

    def test_influence_score_with_connections(self):
        """Test influence score increases with more connections."""
        graph = SocialGraph()
        # Create a hub node
        for i in range(5):
            graph.update_relationship(f"agent-{i}", "hub", affinity_delta=0.8)

        # Create a less connected node
        graph.update_relationship("agent-0", "loner", affinity_delta=0.5)

        hub_score = graph.get_influence_score("hub")
        loner_score = graph.get_influence_score("loner")

        # Hub should have higher influence
        assert hub_score > loner_score

    def test_influence_score_non_existent(self):
        """Test influence score for non-existent agent."""
        graph = SocialGraph()
        score = graph.get_influence_score("non-existent")
        assert score == 0.0

    def test_communities_detection(self):
        """Test community detection."""
        graph = SocialGraph()
        # Create two separate communities
        graph.update_relationship("a1", "a2", affinity_delta=0.5)
        graph.update_relationship("a2", "a3", affinity_delta=0.5)

        graph.update_relationship("b1", "b2", affinity_delta=0.5)
        graph.update_relationship("b2", "b3", affinity_delta=0.5)

        communities = graph.get_communities()
        assert len(communities) == 2

        # Find which community has a1
        comm_a = None
        comm_b = None
        for comm in communities:
            if "a1" in comm:
                comm_a = comm
            if "b1" in comm:
                comm_b = comm

        assert comm_a is not None
        assert comm_b is not None
        assert "a1" in comm_a and "a2" in comm_a and "a3" in comm_a
        assert "b1" in comm_b and "b2" in comm_b and "b3" in comm_b

    def test_communities_single_component(self):
        """Test community detection with fully connected graph."""
        graph = SocialGraph()
        graph.update_relationship("agent-1", "agent-2", affinity_delta=0.5)
        graph.update_relationship("agent-2", "agent-3", affinity_delta=0.5)
        graph.update_relationship("agent-3", "agent-1", affinity_delta=0.5)

        communities = graph.get_communities()
        assert len(communities) == 1
        assert len(communities[0]) == 3

    def test_social_distance(self):
        """Test shortest path calculation."""
        graph = SocialGraph()
        graph.update_relationship("agent-1", "agent-2", affinity_delta=0.5)
        graph.update_relationship("agent-2", "agent-3", affinity_delta=0.5)
        graph.update_relationship("agent-3", "agent-4", affinity_delta=0.5)

        # Direct connection
        assert graph.get_social_distance("agent-1", "agent-2") == 1

        # Two hops
        assert graph.get_social_distance("agent-1", "agent-3") == 2

        # Three hops
        assert graph.get_social_distance("agent-1", "agent-4") == 3

    def test_social_distance_no_path(self):
        """Test social distance when no path exists."""
        graph = SocialGraph()
        graph.add_agent("agent-1")
        graph.add_agent("agent-2")

        # No connection
        assert graph.get_social_distance("agent-1", "agent-2") is None

    def test_social_distance_non_existent_agents(self):
        """Test social distance with non-existent agents."""
        graph = SocialGraph()
        assert graph.get_social_distance("non-existent-1", "non-existent-2") is None

    def test_serialization_round_trip(self):
        """Test serializing and deserializing graph."""
        graph = SocialGraph()
        graph.update_relationship("agent-1", "agent-2", affinity_delta=0.5, trust_delta=0.2)
        graph.update_relationship("agent-2", "agent-3", affinity_delta=-0.3, trust_delta=-0.1)

        # Add a relationship with tags
        relation = SocialRelation(
            source_id="agent-3",
            target_id="agent-1",
            affinity=0.8,
            trust=0.9,
            interaction_count=10,
            last_interaction=datetime.now(UTC),
            tags=["friend", "ally"],
        )
        graph.set_relationship(relation)

        # Serialize
        data = graph.to_dict()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 3

        # Deserialize
        restored = SocialGraph.from_dict(data)
        assert restored.agent_count == 3
        assert restored.edge_count == 3

        # Check relationships are preserved
        rel_12 = restored.get_relationship("agent-1", "agent-2")
        assert rel_12 is not None
        assert rel_12.affinity == 0.5
        assert abs(rel_12.trust - 0.7) < 0.01

        rel_31 = restored.get_relationship("agent-3", "agent-1")
        assert rel_31 is not None
        assert rel_31.affinity == 0.8
        assert rel_31.trust == 0.9
        assert rel_31.interaction_count == 10
        assert len(rel_31.tags) == 2

    def test_serialization_empty_graph(self):
        """Test serializing empty graph."""
        graph = SocialGraph()
        data = graph.to_dict()
        assert data["nodes"] == []
        assert data["edges"] == []

        restored = SocialGraph.from_dict(data)
        assert restored.agent_count == 0
        assert restored.edge_count == 0

    def test_serialization_preserves_timestamps(self):
        """Test that timestamps are preserved in serialization."""
        graph = SocialGraph()
        now = datetime.now(UTC)
        relation = SocialRelation(
            source_id="agent-1",
            target_id="agent-2",
            affinity=0.5,
            trust=0.7,
            interaction_count=3,
            last_interaction=now,
        )
        graph.set_relationship(relation)

        # Serialize and deserialize
        data = graph.to_dict()
        restored = SocialGraph.from_dict(data)

        rel = restored.get_relationship("agent-1", "agent-2")
        assert rel is not None
        assert rel.last_interaction is not None
        # Compare timestamps (allow small difference due to serialization)
        time_diff = abs((rel.last_interaction - now).total_seconds())
        assert time_diff < 1

    def test_directed_graph(self):
        """Test that the graph is directed (A->B != B->A)."""
        graph = SocialGraph()
        graph.update_relationship("agent-1", "agent-2", affinity_delta=0.8)

        # agent-1 -> agent-2 exists
        rel_12 = graph.get_relationship("agent-1", "agent-2")
        assert rel_12 is not None
        assert rel_12.affinity == 0.8

        # agent-2 -> agent-1 does not exist
        rel_21 = graph.get_relationship("agent-2", "agent-1")
        assert rel_21 is None

    def test_affinity_updates_accumulate(self):
        """Test that affinity updates accumulate correctly."""
        graph = SocialGraph()
        graph.update_relationship("agent-1", "agent-2", affinity_delta=0.3)
        graph.update_relationship("agent-1", "agent-2", affinity_delta=0.2)
        graph.update_relationship("agent-1", "agent-2", affinity_delta=-0.1)

        rel = graph.get_relationship("agent-1", "agent-2")
        assert rel is not None
        assert abs(rel.affinity - 0.4) < 0.01  # 0.3 + 0.2 - 0.1
        assert rel.interaction_count == 3

    def test_trust_updates_accumulate(self):
        """Test that trust updates accumulate correctly."""
        graph = SocialGraph()
        graph.update_relationship("agent-1", "agent-2", trust_delta=0.2)
        graph.update_relationship("agent-1", "agent-2", trust_delta=0.1)

        rel = graph.get_relationship("agent-1", "agent-2")
        assert rel is not None
        assert abs(rel.trust - 0.8) < 0.01  # 0.5 + 0.2 + 0.1

    def test_extreme_affinity_clamping_in_updates(self):
        """Test that affinity stays clamped during multiple updates."""
        graph = SocialGraph()
        # Push affinity beyond upper limit
        graph.update_relationship("agent-1", "agent-2", affinity_delta=0.6)
        graph.update_relationship("agent-1", "agent-2", affinity_delta=0.6)

        rel = graph.get_relationship("agent-1", "agent-2")
        assert rel is not None
        assert rel.affinity == 1.0  # Clamped at maximum

        # Push affinity beyond lower limit
        graph.update_relationship("agent-1", "agent-3", affinity_delta=-0.6)
        graph.update_relationship("agent-1", "agent-3", affinity_delta=-0.6)

        rel = graph.get_relationship("agent-1", "agent-3")
        assert rel is not None
        assert rel.affinity == -1.0  # Clamped at minimum

    def test_extreme_trust_clamping_in_updates(self):
        """Test that trust stays clamped during multiple updates."""
        graph = SocialGraph()
        # Push trust beyond upper limit
        graph.update_relationship("agent-1", "agent-2", trust_delta=0.4)
        graph.update_relationship("agent-1", "agent-2", trust_delta=0.4)

        rel = graph.get_relationship("agent-1", "agent-2")
        assert rel is not None
        assert rel.trust == 1.0  # Clamped at maximum

        # Push trust beyond lower limit
        graph.update_relationship("agent-1", "agent-3", trust_delta=-0.4)
        graph.update_relationship("agent-1", "agent-3", trust_delta=-0.4)

        rel = graph.get_relationship("agent-1", "agent-3")
        assert rel is not None
        assert rel.trust == 0.0  # Clamped at minimum
