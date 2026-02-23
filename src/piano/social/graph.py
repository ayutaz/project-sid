"""NetworkX-based social graph for tracking agent relationships.

This module implements a directed weighted graph where nodes represent agents
and edges represent relationships with attributes like affinity and trust.
"""

from __future__ import annotations

__all__ = ["SocialGraph", "SocialRelation"]

from datetime import UTC, datetime
from typing import Any

import networkx as nx
import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)


class SocialRelation(BaseModel):
    """A relationship between two agents.

    Represents a directed edge in the social graph with attributes
    for affinity, trust, and interaction history.
    """

    source_id: str
    target_id: str
    affinity: float = 0.0  # -1.0 (hate) to 1.0 (love)
    trust: float = 0.5  # 0.0 (no trust) to 1.0 (complete trust)
    interaction_count: int = 0
    last_interaction: datetime | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("affinity")
    @classmethod
    def clamp_affinity(cls, v: float) -> float:
        """Clamp affinity to [-1.0, 1.0] range."""
        return max(-1.0, min(1.0, v))

    @field_validator("trust")
    @classmethod
    def clamp_trust(cls, v: float) -> float:
        """Clamp trust to [0.0, 1.0] range."""
        return max(0.0, min(1.0, v))


class SocialGraph:
    """Directed weighted graph for tracking social relationships.

    Each node represents an agent, and each edge represents a directed
    relationship with attributes like affinity and trust.
    """

    def __init__(self) -> None:
        """Initialize an empty social graph."""
        self._graph = nx.DiGraph()
        logger.info("social_graph_initialized")

    def add_agent(self, agent_id: str) -> None:
        """Add an agent node to the graph.

        Args:
            agent_id: Unique identifier for the agent
        """
        if agent_id not in self._graph:
            self._graph.add_node(agent_id)
            logger.debug("agent_added", agent_id=agent_id)

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent node and all its relationships.

        Args:
            agent_id: Agent to remove
        """
        if agent_id in self._graph:
            self._graph.remove_node(agent_id)
            logger.debug("agent_removed", agent_id=agent_id)

    def update_relationship(
        self,
        source: str,
        target: str,
        affinity_delta: float = 0.0,
        trust_delta: float = 0.0,
    ) -> SocialRelation:
        """Update or create a relationship between two agents.

        Args:
            source: Source agent ID
            target: Target agent ID
            affinity_delta: Change in affinity (-1 to 1)
            trust_delta: Change in trust (0 to 1)

        Returns:
            Updated SocialRelation
        """
        # Ensure both agents exist
        self.add_agent(source)
        self.add_agent(target)

        # Get existing relationship or create new one
        if self._graph.has_edge(source, target):
            existing = self._graph[source][target]["relation"]
            new_affinity = existing.affinity + affinity_delta
            new_trust = existing.trust + trust_delta
            new_count = existing.interaction_count + 1
            tags = existing.tags
        else:
            new_affinity = affinity_delta
            new_trust = 0.5 + trust_delta
            new_count = 1
            tags = []

        # Create updated relation
        relation = SocialRelation(
            source_id=source,
            target_id=target,
            affinity=new_affinity,
            trust=new_trust,
            interaction_count=new_count,
            last_interaction=datetime.now(UTC),
            tags=tags,
        )

        self._graph.add_edge(source, target, relation=relation)
        logger.debug(
            "relationship_updated",
            source=source,
            target=target,
            affinity=relation.affinity,
            trust=relation.trust,
        )

        return relation

    def set_relationship(self, relation: SocialRelation) -> None:
        """Set a relationship directly.

        Args:
            relation: Complete relationship to set
        """
        self.add_agent(relation.source_id)
        self.add_agent(relation.target_id)
        self._graph.add_edge(relation.source_id, relation.target_id, relation=relation)
        logger.debug(
            "relationship_set",
            source=relation.source_id,
            target=relation.target_id,
        )

    def get_relationship(self, source: str, target: str) -> SocialRelation | None:
        """Get the relationship from source to target.

        Args:
            source: Source agent ID
            target: Target agent ID

        Returns:
            SocialRelation if it exists, None otherwise
        """
        if self._graph.has_edge(source, target):
            return self._graph[source][target]["relation"]
        return None

    def get_friends(self, agent_id: str, min_affinity: float = 0.3) -> list[str]:
        """Get agents that this agent has positive affinity toward.

        Args:
            agent_id: Agent to get friends for
            min_affinity: Minimum affinity threshold (default 0.3)

        Returns:
            List of agent IDs
        """
        friends = []
        if agent_id not in self._graph:
            return friends

        for _, target in self._graph.out_edges(agent_id):
            relation = self._graph[agent_id][target]["relation"]
            if relation.affinity >= min_affinity:
                friends.append(target)

        return friends

    def get_enemies(self, agent_id: str, max_affinity: float = -0.3) -> list[str]:
        """Get agents that this agent has negative affinity toward.

        Args:
            agent_id: Agent to get enemies for
            max_affinity: Maximum affinity threshold (default -0.3)

        Returns:
            List of agent IDs
        """
        enemies = []
        if agent_id not in self._graph:
            return enemies

        for _, target in self._graph.out_edges(agent_id):
            relation = self._graph[agent_id][target]["relation"]
            if relation.affinity <= max_affinity:
                enemies.append(target)

        return enemies

    def get_strangers(self, agent_id: str) -> list[str]:
        """Get agents with no edge to this agent (no relationship).

        Args:
            agent_id: Agent to find strangers for

        Returns:
            List of agent IDs with no relationship
        """
        if agent_id not in self._graph:
            return []

        # Get all agents this agent has relationships with
        connected = set()
        for _, target in self._graph.out_edges(agent_id):
            connected.add(target)

        # Return all other agents (excluding self)
        strangers = [
            node for node in self._graph.nodes() if node != agent_id and node not in connected
        ]
        return strangers

    def get_influence_score(self, agent_id: str) -> float:
        """Calculate influence score using PageRank.

        Higher scores indicate more influence in the network.

        Args:
            agent_id: Agent to calculate influence for

        Returns:
            Influence score (0.0 to 1.0)
        """
        if agent_id not in self._graph:
            return 0.0

        if self._graph.number_of_edges() == 0:
            # No edges - all agents have equal influence
            return 1.0 / self._graph.number_of_nodes() if self._graph.number_of_nodes() > 0 else 0.0

        # Use affinity as edge weights (convert to 0-1 range)
        # affinity: -1 to 1 -> weight: 0 to 1
        weighted_graph = nx.DiGraph()
        for source, target in self._graph.edges():
            relation = self._graph[source][target]["relation"]
            weight = (relation.affinity + 1.0) / 2.0  # Convert -1..1 to 0..1
            weighted_graph.add_edge(source, target, weight=max(0.01, weight))

        try:
            pagerank = nx.pagerank(weighted_graph, weight="weight")
            return pagerank.get(agent_id, 0.0)
        except Exception as exc:
            logger.warning("pagerank_failed", error=str(exc))
            return 0.0

    def get_communities(self) -> list[set[str]]:
        """Detect communities using weakly connected components.

        Returns:
            List of sets, each containing agent IDs in a community
        """
        communities = list(nx.weakly_connected_components(self._graph))
        logger.debug("communities_detected", count=len(communities))
        return communities

    def get_social_distance(self, source: str, target: str) -> int | None:
        """Calculate shortest path length between two agents.

        Args:
            source: Source agent ID
            target: Target agent ID

        Returns:
            Shortest path length, or None if no path exists
        """
        if source not in self._graph or target not in self._graph:
            return None

        try:
            return nx.shortest_path_length(self._graph, source, target)
        except nx.NetworkXNoPath:
            return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to a dictionary.

        Returns:
            Dictionary with nodes and edges
        """
        nodes = list(self._graph.nodes())
        edges = []

        for source, target in self._graph.edges():
            relation = self._graph[source][target]["relation"]
            edges.append(
                {
                    "source_id": relation.source_id,
                    "target_id": relation.target_id,
                    "affinity": relation.affinity,
                    "trust": relation.trust,
                    "interaction_count": relation.interaction_count,
                    "last_interaction": (
                        relation.last_interaction.isoformat() if relation.last_interaction else None
                    ),
                    "tags": relation.tags,
                }
            )

        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SocialGraph:
        """Restore graph from a dictionary.

        Args:
            data: Dictionary with nodes and edges

        Returns:
            Restored SocialGraph
        """
        graph = cls()

        # Add nodes
        for node_id in data.get("nodes", []):
            graph.add_agent(node_id)

        # Add edges
        for edge_data in data.get("edges", []):
            # Parse last_interaction if present
            last_interaction = edge_data.get("last_interaction")
            if last_interaction:
                last_interaction = datetime.fromisoformat(last_interaction)

            relation = SocialRelation(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                affinity=edge_data["affinity"],
                trust=edge_data["trust"],
                interaction_count=edge_data["interaction_count"],
                last_interaction=last_interaction,
                tags=edge_data.get("tags", []),
            )
            graph.set_relationship(relation)

        logger.info(
            "social_graph_restored",
            node_count=graph.agent_count,
            edge_count=graph.edge_count,
        )
        return graph

    @property
    def agent_count(self) -> int:
        """Number of agents in the graph."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Number of relationships in the graph."""
        return self._graph.number_of_edges()
