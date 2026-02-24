"""Influencer mechanism for modeling social influence propagation.

This module implements influence calculation, emotion propagation, and vote
influence prediction based on the social graph structure. Key features:
- PageRank-based influence scoring
- SIR-model-inspired emotion contagion with personality-based susceptibility
- Vote influence prediction through weighted social connections

Reference: docs/implementation/06-social-cognition.md
"""

from __future__ import annotations

__all__ = [
    "EmotionPropagation",
    "InfluenceConfig",
    "InfluencerModel",
    "VoteInfluence",
]

from collections import deque
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from piano.social.graph import SocialGraph
    from piano.social.personality import PersonalityProfile

logger = structlog.get_logger(__name__)


class InfluenceConfig(BaseModel):
    """Configuration for the influencer model.

    Attributes:
        decay_factor: Exponential decay factor per hop (0-1). Lower values
            mean faster decay with distance.
        max_hops: Maximum number of hops for influence propagation.
        emotion_contagion_rate: Base rate of emotion spread between agents (0-1).
        influence_threshold: Minimum influence level to consider an agent
            as being influenced.
    """

    decay_factor: float = 0.5
    max_hops: int = 3
    emotion_contagion_rate: float = 0.3
    influence_threshold: float = 0.1

    @field_validator("decay_factor")
    @classmethod
    def validate_decay_factor(cls, v: float) -> float:
        """Ensure decay_factor is in [0.0, 1.0] range."""
        return max(0.0, min(1.0, v))

    @field_validator("max_hops")
    @classmethod
    def validate_max_hops(cls, v: int) -> int:
        """Ensure max_hops is positive."""
        return max(1, v)

    @field_validator("emotion_contagion_rate")
    @classmethod
    def validate_contagion_rate(cls, v: float) -> float:
        """Ensure emotion_contagion_rate is in [0.0, 1.0] range."""
        return max(0.0, min(1.0, v))

    @field_validator("influence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Ensure influence_threshold is in [0.0, 1.0] range."""
        return max(0.0, min(1.0, v))


class VoteInfluence(BaseModel):
    """Prediction of how social influence affects an agent's voting preference.

    Attributes:
        agent_id: The agent whose vote is being predicted.
        proposal_id: Identifier for the proposal being voted on.
        base_preference: The agent's original preference before influence (-1 to 1).
        influenced_preference: The preference after accounting for social influence (-1 to 1).
        influencers: List of agent IDs that contributed to the influence.
    """

    agent_id: str
    proposal_id: str
    base_preference: float = Field(default=0.0, ge=-1.0, le=1.0)
    influenced_preference: float = Field(default=0.0, ge=-1.0, le=1.0)
    influencers: list[str] = Field(default_factory=list)


class EmotionPropagation(BaseModel):
    """Result of emotion propagation through the social network.

    Uses an SIR-model-inspired approach where:
    - Susceptible: Agents who have not yet been affected
    - Infected: Agents currently experiencing the propagated emotion
    - Recovered: Agents who have been affected and returned to baseline

    Attributes:
        source_id: The agent originating the emotion.
        emotion: Name of the emotion being propagated.
        affected_agents: Mapping of agent IDs to received emotion intensity (0-1).
        propagation_path: Ordered list of (agent_id, hop_number) showing spread.
    """

    source_id: str
    emotion: str
    affected_agents: dict[str, float] = Field(default_factory=dict)
    propagation_path: list[tuple[str, int]] = Field(default_factory=list)


class InfluencerModel:
    """Model for calculating social influence and propagating effects.

    Combines PageRank-based global influence with local connection weights
    (affinity x trust) and distance-based decay to model realistic social
    influence patterns.

    Args:
        config: Configuration parameters for influence calculations.
        personalities: Mapping of agent IDs to their personality profiles.
            Used for emotion susceptibility calculations.
    """

    def __init__(
        self,
        config: InfluenceConfig | None = None,
        personalities: dict[str, PersonalityProfile] | None = None,
    ) -> None:
        """Initialize the influencer model.

        Args:
            config: Influence configuration. Uses defaults if None.
            personalities: Agent personality profiles for susceptibility.
        """
        self._config = config or InfluenceConfig()
        self._personalities = personalities or {}
        logger.info(
            "influencer_model_initialized",
            decay_factor=self._config.decay_factor,
            max_hops=self._config.max_hops,
            emotion_contagion_rate=self._config.emotion_contagion_rate,
        )

    @property
    def config(self) -> InfluenceConfig:
        """Current influence configuration."""
        return self._config

    def set_personality(self, agent_id: str, profile: PersonalityProfile) -> None:
        """Set or update an agent's personality profile.

        Args:
            agent_id: Agent identifier.
            profile: Personality profile for susceptibility calculations.
        """
        self._personalities[agent_id] = profile

    def calculate_influence(self, agent_id: str, social_graph: SocialGraph) -> float:
        """Calculate an agent's influence score in the social network.

        Combines PageRank centrality with direct connection strength to produce
        a normalized influence score. The score reflects both global network
        position and local relationship quality.

        Args:
            agent_id: Agent to calculate influence for.
            social_graph: The social graph representing the network.

        Returns:
            Influence score in [0.0, 1.0] range. Returns 0.0 if the agent
            is not in the graph.
        """
        if not social_graph.has_agent(agent_id):
            logger.debug("influence_agent_not_found", agent_id=agent_id)
            return 0.0

        if social_graph.agent_count == 0:
            return 0.0

        # Component 1: PageRank-based global influence
        pagerank_score = social_graph.get_influence_score(agent_id)

        # Component 2: Direct connection weight (average affinity x trust of incoming edges)
        direct_score = self._calculate_direct_influence(agent_id, social_graph)

        # Normalize to [0, 1] range
        # PageRank values are typically small fractions; scale up
        n_agents = social_graph.agent_count
        if n_agents > 1:
            # Scale PageRank relative to uniform distribution (1/N)
            uniform = 1.0 / n_agents
            scaled_pr = min(1.0, pagerank_score / (2.0 * uniform)) if uniform > 0 else 0.0
            normalized = 0.6 * scaled_pr + 0.4 * direct_score
        else:
            normalized = 1.0 if social_graph.agent_count == 1 else 0.0

        result = max(0.0, min(1.0, normalized))

        logger.debug(
            "influence_calculated",
            agent_id=agent_id,
            pagerank=pagerank_score,
            direct=direct_score,
            result=result,
        )

        return result

    def propagate_emotion(
        self,
        source_id: str,
        emotion: str,
        social_graph: SocialGraph,
        *,
        time_decay: float = 1.0,
    ) -> dict[str, float]:
        """Propagate an emotion from a source agent through the social network.

        Uses a BFS-based SIR-model approach where:
        - Emotion intensity decays exponentially with hop distance
        - Each agent's susceptibility is modulated by their neuroticism trait
        - The contagion rate determines base transmission probability
        - Only agents above the influence threshold are included

        Args:
            source_id: Agent originating the emotion.
            emotion: Name of the emotion to propagate.
            social_graph: The social graph.
            time_decay: Temporal decay multiplier (0-1). Lower values simulate
                older emotions that have weaker propagation.

        Returns:
            Mapping of affected agent IDs to received emotion intensity (0-1).
            Does not include the source agent.
        """
        if not social_graph.has_agent(source_id):
            logger.debug("propagate_source_not_found", source_id=source_id)
            return {}

        time_decay = max(0.0, min(1.0, time_decay))
        affected: dict[str, float] = {}
        visited: set[str] = {source_id}

        # BFS propagation
        queue: deque[tuple[str, int, float]] = deque()
        # Initialize with direct neighbors of source
        queue.append((source_id, 0, 1.0))

        while queue:
            current_id, current_hop, current_intensity = queue.popleft()

            if current_hop >= self._config.max_hops:
                continue

            # Get outgoing edges from current agent
            if not social_graph.has_agent(current_id):
                continue

            for target_id in social_graph.get_outgoing_neighbors(current_id):
                if target_id in visited:
                    continue

                visited.add(target_id)

                # Calculate transmission intensity
                relation = social_graph.get_relationship(current_id, target_id)
                if relation is None:
                    continue

                # Edge weight: affinity (0-1 normalized) * trust
                affinity_normalized = (relation.affinity + 1.0) / 2.0
                edge_weight = affinity_normalized * relation.trust

                # Per-hop decay: apply decay_factor once per hop (not cumulative
                # with current_intensity which already carries prior decay).
                hop_number = current_hop + 1
                per_hop_decay = self._config.decay_factor

                # Susceptibility based on neuroticism
                susceptibility = self._get_susceptibility(target_id)

                # Final intensity
                intensity = (
                    current_intensity
                    * edge_weight
                    * per_hop_decay
                    * self._config.emotion_contagion_rate
                    * susceptibility
                    * time_decay
                )

                if intensity >= self._config.influence_threshold:
                    affected[target_id] = max(0.0, min(1.0, intensity))
                    queue.append((target_id, hop_number, intensity))

        logger.info(
            "emotion_propagated",
            source_id=source_id,
            emotion=emotion,
            affected_count=len(affected),
        )

        return affected

    def predict_vote_influence(
        self,
        agent_id: str,
        proposal: str,
        social_graph: SocialGraph,
        agent_preferences: dict[str, float] | None = None,
    ) -> VoteInfluence:
        """Predict how social influence affects an agent's voting preference.

        Calculates the weighted average of neighboring agents' preferences,
        where weights are determined by connection strength (affinity x trust)
        and distance decay. The agent's base preference is then blended with
        the social influence.

        Args:
            agent_id: The agent whose vote influence to predict.
            proposal: Identifier for the proposal being voted on.
            social_graph: The social graph.
            agent_preferences: Known preferences of agents for this proposal.
                Maps agent_id to preference score (-1 to 1).

        Returns:
            VoteInfluence with base and influenced preferences.
        """
        preferences = agent_preferences or {}
        base_preference = preferences.get(agent_id, 0.0)
        base_preference = max(-1.0, min(1.0, base_preference))

        if not social_graph.has_agent(agent_id):
            return VoteInfluence(
                agent_id=agent_id,
                proposal_id=proposal,
                base_preference=base_preference,
                influenced_preference=base_preference,
                influencers=[],
            )

        # Gather influence from neighbors within max_hops
        influence_sum = 0.0
        weight_sum = 0.0
        influencers: list[str] = []

        visited: set[str] = {agent_id}
        queue: deque[tuple[str, int]] = deque()
        queue.append((agent_id, 0))

        while queue:
            current_id, current_hop = queue.popleft()

            if current_hop >= self._config.max_hops:
                continue

            if not social_graph.has_agent(current_id):
                continue

            # Check incoming edges (agents that influence the target)
            for neighbor_id in social_graph.get_incoming_neighbors(current_id):
                if neighbor_id in visited:
                    continue
                if neighbor_id == agent_id:
                    continue

                visited.add(neighbor_id)

                # Get relationship from neighbor to current agent
                relation = social_graph.get_relationship(neighbor_id, current_id)
                if relation is None:
                    continue

                # Connection weight
                affinity_normalized = (relation.affinity + 1.0) / 2.0
                edge_weight = affinity_normalized * relation.trust

                # Distance decay
                hop_number = current_hop + 1
                distance_decay = self._config.decay_factor**hop_number

                weight = edge_weight * distance_decay

                if weight < self._config.influence_threshold:
                    continue

                # Get neighbor's preference if known
                neighbor_pref = preferences.get(neighbor_id, 0.0)
                neighbor_pref = max(-1.0, min(1.0, neighbor_pref))

                influence_sum += neighbor_pref * weight
                weight_sum += weight
                influencers.append(neighbor_id)

                queue.append((neighbor_id, hop_number))

        # Blend base preference with social influence
        if weight_sum > 0:
            social_influence = influence_sum / weight_sum
            # 70% own preference, 30% social influence
            influenced = 0.7 * base_preference + 0.3 * social_influence
        else:
            influenced = base_preference

        influenced = max(-1.0, min(1.0, influenced))

        logger.info(
            "vote_influence_predicted",
            agent_id=agent_id,
            proposal=proposal,
            base_preference=base_preference,
            influenced_preference=influenced,
            influencer_count=len(influencers),
        )

        return VoteInfluence(
            agent_id=agent_id,
            proposal_id=proposal,
            base_preference=base_preference,
            influenced_preference=influenced,
            influencers=influencers,
        )

    def _calculate_direct_influence(self, agent_id: str, social_graph: SocialGraph) -> float:
        """Calculate influence from direct incoming connections.

        Averages the product of affinity (normalized to 0-1) and trust for
        all incoming edges to the agent.

        Args:
            agent_id: Agent to calculate direct influence for.
            social_graph: The social graph.

        Returns:
            Direct influence score in [0.0, 1.0] range.
        """
        if not social_graph.has_agent(agent_id):
            return 0.0

        in_neighbors = social_graph.get_incoming_neighbors(agent_id)
        if not in_neighbors:
            return 0.0

        total_weight = 0.0
        for source in in_neighbors:
            relation = social_graph.get_relationship(source, agent_id)
            if relation is not None:
                affinity_normalized = (relation.affinity + 1.0) / 2.0
                total_weight += affinity_normalized * relation.trust

        return min(1.0, total_weight / len(in_neighbors))

    def _get_susceptibility(self, agent_id: str) -> float:
        """Get an agent's emotional susceptibility based on neuroticism.

        Higher neuroticism leads to higher susceptibility to emotional
        contagion. Agents without a personality profile get a default
        moderate susceptibility.

        Args:
            agent_id: Agent to get susceptibility for.

        Returns:
            Susceptibility score in [0.3, 1.0] range.
        """
        profile = self._personalities.get(agent_id)
        if profile is None:
            return 0.65  # Default moderate susceptibility

        # Neuroticism 0.0 -> susceptibility 0.3 (emotionally stable)
        # Neuroticism 1.0 -> susceptibility 1.0 (emotionally reactive)
        return 0.3 + 0.7 * profile.neuroticism
