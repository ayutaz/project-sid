"""Social cognition evaluation metrics for PIANO architecture.

This module implements metrics for:
1. Social cognition accuracy (Pearson correlation, sentiment prediction)
2. Emergent specialization (role entropy, role distribution)
3. Observer threshold analysis (wisdom of crowds effect)
4. Role persistence and stability over time
"""

import math
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field
from scipy import stats


class SentimentPrediction(BaseModel):
    """A single sentiment prediction by an agent about a target agent."""

    agent_id: str = Field(description="ID of the agent making the prediction")
    target_id: str = Field(description="ID of the target agent being evaluated")
    predicted: float = Field(description="Predicted sentiment/likeability score")
    actual: float = Field(description="Actual sentiment/likeability score")


class RoleAssignment(BaseModel):
    """Role assignment for an agent at a point in time."""

    agent_id: str = Field(description="ID of the agent")
    role: str = Field(description="Assigned role (e.g., farmer, miner, guard)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in assignment")


class SocialCognitionMetrics:
    """Metrics for social cognition accuracy.

    Measures how accurately agents can predict other agents' sentiments/likeability.
    Based on paper section 6 (Social Cognition Evaluation).
    """

    @staticmethod
    def pearson_correlation(predictions: list[SentimentPrediction]) -> float:
        """Calculate Pearson correlation between predicted and actual sentiments.

        Args:
            predictions: List of sentiment predictions

        Returns:
            Pearson correlation coefficient (r). Returns 0.0 if insufficient data.

        Note:
            Paper achieved r=0.807 with social cognition module,
            r=0.617 without it (50 agents, 4+ hour simulation).
        """
        if len(predictions) < 2:
            return 0.0

        predicted_values = [p.predicted for p in predictions]
        actual_values = [p.actual for p in predictions]

        # scipy.stats.pearsonr returns (correlation, p_value)
        r, _ = stats.pearsonr(predicted_values, actual_values)
        return float(r)

    @staticmethod
    def observer_threshold_analysis(
        predictions_by_observer_count: dict[int, list[SentimentPrediction]],
    ) -> dict[int, float]:
        """Analyze how accuracy changes with number of observers (wisdom of crowds).

        Args:
            predictions_by_observer_count: Dict mapping observer threshold to predictions
                e.g., {1: [all predictions], 5: [predictions with 5+ observers], ...}

        Returns:
            Dict mapping observer threshold to Pearson correlation coefficient

        Note:
            Paper shows correlation improves with observer threshold:
            - threshold=1: r=0.646 (n=46)
            - threshold=5: r=0.807
            - threshold=11: r=0.907 (n=18)
        """
        results = {}
        for threshold, preds in predictions_by_observer_count.items():
            results[threshold] = SocialCognitionMetrics.pearson_correlation(preds)
        return results

    @staticmethod
    def sentiment_mae(predictions: list[SentimentPrediction]) -> float:
        """Calculate mean absolute error of sentiment predictions.

        Args:
            predictions: List of sentiment predictions

        Returns:
            Mean absolute error
        """
        if not predictions:
            return 0.0

        errors = [abs(p.predicted - p.actual) for p in predictions]
        return sum(errors) / len(errors)

    @staticmethod
    def sentiment_accuracy(predictions: list[SentimentPrediction], tolerance: float = 0.5) -> float:
        """Calculate fraction of predictions within tolerance of actual value.

        Args:
            predictions: List of sentiment predictions
            tolerance: Maximum allowed error to count as accurate

        Returns:
            Accuracy as fraction in [0, 1]
        """
        if not predictions:
            return 0.0

        correct = sum(1 for p in predictions if abs(p.predicted - p.actual) <= tolerance)
        return correct / len(predictions)


class SpecializationMetrics:
    """Metrics for emergent role specialization.

    Measures diversity and stability of role assignments across agent population.
    Based on paper section 2 (Specialization Metrics).
    """

    @staticmethod
    def role_entropy(assignments: list[RoleAssignment]) -> float:
        """Calculate Shannon entropy of role distribution.

        Args:
            assignments: List of role assignments for all agents

        Returns:
            Shannon entropy in bits. 0 = all same role, log2(n_roles) = uniform distribution

        Formula:
            H(R) = -Σ p(r_i) * log₂(p(r_i))
            where p(r_i) is the fraction of agents assigned to role r_i
        """
        if not assignments:
            return 0.0

        # Count role occurrences
        role_counts = Counter(a.role for a in assignments)
        total = len(assignments)

        # Calculate probabilities and entropy
        entropy = 0.0
        for count in role_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    @staticmethod
    def normalized_role_entropy(assignments: list[RoleAssignment]) -> float:
        """Calculate normalized role entropy (0-1 scale).

        Args:
            assignments: List of role assignments for all agents

        Returns:
            Normalized entropy in [0, 1]. 0 = no diversity, 1 = maximum diversity

        Note:
            Interpretation guide (paper section 2.1):
            - 0.0-0.2: No specialization (all homogeneous)
            - 0.2-0.5: Partial specialization (dominant roles)
            - 0.5-0.8: Moderate specialization
            - 0.8-1.0: High specialization (diverse roles evenly distributed)
        """
        if not assignments:
            return 0.0

        entropy = SpecializationMetrics.role_entropy(assignments)
        num_unique_roles = len(set(a.role for a in assignments))

        # Maximum entropy is log2(number of unique roles)
        max_entropy = math.log2(num_unique_roles) if num_unique_roles > 1 else 0.0

        if max_entropy == 0.0:
            return 0.0

        return entropy / max_entropy

    @staticmethod
    def role_distribution(assignments: list[RoleAssignment]) -> dict[str, float]:
        """Calculate distribution of roles across agents.

        Args:
            assignments: List of role assignments for all agents

        Returns:
            Dict mapping role name to fraction of agents (sums to 1.0)
        """
        if not assignments:
            return {}

        role_counts = Counter(a.role for a in assignments)
        total = len(assignments)

        return {role: count / total for role, count in role_counts.items()}

    @staticmethod
    def role_persistence(role_history: list[list[RoleAssignment]]) -> float:
        """Calculate role stability over time.

        Args:
            role_history: List of role assignment snapshots over time
                e.g., [[assignments at t0], [assignments at t1], ...]

        Returns:
            Persistence score in [0, 1]. 1.0 = perfectly stable, 0.0 = completely random

        Note:
            Measures the average fraction of agents that maintain the same role
            between consecutive time steps.
        """
        if len(role_history) < 2:
            return 1.0  # Single snapshot = perfectly stable (no changes observed)

        stability_scores = []

        for i in range(len(role_history) - 1):
            current_roles = {a.agent_id: a.role for a in role_history[i]}
            next_roles = {a.agent_id: a.role for a in role_history[i + 1]}

            # Find agents present in both snapshots
            common_agents = set(current_roles.keys()) & set(next_roles.keys())

            if not common_agents:
                continue

            # Count how many maintained their role
            maintained = sum(
                1 for agent_id in common_agents if current_roles[agent_id] == next_roles[agent_id]
            )

            stability_scores.append(maintained / len(common_agents))

        if not stability_scores:
            return 1.0

        return sum(stability_scores) / len(stability_scores)

    @staticmethod
    def num_unique_roles(assignments: list[RoleAssignment]) -> int:
        """Count number of unique roles in the population.

        Args:
            assignments: List of role assignments for all agents

        Returns:
            Number of unique roles
        """
        return len(set(a.role for a in assignments))


class MetricsReport(BaseModel):
    """Comprehensive metrics report combining social cognition and specialization."""

    social_cognition_r: float = Field(
        description="Pearson correlation for social cognition accuracy"
    )
    normalized_entropy: float = Field(ge=0.0, le=1.0, description="Normalized role entropy (0-1)")
    num_roles: int = Field(ge=0, description="Number of unique roles")
    timestamp: datetime = Field(description="When the report was generated")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context and metrics"
    )


def generate_report(
    predictions: list[SentimentPrediction],
    assignments: list[RoleAssignment],
    metadata: dict[str, Any] | None = None,
) -> MetricsReport:
    """Generate comprehensive metrics report.

    Args:
        predictions: Sentiment predictions for social cognition analysis
        assignments: Role assignments for specialization analysis
        metadata: Optional additional metadata to include

    Returns:
        MetricsReport with all computed metrics

    Example:
        >>> predictions = [
        ...     SentimentPrediction(agent_id="a1", target_id="a2", predicted=0.8, actual=0.7),
        ...     SentimentPrediction(agent_id="a2", target_id="a1", predicted=0.6, actual=0.5),
        ... ]
        >>> assignments = [
        ...     RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),
        ...     RoleAssignment(agent_id="a2", role="miner", confidence=0.8),
        ... ]
        >>> report = generate_report(predictions, assignments)
        >>> print(f"Social cognition: r={report.social_cognition_r:.3f}")
        >>> print(f"Role diversity: {report.normalized_entropy:.3f}")
    """
    # Compute core metrics
    social_r = SocialCognitionMetrics.pearson_correlation(predictions)
    norm_entropy = SpecializationMetrics.normalized_role_entropy(assignments)
    n_roles = SpecializationMetrics.num_unique_roles(assignments)

    # Prepare extended metadata
    extended_metadata = metadata.copy() if metadata else {}
    extended_metadata.update(
        {
            "sentiment_mae": SocialCognitionMetrics.sentiment_mae(predictions),
            "sentiment_accuracy": SocialCognitionMetrics.sentiment_accuracy(predictions),
            "role_entropy": SpecializationMetrics.role_entropy(assignments),
            "role_distribution": SpecializationMetrics.role_distribution(assignments),
            "num_predictions": len(predictions),
            "num_assignments": len(assignments),
        }
    )

    return MetricsReport(
        social_cognition_r=social_r,
        normalized_entropy=norm_entropy,
        num_roles=n_roles,
        timestamp=datetime.now(UTC),
        metadata=extended_metadata,
    )


__all__ = [
    "MetricsReport",
    "RoleAssignment",
    "SentimentPrediction",
    "SocialCognitionMetrics",
    "SpecializationMetrics",
    "generate_report",
]
