"""Collective Intelligence mechanism for PIANO architecture.

Implements observer threshold-based accuracy improvement by aggregating
observations from multiple agents. Based on the "wisdom of crowds" effect
described in the paper (Section 4.3): as more agents observe a target,
the aggregated prediction accuracy improves.

Key findings from the paper:
- threshold=1 (all predictions): r=0.646 (n=46)
- threshold=5: r=0.807
- threshold=11: r=0.907 (n=18)

Reference: Paper Section 4.3, docs/implementation/06-social-cognition.md
"""

from __future__ import annotations

__all__ = [
    "AggregatedResult",
    "AggregationMethod",
    "CollectiveIntelligence",
    "Observation",
    "ThresholdResult",
]

import math
import statistics
from collections import defaultdict
from datetime import datetime  # noqa: TC003 (needed at runtime by Pydantic)
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field

try:
    from scipy import stats as _scipy_stats

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

logger = structlog.get_logger(__name__)


class AggregationMethod(StrEnum):
    """Supported observation aggregation methods."""

    SIMPLE_MEAN = "simple_mean"
    WEIGHTED_MEAN = "weighted_mean"
    MEDIAN = "median"


class Observation(BaseModel):
    """A single observation by one agent about a target.

    Attributes:
        observer_id: ID of the agent making the observation.
        target_id: ID of the target agent being observed.
        predicted_value: Predicted value (e.g., sentiment/likeability score).
        timestamp: When the observation was made.
    """

    observer_id: str
    target_id: str
    predicted_value: float
    timestamp: datetime


class AggregatedResult(BaseModel):
    """Result of aggregating multiple observations for a single target.

    Attributes:
        target_id: ID of the target agent.
        aggregated_value: The aggregated prediction value.
        num_observers: Number of distinct observers contributing.
        confidence: Confidence score in [0.0, 1.0].
        method: Aggregation method used.
    """

    target_id: str
    aggregated_value: float
    num_observers: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    method: str


class ThresholdResult(BaseModel):
    """Result of threshold analysis at a specific observer count.

    Attributes:
        threshold: Minimum number of observers required.
        pearson_r: Pearson correlation coefficient at this threshold.
        num_predictions: Number of target predictions meeting the threshold.
        mae: Mean absolute error at this threshold.
    """

    threshold: int = Field(ge=1)
    pearson_r: float
    num_predictions: int = Field(ge=0)
    mae: float = Field(ge=0.0)


class CollectiveIntelligence:
    """Aggregates observations from multiple agents to improve prediction accuracy.

    Implements the "wisdom of crowds" effect: as more agents observe a target,
    the aggregated prediction becomes more accurate (higher Pearson correlation
    with the actual value).

    Attributes:
        default_method: Default aggregation method for observations.
        observer_weights: Optional per-observer accuracy weights for weighted_mean.
    """

    def __init__(
        self,
        default_method: AggregationMethod = AggregationMethod.SIMPLE_MEAN,
        observer_weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize collective intelligence engine.

        Args:
            default_method: Default aggregation method.
            observer_weights: Optional mapping of observer_id to accuracy weight
                (higher = more trusted). Used only with WEIGHTED_MEAN method.
        """
        self.default_method = default_method
        self.observer_weights: dict[str, float] = observer_weights or {}

        logger.info(
            "collective_intelligence_initialized",
            default_method=default_method,
            num_observer_weights=len(self.observer_weights),
        )

    def aggregate_observations(
        self,
        observations: list[Observation],
        *,
        method: AggregationMethod | None = None,
    ) -> AggregatedResult:
        """Aggregate observations about a single target.

        All observations must be about the same target. If observations are
        for multiple targets, only the first target's observations are used.

        Args:
            observations: List of observations (must all share the same target_id).
            method: Aggregation method to use (defaults to self.default_method).

        Returns:
            AggregatedResult with the aggregated prediction.

        Raises:
            ValueError: If observations list is empty.
        """
        if not observations:
            raise ValueError("Cannot aggregate empty observations list.")

        method = method or self.default_method
        target_id = observations[0].target_id

        # Filter to only observations for this target
        target_obs = [o for o in observations if o.target_id == target_id]

        # Deduplicate by observer (keep latest observation per observer)
        latest_by_observer: dict[str, Observation] = {}
        for obs in target_obs:
            existing = latest_by_observer.get(obs.observer_id)
            if existing is None or obs.timestamp > existing.timestamp:
                latest_by_observer[obs.observer_id] = obs

        unique_observations = list(latest_by_observer.values())
        num_observers = len(unique_observations)
        values = [o.predicted_value for o in unique_observations]

        # Compute aggregated value
        if method == AggregationMethod.SIMPLE_MEAN:
            aggregated_value = statistics.mean(values)
        elif method == AggregationMethod.WEIGHTED_MEAN:
            aggregated_value = self._weighted_mean(unique_observations)
        elif method == AggregationMethod.MEDIAN:
            aggregated_value = statistics.median(values)
        else:
            aggregated_value = statistics.mean(values)

        confidence = self.get_confidence_from_count(num_observers)

        logger.debug(
            "observations_aggregated",
            target_id=target_id,
            num_observers=num_observers,
            method=method,
            aggregated_value=aggregated_value,
            confidence=confidence,
        )

        return AggregatedResult(
            target_id=target_id,
            aggregated_value=aggregated_value,
            num_observers=num_observers,
            confidence=confidence,
            method=method,
        )

    def apply_threshold(
        self,
        observations: list[Observation],
        threshold: int,
        *,
        method: AggregationMethod | None = None,
    ) -> AggregatedResult:
        """Aggregate observations, filtering targets with fewer than threshold observers.

        Groups observations by target_id, filters to targets with at least
        `threshold` unique observers, then aggregates the remaining observations.

        If no targets meet the threshold, returns an AggregatedResult with
        num_observers=0 and aggregated_value=0.0.

        Args:
            observations: List of observations across potentially multiple targets.
            threshold: Minimum number of unique observers required per target.
            method: Aggregation method to use.

        Returns:
            AggregatedResult for the first target meeting the threshold,
            or empty result if none qualify.

        Raises:
            ValueError: If threshold is less than 1.
        """
        if threshold < 1:
            raise ValueError("Threshold must be at least 1.")

        method = method or self.default_method

        # Group observations by target
        by_target = self._group_by_target(observations)

        # Find a target meeting the threshold
        for _target_id, target_obs in by_target.items():
            unique_observers = {o.observer_id for o in target_obs}
            if len(unique_observers) >= threshold:
                return self.aggregate_observations(target_obs, method=method)

        # No target meets threshold
        logger.debug(
            "threshold_not_met",
            threshold=threshold,
            num_targets=len(by_target),
        )
        return AggregatedResult(
            target_id="",
            aggregated_value=0.0,
            num_observers=0,
            confidence=0.0,
            method=method,
        )

    def get_confidence(self, result: AggregatedResult) -> float:
        """Get confidence score for an aggregated result.

        Confidence is based on the number of observers. More observers
        yield higher confidence, following a logarithmic saturation curve.

        Args:
            result: An aggregated result.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        return self.get_confidence_from_count(result.num_observers)

    @staticmethod
    def get_confidence_from_count(num_observers: int) -> float:
        """Calculate confidence from observer count.

        Uses a logarithmic saturation curve:
          confidence = min(1.0, log2(num_observers + 1) / log2(12))

        This yields approximately:
          1 observer  -> 0.28
          3 observers -> 0.56
          5 observers -> 0.72
          7 observers -> 0.83
          11 observers -> 0.95
          12+ observers -> 1.0

        Args:
            num_observers: Number of distinct observers.

        Returns:
            Confidence score in [0.0, 1.0].
        """
        if num_observers <= 0:
            return 0.0

        # log2(12) ~ 3.585
        max_log = math.log2(12)
        raw = math.log2(num_observers + 1) / max_log
        return min(1.0, raw)

    def threshold_analysis(
        self,
        observations: list[Observation],
        actual_values: dict[str, float],
        thresholds: list[int] | None = None,
    ) -> list[ThresholdResult]:
        """Analyze prediction accuracy at different observer thresholds.

        For each threshold, filters to targets with >= threshold unique observers,
        aggregates their observations, and computes Pearson correlation and MAE
        against actual values.

        Args:
            observations: All observations across multiple targets.
            actual_values: Mapping of target_id to actual ground-truth value.
            thresholds: List of thresholds to test. Defaults to [1, 3, 5, 7, 9, 11].

        Returns:
            List of ThresholdResult, one per threshold.
        """
        if thresholds is None:
            thresholds = [1, 3, 5, 7, 9, 11]

        # Group observations by target
        by_target = self._group_by_target(observations)

        results: list[ThresholdResult] = []

        for threshold in thresholds:
            predicted_list: list[float] = []
            actual_list: list[float] = []

            for target_id, target_obs in by_target.items():
                unique_observers = {o.observer_id for o in target_obs}
                if len(unique_observers) < threshold:
                    continue

                if target_id not in actual_values:
                    continue

                # Aggregate for this target
                agg = self.aggregate_observations(target_obs)
                predicted_list.append(agg.aggregated_value)
                actual_list.append(actual_values[target_id])

            num_predictions = len(predicted_list)

            # Compute Pearson correlation
            pearson_r = _safe_pearson(predicted_list, actual_list) if num_predictions >= 2 else 0.0

            # Compute MAE
            if num_predictions > 0:
                mae = sum(
                    abs(p - a) for p, a in zip(predicted_list, actual_list, strict=True)
                ) / num_predictions
            else:
                mae = 0.0

            results.append(
                ThresholdResult(
                    threshold=threshold,
                    pearson_r=pearson_r,
                    num_predictions=num_predictions,
                    mae=mae,
                )
            )

            logger.debug(
                "threshold_analysis_step",
                threshold=threshold,
                num_predictions=num_predictions,
                pearson_r=pearson_r,
                mae=mae,
            )

        return results

    def aggregate_all_targets(
        self,
        observations: list[Observation],
        *,
        min_observers: int = 1,
        method: AggregationMethod | None = None,
    ) -> list[AggregatedResult]:
        """Aggregate observations for all targets meeting minimum observer count.

        Args:
            observations: All observations across multiple targets.
            min_observers: Minimum unique observers required per target.
            method: Aggregation method to use.

        Returns:
            List of AggregatedResult, one per qualifying target.
        """
        method = method or self.default_method
        by_target = self._group_by_target(observations)

        results: list[AggregatedResult] = []
        for _target_id, target_obs in by_target.items():
            unique_observers = {o.observer_id for o in target_obs}
            if len(unique_observers) >= min_observers:
                agg = self.aggregate_observations(target_obs, method=method)
                results.append(agg)

        return results

    def _weighted_mean(self, observations: list[Observation]) -> float:
        """Compute weighted mean using observer weights.

        Observers without weights default to 1.0.

        Args:
            observations: Deduplicated observations.

        Returns:
            Weighted mean value.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for obs in observations:
            weight = self.observer_weights.get(obs.observer_id, 1.0)
            weighted_sum += obs.predicted_value * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0

        return weighted_sum / total_weight

    @staticmethod
    def _group_by_target(
        observations: list[Observation],
    ) -> dict[str, list[Observation]]:
        """Group observations by target_id.

        Args:
            observations: List of observations.

        Returns:
            Dict mapping target_id to list of observations.
        """
        by_target: dict[str, list[Observation]] = defaultdict(list)
        for obs in observations:
            by_target[obs.target_id].append(obs)
        return dict(by_target)


def _safe_pearson(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson r, using scipy if available, else manual implementation.

    Args:
        xs: First variable.
        ys: Second variable (same length as *xs*).

    Returns:
        Pearson correlation coefficient, or 0.0 on degenerate input.
    """
    if _HAS_SCIPY:
        r, _ = _scipy_stats.pearsonr(xs, ys)
        return float(r)

    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return 0.0

    return cov / denom
