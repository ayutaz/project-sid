"""Tests for collective intelligence mechanism."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from piano.social.collective import (
    AggregatedResult,
    AggregationMethod,
    CollectiveIntelligence,
    Observation,
    ThresholdResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(
    observer_id: str,
    target_id: str,
    predicted_value: float,
    *,
    ts: datetime | None = None,
) -> Observation:
    """Create an Observation with sensible defaults."""
    return Observation(
        observer_id=observer_id,
        target_id=target_id,
        predicted_value=predicted_value,
        timestamp=ts or datetime.now(UTC),
    )


def _make_observations_for_target(
    target_id: str,
    values: list[float],
    *,
    observer_prefix: str = "agent",
) -> list[Observation]:
    """Create N observations for a single target from distinct observers."""
    base = datetime.now(UTC)
    return [
        _make_obs(
            f"{observer_prefix}_{i}",
            target_id,
            v,
            ts=base + timedelta(seconds=i),
        )
        for i, v in enumerate(values)
    ]


# ---------------------------------------------------------------------------
# Observation model tests
# ---------------------------------------------------------------------------

class TestObservationModel:
    """Tests for the Observation Pydantic model."""

    def test_observation_creation(self) -> None:
        """Test basic observation creation."""
        ts = datetime.now(UTC)
        obs = Observation(
            observer_id="a1",
            target_id="a2",
            predicted_value=0.75,
            timestamp=ts,
        )
        assert obs.observer_id == "a1"
        assert obs.target_id == "a2"
        assert obs.predicted_value == 0.75
        assert obs.timestamp == ts

    def test_observation_negative_value(self) -> None:
        """Test observation with negative predicted value."""
        obs = _make_obs("a1", "a2", -0.5)
        assert obs.predicted_value == -0.5

    def test_observation_serialization(self) -> None:
        """Test observation serialization round-trip."""
        obs = _make_obs("a1", "a2", 0.8)
        data = obs.model_dump()
        restored = Observation.model_validate(data)
        assert restored.observer_id == obs.observer_id
        assert restored.predicted_value == obs.predicted_value


# ---------------------------------------------------------------------------
# AggregatedResult model tests
# ---------------------------------------------------------------------------

class TestAggregatedResult:
    """Tests for the AggregatedResult model."""

    def test_aggregated_result_creation(self) -> None:
        """Test basic result creation."""
        result = AggregatedResult(
            target_id="t1",
            aggregated_value=0.5,
            num_observers=3,
            confidence=0.56,
            method="simple_mean",
        )
        assert result.target_id == "t1"
        assert result.num_observers == 3
        assert result.confidence == 0.56

    def test_confidence_range_validation(self) -> None:
        """Test confidence must be in [0, 1]."""
        with pytest.raises(ValidationError):
            AggregatedResult(
                target_id="t1",
                aggregated_value=0.5,
                num_observers=1,
                confidence=1.5,
                method="simple_mean",
            )

    def test_num_observers_non_negative(self) -> None:
        """Test num_observers must be >= 0."""
        with pytest.raises(ValidationError):
            AggregatedResult(
                target_id="t1",
                aggregated_value=0.5,
                num_observers=-1,
                confidence=0.5,
                method="simple_mean",
            )


# ---------------------------------------------------------------------------
# ThresholdResult model tests
# ---------------------------------------------------------------------------

class TestThresholdResult:
    """Tests for the ThresholdResult model."""

    def test_threshold_result_creation(self) -> None:
        """Test basic threshold result creation."""
        result = ThresholdResult(
            threshold=5,
            pearson_r=0.807,
            num_predictions=30,
            mae=0.12,
        )
        assert result.threshold == 5
        assert result.pearson_r == pytest.approx(0.807)
        assert result.num_predictions == 30

    def test_threshold_must_be_positive(self) -> None:
        """Test threshold must be >= 1."""
        with pytest.raises(ValidationError):
            ThresholdResult(threshold=0, pearson_r=0.5, num_predictions=10, mae=0.1)


# ---------------------------------------------------------------------------
# CollectiveIntelligence — aggregate_observations
# ---------------------------------------------------------------------------

class TestAggregateObservations:
    """Tests for CollectiveIntelligence.aggregate_observations."""

    def test_simple_mean_single_observer(self) -> None:
        """Test aggregation with a single observer returns that value."""
        ci = CollectiveIntelligence()
        obs = [_make_obs("a1", "t1", 0.8)]
        result = ci.aggregate_observations(obs)

        assert result.target_id == "t1"
        assert result.aggregated_value == pytest.approx(0.8)
        assert result.num_observers == 1
        assert result.method == AggregationMethod.SIMPLE_MEAN

    def test_simple_mean_multiple_observers(self) -> None:
        """Test simple mean with multiple observers."""
        ci = CollectiveIntelligence()
        obs = _make_observations_for_target("t1", [0.6, 0.8, 1.0])
        result = ci.aggregate_observations(obs)

        assert result.aggregated_value == pytest.approx(0.8)
        assert result.num_observers == 3

    def test_median_aggregation(self) -> None:
        """Test median aggregation is robust to outliers."""
        ci = CollectiveIntelligence(default_method=AggregationMethod.MEDIAN)
        # Outlier at 10.0
        obs = _make_observations_for_target("t1", [0.5, 0.6, 0.7, 0.6, 10.0])
        result = ci.aggregate_observations(obs)

        assert result.aggregated_value == pytest.approx(0.6)
        assert result.method == AggregationMethod.MEDIAN

    def test_weighted_mean(self) -> None:
        """Test weighted mean uses observer weights."""
        weights = {"agent_0": 2.0, "agent_1": 1.0}
        ci = CollectiveIntelligence(
            default_method=AggregationMethod.WEIGHTED_MEAN,
            observer_weights=weights,
        )
        obs = _make_observations_for_target("t1", [0.8, 0.2])
        result = ci.aggregate_observations(obs)

        # Weighted: (0.8*2 + 0.2*1) / (2+1) = 1.8/3 = 0.6
        assert result.aggregated_value == pytest.approx(0.6)
        assert result.method == AggregationMethod.WEIGHTED_MEAN

    def test_weighted_mean_default_weight(self) -> None:
        """Test weighted mean defaults to 1.0 for unknown observers."""
        weights = {"agent_0": 3.0}
        ci = CollectiveIntelligence(
            default_method=AggregationMethod.WEIGHTED_MEAN,
            observer_weights=weights,
        )
        obs = _make_observations_for_target("t1", [1.0, 0.0])
        result = ci.aggregate_observations(obs)

        # Weighted: (1.0*3 + 0.0*1) / (3+1) = 3.0/4 = 0.75
        assert result.aggregated_value == pytest.approx(0.75)

    def test_empty_observations_raises(self) -> None:
        """Test empty observations list raises ValueError."""
        ci = CollectiveIntelligence()
        with pytest.raises(ValueError, match="empty"):
            ci.aggregate_observations([])

    def test_deduplication_keeps_latest(self) -> None:
        """Test that duplicate observers are deduplicated, keeping latest."""
        ci = CollectiveIntelligence()
        base = datetime.now(UTC)
        obs = [
            _make_obs("a1", "t1", 0.2, ts=base),
            _make_obs("a1", "t1", 0.9, ts=base + timedelta(seconds=10)),
        ]
        result = ci.aggregate_observations(obs)

        # Should keep the latest observation (0.9), not the first (0.2)
        assert result.aggregated_value == pytest.approx(0.9)
        assert result.num_observers == 1

    def test_method_override(self) -> None:
        """Test per-call method override works."""
        ci = CollectiveIntelligence(default_method=AggregationMethod.SIMPLE_MEAN)
        obs = _make_observations_for_target("t1", [0.5, 0.6, 0.7, 0.6, 10.0])

        result_median = ci.aggregate_observations(obs, method=AggregationMethod.MEDIAN)
        assert result_median.method == AggregationMethod.MEDIAN
        assert result_median.aggregated_value == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# CollectiveIntelligence — apply_threshold
# ---------------------------------------------------------------------------

class TestApplyThreshold:
    """Tests for CollectiveIntelligence.apply_threshold."""

    def test_threshold_met(self) -> None:
        """Test apply_threshold when threshold is met."""
        ci = CollectiveIntelligence()
        obs = _make_observations_for_target("t1", [0.5, 0.6, 0.7])
        result = ci.apply_threshold(obs, threshold=2)

        assert result.num_observers == 3
        assert result.target_id == "t1"

    def test_threshold_not_met(self) -> None:
        """Test apply_threshold when threshold is not met returns empty."""
        ci = CollectiveIntelligence()
        obs = _make_observations_for_target("t1", [0.5, 0.6])
        result = ci.apply_threshold(obs, threshold=5)

        assert result.num_observers == 0
        assert result.target_id == ""

    def test_threshold_invalid_raises(self) -> None:
        """Test threshold < 1 raises ValueError."""
        ci = CollectiveIntelligence()
        with pytest.raises(ValueError, match="at least 1"):
            ci.apply_threshold([], threshold=0)

    def test_threshold_exactly_met(self) -> None:
        """Test that exactly threshold observers qualifies."""
        ci = CollectiveIntelligence()
        obs = _make_observations_for_target("t1", [0.3, 0.5, 0.7])
        result = ci.apply_threshold(obs, threshold=3)

        assert result.num_observers == 3


# ---------------------------------------------------------------------------
# CollectiveIntelligence — get_confidence
# ---------------------------------------------------------------------------

class TestGetConfidence:
    """Tests for confidence calculation."""

    def test_confidence_zero_observers(self) -> None:
        """Test confidence is 0 with zero observers."""
        assert CollectiveIntelligence.get_confidence_from_count(0) == 0.0

    def test_confidence_single_observer(self) -> None:
        """Test confidence is low with single observer."""
        c = CollectiveIntelligence.get_confidence_from_count(1)
        assert 0.2 < c < 0.4  # log2(2)/log2(12) ~ 0.28

    def test_confidence_many_observers(self) -> None:
        """Test confidence approaches 1.0 with many observers."""
        c = CollectiveIntelligence.get_confidence_from_count(11)
        assert c >= 0.9

    def test_confidence_capped_at_one(self) -> None:
        """Test confidence never exceeds 1.0."""
        c = CollectiveIntelligence.get_confidence_from_count(100)
        assert c == 1.0

    def test_confidence_monotonically_increases(self) -> None:
        """Test confidence increases with more observers."""
        values = [
            CollectiveIntelligence.get_confidence_from_count(n)
            for n in range(1, 13)
        ]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]

    def test_get_confidence_from_result(self) -> None:
        """Test get_confidence method on AggregatedResult."""
        ci = CollectiveIntelligence()
        result = AggregatedResult(
            target_id="t1",
            aggregated_value=0.5,
            num_observers=5,
            confidence=0.0,  # Will be recalculated
            method="simple_mean",
        )
        conf = ci.get_confidence(result)
        assert 0.7 < conf < 0.8


# ---------------------------------------------------------------------------
# CollectiveIntelligence — threshold_analysis
# ---------------------------------------------------------------------------

class TestThresholdAnalysis:
    """Tests for threshold_analysis with synthetic data."""

    @staticmethod
    def _build_synthetic_data(
        num_targets: int = 20,
        max_observers: int = 12,
    ) -> tuple[list[Observation], dict[str, float]]:
        """Build synthetic observations where more observers = better accuracy.

        Each target has a ground-truth value. Observers add noise.
        Targets with more observers have aggregations closer to truth.
        """
        import random

        random.seed(42)
        observations: list[Observation] = []
        actual_values: dict[str, float] = {}
        base = datetime.now(UTC)

        for t_idx in range(num_targets):
            target_id = f"target_{t_idx}"
            actual = random.uniform(-1.0, 1.0)
            actual_values[target_id] = actual

            # Number of observers varies (1 to max_observers)
            num_obs = random.randint(1, max_observers)
            for o_idx in range(num_obs):
                # Add Gaussian noise; more observers -> individual noise cancels out
                noise = random.gauss(0, 0.3)
                predicted = actual + noise
                observations.append(
                    _make_obs(
                        f"observer_{o_idx}",
                        target_id,
                        predicted,
                        ts=base + timedelta(seconds=t_idx * 100 + o_idx),
                    )
                )

        return observations, actual_values

    def test_threshold_analysis_returns_all_thresholds(self) -> None:
        """Test that threshold_analysis returns one result per threshold."""
        ci = CollectiveIntelligence()
        obs, actuals = self._build_synthetic_data()

        thresholds = [1, 3, 5, 7]
        results = ci.threshold_analysis(obs, actuals, thresholds=thresholds)

        assert len(results) == len(thresholds)
        for result, expected_t in zip(results, thresholds, strict=True):
            assert result.threshold == expected_t

    def test_threshold_analysis_default_thresholds(self) -> None:
        """Test that default thresholds are [1, 3, 5, 7, 9, 11]."""
        ci = CollectiveIntelligence()
        obs, actuals = self._build_synthetic_data()

        results = ci.threshold_analysis(obs, actuals)

        assert len(results) == 6
        assert [r.threshold for r in results] == [1, 3, 5, 7, 9, 11]

    def test_higher_threshold_fewer_predictions(self) -> None:
        """Test that higher thresholds generally have fewer qualifying predictions."""
        ci = CollectiveIntelligence()
        obs, actuals = self._build_synthetic_data(num_targets=50, max_observers=12)

        results = ci.threshold_analysis(obs, actuals)

        # With random observer counts 1-12, higher thresholds should have
        # fewer or equal qualifying targets
        for i in range(len(results) - 1):
            assert results[i].num_predictions >= results[i + 1].num_predictions

    def test_threshold_analysis_empty_observations(self) -> None:
        """Test threshold analysis with empty observations."""
        ci = CollectiveIntelligence()
        results = ci.threshold_analysis([], {})

        assert len(results) == 6  # Default thresholds
        for r in results:
            assert r.num_predictions == 0
            assert r.pearson_r == 0.0
            assert r.mae == 0.0

    def test_threshold_analysis_pearson_r_type(self) -> None:
        """Test that pearson_r is a valid float."""
        ci = CollectiveIntelligence()
        obs, actuals = self._build_synthetic_data()
        results = ci.threshold_analysis(obs, actuals)

        for r in results:
            assert isinstance(r.pearson_r, float)
            # Pearson r is in [-1, 1]
            assert -1.0 <= r.pearson_r <= 1.0 or r.num_predictions < 2


# ---------------------------------------------------------------------------
# CollectiveIntelligence — aggregate_all_targets
# ---------------------------------------------------------------------------

class TestAggregateAllTargets:
    """Tests for aggregate_all_targets."""

    def test_aggregate_all_targets_basic(self) -> None:
        """Test aggregating all targets with multiple targets."""
        ci = CollectiveIntelligence()
        obs_t1 = _make_observations_for_target("t1", [0.5, 0.6, 0.7])
        obs_t2 = _make_observations_for_target("t2", [0.1, 0.2])
        all_obs = obs_t1 + obs_t2

        results = ci.aggregate_all_targets(all_obs)

        assert len(results) == 2
        target_ids = {r.target_id for r in results}
        assert target_ids == {"t1", "t2"}

    def test_aggregate_all_targets_with_min_observers(self) -> None:
        """Test filtering by minimum observers."""
        ci = CollectiveIntelligence()
        obs_t1 = _make_observations_for_target("t1", [0.5, 0.6, 0.7])
        obs_t2 = _make_observations_for_target("t2", [0.1])
        all_obs = obs_t1 + obs_t2

        results = ci.aggregate_all_targets(all_obs, min_observers=2)

        assert len(results) == 1
        assert results[0].target_id == "t1"

    def test_aggregate_all_targets_empty(self) -> None:
        """Test aggregating empty observation list."""
        ci = CollectiveIntelligence()
        results = ci.aggregate_all_targets([])
        assert results == []


# ---------------------------------------------------------------------------
# Integration / edge-case tests
# ---------------------------------------------------------------------------

class TestCollectiveIntelligenceIntegration:
    """Integration and edge-case tests."""

    def test_aggregation_method_enum(self) -> None:
        """Test AggregationMethod enum values."""
        assert AggregationMethod.SIMPLE_MEAN == "simple_mean"
        assert AggregationMethod.WEIGHTED_MEAN == "weighted_mean"
        assert AggregationMethod.MEDIAN == "median"

    def test_all_exports(self) -> None:
        """Test __all__ contains expected exports."""
        from piano.social import collective

        expected = {
            "AggregatedResult",
            "AggregationMethod",
            "CollectiveIntelligence",
            "Observation",
            "ThresholdResult",
        }
        assert set(collective.__all__) == expected

    def test_large_observation_set(self) -> None:
        """Test aggregation with a large number of observations."""
        ci = CollectiveIntelligence()
        obs = _make_observations_for_target(
            "t1",
            [float(i) / 100.0 for i in range(100)],
        )
        result = ci.aggregate_observations(obs)

        assert result.num_observers == 100
        # Mean of 0..99/100 = 49.5/100 = 0.495
        assert result.aggregated_value == pytest.approx(0.495)
        assert result.confidence == 1.0  # 100 observers -> max confidence
