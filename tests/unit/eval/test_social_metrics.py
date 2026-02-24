"""Tests for social cognition evaluation metrics."""

import math
from datetime import UTC

import pytest

from piano.eval.social_metrics import (
    MetricsReport,
    RoleAssignment,
    SentimentPrediction,
    SocialCognitionMetrics,
    SpecializationMetrics,
    generate_report,
)


class TestSentimentPrediction:
    """Tests for SentimentPrediction model."""

    def test_sentiment_prediction_creation(self):
        pred = SentimentPrediction(agent_id="alice", target_id="bob", predicted=0.7, actual=0.8)
        assert pred.agent_id == "alice"
        assert pred.target_id == "bob"
        assert pred.predicted == 0.7
        assert pred.actual == 0.8


class TestRoleAssignment:
    """Tests for RoleAssignment model."""

    def test_role_assignment_creation(self):
        assignment = RoleAssignment(agent_id="alice", role="farmer", confidence=0.9)
        assert assignment.agent_id == "alice"
        assert assignment.role == "farmer"
        assert assignment.confidence == 0.9

    def test_confidence_bounds(self):
        # Valid bounds
        RoleAssignment(agent_id="alice", role="farmer", confidence=0.0)
        RoleAssignment(agent_id="alice", role="farmer", confidence=1.0)

        # Invalid bounds
        with pytest.raises(ValueError):
            RoleAssignment(agent_id="alice", role="farmer", confidence=-0.1)
        with pytest.raises(ValueError):
            RoleAssignment(agent_id="alice", role="farmer", confidence=1.1)


class TestSocialCognitionMetrics:
    """Tests for social cognition accuracy metrics."""

    def test_pearson_correlation_perfect_positive(self):
        """Perfect correlation (r=1.0) when predicted equals actual."""
        predictions = [
            SentimentPrediction(
                agent_id=f"a{i}", target_id=f"t{i}", predicted=i * 0.1, actual=i * 0.1
            )
            for i in range(10)
        ]
        r = SocialCognitionMetrics.pearson_correlation(predictions)
        assert abs(r - 1.0) < 1e-10

    def test_pearson_correlation_perfect_negative(self):
        """Perfect negative correlation (r=-1.0)."""
        predictions = [
            SentimentPrediction(
                agent_id=f"a{i}",
                target_id=f"t{i}",
                predicted=i * 0.1,
                actual=1.0 - i * 0.1,
            )
            for i in range(10)
        ]
        r = SocialCognitionMetrics.pearson_correlation(predictions)
        assert abs(r - (-1.0)) < 1e-10

    def test_pearson_correlation_no_correlation(self):
        """Zero correlation when predicted and actual are independent."""
        import warnings

        # Predicted constant, actual varies -> r ≈ 0
        predictions = [
            SentimentPrediction(agent_id=f"a{i}", target_id=f"t{i}", predicted=0.5, actual=i * 0.1)
            for i in range(10)
        ]

        # Suppress scipy warning about constant input
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = SocialCognitionMetrics.pearson_correlation(predictions)

        # With constant predicted values, correlation should be NaN or 0
        # scipy returns NaN for constant array, we handle it
        assert math.isnan(r) or abs(r) < 1e-10

    def test_pearson_correlation_insufficient_data(self):
        """Return 0.0 when insufficient data points."""
        assert SocialCognitionMetrics.pearson_correlation([]) == 0.0

        single = [SentimentPrediction(agent_id="a1", target_id="t1", predicted=0.5, actual=0.6)]
        assert SocialCognitionMetrics.pearson_correlation(single) == 0.0

    def test_observer_threshold_analysis(self):
        """Test observer threshold analysis (wisdom of crowds)."""
        # Create predictions with varying observer counts
        predictions_all = [
            SentimentPrediction(
                agent_id=f"a{i}", target_id=f"t{i}", predicted=i * 0.1, actual=i * 0.1
            )
            for i in range(10)
        ]
        predictions_subset = predictions_all[:5]

        predictions_by_count = {1: predictions_all, 5: predictions_subset}

        results = SocialCognitionMetrics.observer_threshold_analysis(predictions_by_count)

        assert 1 in results
        assert 5 in results
        # Both should have high correlation (perfect linear relationship)
        assert abs(results[1] - 1.0) < 1e-10
        assert abs(results[5] - 1.0) < 1e-10

    def test_observer_threshold_analysis_empty(self):
        """Handle empty predictions in threshold analysis."""
        predictions_by_count = {1: [], 5: []}
        results = SocialCognitionMetrics.observer_threshold_analysis(predictions_by_count)
        assert results[1] == 0.0
        assert results[5] == 0.0

    def test_sentiment_mae(self):
        """Test mean absolute error calculation."""
        predictions = [
            SentimentPrediction(
                agent_id="a1", target_id="t1", predicted=0.5, actual=0.7
            ),  # error = 0.2
            SentimentPrediction(
                agent_id="a2", target_id="t2", predicted=0.8, actual=0.6
            ),  # error = 0.2
            SentimentPrediction(
                agent_id="a3", target_id="t3", predicted=0.4, actual=0.5
            ),  # error = 0.1
        ]
        mae = SocialCognitionMetrics.sentiment_mae(predictions)
        assert abs(mae - 0.166667) < 1e-5

    def test_sentiment_mae_empty(self):
        """MAE is 0.0 for empty predictions."""
        assert SocialCognitionMetrics.sentiment_mae([]) == 0.0

    def test_sentiment_accuracy_default_tolerance(self):
        """Test accuracy with default tolerance (0.5)."""
        predictions = [
            SentimentPrediction(
                agent_id="a1", target_id="t1", predicted=0.5, actual=0.7
            ),  # within 0.5
            SentimentPrediction(
                agent_id="a2", target_id="t2", predicted=0.8, actual=0.9
            ),  # within 0.5
            SentimentPrediction(
                agent_id="a3", target_id="t3", predicted=0.2, actual=0.9
            ),  # outside 0.5
        ]
        accuracy = SocialCognitionMetrics.sentiment_accuracy(predictions)
        assert abs(accuracy - (2 / 3)) < 1e-10

    def test_sentiment_accuracy_custom_tolerance(self):
        """Test accuracy with custom tolerance."""
        predictions = [
            SentimentPrediction(
                agent_id="a1", target_id="t1", predicted=0.5, actual=0.6
            ),  # 0.1 diff
            SentimentPrediction(
                agent_id="a2", target_id="t2", predicted=0.5, actual=0.8
            ),  # 0.3 diff
        ]
        # With tolerance=0.2, only first is accurate
        accuracy = SocialCognitionMetrics.sentiment_accuracy(predictions, tolerance=0.2)
        assert accuracy == 0.5

    def test_sentiment_accuracy_empty(self):
        """Accuracy is 0.0 for empty predictions."""
        assert SocialCognitionMetrics.sentiment_accuracy([]) == 0.0


class TestSocialCognitionWithoutScipy:
    """Tests for social cognition metrics using manual Pearson (no scipy)."""

    def test_manual_pearson_perfect_positive(self, monkeypatch):
        """Manual Pearson returns 1.0 for perfect positive correlation."""
        import piano.eval.social_metrics as sm

        monkeypatch.setattr(sm, "_HAS_SCIPY", False)

        predictions = [
            SentimentPrediction(
                agent_id=f"a{i}", target_id=f"t{i}", predicted=i * 0.1, actual=i * 0.1
            )
            for i in range(10)
        ]
        r = SocialCognitionMetrics.pearson_correlation(predictions)
        assert abs(r - 1.0) < 1e-10

    def test_manual_pearson_perfect_negative(self, monkeypatch):
        """Manual Pearson returns -1.0 for perfect negative correlation."""
        import piano.eval.social_metrics as sm

        monkeypatch.setattr(sm, "_HAS_SCIPY", False)

        predictions = [
            SentimentPrediction(
                agent_id=f"a{i}",
                target_id=f"t{i}",
                predicted=i * 0.1,
                actual=1.0 - i * 0.1,
            )
            for i in range(10)
        ]
        r = SocialCognitionMetrics.pearson_correlation(predictions)
        assert abs(r - (-1.0)) < 1e-10

    def test_manual_pearson_constant_input(self, monkeypatch):
        """Manual Pearson returns 0.0 for constant predicted values."""
        import piano.eval.social_metrics as sm

        monkeypatch.setattr(sm, "_HAS_SCIPY", False)

        predictions = [
            SentimentPrediction(
                agent_id=f"a{i}", target_id=f"t{i}", predicted=0.5, actual=i * 0.1
            )
            for i in range(10)
        ]
        r = SocialCognitionMetrics.pearson_correlation(predictions)
        assert r == 0.0


class TestSpecializationMetrics:
    """Tests for role specialization metrics."""

    def test_role_entropy_single_role(self):
        """Entropy is 0 when all agents have the same role."""
        assignments = [
            RoleAssignment(agent_id=f"a{i}", role="farmer", confidence=0.9) for i in range(10)
        ]
        entropy = SpecializationMetrics.role_entropy(assignments)
        assert entropy == 0.0

    def test_role_entropy_uniform_distribution(self):
        """Entropy is log2(n) for uniform distribution over n roles."""
        # 4 roles with 2 agents each = uniform distribution
        assignments = [
            RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a2", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a3", role="miner", confidence=0.9),
            RoleAssignment(agent_id="a4", role="miner", confidence=0.9),
            RoleAssignment(agent_id="a5", role="guard", confidence=0.9),
            RoleAssignment(agent_id="a6", role="guard", confidence=0.9),
            RoleAssignment(agent_id="a7", role="explorer", confidence=0.9),
            RoleAssignment(agent_id="a8", role="explorer", confidence=0.9),
        ]
        entropy = SpecializationMetrics.role_entropy(assignments)
        expected = math.log2(4)  # 4 roles
        assert abs(entropy - expected) < 1e-10

    def test_role_entropy_empty(self):
        """Entropy is 0 for empty assignments."""
        assert SpecializationMetrics.role_entropy([]) == 0.0

    def test_normalized_role_entropy_bounds(self):
        """Normalized entropy is in [0, 1]."""
        # All same role -> 0.0
        same_role = [
            RoleAssignment(agent_id=f"a{i}", role="farmer", confidence=0.9) for i in range(5)
        ]
        assert SpecializationMetrics.normalized_role_entropy(same_role) == 0.0

        # Uniform distribution -> 1.0
        uniform = [
            RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a2", role="miner", confidence=0.9),
            RoleAssignment(agent_id="a3", role="guard", confidence=0.9),
        ]
        norm_entropy = SpecializationMetrics.normalized_role_entropy(uniform)
        assert abs(norm_entropy - 1.0) < 1e-10

    def test_normalized_role_entropy_interpretation(self):
        """Test interpretation guide from paper."""
        # No specialization (0.0-0.2): all same role
        no_spec = [
            RoleAssignment(agent_id=f"a{i}", role="farmer", confidence=0.9) for i in range(10)
        ]
        assert SpecializationMetrics.normalized_role_entropy(no_spec) < 0.2

        # High specialization (0.8-1.0): diverse roles evenly distributed
        high_spec = [
            RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a2", role="miner", confidence=0.9),
            RoleAssignment(agent_id="a3", role="guard", confidence=0.9),
            RoleAssignment(agent_id="a4", role="explorer", confidence=0.9),
        ]
        assert SpecializationMetrics.normalized_role_entropy(high_spec) > 0.8

    def test_role_distribution(self):
        """Test role distribution calculation."""
        assignments = [
            RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a2", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a3", role="miner", confidence=0.9),
            RoleAssignment(agent_id="a4", role="guard", confidence=0.9),
        ]
        dist = SpecializationMetrics.role_distribution(assignments)

        assert dist["farmer"] == 0.5
        assert dist["miner"] == 0.25
        assert dist["guard"] == 0.25
        # Should sum to 1.0
        assert abs(sum(dist.values()) - 1.0) < 1e-10

    def test_role_distribution_empty(self):
        """Empty distribution for empty assignments."""
        assert SpecializationMetrics.role_distribution([]) == {}

    def test_role_persistence_stable(self):
        """Persistence is 1.0 when roles are perfectly stable."""
        t0 = [
            RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a2", role="miner", confidence=0.9),
        ]
        t1 = [
            RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a2", role="miner", confidence=0.9),
        ]
        t2 = [
            RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a2", role="miner", confidence=0.9),
        ]

        persistence = SpecializationMetrics.role_persistence([t0, t1, t2])
        assert persistence == 1.0

    def test_role_persistence_complete_turnover(self):
        """Persistence is 0.0 when all roles change at each step."""
        t0 = [
            RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a2", role="miner", confidence=0.9),
        ]
        t1 = [
            RoleAssignment(agent_id="a1", role="miner", confidence=0.9),
            RoleAssignment(agent_id="a2", role="farmer", confidence=0.9),
        ]
        t2 = [
            RoleAssignment(agent_id="a1", role="guard", confidence=0.9),
            RoleAssignment(agent_id="a2", role="explorer", confidence=0.9),
        ]

        persistence = SpecializationMetrics.role_persistence([t0, t1, t2])
        assert persistence == 0.0

    def test_role_persistence_partial(self):
        """Test partial role changes."""
        t0 = [
            RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a2", role="miner", confidence=0.9),
        ]
        t1 = [
            RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),  # stays farmer
            RoleAssignment(agent_id="a2", role="guard", confidence=0.9),  # changes to guard
        ]

        persistence = SpecializationMetrics.role_persistence([t0, t1])
        # 1 out of 2 maintained their role = 0.5
        assert persistence == 0.5

    def test_role_persistence_single_snapshot(self):
        """Single snapshot is considered perfectly stable."""
        t0 = [RoleAssignment(agent_id="a1", role="farmer", confidence=0.9)]
        assert SpecializationMetrics.role_persistence([t0]) == 1.0

    def test_role_persistence_agent_changes(self):
        """Handle agents appearing/disappearing between snapshots."""
        t0 = [
            RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a2", role="miner", confidence=0.9),
        ]
        t1 = [
            RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),  # a1 stays
            RoleAssignment(agent_id="a3", role="guard", confidence=0.9),  # a3 new, a2 left
        ]

        persistence = SpecializationMetrics.role_persistence([t0, t1])
        # Only a1 is common, and maintained role = 1.0
        assert persistence == 1.0

    def test_num_unique_roles(self):
        """Test counting unique roles."""
        assignments = [
            RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a2", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a3", role="miner", confidence=0.9),
            RoleAssignment(agent_id="a4", role="guard", confidence=0.9),
        ]
        assert SpecializationMetrics.num_unique_roles(assignments) == 3

    def test_num_unique_roles_empty(self):
        """Empty assignments have 0 unique roles."""
        assert SpecializationMetrics.num_unique_roles([]) == 0


class TestMetricsReport:
    """Tests for MetricsReport model."""

    def test_metrics_report_creation(self):
        """Test creating a metrics report."""
        from datetime import datetime

        report = MetricsReport(
            social_cognition_r=0.807,
            normalized_entropy=0.75,
            num_roles=5,
            timestamp=datetime.now(UTC),
            metadata={"experiment": "test1"},
        )

        assert report.social_cognition_r == 0.807
        assert report.normalized_entropy == 0.75
        assert report.num_roles == 5
        assert report.metadata["experiment"] == "test1"

    def test_metrics_report_bounds(self):
        """Test validation of bounded fields."""
        from datetime import datetime

        # Valid bounds
        MetricsReport(
            social_cognition_r=0.5,
            normalized_entropy=0.0,
            num_roles=0,
            timestamp=datetime.now(UTC),
        )
        MetricsReport(
            social_cognition_r=1.0,
            normalized_entropy=1.0,
            num_roles=100,
            timestamp=datetime.now(UTC),
        )

        # Invalid normalized_entropy
        with pytest.raises(ValueError):
            MetricsReport(
                social_cognition_r=0.5,
                normalized_entropy=-0.1,
                num_roles=5,
                timestamp=datetime.now(UTC),
            )

        with pytest.raises(ValueError):
            MetricsReport(
                social_cognition_r=0.5,
                normalized_entropy=1.1,
                num_roles=5,
                timestamp=datetime.now(UTC),
            )

        # Invalid num_roles
        with pytest.raises(ValueError):
            MetricsReport(
                social_cognition_r=0.5,
                normalized_entropy=0.5,
                num_roles=-1,
                timestamp=datetime.now(UTC),
            )


class TestGenerateReport:
    """Tests for generate_report function."""

    def test_generate_report_basic(self):
        """Test generating a basic report."""
        predictions = [
            SentimentPrediction(agent_id="a1", target_id="t1", predicted=0.5, actual=0.6),
            SentimentPrediction(agent_id="a2", target_id="t2", predicted=0.7, actual=0.8),
        ]
        assignments = [
            RoleAssignment(agent_id="a1", role="farmer", confidence=0.9),
            RoleAssignment(agent_id="a2", role="miner", confidence=0.8),
        ]

        report = generate_report(predictions, assignments)

        assert isinstance(report, MetricsReport)
        assert report.num_roles == 2
        assert "sentiment_mae" in report.metadata
        assert "sentiment_accuracy" in report.metadata
        assert "role_entropy" in report.metadata
        assert "role_distribution" in report.metadata
        assert report.metadata["num_predictions"] == 2
        assert report.metadata["num_assignments"] == 2

    def test_generate_report_with_metadata(self):
        """Test generating report with custom metadata."""
        predictions = [
            SentimentPrediction(agent_id="a1", target_id="t1", predicted=0.5, actual=0.5)
        ]
        assignments = [RoleAssignment(agent_id="a1", role="farmer", confidence=0.9)]
        custom_metadata = {"experiment_id": "exp123", "duration": 3600}

        report = generate_report(predictions, assignments, metadata=custom_metadata)

        assert report.metadata["experiment_id"] == "exp123"
        assert report.metadata["duration"] == 3600
        # Should also have auto-generated metadata
        assert "sentiment_mae" in report.metadata

    def test_generate_report_empty_inputs(self):
        """Test generating report with empty inputs."""
        report = generate_report([], [])

        assert report.social_cognition_r == 0.0
        assert report.normalized_entropy == 0.0
        assert report.num_roles == 0
        assert report.metadata["num_predictions"] == 0
        assert report.metadata["num_assignments"] == 0

    def test_generate_report_timestamp(self):
        """Test that report includes timestamp."""
        from datetime import datetime

        predictions = [
            SentimentPrediction(agent_id="a1", target_id="t1", predicted=0.5, actual=0.5)
        ]
        assignments = [RoleAssignment(agent_id="a1", role="farmer", confidence=0.9)]

        before = datetime.now(UTC)
        report = generate_report(predictions, assignments)
        after = datetime.now(UTC)

        assert before <= report.timestamp <= after
