"""Tests for performance benchmark framework."""

from __future__ import annotations

import time

import pytest

from piano.eval.performance import (
    BenchmarkResult,
    LLMCallRecord,
    PerformanceBenchmark,
    PerformanceConfig,
    RegressionDetector,
    RegressionItem,
    RegressionReport,
    TickRecord,
    compute_percentile,
)

# ---------------------------------------------------------------------------
# compute_percentile
# ---------------------------------------------------------------------------


class TestComputePercentile:
    """Tests for the percentile helper function."""

    def test_empty_list_returns_zero(self):
        assert compute_percentile([], 50) == 0.0

    def test_single_element(self):
        assert compute_percentile([42.0], 50) == 42.0
        assert compute_percentile([42.0], 0) == 42.0
        assert compute_percentile([42.0], 100) == 42.0

    def test_p50_even_count(self):
        """Median of [1,2,3,4] should interpolate to 2.5."""
        values = [1.0, 2.0, 3.0, 4.0]
        result = compute_percentile(values, 50)
        assert abs(result - 2.5) < 1e-10

    def test_p50_odd_count(self):
        """Median of [1,2,3,4,5] should be exactly 3."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_percentile(values, 50)
        assert abs(result - 3.0) < 1e-10

    def test_p0_returns_minimum(self):
        values = [10.0, 20.0, 30.0]
        assert compute_percentile(values, 0) == 10.0

    def test_p100_returns_maximum(self):
        values = [10.0, 20.0, 30.0]
        assert compute_percentile(values, 100) == 30.0

    def test_p95_large_list(self):
        """p95 of 0..99 -> rank=94.05, expected=94.05."""
        values = [float(i) for i in range(100)]
        result = compute_percentile(values, 95)
        assert abs(result - 94.05) < 1e-10

    def test_p99_large_list(self):
        """p99 of 0..99 -> rank=98.01, expected=98.01."""
        values = [float(i) for i in range(100)]
        result = compute_percentile(values, 99)
        assert abs(result - 98.01) < 1e-10

    def test_clamps_out_of_range(self):
        """Percentiles below 0 or above 100 are clamped."""
        values = [1.0, 2.0, 3.0]
        assert compute_percentile(values, -10) == 1.0
        assert compute_percentile(values, 200) == 3.0


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestTickRecord:
    """Tests for TickRecord model."""

    def test_creation(self):
        rec = TickRecord(agent_id="agent-001", duration_ms=15.2, timestamp=1000.0)
        assert rec.agent_id == "agent-001"
        assert rec.duration_ms == 15.2

    def test_negative_duration_rejected(self):
        with pytest.raises(ValueError):
            TickRecord(agent_id="a", duration_ms=-1.0, timestamp=0.0)


class TestLLMCallRecord:
    """Tests for LLMCallRecord model."""

    def test_creation(self):
        rec = LLMCallRecord(
            provider="openai", latency_ms=320.0, cost_usd=0.003, tokens=500, timestamp=0.0
        )
        assert rec.provider == "openai"
        assert rec.tokens == 500

    def test_negative_cost_rejected(self):
        with pytest.raises(ValueError):
            LLMCallRecord(
                provider="openai", latency_ms=1.0, cost_usd=-0.01, tokens=10, timestamp=0.0
            )


class TestBenchmarkResult:
    """Tests for BenchmarkResult model."""

    def test_creation(self):
        result = BenchmarkResult(
            duration_seconds=60.0,
            tps=100.0,
            avg_tick_latency_ms=10.0,
            p50_tick_latency_ms=8.0,
            p95_tick_latency_ms=18.0,
            p99_tick_latency_ms=25.0,
            total_llm_calls=50,
            avg_llm_latency_ms=300.0,
            total_cost_usd=0.15,
            total_bridge_commands=200,
            agent_count=10,
        )
        assert result.tps == 100.0
        assert result.agent_count == 10


class TestPerformanceConfig:
    """Tests for PerformanceConfig model."""

    def test_defaults(self):
        config = PerformanceConfig()
        assert config.tps_threshold_pct == 20.0
        assert config.latency_threshold_pct == 30.0
        assert config.cost_threshold_pct == 50.0

    def test_custom_values(self):
        config = PerformanceConfig(
            tps_threshold_pct=10.0,
            latency_threshold_pct=15.0,
            cost_threshold_pct=25.0,
        )
        assert config.tps_threshold_pct == 10.0


# ---------------------------------------------------------------------------
# PerformanceBenchmark
# ---------------------------------------------------------------------------


class TestPerformanceBenchmark:
    """Tests for the PerformanceBenchmark class."""

    def test_start_stop_lifecycle(self):
        bench = PerformanceBenchmark()
        bench.start()
        bench.record_tick("a1", 10.0)
        result = bench.stop()
        assert isinstance(result, BenchmarkResult)
        assert result.duration_seconds > 0.0

    def test_stop_without_start_raises(self):
        bench = PerformanceBenchmark()
        with pytest.raises(RuntimeError, match="not running"):
            bench.stop()

    def test_double_start_raises(self):
        bench = PerformanceBenchmark()
        bench.start()
        with pytest.raises(RuntimeError, match="already running"):
            bench.start()
        bench.stop()  # cleanup

    def test_record_tick_without_start_raises(self):
        bench = PerformanceBenchmark()
        with pytest.raises(RuntimeError, match="not running"):
            bench.record_tick("a1", 5.0)

    def test_record_llm_without_start_raises(self):
        bench = PerformanceBenchmark()
        with pytest.raises(RuntimeError, match="not running"):
            bench.record_llm_call("openai", 100.0, 0.01, 100)

    def test_record_bridge_without_start_raises(self):
        bench = PerformanceBenchmark()
        with pytest.raises(RuntimeError, match="not running"):
            bench.record_bridge_command("move", 50.0)

    def test_tps_calculation(self):
        bench = PerformanceBenchmark()
        bench.start()
        for i in range(100):
            bench.record_tick(f"agent-{i % 5}", 10.0)
        # Introduce a small sleep so duration > 0
        time.sleep(0.01)
        result = bench.stop()
        assert result.tps > 0.0
        # 100 ticks in ~0.01s -> TPS should be high
        assert result.tps > 100.0

    def test_avg_tick_latency(self):
        bench = PerformanceBenchmark()
        bench.start()
        bench.record_tick("a1", 10.0)
        bench.record_tick("a1", 20.0)
        bench.record_tick("a1", 30.0)
        result = bench.stop()
        assert abs(result.avg_tick_latency_ms - 20.0) < 1e-10

    def test_percentile_latencies(self):
        bench = PerformanceBenchmark()
        bench.start()
        # Record 100 ticks with durations 1..100
        for i in range(1, 101):
            bench.record_tick("a1", float(i))
        result = bench.stop()
        # p50 should be around 50.5
        assert 49.0 < result.p50_tick_latency_ms < 52.0
        # p95 should be around 95.05
        assert 93.0 < result.p95_tick_latency_ms < 97.0
        # p99 should be around 99.01
        assert 97.0 < result.p99_tick_latency_ms < 101.0

    def test_llm_call_tracking(self):
        bench = PerformanceBenchmark()
        bench.start()
        bench.record_llm_call("openai", latency_ms=100.0, cost_usd=0.01, tokens=500)
        bench.record_llm_call("openai", latency_ms=200.0, cost_usd=0.02, tokens=1000)
        bench.record_llm_call("anthropic", latency_ms=300.0, cost_usd=0.03, tokens=800)
        result = bench.stop()
        assert result.total_llm_calls == 3
        assert abs(result.avg_llm_latency_ms - 200.0) < 1e-10
        assert abs(result.total_cost_usd - 0.06) < 1e-10

    def test_bridge_command_tracking(self):
        bench = PerformanceBenchmark()
        bench.start()
        bench.record_bridge_command("move", 30.0)
        bench.record_bridge_command("mine", 45.0)
        bench.record_bridge_command("chat", 15.0)
        result = bench.stop()
        assert result.total_bridge_commands == 3

    def test_agent_count(self):
        bench = PerformanceBenchmark()
        bench.start()
        bench.record_tick("agent-001", 10.0)
        bench.record_tick("agent-002", 12.0)
        bench.record_tick("agent-001", 11.0)
        bench.record_tick("agent-003", 9.0)
        result = bench.stop()
        assert result.agent_count == 3

    def test_no_ticks_zero_metrics(self):
        bench = PerformanceBenchmark()
        bench.start()
        time.sleep(0.01)
        result = bench.stop()
        assert result.tps == 0.0
        assert result.avg_tick_latency_ms == 0.0
        assert result.p50_tick_latency_ms == 0.0
        assert result.agent_count == 0

    def test_restart_clears_data(self):
        """A new start() clears previous data."""
        bench = PerformanceBenchmark()
        bench.start()
        bench.record_tick("a1", 999.0)
        bench.stop()

        bench.start()
        bench.record_tick("a2", 5.0)
        result = bench.stop()
        # Only the second run's tick should count
        assert result.agent_count == 1
        assert abs(result.avg_tick_latency_ms - 5.0) < 1e-10

    def test_compute_result_raises_runtime_error_without_start(self):
        """_compute_result raises RuntimeError when start_time is None."""
        bench = PerformanceBenchmark()
        with pytest.raises(RuntimeError, match="start_time is None"):
            bench._compute_result()

    def test_compute_result_raises_runtime_error_without_stop(self):
        """_compute_result raises RuntimeError when stop_time is None."""
        bench = PerformanceBenchmark()
        bench.start()
        with pytest.raises(RuntimeError, match="stop_time is None"):
            bench._compute_result()
        bench.stop()  # cleanup


# ---------------------------------------------------------------------------
# RegressionDetector
# ---------------------------------------------------------------------------


def _make_result(**overrides: float | int) -> BenchmarkResult:
    """Helper to build a BenchmarkResult with sensible defaults."""
    defaults: dict[str, float | int] = {
        "duration_seconds": 60.0,
        "tps": 100.0,
        "avg_tick_latency_ms": 10.0,
        "p50_tick_latency_ms": 8.0,
        "p95_tick_latency_ms": 18.0,
        "p99_tick_latency_ms": 25.0,
        "total_llm_calls": 50,
        "avg_llm_latency_ms": 300.0,
        "total_cost_usd": 0.10,
        "total_bridge_commands": 200,
        "agent_count": 10,
    }
    defaults.update(overrides)
    return BenchmarkResult(**defaults)  # type: ignore[arg-type]


class TestRegressionDetector:
    """Tests for the RegressionDetector class."""

    def test_no_baseline_raises(self):
        detector = RegressionDetector()
        with pytest.raises(RuntimeError, match="No baseline"):
            detector.check_regression(_make_result())

    def test_identical_results_pass(self):
        detector = RegressionDetector()
        baseline = _make_result()
        detector.set_baseline(baseline)
        report = detector.check_regression(baseline)
        assert report.passed
        assert len(report.regressions) == 0

    def test_tps_regression_detected(self):
        """A >20% TPS drop should be flagged."""
        detector = RegressionDetector()
        detector.set_baseline(_make_result(tps=100.0))
        report = detector.check_regression(_make_result(tps=70.0))  # 30% drop
        assert not report.passed
        tps_items = [r for r in report.regressions if r.metric_name == "tps"]
        assert len(tps_items) == 1
        assert tps_items[0].change_pct == pytest.approx(30.0)

    def test_tps_within_threshold_passes(self):
        """A 15% TPS drop (under 20% threshold) should pass."""
        detector = RegressionDetector()
        detector.set_baseline(_make_result(tps=100.0))
        report = detector.check_regression(_make_result(tps=85.0))
        tps_items = [r for r in report.regressions if r.metric_name == "tps"]
        assert len(tps_items) == 0

    def test_latency_regression_detected(self):
        """A >30% latency increase should be flagged."""
        detector = RegressionDetector()
        detector.set_baseline(_make_result(avg_tick_latency_ms=10.0))
        report = detector.check_regression(_make_result(avg_tick_latency_ms=14.0))  # +40%
        lat_items = [r for r in report.regressions if r.metric_name == "avg_tick_latency_ms"]
        assert len(lat_items) == 1

    def test_cost_regression_detected(self):
        """A >50% cost increase should be flagged."""
        detector = RegressionDetector()
        detector.set_baseline(_make_result(total_cost_usd=0.10))
        report = detector.check_regression(_make_result(total_cost_usd=0.20))  # +100%
        cost_items = [r for r in report.regressions if r.metric_name == "total_cost_usd"]
        assert len(cost_items) == 1
        assert cost_items[0].change_pct == pytest.approx(100.0)

    def test_cost_within_threshold_passes(self):
        """A 40% cost increase (under 50% threshold) should pass."""
        detector = RegressionDetector()
        detector.set_baseline(_make_result(total_cost_usd=0.10))
        report = detector.check_regression(_make_result(total_cost_usd=0.14))
        cost_items = [r for r in report.regressions if r.metric_name == "total_cost_usd"]
        assert len(cost_items) == 0

    def test_multiple_regressions(self):
        """Multiple regressions can be reported at once."""
        detector = RegressionDetector()
        baseline = _make_result(tps=100.0, avg_tick_latency_ms=10.0, total_cost_usd=0.10)
        detector.set_baseline(baseline)
        report = detector.check_regression(
            _make_result(tps=50.0, avg_tick_latency_ms=20.0, total_cost_usd=0.30)
        )
        assert not report.passed
        regressed_metrics = {r.metric_name for r in report.regressions}
        assert "tps" in regressed_metrics
        assert "avg_tick_latency_ms" in regressed_metrics
        assert "total_cost_usd" in regressed_metrics

    def test_custom_config(self):
        """Custom thresholds are respected."""
        config = PerformanceConfig(tps_threshold_pct=5.0)
        detector = RegressionDetector(config=config)
        detector.set_baseline(_make_result(tps=100.0))
        # 10% drop now exceeds 5% threshold
        report = detector.check_regression(_make_result(tps=90.0))
        tps_items = [r for r in report.regressions if r.metric_name == "tps"]
        assert len(tps_items) == 1

    def test_is_within_threshold_static(self):
        assert RegressionDetector.is_within_threshold(100.0, 100.0, 10.0) is True
        assert RegressionDetector.is_within_threshold(110.0, 100.0, 10.0) is True
        assert RegressionDetector.is_within_threshold(111.0, 100.0, 10.0) is False
        assert RegressionDetector.is_within_threshold(89.0, 100.0, 10.0) is False

    def test_is_within_threshold_zero_baseline(self):
        """Zero baseline: only zero current is within threshold."""
        assert RegressionDetector.is_within_threshold(0.0, 0.0, 10.0) is True
        assert RegressionDetector.is_within_threshold(1.0, 0.0, 10.0) is False

    def test_baseline_property(self):
        detector = RegressionDetector()
        assert detector.baseline is None
        baseline = _make_result()
        detector.set_baseline(baseline)
        assert detector.baseline is baseline

    def test_config_property(self):
        config = PerformanceConfig(tps_threshold_pct=5.0)
        detector = RegressionDetector(config=config)
        assert detector.config.tps_threshold_pct == 5.0

    def test_zero_baseline_tps_skip(self):
        """Zero baseline TPS should not trigger a regression (can't compute %)."""
        detector = RegressionDetector()
        detector.set_baseline(_make_result(tps=0.0))
        report = detector.check_regression(_make_result(tps=50.0))
        tps_items = [r for r in report.regressions if r.metric_name == "tps"]
        assert len(tps_items) == 0

    def test_zero_baseline_latency_increase(self):
        """Zero baseline latency with nonzero current should flag regression."""
        detector = RegressionDetector()
        detector.set_baseline(_make_result(avg_tick_latency_ms=0.0))
        report = detector.check_regression(_make_result(avg_tick_latency_ms=5.0))
        lat_items = [r for r in report.regressions if r.metric_name == "avg_tick_latency_ms"]
        assert len(lat_items) == 1

    def test_zero_baseline_uses_finite_change_pct(self):
        """Zero baseline uses finite 99999.0 instead of float('inf')."""
        detector = RegressionDetector()
        detector.set_baseline(_make_result(avg_tick_latency_ms=0.0))
        report = detector.check_regression(_make_result(avg_tick_latency_ms=5.0))
        lat_items = [r for r in report.regressions if r.metric_name == "avg_tick_latency_ms"]
        assert len(lat_items) == 1
        assert lat_items[0].change_pct == 99999.0
        import math

        assert not math.isinf(lat_items[0].change_pct)


# ---------------------------------------------------------------------------
# RegressionReport / RegressionItem models
# ---------------------------------------------------------------------------


class TestRegressionModels:
    """Tests for RegressionReport and RegressionItem models."""

    def test_regression_item_creation(self):
        item = RegressionItem(
            metric_name="tps",
            baseline=100.0,
            current=70.0,
            change_pct=30.0,
            threshold_pct=20.0,
        )
        assert item.metric_name == "tps"
        assert item.change_pct == 30.0

    def test_regression_report_passed(self):
        report = RegressionReport(passed=True, regressions=[])
        assert report.passed
        assert len(report.regressions) == 0

    def test_regression_report_failed(self):
        item = RegressionItem(
            metric_name="tps",
            baseline=100.0,
            current=50.0,
            change_pct=50.0,
            threshold_pct=20.0,
        )
        report = RegressionReport(passed=False, regressions=[item])
        assert not report.passed
        assert len(report.regressions) == 1
