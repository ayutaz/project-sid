"""Tests for the Prometheus metrics exporter."""

from __future__ import annotations

import re
import threading

import pytest

from piano.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    PianoMetrics,
    _escape_label_value,
    _format_labels,
    _format_value,
    _labels_key,
)

# ---------------------------------------------------------------------------
# Counter tests
# ---------------------------------------------------------------------------


class TestCounter:
    """Tests for the Counter metric type."""

    def test_default_value_is_zero(self) -> None:
        c = Counter("test_total", "A test counter", [])
        assert c.get() == 0.0

    def test_inc_default(self) -> None:
        c = Counter("test_total", "A test counter", [])
        c.inc()
        assert c.get() == 1.0

    def test_inc_custom_value(self) -> None:
        c = Counter("test_total", "A test counter", [])
        c.inc(5.0)
        assert c.get() == 5.0

    def test_inc_accumulates(self) -> None:
        c = Counter("test_total", "A test counter", [])
        c.inc(3.0)
        c.inc(2.0)
        assert c.get() == 5.0

    def test_inc_negative_raises(self) -> None:
        c = Counter("test_total", "A test counter", [])
        with pytest.raises(ValueError, match="non-negative"):
            c.inc(-1.0)

    def test_labelled_counters_are_independent(self) -> None:
        c = Counter("http_total", "Requests", ["method"])
        c.inc(labels={"method": "GET"})
        c.inc(labels={"method": "POST"})
        c.inc(labels={"method": "GET"})

        assert c.get(labels={"method": "GET"}) == 2.0
        assert c.get(labels={"method": "POST"}) == 1.0

    def test_reset_clears_all(self) -> None:
        c = Counter("test_total", "A test counter", [])
        c.inc(10.0)
        c.reset()
        assert c.get() == 0.0

    def test_export_lines_format(self) -> None:
        c = Counter("my_counter", "My description", [])
        c.inc(42.0)
        lines = c.export_lines()
        assert lines[0] == "# HELP my_counter My description"
        assert lines[1] == "# TYPE my_counter counter"
        assert lines[2] == "my_counter 42"

    def test_export_lines_with_labels(self) -> None:
        c = Counter("req_total", "Requests", ["method", "status"])
        c.inc(labels={"method": "GET", "status": "200"})
        lines = c.export_lines()
        assert any('method="GET"' in line and 'status="200"' in line for line in lines)


# ---------------------------------------------------------------------------
# Gauge tests
# ---------------------------------------------------------------------------


class TestGauge:
    """Tests for the Gauge metric type."""

    def test_default_value_is_zero(self) -> None:
        g = Gauge("test_gauge", "A test gauge", [])
        assert g.get() == 0.0

    def test_set_value(self) -> None:
        g = Gauge("test_gauge", "A test gauge", [])
        g.set(42.0)
        assert g.get() == 42.0

    def test_inc_default(self) -> None:
        g = Gauge("test_gauge", "A test gauge", [])
        g.inc()
        assert g.get() == 1.0

    def test_dec_default(self) -> None:
        g = Gauge("test_gauge", "A test gauge", [])
        g.set(5.0)
        g.dec()
        assert g.get() == 4.0

    def test_inc_custom_value(self) -> None:
        g = Gauge("test_gauge", "A test gauge", [])
        g.inc(3.5)
        assert g.get() == 3.5

    def test_dec_custom_value(self) -> None:
        g = Gauge("test_gauge", "A test gauge", [])
        g.set(10.0)
        g.dec(3.0)
        assert g.get() == 7.0

    def test_gauge_can_go_negative(self) -> None:
        g = Gauge("test_gauge", "A test gauge", [])
        g.dec(5.0)
        assert g.get() == -5.0

    def test_labelled_gauges_are_independent(self) -> None:
        g = Gauge("active", "Active items", ["type"])
        g.set(10.0, labels={"type": "a"})
        g.set(20.0, labels={"type": "b"})
        assert g.get(labels={"type": "a"}) == 10.0
        assert g.get(labels={"type": "b"}) == 20.0

    def test_reset_clears_all(self) -> None:
        g = Gauge("test_gauge", "A test gauge", [])
        g.set(99.0)
        g.reset()
        assert g.get() == 0.0

    def test_export_lines_format(self) -> None:
        g = Gauge("queue_size", "Queue size", [])
        g.set(7.0)
        lines = g.export_lines()
        assert lines[0] == "# HELP queue_size Queue size"
        assert lines[1] == "# TYPE queue_size gauge"
        assert lines[2] == "queue_size 7"


# ---------------------------------------------------------------------------
# Histogram tests
# ---------------------------------------------------------------------------


class TestHistogram:
    """Tests for the Histogram metric type."""

    def test_default_count_is_zero(self) -> None:
        h = Histogram("dur", "Duration", [], buckets=(1.0, 5.0, float("inf")))
        assert h.get_count() == 0
        assert h.get_sum() == 0.0

    def test_observe_updates_count_and_sum(self) -> None:
        h = Histogram("dur", "Duration", [], buckets=(1.0, 5.0, float("inf")))
        h.observe(0.5)
        h.observe(2.0)
        assert h.get_count() == 2
        assert h.get_sum() == pytest.approx(2.5)

    def test_buckets_are_cumulative(self) -> None:
        h = Histogram("dur", "Duration", [], buckets=(1.0, 5.0, 10.0, float("inf")))
        h.observe(0.5)  # <= 1.0, 5.0, 10.0, +Inf
        h.observe(3.0)  # <= 5.0, 10.0, +Inf
        h.observe(7.0)  # <= 10.0, +Inf
        h.observe(20.0)  # <= +Inf

        buckets = h.get_buckets()
        # (1.0, 1), (5.0, 2), (10.0, 3), (+Inf, 4)
        assert buckets[0] == (1.0, 1)
        assert buckets[1] == (5.0, 2)
        assert buckets[2] == (10.0, 3)
        assert buckets[3] == (float("inf"), 4)

    def test_inf_bucket_always_added(self) -> None:
        h = Histogram("dur", "Duration", [], buckets=(1.0, 5.0))
        # +Inf should be added automatically
        buckets = h.get_buckets()
        bounds = [b[0] for b in buckets]
        assert float("inf") in bounds

    def test_labelled_histograms_are_independent(self) -> None:
        h = Histogram("lat", "Latency", ["tier"], buckets=(1.0, float("inf")))
        h.observe(0.5, labels={"tier": "fast"})
        h.observe(2.0, labels={"tier": "slow"})

        assert h.get_count(labels={"tier": "fast"}) == 1
        assert h.get_count(labels={"tier": "slow"}) == 1
        assert h.get_sum(labels={"tier": "fast"}) == pytest.approx(0.5)
        assert h.get_sum(labels={"tier": "slow"}) == pytest.approx(2.0)

    def test_reset_clears_all(self) -> None:
        h = Histogram("dur", "Duration", [], buckets=(1.0, float("inf")))
        h.observe(1.0)
        h.reset()
        assert h.get_count() == 0
        assert h.get_sum() == 0.0

    def test_export_lines_contain_buckets(self) -> None:
        h = Histogram("req_dur", "Request duration", [], buckets=(0.5, 1.0, float("inf")))
        h.observe(0.3)
        lines = h.export_lines()

        assert lines[0] == "# HELP req_dur Request duration"
        assert lines[1] == "# TYPE req_dur histogram"
        # Should contain _bucket, _count, _sum lines
        text = "\n".join(lines)
        assert "req_dur_bucket" in text
        assert "req_dur_count" in text
        assert "req_dur_sum" in text
        assert "+Inf" in text

    def test_get_buckets_empty_labels(self) -> None:
        """get_buckets returns zeroed buckets for unobserved labels."""
        h = Histogram("dur", "Duration", ["x"], buckets=(1.0, float("inf")))
        buckets = h.get_buckets(labels={"x": "never"})
        assert all(count == 0 for _, count in buckets)


# ---------------------------------------------------------------------------
# MetricsRegistry tests
# ---------------------------------------------------------------------------


class TestMetricsRegistry:
    """Tests for the MetricsRegistry class."""

    def test_register_counter(self) -> None:
        reg = MetricsRegistry()
        c = reg.counter("test_total", "A counter")
        assert isinstance(c, Counter)

    def test_register_gauge(self) -> None:
        reg = MetricsRegistry()
        g = reg.gauge("test_gauge", "A gauge")
        assert isinstance(g, Gauge)

    def test_register_histogram(self) -> None:
        reg = MetricsRegistry()
        h = reg.histogram("test_hist", "A histogram")
        assert isinstance(h, Histogram)

    def test_duplicate_registration_returns_same(self) -> None:
        reg = MetricsRegistry()
        c1 = reg.counter("test_total", "A counter")
        c2 = reg.counter("test_total", "A counter")
        assert c1 is c2

    def test_conflicting_type_raises(self) -> None:
        reg = MetricsRegistry()
        reg.counter("test_total", "A counter")
        with pytest.raises(TypeError, match="already registered"):
            reg.gauge("test_total", "Conflicting gauge")

    def test_get_existing(self) -> None:
        reg = MetricsRegistry()
        c = reg.counter("test_total", "A counter")
        assert reg.get("test_total") is c

    def test_get_nonexistent(self) -> None:
        reg = MetricsRegistry()
        assert reg.get("nonexistent") is None

    def test_export_empty(self) -> None:
        reg = MetricsRegistry()
        assert reg.export() == ""

    def test_export_includes_all_metrics(self) -> None:
        reg = MetricsRegistry()
        c = reg.counter("req_total", "Requests")
        g = reg.gauge("queue_size", "Queue")
        c.inc()
        g.set(5.0)
        text = reg.export()
        assert "req_total" in text
        assert "queue_size" in text

    def test_reset_clears_all_metrics(self) -> None:
        reg = MetricsRegistry()
        c = reg.counter("test_total", "A counter")
        g = reg.gauge("test_gauge", "A gauge")
        c.inc(10.0)
        g.set(42.0)
        reg.reset()
        assert c.get() == 0.0
        assert g.get() == 0.0

    def test_histogram_with_custom_buckets(self) -> None:
        reg = MetricsRegistry()
        h = reg.histogram("lat", "Latency", buckets=(0.1, 0.5, 1.0))
        assert isinstance(h, Histogram)
        h.observe(0.3)
        assert h.get_count() == 1

    def test_len(self) -> None:
        reg = MetricsRegistry()
        assert len(reg) == 0
        reg.counter("c1", "Counter 1")
        assert len(reg) == 1
        reg.gauge("g1", "Gauge 1")
        assert len(reg) == 2
        reg.histogram("h1", "Histogram 1")
        assert len(reg) == 3


# ---------------------------------------------------------------------------
# PianoMetrics tests
# ---------------------------------------------------------------------------


class TestPianoMetrics:
    """Tests for the pre-defined PIANO metrics."""

    def test_all_metrics_registered(self) -> None:
        m = PianoMetrics()
        expected_names = [
            "piano_agent_tick_total",
            "piano_llm_requests_total",
            "piano_llm_cost_usd_total",
            "piano_llm_latency_seconds",
            "piano_agent_count",
            "piano_worker_count",
            "piano_memory_stm_entries",
            "piano_bridge_commands_total",
            "piano_checkpoint_duration_seconds",
        ]
        for name in expected_names:
            assert m.registry.get(name) is not None, f"Metric {name} not registered"

    def test_agent_tick_counter(self) -> None:
        m = PianoMetrics()
        m.agent_tick_total.inc(labels={"agent_id": "agent-001"})
        assert m.agent_tick_total.get(labels={"agent_id": "agent-001"}) == 1.0

    def test_llm_requests_counter(self) -> None:
        m = PianoMetrics()
        labels = {"provider": "openai", "tier": "slow", "status": "ok"}
        m.llm_requests_total.inc(labels=labels)
        m.llm_requests_total.inc(labels=labels)
        assert m.llm_requests_total.get(labels=labels) == 2.0

    def test_llm_cost_counter(self) -> None:
        m = PianoMetrics()
        m.llm_cost_usd_total.inc(0.003, labels={"provider": "openai", "tier": "slow"})
        cost = m.llm_cost_usd_total.get(labels={"provider": "openai", "tier": "slow"})
        assert cost == pytest.approx(0.003)

    def test_agent_count_gauge(self) -> None:
        m = PianoMetrics()
        m.agent_count.set(10.0)
        m.agent_count.dec(2.0)
        assert m.agent_count.get() == 8.0

    def test_worker_count_gauge(self) -> None:
        m = PianoMetrics()
        m.worker_count.set(4.0)
        assert m.worker_count.get() == 4.0

    def test_memory_stm_entries_gauge(self) -> None:
        m = PianoMetrics()
        m.memory_stm_entries.set(25.0, labels={"agent_id": "agent-001"})
        assert m.memory_stm_entries.get(labels={"agent_id": "agent-001"}) == 25.0

    def test_bridge_commands_counter(self) -> None:
        m = PianoMetrics()
        m.bridge_commands_total.inc(labels={"action": "move", "status": "ok"})
        assert m.bridge_commands_total.get(labels={"action": "move", "status": "ok"}) == 1.0

    def test_llm_latency_histogram(self) -> None:
        m = PianoMetrics()
        m.llm_latency_seconds.observe(1.5, labels={"provider": "openai", "tier": "slow"})
        assert m.llm_latency_seconds.get_count(labels={"provider": "openai", "tier": "slow"}) == 1

    def test_checkpoint_duration_histogram(self) -> None:
        m = PianoMetrics()
        m.checkpoint_duration_seconds.observe(0.25, labels={"agent_id": "agent-001"})
        assert m.checkpoint_duration_seconds.get_count(labels={"agent_id": "agent-001"}) == 1
        cp_sum = m.checkpoint_duration_seconds.get_sum(labels={"agent_id": "agent-001"})
        assert cp_sum == pytest.approx(0.25)

    def test_custom_registry(self) -> None:
        reg = MetricsRegistry()
        m = PianoMetrics(registry=reg)
        assert m.registry is reg
        assert reg.get("piano_agent_tick_total") is not None

    def test_full_export_format(self) -> None:
        m = PianoMetrics()
        m.agent_tick_total.inc(labels={"agent_id": "agent-001"})
        m.agent_count.set(5.0)
        text = m.registry.export()
        # Should contain HELP and TYPE lines
        assert "# HELP piano_agent_tick_total" in text
        assert "# TYPE piano_agent_tick_total counter" in text
        assert "# HELP piano_agent_count" in text
        assert "# TYPE piano_agent_count gauge" in text


# ---------------------------------------------------------------------------
# Label / formatting helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for internal helper functions."""

    def test_labels_key_empty(self) -> None:
        assert _labels_key({}) == ()

    def test_labels_key_sorted(self) -> None:
        result = _labels_key({"b": "2", "a": "1"})
        assert result == (("a", "1"), ("b", "2"))

    def test_format_labels_empty(self) -> None:
        assert _format_labels({}) == ""

    def test_format_labels_single(self) -> None:
        assert _format_labels({"method": "GET"}) == '{method="GET"}'

    def test_format_labels_multiple_sorted(self) -> None:
        result = _format_labels({"status": "200", "method": "GET"})
        assert result == '{method="GET",status="200"}'

    def test_escape_label_value_backslash(self) -> None:
        assert _escape_label_value("a\\b") == "a\\\\b"

    def test_escape_label_value_double_quote(self) -> None:
        assert _escape_label_value('say "hello"') == 'say \\"hello\\"'

    def test_escape_label_value_newline(self) -> None:
        assert _escape_label_value("line1\nline2") == "line1\\nline2"

    def test_escape_label_value_combined(self) -> None:
        assert _escape_label_value('a\\b\n"c"') == 'a\\\\b\\n\\"c\\"'

    def test_format_labels_escapes_values(self) -> None:
        result = _format_labels({"msg": 'say "hi"'})
        assert result == '{msg="say \\"hi\\""}'

    def test_format_value_integer(self) -> None:
        assert _format_value(42.0) == "42"

    def test_format_value_float(self) -> None:
        assert _format_value(3.14) == "3.14"

    def test_format_value_inf(self) -> None:
        assert _format_value(float("inf")) == "+Inf"

    def test_format_value_nan(self) -> None:
        assert _format_value(float("nan")) == "NaN"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for concurrent access to metrics."""

    def test_counter_concurrent_increments(self) -> None:
        """Multiple threads increment a counter without data loss."""
        c = Counter("conc_total", "Concurrent", [])
        n_threads = 10
        n_increments = 1000

        def worker() -> None:
            for _ in range(n_increments):
                c.inc()

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert c.get() == n_threads * n_increments

    def test_gauge_concurrent_sets(self) -> None:
        """Multiple threads setting a gauge do not crash."""
        g = Gauge("conc_gauge", "Concurrent", [])
        n_threads = 10

        def worker(val: float) -> None:
            for _ in range(100):
                g.set(val)

        threads = [threading.Thread(target=worker, args=(float(i),)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Just verify it's a valid float (no corruption)
        assert isinstance(g.get(), float)

    def test_histogram_concurrent_observations(self) -> None:
        """Multiple threads observe values without data loss."""
        h = Histogram("conc_hist", "Concurrent", [], buckets=(1.0, float("inf")))
        n_threads = 10
        n_obs = 100

        def worker() -> None:
            for _ in range(n_obs):
                h.observe(0.5)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert h.get_count() == n_threads * n_obs


# ---------------------------------------------------------------------------
# Prometheus text format validation
# ---------------------------------------------------------------------------


class TestPrometheusFormat:
    """Tests that exported text conforms to Prometheus exposition format."""

    def test_counter_line_format(self) -> None:
        reg = MetricsRegistry()
        c = reg.counter("http_requests_total", "Total HTTP requests", labels=["method"])
        c.inc(labels={"method": "GET"})
        text = reg.export()
        assert "# HELP http_requests_total Total HTTP requests\n" in text
        assert "# TYPE http_requests_total counter\n" in text
        assert 'http_requests_total{method="GET"} 1\n' in text

    def test_histogram_bucket_format(self) -> None:
        reg = MetricsRegistry()
        h = reg.histogram("dur_seconds", "Duration", buckets=(0.5, 1.0))
        h.observe(0.3)
        text = reg.export()
        # Check for le= labels in bucket lines
        assert re.search(r'dur_seconds_bucket\{le="0\.5"\} 1', text)
        assert re.search(r'dur_seconds_bucket\{le="1\.0"\} 1', text)
        assert re.search(r'dur_seconds_bucket\{le="\+Inf"\} 1', text)
        assert "dur_seconds_count 1" in text

    def test_export_ends_with_newline(self) -> None:
        reg = MetricsRegistry()
        reg.counter("test_total", "Test").inc()
        text = reg.export()
        assert text.endswith("\n")


# ---------------------------------------------------------------------------
# Label names validation
# ---------------------------------------------------------------------------


class TestLabelNamesValidation:
    """Tests for label_names validation in inc/set/observe."""

    def test_counter_wrong_labels_raises(self) -> None:
        c = Counter("req_total", "Requests", ["method", "status"])
        with pytest.raises(ValueError, match="Label mismatch"):
            c.inc(labels={"method": "GET"})  # missing 'status'

    def test_counter_extra_labels_raises(self) -> None:
        c = Counter("req_total", "Requests", ["method"])
        with pytest.raises(ValueError, match="Label mismatch"):
            c.inc(labels={"method": "GET", "status": "200"})

    def test_counter_correct_labels_ok(self) -> None:
        c = Counter("req_total", "Requests", ["method"])
        c.inc(labels={"method": "GET"})
        assert c.get(labels={"method": "GET"}) == 1.0

    def test_counter_no_label_names_allows_any(self) -> None:
        c = Counter("req_total", "Requests", [])
        c.inc(labels={"anything": "goes"})
        assert c.get(labels={"anything": "goes"}) == 1.0

    def test_gauge_wrong_labels_raises(self) -> None:
        g = Gauge("active", "Active", ["type"])
        with pytest.raises(ValueError, match="Label mismatch"):
            g.set(1.0, labels={"wrong": "key"})

    def test_gauge_inc_wrong_labels_raises(self) -> None:
        g = Gauge("active", "Active", ["type"])
        with pytest.raises(ValueError, match="Label mismatch"):
            g.inc(labels={"wrong": "key"})

    def test_gauge_dec_wrong_labels_raises(self) -> None:
        g = Gauge("active", "Active", ["type"])
        with pytest.raises(ValueError, match="Label mismatch"):
            g.dec(labels={"wrong": "key"})

    def test_histogram_wrong_labels_raises(self) -> None:
        h = Histogram("dur", "Duration", ["tier"], buckets=(1.0, float("inf")))
        with pytest.raises(ValueError, match="Label mismatch"):
            h.observe(0.5, labels={"wrong": "key"})
