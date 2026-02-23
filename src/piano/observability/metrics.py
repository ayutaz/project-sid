"""Lightweight Prometheus metrics exporter for the PIANO architecture.

Provides Counter, Gauge, and Histogram metric types with thread-safe
label handling and Prometheus text format export. Does not depend on
the prometheus_client library -- all formatting is self-contained.

The ``PianoMetrics`` class pre-registers all standard PIANO metrics
so that every component can record telemetry through a single registry.

Reference: docs/implementation/08-infrastructure.md
"""

from __future__ import annotations

__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "PianoMetrics",
]

import math
import threading
from collections import defaultdict
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Default histogram buckets (seconds) -- matches Prometheus defaults
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    float("inf"),
)

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def _labels_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Convert a labels dict to a hashable, sorted tuple."""
    return tuple(sorted(labels.items()))


def _format_labels(labels: dict[str, str]) -> str:
    """Format labels as a Prometheus label string ``{k1="v1",k2="v2"}``."""
    if not labels:
        return ""
    parts = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + parts + "}"


# ---------------------------------------------------------------------------
# Metric types
# ---------------------------------------------------------------------------


class Counter:
    """Monotonically increasing counter.

    Supports labelled time series: each unique label combination is
    tracked independently.
    """

    def __init__(self, name: str, description: str, label_names: list[str]) -> None:
        self.name = name
        self.description = description
        self.label_names = label_names
        self._lock = threading.Lock()
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)

    def inc(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment the counter.

        Args:
            value: Amount to increment (must be >= 0).
            labels: Optional label dict.

        Raises:
            ValueError: If *value* is negative.
        """
        if value < 0:
            msg = "Counter increment must be non-negative"
            raise ValueError(msg)
        key = _labels_key(labels or {})
        with self._lock:
            self._values[key] += value

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Return current counter value for the given labels."""
        key = _labels_key(labels or {})
        with self._lock:
            return self._values[key]

    def reset(self) -> None:
        """Reset all label series to zero."""
        with self._lock:
            self._values.clear()

    def export_lines(self) -> list[str]:
        """Return Prometheus text-format lines for this counter."""
        lines: list[str] = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            for key, value in sorted(self._values.items()):
                label_dict = dict(key)
                label_str = _format_labels(label_dict)
                lines.append(f"{self.name}{label_str} {_format_value(value)}")
        return lines


class Gauge:
    """Metric that can go up and down.

    Supports labelled time series.
    """

    def __init__(self, name: str, description: str, label_names: list[str]) -> None:
        self.name = name
        self.description = description
        self.label_names = label_names
        self._lock = threading.Lock()
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Set the gauge to an arbitrary value."""
        key = _labels_key(labels or {})
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment the gauge."""
        key = _labels_key(labels or {})
        with self._lock:
            self._values[key] += value

    def dec(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Decrement the gauge."""
        key = _labels_key(labels or {})
        with self._lock:
            self._values[key] -= value

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Return current gauge value for the given labels."""
        key = _labels_key(labels or {})
        with self._lock:
            return self._values[key]

    def reset(self) -> None:
        """Reset all label series to zero."""
        with self._lock:
            self._values.clear()

    def export_lines(self) -> list[str]:
        """Return Prometheus text-format lines for this gauge."""
        lines: list[str] = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
        ]
        with self._lock:
            for key, value in sorted(self._values.items()):
                label_dict = dict(key)
                label_str = _format_labels(label_dict)
                lines.append(f"{self.name}{label_str} {_format_value(value)}")
        return lines


class _HistogramData:
    """Internal per-label-set accumulator for a histogram."""

    __slots__ = ("bucket_bounds", "bucket_counts", "count", "total")

    def __init__(self, bucket_bounds: tuple[float, ...]) -> None:
        self.bucket_bounds = bucket_bounds
        self.bucket_counts: list[int] = [0] * len(bucket_bounds)
        self.count: int = 0
        self.total: float = 0.0

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        for i, bound in enumerate(self.bucket_bounds):
            if value <= bound:
                self.bucket_counts[i] += 1
                break


class Histogram:
    """Cumulative histogram with configurable buckets.

    Buckets are cumulative (each bucket includes all observations <=
    the upper bound).  An ``+Inf`` bucket is always present.
    """

    def __init__(
        self,
        name: str,
        description: str,
        label_names: list[str],
        buckets: tuple[float, ...] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.label_names = label_names
        # Ensure +Inf is always the last bucket
        raw = list(buckets or DEFAULT_BUCKETS)
        if not raw or raw[-1] != float("inf"):
            raw.append(float("inf"))
        self._buckets: tuple[float, ...] = tuple(raw)
        self._lock = threading.Lock()
        self._data: dict[tuple[tuple[str, str], ...], _HistogramData] = {}

    def _get_data(self, key: tuple[tuple[str, str], ...]) -> _HistogramData:
        """Return (or create) accumulator for *key*."""
        if key not in self._data:
            self._data[key] = _HistogramData(self._buckets)
        return self._data[key]

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Record an observation."""
        key = _labels_key(labels or {})
        with self._lock:
            self._get_data(key).observe(value)

    def get_count(self, labels: dict[str, str] | None = None) -> int:
        """Return observation count for the given labels."""
        key = _labels_key(labels or {})
        with self._lock:
            if key not in self._data:
                return 0
            return self._data[key].count

    def get_sum(self, labels: dict[str, str] | None = None) -> float:
        """Return observation sum for the given labels."""
        key = _labels_key(labels or {})
        with self._lock:
            if key not in self._data:
                return 0.0
            return self._data[key].total

    def get_buckets(self, labels: dict[str, str] | None = None) -> list[tuple[float, int]]:
        """Return list of ``(upper_bound, cumulative_count)`` for the given labels."""
        key = _labels_key(labels or {})
        with self._lock:
            if key not in self._data:
                return [(b, 0) for b in self._buckets]
            data = self._data[key]
            cumulative = 0
            result: list[tuple[float, int]] = []
            for i, bound in enumerate(data.bucket_bounds):
                cumulative += data.bucket_counts[i]
                result.append((bound, cumulative))
            return result

    def reset(self) -> None:
        """Reset all label series."""
        with self._lock:
            self._data.clear()

    def export_lines(self) -> list[str]:
        """Return Prometheus text-format lines for this histogram."""
        lines: list[str] = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            for key in sorted(self._data):
                data = self._data[key]
                label_dict = dict(key)
                label_str_base = _format_labels(label_dict)

                # Bucket lines (cumulative)
                cumulative = 0
                for i, bound in enumerate(data.bucket_bounds):
                    cumulative += data.bucket_counts[i]
                    le_label = dict(key)
                    le_label["le"] = _format_bucket_bound(bound)
                    le_str = _format_labels(le_label)
                    lines.append(f"{self.name}_bucket{le_str} {cumulative}")

                # _count and _sum
                lines.append(f"{self.name}_count{label_str_base} {data.count}")
                lines.append(f"{self.name}_sum{label_str_base} {_format_value(data.total)}")
        return lines


# ---------------------------------------------------------------------------
# Value formatting helpers
# ---------------------------------------------------------------------------


def _format_value(value: float) -> str:
    """Format a float for Prometheus output.

    Integers are rendered without a decimal point; floats use up to
    17 significant digits.
    """
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if math.isnan(value):
        return "NaN"
    if value == int(value) and not math.isinf(value):
        return str(int(value))
    return repr(value)


def _format_bucket_bound(bound: float) -> str:
    """Format a histogram bucket upper bound."""
    if math.isinf(bound):
        return "+Inf"
    if bound == int(bound):
        return f"{bound:.1f}"
    return repr(bound)


# ---------------------------------------------------------------------------
# MetricsRegistry
# ---------------------------------------------------------------------------


class MetricsRegistry:
    """Central registry for all metric instances.

    Thread-safe: multiple modules may register and update metrics
    concurrently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}

    def counter(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> Counter:
        """Register and return a :class:`Counter`.

        If a counter with the same *name* already exists it is returned
        without modification.

        Raises:
            TypeError: If *name* is already registered with a different type.
        """
        return self._register(Counter, name, description, labels or [])

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> Gauge:
        """Register and return a :class:`Gauge`.

        Raises:
            TypeError: If *name* is already registered with a different type.
        """
        return self._register(Gauge, name, description, labels or [])

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ) -> Histogram:
        """Register and return a :class:`Histogram`.

        Raises:
            TypeError: If *name* is already registered with a different type.
        """
        return self._register(Histogram, name, description, labels or [], buckets=buckets)

    # --- internal ---------------------------------------------------------

    def _register(
        self,
        cls: type,
        name: str,
        description: str,
        label_names: list[str],
        **kwargs: Any,
    ) -> Any:
        with self._lock:
            if name in self._metrics:
                existing = self._metrics[name]
                if not isinstance(existing, cls):
                    msg = (
                        f"Metric {name!r} already registered as "
                        f"{type(existing).__name__}, cannot register as {cls.__name__}"
                    )
                    raise TypeError(msg)
                return existing
            metric = cls(name=name, description=description, label_names=label_names, **kwargs)
            self._metrics[name] = metric
            return metric

    # --- public API -------------------------------------------------------

    def export(self) -> str:
        """Export all metrics in Prometheus text format.

        Returns:
            A string conforming to the Prometheus exposition format.
        """
        with self._lock:
            metrics_snapshot = list(self._metrics.values())

        blocks: list[str] = []
        for metric in metrics_snapshot:
            lines = metric.export_lines()
            if lines:
                blocks.append("\n".join(lines))
        return "\n".join(blocks) + "\n" if blocks else ""

    def reset(self) -> None:
        """Reset every registered metric to its initial state."""
        with self._lock:
            for metric in self._metrics.values():
                metric.reset()

    def get(self, name: str) -> Counter | Gauge | Histogram | None:
        """Look up a metric by name. Returns ``None`` if not found."""
        with self._lock:
            return self._metrics.get(name)


# ---------------------------------------------------------------------------
# PianoMetrics -- pre-defined PIANO metrics
# ---------------------------------------------------------------------------


class PianoMetrics:
    """One-stop shop for all standard PIANO metrics.

    Instantiate once (typically at application startup) and pass the
    instance to components that need to record telemetry.

    Example::

        metrics = PianoMetrics()
        metrics.agent_tick_total.inc(labels={"agent_id": "agent-001"})
        print(metrics.registry.export())
    """

    def __init__(self, registry: MetricsRegistry | None = None) -> None:
        self.registry = registry or MetricsRegistry()

        # --- Counters ---

        self.agent_tick_total: Counter = self.registry.counter(
            "piano_agent_tick_total",
            "Total number of agent ticks executed",
            labels=["agent_id"],
        )

        self.llm_requests_total: Counter = self.registry.counter(
            "piano_llm_requests_total",
            "Total number of LLM requests",
            labels=["provider", "tier", "status"],
        )

        self.llm_cost_usd_total: Counter = self.registry.counter(
            "piano_llm_cost_usd_total",
            "Total LLM cost in USD",
            labels=["provider", "tier"],
        )

        self.bridge_commands_total: Counter = self.registry.counter(
            "piano_bridge_commands_total",
            "Total number of bridge commands sent",
            labels=["action", "status"],
        )

        # --- Gauges ---

        self.agent_count: Gauge = self.registry.gauge(
            "piano_agent_count",
            "Number of currently active agents",
        )

        self.worker_count: Gauge = self.registry.gauge(
            "piano_worker_count",
            "Number of currently active workers",
        )

        self.memory_stm_entries: Gauge = self.registry.gauge(
            "piano_memory_stm_entries",
            "Number of entries in short-term memory",
            labels=["agent_id"],
        )

        # --- Histograms ---

        self.llm_latency_seconds: Histogram = self.registry.histogram(
            "piano_llm_latency_seconds",
            "LLM request latency in seconds",
            labels=["provider", "tier"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, float("inf")),
        )

        self.checkpoint_duration_seconds: Histogram = self.registry.histogram(
            "piano_checkpoint_duration_seconds",
            "Checkpoint save duration in seconds",
            labels=["agent_id"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float("inf")),
        )

        logger.info(
            "piano_metrics_initialized",
            metric_count=len(self.registry._metrics),
        )
