"""Performance benchmark framework for PIANO architecture.

Provides automated regression testing for TPS (ticks per second),
latency percentiles, LLM costs, and bridge command performance.

Usage:
    >>> bench = PerformanceBenchmark()
    >>> bench.start()
    >>> bench.record_tick("agent-001", duration_ms=15.2)
    >>> bench.record_llm_call("openai", latency_ms=320.0, cost_usd=0.003, tokens=500)
    >>> bench.record_bridge_command("move", latency_ms=45.0)
    >>> result = bench.stop()
    >>> print(f"TPS: {result.tps:.1f}, p99 latency: {result.p99_tick_latency_ms:.1f}ms")

    >>> detector = RegressionDetector()
    >>> detector.set_baseline(result)
    >>> report = detector.check_regression(new_result)
    >>> assert report.passed
"""

from __future__ import annotations

__all__ = [
    "BenchmarkResult",
    "LLMCallRecord",
    "PerformanceBenchmark",
    "PerformanceConfig",
    "RegressionDetector",
    "RegressionItem",
    "RegressionReport",
    "TickRecord",
    "compute_percentile",
]

import math
import time

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


class TickRecord(BaseModel):
    """A single recorded tick execution."""

    agent_id: str = Field(description="Agent that executed this tick")
    duration_ms: float = Field(ge=0.0, description="Tick duration in milliseconds")
    timestamp: float = Field(description="Unix timestamp when the tick was recorded")


class LLMCallRecord(BaseModel):
    """A single recorded LLM call."""

    provider: str = Field(description="LLM provider name (e.g., openai, anthropic)")
    latency_ms: float = Field(ge=0.0, description="LLM call latency in milliseconds")
    cost_usd: float = Field(ge=0.0, description="Cost in USD")
    tokens: int = Field(ge=0, description="Total tokens used (prompt + completion)")
    timestamp: float = Field(description="Unix timestamp when the call was recorded")


class BridgeCommandRecord(BaseModel):
    """A single recorded bridge command."""

    command: str = Field(description="Bridge command name (e.g., move, mine, chat)")
    latency_ms: float = Field(ge=0.0, description="Command latency in milliseconds")
    timestamp: float = Field(description="Unix timestamp when the command was recorded")


# ---------------------------------------------------------------------------
# Percentile computation
# ---------------------------------------------------------------------------


def compute_percentile(sorted_values: list[float], percentile: float) -> float:
    """Compute a percentile from a sorted list of values using linear interpolation.

    Uses the C=1 variant (inclusive) of the percentile formula, which
    matches numpy's ``percentile(..., interpolation='linear')`` default.

    Args:
        sorted_values: **Pre-sorted** list of float values (ascending).
        percentile: Percentile to compute, in range [0, 100].

    Returns:
        Interpolated percentile value.  Returns 0.0 for an empty list.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]

    # Clamp percentile to [0, 100]
    percentile = max(0.0, min(100.0, percentile))

    # Rank (0-based fractional index)
    rank = (percentile / 100.0) * (n - 1)
    lower = math.floor(rank)
    upper = min(lower + 1, n - 1)
    fraction = rank - lower

    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------


class BenchmarkResult(BaseModel):
    """Result produced by ``PerformanceBenchmark.stop()``."""

    duration_seconds: float = Field(ge=0.0, description="Total benchmark duration")
    tps: float = Field(ge=0.0, description="Ticks per second (all agents combined)")
    avg_tick_latency_ms: float = Field(ge=0.0, description="Mean tick latency in ms")
    p50_tick_latency_ms: float = Field(ge=0.0, description="Median tick latency")
    p95_tick_latency_ms: float = Field(ge=0.0, description="95th percentile tick latency")
    p99_tick_latency_ms: float = Field(ge=0.0, description="99th percentile tick latency")
    total_llm_calls: int = Field(ge=0, description="Total number of LLM calls")
    avg_llm_latency_ms: float = Field(ge=0.0, description="Mean LLM call latency")
    total_cost_usd: float = Field(ge=0.0, description="Total LLM cost in USD")
    total_bridge_commands: int = Field(ge=0, description="Total bridge commands issued")
    agent_count: int = Field(ge=0, description="Number of distinct agents observed")


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------


class PerformanceBenchmark:
    """Collects tick, LLM, and bridge metrics and produces a ``BenchmarkResult``.

    Thread-safety is **not** guaranteed; designed for single-threaded / async use.
    """

    def __init__(self) -> None:
        self._ticks: list[TickRecord] = []
        self._llm_calls: list[LLMCallRecord] = []
        self._bridge_commands: list[BridgeCommandRecord] = []
        self._start_time: float | None = None
        self._stop_time: float | None = None
        self._running: bool = False

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the benchmark timer.

        Raises:
            RuntimeError: If the benchmark is already running.
        """
        if self._running:
            msg = "Benchmark is already running"
            raise RuntimeError(msg)
        self._ticks.clear()
        self._llm_calls.clear()
        self._bridge_commands.clear()
        # time.monotonic() is used instead of time.time() to avoid
        # clock adjustments affecting duration measurements.
        self._start_time = time.monotonic()
        self._stop_time = None
        self._running = True
        logger.info("performance_benchmark_started")

    def stop(self) -> BenchmarkResult:
        """Stop the benchmark and compute the result.

        Returns:
            A ``BenchmarkResult`` summarising all recorded metrics.

        Raises:
            RuntimeError: If the benchmark has not been started.
        """
        if not self._running or self._start_time is None:
            msg = "Benchmark is not running"
            raise RuntimeError(msg)
        self._stop_time = time.monotonic()
        self._running = False
        result = self._compute_result()
        logger.info(
            "performance_benchmark_stopped",
            duration_s=result.duration_seconds,
            tps=result.tps,
            total_cost=result.total_cost_usd,
        )
        return result

    # -- recording -----------------------------------------------------------

    def record_tick(self, agent_id: str, duration_ms: float) -> None:
        """Record a single tick execution.

        Args:
            agent_id: The agent that executed this tick.
            duration_ms: Tick wall-clock duration in milliseconds.

        Raises:
            RuntimeError: If the benchmark is not running.
        """
        if not self._running:
            msg = "Benchmark is not running"
            raise RuntimeError(msg)
        self._ticks.append(
            TickRecord(agent_id=agent_id, duration_ms=duration_ms, timestamp=time.monotonic())
        )

    def record_llm_call(
        self, provider: str, latency_ms: float, cost_usd: float, tokens: int
    ) -> None:
        """Record a single LLM call.

        Args:
            provider: LLM provider identifier (e.g., ``"openai"``).
            latency_ms: Round-trip latency in milliseconds.
            cost_usd: Estimated cost in USD.
            tokens: Total token count (prompt + completion).

        Raises:
            RuntimeError: If the benchmark is not running.
        """
        if not self._running:
            msg = "Benchmark is not running"
            raise RuntimeError(msg)
        self._llm_calls.append(
            LLMCallRecord(
                provider=provider,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                tokens=tokens,
                timestamp=time.monotonic(),
            )
        )

    def record_bridge_command(self, command: str, latency_ms: float) -> None:
        """Record a single bridge command.

        Args:
            command: Bridge command name (e.g., ``"move"``).
            latency_ms: Command latency in milliseconds.

        Raises:
            RuntimeError: If the benchmark is not running.
        """
        if not self._running:
            msg = "Benchmark is not running"
            raise RuntimeError(msg)
        self._bridge_commands.append(
            BridgeCommandRecord(
                command=command, latency_ms=latency_ms, timestamp=time.monotonic()
            )
        )

    # -- internal ------------------------------------------------------------

    def _compute_result(self) -> BenchmarkResult:
        """Build the benchmark result from collected data."""
        if self._start_time is None:
            msg = "start_time is None: call start() before computing results"
            raise RuntimeError(msg)
        if self._stop_time is None:
            msg = "stop_time is None: call stop() before computing results"
            raise RuntimeError(msg)

        duration = self._stop_time - self._start_time

        # Tick metrics
        tick_durations = sorted(t.duration_ms for t in self._ticks)
        total_ticks = len(tick_durations)
        tps = total_ticks / duration if duration > 0 else 0.0
        avg_tick = (sum(tick_durations) / total_ticks) if total_ticks > 0 else 0.0
        p50 = compute_percentile(tick_durations, 50)
        p95 = compute_percentile(tick_durations, 95)
        p99 = compute_percentile(tick_durations, 99)

        # LLM metrics
        total_llm = len(self._llm_calls)
        avg_llm_latency = (
            sum(c.latency_ms for c in self._llm_calls) / total_llm if total_llm > 0 else 0.0
        )
        total_cost = sum(c.cost_usd for c in self._llm_calls)

        # Bridge metrics
        total_bridge = len(self._bridge_commands)

        # Agent count
        agents = {t.agent_id for t in self._ticks}

        return BenchmarkResult(
            duration_seconds=duration,
            tps=tps,
            avg_tick_latency_ms=avg_tick,
            p50_tick_latency_ms=p50,
            p95_tick_latency_ms=p95,
            p99_tick_latency_ms=p99,
            total_llm_calls=total_llm,
            avg_llm_latency_ms=avg_llm_latency,
            total_cost_usd=total_cost,
            total_bridge_commands=total_bridge,
            agent_count=len(agents),
        )


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


class PerformanceConfig(BaseModel):
    """Thresholds for regression detection.

    Each ``*_threshold_pct`` value represents the maximum allowed **degradation**
    percentage before a metric is flagged as a regression.

    Example:
        With ``tps_threshold_pct = 20.0``, a TPS drop of 25% triggers a regression.
    """

    tps_threshold_pct: float = Field(
        default=20.0,
        ge=0.0,
        description="Max allowed TPS decrease (%)",
    )
    latency_threshold_pct: float = Field(
        default=30.0,
        ge=0.0,
        description="Max allowed latency increase (%)",
    )
    cost_threshold_pct: float = Field(
        default=50.0,
        ge=0.0,
        description="Max allowed cost increase (%)",
    )


class RegressionItem(BaseModel):
    """A single regression finding."""

    metric_name: str = Field(description="Name of the regressed metric")
    baseline: float = Field(description="Baseline value")
    current: float = Field(description="Current value")
    change_pct: float = Field(description="Percentage change (positive = worse)")
    threshold_pct: float = Field(description="Threshold that was exceeded")


class RegressionReport(BaseModel):
    """Overall regression check result."""

    passed: bool = Field(description="True if no regressions detected")
    regressions: list[RegressionItem] = Field(
        default_factory=list, description="List of detected regressions"
    )


class RegressionDetector:
    """Compares a current ``BenchmarkResult`` against a baseline to find regressions.

    Usage:
        >>> detector = RegressionDetector()
        >>> detector.set_baseline(baseline_result)
        >>> report = detector.check_regression(current_result)
        >>> if not report.passed:
        ...     for item in report.regressions:
        ...         print(f"{item.metric_name}: {item.change_pct:+.1f}%")
    """

    def __init__(self, config: PerformanceConfig | None = None) -> None:
        self._config = config or PerformanceConfig()
        self._baseline: BenchmarkResult | None = None

    @property
    def config(self) -> PerformanceConfig:
        """Return the current performance configuration."""
        return self._config

    @property
    def baseline(self) -> BenchmarkResult | None:
        """Return the current baseline, or ``None`` if not set."""
        return self._baseline

    def set_baseline(self, result: BenchmarkResult) -> None:
        """Set the baseline benchmark result for future comparisons.

        Args:
            result: The benchmark result to use as baseline.
        """
        self._baseline = result
        logger.info("regression_baseline_set", tps=result.tps, cost=result.total_cost_usd)

    def check_regression(self, current: BenchmarkResult) -> RegressionReport:
        """Compare *current* against the stored baseline.

        Args:
            current: The benchmark result to check.

        Returns:
            A ``RegressionReport`` indicating pass/fail and any regressions.

        Raises:
            RuntimeError: If no baseline has been set.
        """
        if self._baseline is None:
            msg = "No baseline set. Call set_baseline() first."
            raise RuntimeError(msg)

        regressions: list[RegressionItem] = []

        # TPS regression: lower is worse
        self._check_decrease(
            metric_name="tps",
            baseline_val=self._baseline.tps,
            current_val=current.tps,
            threshold_pct=self._config.tps_threshold_pct,
            regressions=regressions,
        )

        # Latency regressions: higher is worse
        bl = self._baseline
        latency_checks = [
            ("avg_tick_latency_ms", bl.avg_tick_latency_ms, current.avg_tick_latency_ms),
            ("p50_tick_latency_ms", bl.p50_tick_latency_ms, current.p50_tick_latency_ms),
            ("p95_tick_latency_ms", bl.p95_tick_latency_ms, current.p95_tick_latency_ms),
            ("p99_tick_latency_ms", bl.p99_tick_latency_ms, current.p99_tick_latency_ms),
            ("avg_llm_latency_ms", bl.avg_llm_latency_ms, current.avg_llm_latency_ms),
        ]
        for metric_name, baseline_val, current_val in latency_checks:
            self._check_increase(
                metric_name=metric_name,
                baseline_val=baseline_val,
                current_val=current_val,
                threshold_pct=self._config.latency_threshold_pct,
                regressions=regressions,
            )

        # Cost regression: higher is worse
        self._check_increase(
            metric_name="total_cost_usd",
            baseline_val=self._baseline.total_cost_usd,
            current_val=current.total_cost_usd,
            threshold_pct=self._config.cost_threshold_pct,
            regressions=regressions,
        )

        passed = len(regressions) == 0
        report = RegressionReport(passed=passed, regressions=regressions)

        if not passed:
            logger.warning(
                "regression_detected",
                num_regressions=len(regressions),
                metrics=[r.metric_name for r in regressions],
            )
        else:
            logger.info("regression_check_passed")

        return report

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def is_within_threshold(
        metric: float, baseline: float, threshold_pct: float
    ) -> bool:
        """Check whether *metric* is within *threshold_pct* of *baseline*.

        This is a **symmetric** check: the absolute percentage change from
        baseline must not exceed *threshold_pct*.

        Args:
            metric: Current metric value.
            baseline: Baseline metric value.
            threshold_pct: Maximum allowed deviation in percent.

        Returns:
            ``True`` if within threshold, ``False`` otherwise.
        """
        if baseline == 0.0:
            return metric == 0.0
        change_pct = abs(metric - baseline) / abs(baseline) * 100.0
        return change_pct <= threshold_pct

    # -- private helpers -----------------------------------------------------

    @staticmethod
    def _check_decrease(
        metric_name: str,
        baseline_val: float,
        current_val: float,
        threshold_pct: float,
        regressions: list[RegressionItem],
    ) -> None:
        """Flag a regression when *current_val* is lower than *baseline_val* beyond threshold."""
        if baseline_val == 0.0:
            return  # Cannot compute percentage change from zero baseline
        change_pct = (baseline_val - current_val) / baseline_val * 100.0
        if change_pct > threshold_pct:
            regressions.append(
                RegressionItem(
                    metric_name=metric_name,
                    baseline=baseline_val,
                    current=current_val,
                    change_pct=change_pct,
                    threshold_pct=threshold_pct,
                )
            )

    @staticmethod
    def _check_increase(
        metric_name: str,
        baseline_val: float,
        current_val: float,
        threshold_pct: float,
        regressions: list[RegressionItem],
    ) -> None:
        """Flag a regression when *current_val* is higher than *baseline_val* beyond threshold."""
        if baseline_val == 0.0:
            # If baseline was zero and current is non-zero, always flag
            if current_val > 0.0:
                regressions.append(
                    RegressionItem(
                        metric_name=metric_name,
                        baseline=baseline_val,
                        current=current_val,
                        change_pct=99999.0,
                        threshold_pct=threshold_pct,
                    )
                )
            return
        change_pct = (current_val - baseline_val) / baseline_val * 100.0
        if change_pct > threshold_pct:
            regressions.append(
                RegressionItem(
                    metric_name=metric_name,
                    baseline=baseline_val,
                    current=current_val,
                    change_pct=change_pct,
                    threshold_pct=threshold_pct,
                )
            )
