"""Tests for bridge health monitor."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from piano.bridge.health import (
    BridgeHealthMonitor,
    BridgeHealthResult,
    BridgeHealthStatus,
)

if TYPE_CHECKING:
    import pytest

_DEGRADED_LATENCY_THRESHOLD = 500.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bridge(ping_return: bool = True) -> MagicMock:
    """Create a mock BridgeClient with configurable ping."""
    bridge = MagicMock()
    bridge.ping = AsyncMock(return_value=ping_return)
    return bridge


def _make_time_counter(start: float = 100.0, step: float = 0.001):
    """Create an incrementing monotonic counter.

    Each call returns start + call_index * step. This avoids brittle
    fixed-list approaches and is straightforward to reason about.
    """
    call_count = 0

    def _monotonic() -> float:
        nonlocal call_count
        t = start + call_count * step
        call_count += 1
        return t

    return _monotonic


# ---------------------------------------------------------------------------
# BridgeHealthStatus
# ---------------------------------------------------------------------------


def test_health_status_enum_values() -> None:
    assert BridgeHealthStatus.CONNECTED == "connected"
    assert BridgeHealthStatus.DEGRADED == "degraded"
    assert BridgeHealthStatus.DISCONNECTED == "disconnected"
    assert BridgeHealthStatus.STALE == "stale"
    assert len(BridgeHealthStatus) == 4


# ---------------------------------------------------------------------------
# BridgeHealthResult
# ---------------------------------------------------------------------------


def test_health_result_to_dict() -> None:
    result = BridgeHealthResult(
        agent_id="agent-1",
        status=BridgeHealthStatus.CONNECTED,
        latency_ms=42.5,
        last_event_age_s=1.2,
    )
    d = result.to_dict()
    assert d == {
        "agent_id": "agent-1",
        "status": "connected",
        "latency_ms": 42.5,
        "last_event_age_s": 1.2,
    }


# ---------------------------------------------------------------------------
# BridgeHealthMonitor.check_one
# ---------------------------------------------------------------------------


async def test_check_one_connected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ping succeeds with low latency and recent event -> CONNECTED."""
    # start=100.0, step=0.001 -> each call increments by 1ms
    monkeypatch.setattr(
        "piano.bridge.health.time.monotonic",
        _make_time_counter(start=100.0, step=0.001),
    )

    monitor = BridgeHealthMonitor()
    # Simulate a recent event at t=99.5 (0.5s ago relative to start=100.0)
    monitor._last_event_times["a1"] = 99.5

    bridge = _make_bridge(ping_return=True)
    result = await monitor.check_one("a1", bridge)

    assert result.status == BridgeHealthStatus.CONNECTED
    assert result.agent_id == "a1"
    assert result.latency_ms < _DEGRADED_LATENCY_THRESHOLD


async def test_check_one_disconnected() -> None:
    """Ping fails -> DISCONNECTED."""
    bridge = _make_bridge(ping_return=False)

    monitor = BridgeHealthMonitor()
    result = await monitor.check_one("a1", bridge)

    assert result.status == BridgeHealthStatus.DISCONNECTED


async def test_check_one_disconnected_on_exception() -> None:
    """Ping raises exception -> DISCONNECTED."""
    bridge = MagicMock()
    bridge.ping = AsyncMock(side_effect=ConnectionError("lost"))

    monitor = BridgeHealthMonitor()
    result = await monitor.check_one("a1", bridge)

    assert result.status == BridgeHealthStatus.DISCONNECTED


async def test_check_one_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ping ok but no events for >10s -> STALE."""
    # start=120.0 so last_event at 100.0 is 20s ago
    monkeypatch.setattr(
        "piano.bridge.health.time.monotonic",
        _make_time_counter(start=120.0, step=0.001),
    )

    monitor = BridgeHealthMonitor()
    monitor._last_event_times["a1"] = 100.0  # 20s ago

    bridge = _make_bridge(ping_return=True)
    result = await monitor.check_one("a1", bridge)

    assert result.status == BridgeHealthStatus.STALE


async def test_check_one_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ping succeeds but latency >500ms -> DEGRADED."""

    async def _slow_ping() -> bool:
        await asyncio.sleep(0)  # yield control
        return True

    bridge = MagicMock()
    bridge.ping = _slow_ping

    # Simulate 600ms latency: start=100.0, step=0.3
    # call 0 (now): 100.0, call 1 (start): 100.3, call 2 (after-ping): 100.6
    # elapsed_ms = (100.6 - 100.3) * 1000 = 300ms ... need bigger step
    # Actually check_one calls monotonic 3 times: now, start, after-ping
    # elapsed = (after-ping - start) * 1000
    # With step=0.6: call1=100.6, call2=101.2 -> elapsed = 600ms
    monkeypatch.setattr(
        "piano.bridge.health.time.monotonic",
        _make_time_counter(start=100.0, step=0.6),
    )

    monitor = BridgeHealthMonitor()
    monitor._last_event_times["a1"] = 99.9  # recent event

    result = await monitor.check_one("a1", bridge)

    assert result.status == BridgeHealthStatus.DEGRADED
    assert result.latency_ms > _DEGRADED_LATENCY_THRESHOLD


async def test_check_one_degraded_when_no_events_received(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ping ok but no events ever received -> DEGRADED (PUB/SUB may not be working)."""
    monkeypatch.setattr(
        "piano.bridge.health.time.monotonic",
        _make_time_counter(start=100.0, step=0.001),
    )

    monitor = BridgeHealthMonitor()
    # No events recorded for this agent at all
    bridge = _make_bridge(ping_return=True)
    result = await monitor.check_one("a1", bridge)

    assert result.status == BridgeHealthStatus.DEGRADED
    assert result.last_event_age_s == -1.0


async def test_record_event_resets_staleness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording a fresh event via record_event() prevents STALE status."""
    monkeypatch.setattr(
        "piano.bridge.health.time.monotonic",
        _make_time_counter(start=120.0, step=0.001),
    )

    monitor = BridgeHealthMonitor()
    # Old event would be stale
    monitor._last_event_times["a1"] = 100.0

    # Use the actual record_event method (not direct assignment)
    # record_event calls time.monotonic(), which returns 120.001 (2nd call)
    # This makes the event very recent
    monitor.record_event("a1")

    bridge = _make_bridge(ping_return=True)
    result = await monitor.check_one("a1", bridge)

    assert result.status == BridgeHealthStatus.CONNECTED


# ---------------------------------------------------------------------------
# BridgeHealthMonitor.check_all
# ---------------------------------------------------------------------------


async def test_check_all_returns_all_bridges() -> None:
    """check_all returns results for every bridge provided."""
    monitor = BridgeHealthMonitor()

    bridges = {
        "a1": _make_bridge(ping_return=True),
        "a2": _make_bridge(ping_return=False),
        "a3": _make_bridge(ping_return=True),
    }

    results = await monitor.check_all(bridges)

    assert set(results.keys()) == {"a1", "a2", "a3"}
    assert results["a2"].status == BridgeHealthStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# BridgeHealthMonitor.summary
# ---------------------------------------------------------------------------


def test_summary_counts() -> None:
    monitor = BridgeHealthMonitor()
    results = {
        "a1": BridgeHealthResult("a1", BridgeHealthStatus.CONNECTED),
        "a2": BridgeHealthResult("a2", BridgeHealthStatus.CONNECTED),
        "a3": BridgeHealthResult("a3", BridgeHealthStatus.DISCONNECTED),
        "a4": BridgeHealthResult("a4", BridgeHealthStatus.STALE),
    }

    summary = monitor.summary(results)

    assert summary == {
        "connected": 2,
        "degraded": 0,
        "disconnected": 1,
        "stale": 1,
    }
