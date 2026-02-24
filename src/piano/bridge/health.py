"""Bridge health monitor - tracks connection state and latency."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from piano.bridge.client import BridgeClient

logger = logging.getLogger(__name__)

_STALE_THRESHOLD_S = 10.0  # No events for 10s = stale
_DEGRADED_LATENCY_MS = 500.0  # Ping > 500ms = degraded


class BridgeHealthStatus(StrEnum):
    """Bridge connection health status."""

    CONNECTED = "connected"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"
    STALE = "stale"


class BridgeHealthResult:
    """Result of a health check for a single bridge."""

    __slots__ = ("agent_id", "last_event_age_s", "latency_ms", "status")

    def __init__(
        self,
        agent_id: str,
        status: BridgeHealthStatus,
        latency_ms: float = 0.0,
        last_event_age_s: float = 0.0,
    ) -> None:
        self.agent_id = agent_id
        self.status = status
        self.latency_ms = latency_ms
        self.last_event_age_s = last_event_age_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": str(self.status),
            "latency_ms": self.latency_ms,
            "last_event_age_s": self.last_event_age_s,
        }


class BridgeHealthMonitor:
    """Monitors health of all bridge connections.

    Tracks:
    - Ping latency (connected/degraded/disconnected)
    - Event freshness (stale if no events for 10s)
    """

    def __init__(self, stale_threshold_s: float = _STALE_THRESHOLD_S) -> None:
        self._stale_threshold = stale_threshold_s
        self._last_event_times: dict[str, float] = {}

    def record_event(self, agent_id: str) -> None:
        """Record that an event was received from a bridge."""
        self._last_event_times[agent_id] = time.monotonic()

    async def check_one(self, agent_id: str, bridge: BridgeClient) -> BridgeHealthResult:
        """Check health of a single bridge."""
        now = time.monotonic()

        # Check event freshness
        last_event = self._last_event_times.get(agent_id)
        last_event_age = (now - last_event) if last_event else float("inf")

        # Ping the bridge
        start = time.monotonic()
        try:
            alive = await bridge.ping()
        except Exception:
            alive = False
        elapsed_ms = (time.monotonic() - start) * 1000

        if not alive:
            status = BridgeHealthStatus.DISCONNECTED
        elif last_event is not None and last_event_age > self._stale_threshold:
            status = BridgeHealthStatus.STALE
        elif elapsed_ms > _DEGRADED_LATENCY_MS:
            status = BridgeHealthStatus.DEGRADED
        else:
            status = BridgeHealthStatus.CONNECTED

        return BridgeHealthResult(
            agent_id=agent_id,
            status=status,
            latency_ms=elapsed_ms,
            last_event_age_s=last_event_age if last_event else -1.0,
        )

    async def check_all(self, bridges: dict[str, BridgeClient]) -> dict[str, BridgeHealthResult]:
        """Check health of all bridges in parallel."""
        checks = await asyncio.gather(*[self.check_one(aid, br) for aid, br in bridges.items()])
        return {result.agent_id: result for result in checks}

    def summary(self, results: dict[str, BridgeHealthResult]) -> dict[str, int]:
        """Summarize health check results by status."""
        counts: dict[str, int] = {s.value: 0 for s in BridgeHealthStatus}
        for r in results.values():
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts
