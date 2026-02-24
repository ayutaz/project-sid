"""Bridge-specific type definitions.

BridgeStatus vs BridgeHealthStatus (in health.py):
- BridgeStatus: Low-level socket connection state (connected/disconnected/reconnecting).
  Tracks whether the ZMQ sockets are physically connected.
- BridgeHealthStatus: Higher-level health assessment (connected/degraded/disconnected/stale).
  Combines ping latency, event freshness, and connection state to determine overall health.
"""

from __future__ import annotations

from enum import StrEnum


class BridgeStatus(StrEnum):
    """Connection status of the bridge."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
