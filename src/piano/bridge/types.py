"""Bridge-specific type definitions."""

from __future__ import annotations

from enum import StrEnum


class BridgeStatus(StrEnum):
    """Connection status of the bridge."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
