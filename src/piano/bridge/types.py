"""Bridge-specific type definitions."""

from __future__ import annotations

from enum import Enum


class BridgeStatus(str, Enum):
    """Connection status of the bridge."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
