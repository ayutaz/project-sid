"""Bridge perception module - converts Mineflayer events to SAS PerceptData."""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any

from piano.core.module import Module
from piano.core.types import BridgeEvent, ModuleResult, ModuleTier, PerceptData

if TYPE_CHECKING:
    from piano.bridge.client import BridgeClient
    from piano.core.sas import SharedAgentState

logger = logging.getLogger(__name__)

_MAX_BUFFER = 100
_MAX_CHAT_MESSAGES = 20


class BridgePerceptionModule(Module):
    """Converts Bridge PUB/SUB events into SAS PerceptData.

    FAST tier module that:
    1. Receives events from bridge via callback
    2. Buffers them in a deque
    3. Drains buffer on each tick, updating SAS percepts
    """

    def __init__(self, bridge: BridgeClient) -> None:
        self._bridge = bridge
        self._buffer: deque[BridgeEvent] = deque(maxlen=_MAX_BUFFER)
        self._listener_started = False

    @property
    def name(self) -> str:
        return "bridge_perception"

    @property
    def tier(self) -> ModuleTier:
        return ModuleTier.FAST

    async def initialize(self) -> None:
        """Start the bridge event listener."""
        if not self._listener_started:
            await self._bridge.start_event_listener(self._on_event)
            self._listener_started = True

    async def _on_event(self, event: BridgeEvent) -> None:
        """Callback for bridge events - adds to buffer."""
        self._buffer.append(event)

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        """Drain buffer and update SAS percepts."""
        events_processed = 0
        if not self._buffer:
            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                data={"events_processed": 0},
            )

        percepts = await sas.get_percepts()

        while self._buffer:
            event = self._buffer.popleft()
            events_processed += 1
            try:
                self._process_event(event, percepts)
            except Exception:
                logger.exception("Error processing event type=%s", event.event_type)

        await sas.update_percepts(percepts)
        return ModuleResult(
            module_name=self.name,
            tier=self.tier,
            data={"events_processed": events_processed},
        )

    def _process_event(self, event: BridgeEvent, percepts: PerceptData) -> None:
        """Process a single event into percepts."""
        if event.event_type == "perception":
            self._handle_perception(event.data, percepts)
        elif event.event_type == "chat":
            self._handle_chat(event.data, percepts)
        elif event.event_type == "death":
            self._handle_death(percepts)

    def _handle_perception(self, data: dict[str, Any], percepts: PerceptData) -> None:
        """Update percepts from a perception event."""
        if "position" in data:
            pos = data["position"]
            if isinstance(pos, dict) and all(k in pos for k in ("x", "y", "z")):
                percepts.position = pos
            else:
                logger.warning("Invalid position data (missing x/y/z keys): %s", pos)
        if "health" in data:
            percepts.health = float(data["health"])
        if "food" in data:
            percepts.hunger = float(data["food"])
        if "nearby_players" in data:
            percepts.nearby_players = [
                p["name"] if isinstance(p, dict) else str(p) for p in data["nearby_players"]
            ]
        if "time_of_day" in data:
            percepts.time_of_day = int(data["time_of_day"])
        if "is_raining" in data:
            percepts.weather = "rain" if data["is_raining"] else "clear"
        if "nearby_blocks" in data:
            percepts.nearby_blocks = data["nearby_blocks"]
        if "inventory" in data:
            inv = data["inventory"]
            if isinstance(inv, dict):
                percepts.inventory = inv
            elif isinstance(inv, list):
                percepts.inventory = {
                    item["name"]: item.get("count", 1)
                    for item in inv
                    if isinstance(item, dict) and "name" in item
                }

    def _handle_chat(self, data: dict[str, Any], percepts: PerceptData) -> None:
        """Add chat message to percepts."""
        percepts.chat_messages.append(
            {
                "username": data.get("username", "unknown"),
                "message": data.get("message", ""),
            }
        )
        if len(percepts.chat_messages) > _MAX_CHAT_MESSAGES:
            percepts.chat_messages = percepts.chat_messages[-_MAX_CHAT_MESSAGES:]

    def _handle_death(self, percepts: PerceptData) -> None:
        """Handle death event."""
        percepts.health = 0.0

    async def shutdown(self) -> None:
        """Stop event listener and clear buffer."""
        self._listener_started = False
        await self._bridge.stop_event_listener()
        self._buffer.clear()
