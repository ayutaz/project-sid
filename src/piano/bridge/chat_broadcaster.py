"""Chat broadcaster - sends TalkingModule utterances to MC chat via bridge."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from piano.core.module import Module
from piano.core.types import CCDecision, ModuleResult, ModuleTier

if TYPE_CHECKING:
    from piano.bridge.client import BridgeClient
    from piano.core.sas import SharedAgentState

logger = logging.getLogger(__name__)


class ChatBroadcaster(Module):
    """Sends agent utterances to Minecraft chat via the bridge.

    Reacts to CC broadcasts (when speaking) and also polls SAS
    on each tick as a fallback to catch any unsent utterances.
    """

    def __init__(self, bridge: BridgeClient, sas: SharedAgentState | None = None) -> None:
        self._bridge = bridge
        self._sas = sas
        self._sent_count = 0
        self._send_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "chat_broadcaster"

    @property
    def tier(self) -> ModuleTier:
        return ModuleTier.FAST

    async def on_broadcast(self, decision: CCDecision) -> None:
        if not decision.speaking:
            return
        if self._sas is None:
            return
        async with self._send_lock:
            await self._send_and_clear(self._sas)

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        section = await sas.get_section("talking")
        utterance = section.get("latest_utterance")
        if utterance and utterance.get("content"):
            async with self._send_lock:
                await self._send_and_clear(sas)
        return ModuleResult(
            module_name=self.name,
            tier=self.tier,
            data={"sent_count": self._sent_count},
        )

    async def _send_and_clear(self, sas: SharedAgentState) -> None:
        section = await sas.get_section("talking")
        utterance = section.get("latest_utterance")
        if not utterance or not utterance.get("content"):
            return
        content = utterance["content"]
        try:
            await self._bridge.chat(content)
            self._sent_count += 1
        except Exception:
            logger.warning("Failed to send chat message", exc_info=True)
            return
        # Clear after successful send
        section["latest_utterance"] = None
        await sas.update_section("talking", section)
