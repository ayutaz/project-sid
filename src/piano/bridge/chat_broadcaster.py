"""Chat broadcaster - sends TalkingModule utterances to MC chat via bridge."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from piano.core.module import Module
from piano.core.types import CCDecision, LLMRequest, ModuleResult, ModuleTier

if TYPE_CHECKING:
    from piano.bridge.client import BridgeClient
    from piano.core.sas import SharedAgentState
    from piano.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class ChatBroadcaster(Module):
    """Sends agent utterances to Minecraft chat via the bridge.

    Reacts to CC broadcasts (when speaking) and also polls SAS
    on each tick as a fallback to catch any unsent utterances.
    """

    def __init__(
        self,
        bridge: BridgeClient,
        sas: SharedAgentState | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self._bridge = bridge
        self._sas = sas
        self._llm = llm
        self._sent_count = 0
        self._send_lock = asyncio.Lock()
        self._cached_sas: SharedAgentState | None = None

    @property
    def name(self) -> str:
        return "chat_broadcaster"

    @property
    def tier(self) -> ModuleTier:
        return ModuleTier.FAST

    async def on_broadcast(self, decision: CCDecision) -> None:
        if not decision.speaking:
            return
        # Use cached SAS from most recent tick(), or constructor-provided SAS
        sas = self._cached_sas or self._sas
        if sas is None:
            return
        async with self._send_lock:
            await self._send_and_clear(sas)

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        # Cache SAS reference for use in on_broadcast()
        self._cached_sas = sas
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
        translated = await self._translate_to_japanese(content)
        try:
            await self._bridge.chat(translated)
            self._sent_count += 1
        except Exception:
            logger.warning("Failed to send chat message", exc_info=True)
            return
        # Clear after successful send
        section["latest_utterance"] = None
        await sas.update_section("talking", section)

    async def _translate_to_japanese(self, text: str) -> str:
        """Translate chat message to Japanese via LLM."""
        if not self._llm or not text.strip():
            return text
        try:
            request = LLMRequest(
                prompt=f"Translate to natural Japanese:\n{text}",
                system_prompt=(
                    "You are a translator. Translate the given Minecraft chat message "
                    "into natural Japanese. Output ONLY the translated text, nothing else."
                ),
                tier=ModuleTier.FAST,
                temperature=0.3,
                max_tokens=200,
            )
            response = await self._llm.complete(request)
            result = response.content.strip()
            return result if result else text
        except Exception:
            logger.warning("Translation failed, using original message")
            return text
