"""Cognitive Controller (CC) - the central decision-making module.

Implements the Global Workspace Theory bottleneck: reads the SAS
snapshot, compresses it, calls the LLM, parses the response into a
CCDecision, and broadcasts it to output modules.

Reference: docs/implementation/03-cognitive-controller.md
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from piano.cc.broadcast import BroadcastManager, BroadcastResult
from piano.cc.compression import CompressionResult, TemplateCompressor
from piano.core.module import Module
from piano.core.types import CCDecision, LLMRequest, LLMResponse, ModuleResult, ModuleTier

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState
    from piano.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


# -- Valid actions for CC decision validation -----------------------------

VALID_ACTIONS: frozenset[str] = frozenset({
    "mine", "move", "chat", "look", "craft", "build", "explore",
    "get_position", "get_inventory", "idle", "wait",
    "attack", "flee", "defend",
    "place", "equip", "use", "drop", "eat",
    "trade", "gift", "follow", "vote",
    "think", "observe",
    "dig", "gather",
})


# -- CC prompt template ----------------------------------------------------

CC_SYSTEM_PROMPT = """\
You are the Cognitive Controller of an AI agent in a Minecraft world.
Your role is to integrate the current situation and decide what the
agent should do next.

Valid actions: mine, move, chat, look, craft, build, explore, \
get_position, get_inventory, idle, wait, attack, flee, defend, \
place, equip, use, drop, eat, trade, gift, follow, vote, think, observe.

Respond in JSON with the following keys:
- "action": short verb describing the next action (e.g. "mine", "chat", "explore", "idle")
- "action_params": object with optional parameters for the action
- "speaking": string with what the agent should say, or null if silent
- "reasoning": one-sentence rationale for your decision
- "salience_scores": object mapping information categories to 0.0-1.0 importance

Respond ONLY with the JSON object, no markdown fences.
"""


# -- Cognitive Controller --------------------------------------------------


class CognitiveController(Module):
    """GWT-based Cognitive Controller module.

    On each tick the CC:
    1. Takes a SAS snapshot
    2. Compresses it via TemplateCompressor
    3. Sends the compressed prompt to the LLM
    4. Parses the response into a CCDecision
    5. Broadcasts the decision via BroadcastManager
    """

    def __init__(
        self,
        llm: LLMProvider,
        broadcast_manager: BroadcastManager | None = None,
        compressor: TemplateCompressor | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> None:
        self._llm = llm
        self._broadcast = broadcast_manager or BroadcastManager()
        self._compressor = compressor or TemplateCompressor()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._last_decision: CCDecision | None = None
        self._cycle_count: int = 0

    # -- Module ABC implementation -----------------------------------------

    @property
    def name(self) -> str:
        """Module name."""
        return "cognitive_controller"

    @property
    def tier(self) -> ModuleTier:
        """Module tier."""
        return ModuleTier.MID

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        """Execute one CC cycle.

        Args:
            sas: The agent's shared state.

        Returns:
            ModuleResult with the decision data or error info.
        """
        self._cycle_count += 1

        # 1. Snapshot
        snapshot = await sas.snapshot()

        # 2. Compress
        compression = self._compressor.compress(snapshot)

        # 3. Call LLM
        try:
            llm_response = await self._call_llm(compression)
        except Exception as exc:
            logger.warning("LLM call failed (cycle %d): %s", self._cycle_count, exc)
            return await self._fallback_result(str(exc), sas)

        # 4. Parse response
        try:
            decision = self._parse_response(llm_response, compression)
        except Exception as exc:
            logger.warning("Parse failed (cycle %d): %s", self._cycle_count, exc)
            return await self._fallback_result(f"parse error: {exc}", sas)

        # 5. Store & broadcast
        self._last_decision = decision
        await sas.set_cc_decision(decision.model_dump(mode="json"))

        broadcast_result: BroadcastResult | None = None
        if self._broadcast.listener_names:
            broadcast_result = await self._broadcast.broadcast(decision)

        return ModuleResult(
            module_name=self.name,
            tier=self.tier,
            data={
                "cycle_id": str(decision.cycle_id),
                "action": decision.action,
                "speaking": decision.speaking,
                "token_estimate": compression.token_estimate,
                "retention_score": compression.retention_score,
                "broadcast_success": (
                    broadcast_result.success_count if broadcast_result else 0
                ),
                "broadcast_failures": (
                    broadcast_result.failure_count if broadcast_result else 0
                ),
            },
        )

    # -- Internals ---------------------------------------------------------

    async def _call_llm(self, compression: CompressionResult) -> LLMResponse:
        """Build LLMRequest from the compressed prompt and call the provider."""
        request = LLMRequest(
            prompt=compression.text,
            system_prompt=CC_SYSTEM_PROMPT,
            tier=ModuleTier.MID,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            json_mode=True,
        )
        return await asyncio.wait_for(self._llm.complete(request), timeout=30.0)

    def _parse_response(
        self,
        response: LLMResponse,
        compression: CompressionResult,
    ) -> CCDecision:
        """Parse LLM JSON response into a CCDecision with validation."""
        raw = response.content.strip()
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract a JSON object from the raw text
            data = self._extract_json_fields(raw)
            if not data:
                raise

        # Validate action field
        action = data.get("action", "idle")
        if action not in VALID_ACTIONS:
            logger.warning(
                "Invalid action '%s' not in VALID_ACTIONS (cycle %d), "
                "falling back to idle",
                action,
                self._cycle_count,
            )
            action = "idle"

        # Validate action_params is a dict
        action_params = data.get("action_params", {})
        if not isinstance(action_params, dict):
            logger.warning(
                "action_params is not a dict (type: %s) in cycle %d, using empty dict",
                type(action_params).__name__,
                self._cycle_count,
            )
            action_params = {}

        return CCDecision(
            cycle_id=uuid4(),
            timestamp=datetime.now(UTC),
            summary=compression.text[:200],
            action=action,
            action_params=action_params,
            speaking=data.get("speaking"),
            reasoning=data.get("reasoning", ""),
            salience_scores=data.get("salience_scores", {}),
            raw_llm_response=raw,
        )

    @staticmethod
    def _extract_json_fields(raw: str) -> dict[str, Any]:
        """Try to extract a JSON object from malformed LLM output.

        Looks for a JSON substring enclosed in braces within the raw text.
        Returns an empty dict if extraction fails.
        """
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {}

    async def _fallback_result(self, error_msg: str, sas: SharedAgentState) -> ModuleResult:
        """Return a fallback result reusing the previous decision and broadcast it."""
        if self._last_decision is not None:
            # Broadcast the fallback decision so output modules still receive it
            if self._broadcast.listener_names:
                try:
                    await self._broadcast.broadcast(self._last_decision)
                except Exception as exc:
                    logger.warning("Fallback broadcast failed: %s", exc)

            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                data={
                    "cycle_id": str(self._last_decision.cycle_id),
                    "action": self._last_decision.action,
                    "speaking": self._last_decision.speaking,
                    "fallback": True,
                    "error": error_msg,
                },
            )
        return ModuleResult(
            module_name=self.name,
            tier=self.tier,
            error=error_msg,
        )

    # -- Public helpers ----------------------------------------------------

    @property
    def last_decision(self) -> CCDecision | None:
        """Return the most recent CCDecision, or ``None``."""
        return self._last_decision

    @property
    def cycle_count(self) -> int:
        """Number of CC cycles executed so far."""
        return self._cycle_count

    @property
    def broadcast_manager(self) -> BroadcastManager:
        """Access the broadcast manager (e.g. to register listeners)."""
        return self._broadcast
