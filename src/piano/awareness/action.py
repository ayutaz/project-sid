"""Rule-based action awareness module (Phase 0).

Compares expected action outcomes against actual percepts to detect
discrepancies. This is the most critical module in the PIANO architecture --
removing it causes significant degradation in item acquisition performance.

Phase 0 uses deterministic rules instead of a neural network.
The NN-based implementation will replace this in Phase 1 once sufficient
training data has been collected.

Reference: docs/implementation/07-goal-planning.md Section 3
"""

from __future__ import annotations

import json
import math
from typing import Any

from piano.core.module import Module
from piano.core.sas import SharedAgentState
from piano.core.types import ActionHistoryEntry, ModuleResult, ModuleTier, PerceptData


class ActionAwareness(Module):
    """Rule-based action awareness for Phase 0 MVP.

    Reads the latest action from SAS action_history, compares its expected
    result against actual percepts, and writes back the comparison result.
    Tracks consecutive failures and raises alerts when the threshold is reached.
    """

    CONSECUTIVE_FAILURE_THRESHOLD = 3

    @property
    def name(self) -> str:
        """Unique module name."""
        return "action_awareness"

    @property
    def tier(self) -> ModuleTier:
        """Execution tier -- FAST (non-LLM, <100ms)."""
        return ModuleTier.FAST

    def __init__(self) -> None:
        self._consecutive_failures: int = 0

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        """Execute one tick of action awareness.

        1. Read action_history and percepts from SAS.
        2. Compare the latest action's expected result against actual percepts.
        3. Build discrepancy list and update action_history entry.
        4. Return ModuleResult with discrepancy data.
        """
        history = await sas.get_action_history(limit=1)
        if not history:
            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                data={
                    "discrepancies": [],
                    "last_action_success": True,
                    "consecutive_failures": self._consecutive_failures,
                },
            )

        latest = history[0]
        percepts = await sas.get_percepts()

        success = self._evaluate(latest, percepts)
        discrepancies = self._build_discrepancies(latest, percepts, success)

        # Update consecutive failure tracking
        if success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

        # Write back the evaluation result to the action history entry
        updated = ActionHistoryEntry(
            timestamp=latest.timestamp,
            action=latest.action,
            expected_result=latest.expected_result,
            actual_result=latest.actual_result,
            success=success,
        )
        await sas.add_action(updated)

        data: dict[str, Any] = {
            "discrepancies": discrepancies,
            "last_action_success": success,
            "consecutive_failures": self._consecutive_failures,
        }

        if self._consecutive_failures >= self.CONSECUTIVE_FAILURE_THRESHOLD:
            data["alert"] = "consecutive_failures"
            data["count"] = self._consecutive_failures

        return ModuleResult(
            module_name=self.name,
            tier=self.tier,
            data=data,
        )

    # ------------------------------------------------------------------
    # Evaluation dispatch
    # ------------------------------------------------------------------

    def _evaluate(self, entry: ActionHistoryEntry, percepts: PerceptData) -> bool:
        """Dispatch to the appropriate check method based on action type."""
        action = entry.action.lower()
        expected = _parse_json_safe(entry.expected_result)

        if action == "move":
            return self.check_move(expected, percepts.position)
        if action == "mine":
            return self.check_mine(expected, percepts)
        if action == "craft":
            return self.check_craft(expected, percepts.inventory)
        if action == "chat":
            return self.check_chat(expected, percepts.chat_messages)

        # Unknown action type -- assume success (no rule to verify)
        return True

    # ------------------------------------------------------------------
    # Check methods (public for testability)
    # ------------------------------------------------------------------

    def check_move(
        self, expected: dict[str, Any], actual_position: dict[str, float]
    ) -> bool:
        """Check if the agent arrived within 5 blocks of the target position."""
        target_x = float(expected.get("x", 0))
        target_y = float(expected.get("y", 0))
        target_z = float(expected.get("z", 0))

        actual_x = float(actual_position.get("x", 0))
        actual_y = float(actual_position.get("y", 0))
        actual_z = float(actual_position.get("z", 0))

        distance = math.sqrt(
            (target_x - actual_x) ** 2
            + (target_y - actual_y) ** 2
            + (target_z - actual_z) ** 2
        )
        return distance <= 5.0

    def check_mine(self, expected: dict[str, Any], percepts: PerceptData) -> bool:
        """Check if the target block was removed from nearby_blocks."""
        target_pos = expected.get("position", {})
        target_x = target_pos.get("x")
        target_y = target_pos.get("y")
        target_z = target_pos.get("z")

        if target_x is None or target_y is None or target_z is None:
            return False

        # The block should no longer be present at the target position
        for block in percepts.nearby_blocks:
            bpos = block.get("position", {})
            if (
                bpos.get("x") == target_x
                and bpos.get("y") == target_y
                and bpos.get("z") == target_z
            ):
                # Block still exists at target position -- mining failed
                return False

        return True

    def check_craft(
        self, expected: dict[str, Any], inventory: dict[str, int]
    ) -> bool:
        """Check if the expected item was added to the inventory."""
        item_name = expected.get("item", "")
        expected_count = int(expected.get("count", 1))

        actual_count = inventory.get(item_name, 0)
        return actual_count >= expected_count

    def check_chat(
        self, expected: dict[str, Any], chat_messages: list[dict[str, Any]]
    ) -> bool:
        """Check if the expected message appears in the chat log."""
        expected_message = expected.get("message", "")
        if not expected_message:
            return False

        for msg in chat_messages:
            if expected_message in msg.get("text", ""):
                return True

        return False

    # ------------------------------------------------------------------
    # Discrepancy builder
    # ------------------------------------------------------------------

    def _build_discrepancies(
        self,
        entry: ActionHistoryEntry,
        percepts: PerceptData,
        success: bool,
    ) -> list[dict[str, Any]]:
        """Build a list of discrepancy dicts for the ModuleResult."""
        if success:
            return []

        expected = _parse_json_safe(entry.expected_result)
        return [
            {
                "action": entry.action,
                "type": f"{entry.action}_failure",
                "expected": expected,
                "actual": _get_actual_summary(entry.action.lower(), percepts),
            }
        ]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_json_safe(text: str) -> dict[str, Any]:
    """Parse a JSON string, returning an empty dict on failure."""
    if not text:
        return {}
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        return {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _get_actual_summary(action: str, percepts: PerceptData) -> dict[str, Any]:
    """Build a summary of the actual state relevant to the action."""
    if action == "move":
        return {"position": percepts.position}
    if action == "mine":
        return {"nearby_blocks_count": len(percepts.nearby_blocks)}
    if action == "craft":
        return {"inventory": percepts.inventory}
    if action == "chat":
        return {"chat_messages_count": len(percepts.chat_messages)}
    return {}
