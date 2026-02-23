"""Skill executor module for the PIANO architecture.

Receives CC broadcast decisions and dispatches the appropriate skill
from the registry. Handles timeouts, cancellation of in-flight skills,
and records action history to SAS.

Reference: docs/implementation/05-minecraft-platform.md Section 3
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from piano.core.module import Module
from piano.core.types import ActionHistoryEntry, CCDecision, ModuleResult, ModuleTier
from piano.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState
    from piano.skills.basic import BridgeClient

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 30.0


class SkillExecutor(Module):
    """Executes skills in response to CC broadcast decisions.

    This module:
    - Listens for CC broadcasts via on_broadcast()
    - Looks up the action in the SkillRegistry
    - Executes the skill with action_params via the BridgeClient
    - Records results to SAS action_history
    - Cancels in-flight skills when a new broadcast arrives
    - Enforces per-skill timeouts

    Attributes:
        registry: The skill registry to look up skills.
        bridge: The bridge client for Mineflayer communication.
        timeout_s: Default timeout in seconds for skill execution.
    """

    def __init__(
        self,
        registry: SkillRegistry,
        bridge: BridgeClient,
        sas: SharedAgentState,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._registry = registry
        self._bridge = bridge
        self._sas = sas
        self._timeout_s = timeout_s
        self._current_task: asyncio.Task[dict[str, Any]] | None = None
        self._last_decision: CCDecision | None = None

    @property
    def name(self) -> str:
        """Unique module name."""
        return "skill_executor"

    @property
    def tier(self) -> ModuleTier:
        """Execution tier - FAST since no LLM calls."""
        return ModuleTier.FAST

    @property
    def registry(self) -> SkillRegistry:
        """The skill registry."""
        return self._registry

    @property
    def bridge(self) -> BridgeClient:
        """The bridge client."""
        return self._bridge

    @property
    def timeout_s(self) -> float:
        """Default timeout for skill execution."""
        return self._timeout_s

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        """Report current execution status on each tick.

        The actual execution is triggered by on_broadcast, not tick.
        Tick only reports whether a skill is currently running.
        """
        is_running = self._current_task is not None and not self._current_task.done()
        action_name = ""
        if self._last_decision and is_running:
            action_name = self._last_decision.action

        return ModuleResult(
            module_name=self.name,
            tier=self.tier,
            data={
                "executing": is_running,
                "current_action": action_name,
            },
        )

    async def on_broadcast(self, decision: CCDecision) -> None:
        """Handle a CC broadcast decision by executing the requested skill.

        If a skill is currently running, it is cancelled before starting
        the new one.

        Args:
            decision: The CC decision containing action and action_params.
        """
        action = decision.action
        if not action or action == "idle":
            await self._cancel_current()
            self._last_decision = decision
            return

        # Cancel any in-flight skill
        await self._cancel_current()
        self._last_decision = decision

        # Look up skill
        if action not in self._registry:
            logger.warning("Unknown skill '%s', skipping execution", action)
            await self._record_action(
                action=action,
                expected="execute skill",
                actual="skill not found in registry",
                success=False,
            )
            return

        # Launch skill execution as a background task
        self._current_task = asyncio.create_task(
            self._execute_skill(action, decision.action_params)
        )

    async def _execute_skill(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a single skill with timeout and error handling.

        Args:
            action: The skill name.
            params: Parameters to pass to the skill function.

        Returns:
            The result dict from the skill execution.
        """
        skill = self._registry.get(action)
        try:
            result = await asyncio.wait_for(
                skill.execute_fn(self._bridge, **params),
                timeout=self._timeout_s,
            )
            await self._record_action(
                action=action,
                expected=f"execute {action}",
                actual=str(result),
                success=result.get("success", True),
            )
            return result
        except TimeoutError:
            logger.warning("Skill '%s' timed out after %.1fs", action, self._timeout_s)
            await self._record_action(
                action=action,
                expected=f"execute {action}",
                actual=f"timeout after {self._timeout_s}s",
                success=False,
            )
            return {"success": False, "error": "timeout"}
        except asyncio.CancelledError:
            logger.info("Skill '%s' was cancelled", action)
            await self._record_action(
                action=action,
                expected=f"execute {action}",
                actual="cancelled",
                success=False,
            )
            return {"success": False, "error": "cancelled"}
        except Exception as exc:
            logger.exception("Skill '%s' failed with error: %s", action, exc)
            await self._record_action(
                action=action,
                expected=f"execute {action}",
                actual=f"error: {exc}",
                success=False,
            )
            return {"success": False, "error": str(exc)}

    async def _cancel_current(self) -> None:
        """Cancel the currently running skill task, if any."""
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
            self._current_task = None

    async def _record_action(
        self,
        action: str,
        expected: str,
        actual: str,
        success: bool,
    ) -> None:
        """Record an action to SAS action history.

        Args:
            action: The skill name.
            expected: What was expected.
            actual: What actually happened.
            success: Whether the action succeeded.
        """
        entry = ActionHistoryEntry(
            action=action,
            expected_result=expected,
            actual_result=actual,
            success=success,
        )
        await self._sas.add_action(entry)

    async def shutdown(self) -> None:
        """Cancel any running skill on shutdown."""
        await self._cancel_current()
