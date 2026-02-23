"""Broadcast manager for CC decisions.

Distributes CCDecision to all registered output modules by invoking
their ``on_broadcast`` method concurrently.

Reference: docs/implementation/03-cognitive-controller.md Section 3
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from piano.core.module import Module
    from piano.core.types import CCDecision

logger = logging.getLogger(__name__)


@dataclass
class BroadcastResult:
    """Aggregated result of a broadcast round."""

    decision: CCDecision
    total_listeners: int = 0
    success_count: int = 0
    failure_count: int = 0
    errors: dict[str, str] = field(default_factory=dict)


class BroadcastManager:
    """Manage CC decision broadcast to output modules.

    Registers listener modules and fans out each CCDecision to all of
    them concurrently via ``Module.on_broadcast``.
    """

    def __init__(self) -> None:
        self._listeners: dict[str, Module] = {}
        self._latest: CCDecision | None = None

    def register(self, module: Module) -> None:
        """Register an output module as a broadcast listener.

        Args:
            module: A Module instance whose ``on_broadcast`` will be called.
        """
        self._listeners[module.name] = module

    def unregister(self, module_name: str) -> None:
        """Remove a listener by name.

        Args:
            module_name: Name of the module to remove.
        """
        self._listeners.pop(module_name, None)

    @property
    def listener_names(self) -> list[str]:
        """Return the names of all registered listeners."""
        return list(self._listeners.keys())

    @property
    def latest_decision(self) -> CCDecision | None:
        """Return the most recently broadcast decision, or ``None``."""
        return self._latest

    async def broadcast(self, decision: CCDecision) -> BroadcastResult:
        """Broadcast a CC decision to all registered listeners.

        Each listener's ``on_broadcast`` is called concurrently.
        Individual listener failures are captured but do not prevent
        delivery to remaining listeners.

        Args:
            decision: The CCDecision to broadcast.

        Returns:
            BroadcastResult summarising delivery outcomes.
        """
        self._latest = decision
        total = len(self._listeners)

        if total == 0:
            return BroadcastResult(decision=decision, total_listeners=0)

        tasks: dict[str, asyncio.Task[None]] = {}
        for name, module in self._listeners.items():
            tasks[name] = asyncio.create_task(
                module.on_broadcast(decision),
                name=f"broadcast-{name}",
            )

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        errors: dict[str, str] = {}
        success = 0
        for name, result in zip(tasks, results, strict=True):
            if isinstance(result, BaseException):
                errors[name] = str(result)
                logger.warning("Broadcast to %s failed: %s", name, result)
            else:
                success += 1

        return BroadcastResult(
            decision=decision,
            total_listeners=total,
            success_count=success,
            failure_count=len(errors),
            errors=errors,
        )
