"""Base module interface for the PIANO architecture.

All PIANO modules (Goal Generation, Planning, Social Awareness, etc.)
implement this interface. Modules are stateless - they read from and
write to the Shared Agent State (SAS).

Reference: docs/implementation/01-system-architecture.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from piano.core.types import CCDecision, ModuleResult, ModuleTier

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState


class Module(ABC):
    """Abstract base class for all PIANO modules.

    Modules are stateless processors that:
    1. Read from SAS on each tick
    2. Perform computation (possibly calling LLM)
    3. Write results back to SAS
    4. Optionally react to CC broadcast decisions
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique module name (e.g., 'goal_generation', 'action_awareness')."""
        ...

    @property
    @abstractmethod
    def tier(self) -> ModuleTier:
        """Execution tier determining scheduling priority and frequency."""
        ...

    @abstractmethod
    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        """Execute one tick of this module.

        Read from SAS, perform computation, write results back to SAS,
        and return a ModuleResult summarizing what happened.
        """
        ...

    async def on_broadcast(self, decision: CCDecision) -> None:
        """Handle a CC broadcast decision.

        Output modules (talking, skill execution) implement this to
        react to CC decisions. Input/processing modules can ignore it.
        """

    async def initialize(self) -> None:
        """One-time initialization (called before first tick)."""

    async def shutdown(self) -> None:
        """Cleanup (called on agent shutdown)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, tier={self.tier.value})"
