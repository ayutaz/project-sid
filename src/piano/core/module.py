"""Base module interface for the PIANO architecture.

All PIANO modules (Goal Generation, Planning, Social Awareness, etc.)
implement this interface. Modules are stateless - they read from and
write to the Shared Agent State (SAS).

Reference: docs/implementation/01-system-architecture.md
"""

from __future__ import annotations

__all__ = ["Module"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState
    from piano.core.types import CCDecision, ModuleResult, ModuleTier


class Module(ABC):
    """Abstract base class for all PIANO modules.

    Modules are stateless processors that:
    1. Read from SAS on each tick
    2. Perform computation (possibly calling LLM)
    3. Write results DIRECTLY back to SAS (primary state update path)
    4. Optionally react to CC broadcast decisions (for output modules)

    The ModuleResult returned from tick() is used for:
    - CC information bottleneck input (compression)
    - Debug metadata and telemetry
    - NOT for state updates (modules write directly to SAS)
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

        Read from SAS, perform computation, and WRITE DIRECTLY to SAS.
        The returned ModuleResult is for CC compression input and debug
        metadata only - it is NOT used for state updates.

        Returns:
            ModuleResult summarizing what happened (for CC and telemetry).
        """
        ...

    async def on_broadcast(self, decision: CCDecision) -> None:  # noqa: B027
        """Handle a CC broadcast decision.

        Output modules (talking, skill execution) implement this to
        react to CC decisions and trigger side effects (e.g., executing
        a skill, sending a chat message). Input/processing modules can
        ignore this method.
        """

    async def initialize(self) -> None:  # noqa: B027
        """One-time initialization (called before first tick)."""

    async def shutdown(self) -> None:  # noqa: B027
        """Cleanup (called on agent shutdown)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, tier={self.tier.value})"
