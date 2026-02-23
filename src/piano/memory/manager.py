"""Memory Manager - unified facade over WM and STM.

The MemoryManager implements the ``Module`` ABC (tier=MID) and orchestrates
both Working Memory and Short-Term Memory. On each tick it reads new
perceptions / actions from SAS and routes them to the appropriate store.

Reference: docs/implementation/04-memory-system.md Section 4.8
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from piano.core.module import Module
from piano.core.types import MemoryEntry, ModuleResult, ModuleTier
from piano.memory.stm import ShortTermMemory
from piano.memory.working import WorkingMemory

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState


class MemoryManager(Module):
    """Orchestrates Working Memory and Short-Term Memory.

    Responsibilities:
    * Route incoming perceptions, actions, and social events to the right store.
    * Keep WM within its capacity budget.
    * Expose a simple query API that other modules can use.
    """

    def __init__(
        self,
        wm: WorkingMemory | None = None,
        stm: ShortTermMemory | None = None,
    ) -> None:
        self._wm = wm or WorkingMemory()
        self._stm = stm or ShortTermMemory()

    # --- Module ABC ---

    @property
    def name(self) -> str:
        return "memory_manager"

    @property
    def tier(self) -> ModuleTier:
        return ModuleTier.MID

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        """Read new information from SAS and distribute to WM / STM."""
        added_count = 0
        try:
            # 1. Ingest latest percepts as a WM entry.
            percepts = await sas.get_percepts()
            percept_entry = MemoryEntry(
                timestamp=datetime.now(timezone.utc),
                content=_percepts_summary(percepts),
                category="perception",
                importance=0.3,
                source_module=self.name,
            )
            self._wm.add(percept_entry)
            added_count += 1

            # 2. Read recent action history and store the newest one in STM.
            actions = await sas.get_action_history(limit=1)
            if actions:
                latest = actions[0]
                action_entry = MemoryEntry(
                    timestamp=latest.timestamp,
                    content=f"Action: {latest.action} -> {latest.actual_result}",
                    category="action",
                    importance=0.3 if latest.success else 0.5,
                    source_module=self.name,
                    metadata={"success": latest.success},
                )
                self._stm.add(action_entry)
                added_count += 1

            # 3. Sync WM back to SAS so CC can read it.
            self._wm.bind_sas(sas)
            await self._wm.sync_to_sas()

        except Exception as exc:  # noqa: BLE001
            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                error=str(exc),
            )

        return ModuleResult(
            module_name=self.name,
            tier=self.tier,
            data={"added": added_count},
        )

    # --- Public convenience accessors ---

    @property
    def wm(self) -> WorkingMemory:
        """Direct access to the WorkingMemory instance."""
        return self._wm

    @property
    def stm(self) -> ShortTermMemory:
        """Direct access to the ShortTermMemory instance."""
        return self._stm

    def store(self, entry: MemoryEntry, *, to_wm: bool = False) -> None:
        """Manually store an entry in STM (or WM if *to_wm* is True)."""
        if to_wm:
            self._wm.add(entry)
        else:
            self._stm.add(entry)


def _percepts_summary(percepts: object) -> str:
    """Build a short textual summary of percept data for WM."""
    # PerceptData is a Pydantic model; grab the most useful fields.
    try:
        p = percepts  # type: ignore[assignment]
        parts: list[str] = []
        pos = getattr(p, "position", {})
        if pos:
            parts.append(f"pos=({pos.get('x',0)},{pos.get('y',0)},{pos.get('z',0)})")
        players = getattr(p, "nearby_players", [])
        if players:
            parts.append(f"nearby={players}")
        parts.append(f"hp={getattr(p, 'health', '?')}")
        return " | ".join(parts) if parts else "no percepts"
    except Exception:  # noqa: BLE001
        return str(percepts)
