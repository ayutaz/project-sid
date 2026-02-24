"""Working Memory (WM) for the PIANO architecture.

Working Memory holds the agent's immediate context: the latest CC broadcast
result, current task context, and recent perceptions. It is the fastest
memory tier, accessed every tick by the Cognitive Controller.

Capacity: max 10 MemoryEntry items.
Eviction: lowest (importance * recency) score is dropped first.

Reference: docs/implementation/04-memory-system.md Section 4.2.1
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState
    from piano.core.types import MemoryEntry

_MAX_CAPACITY = 10

# Exponential decay rate for recency scoring (per hour).
# Half-life ≈ 1.4 hours (ln(2) / 0.5 ≈ 1.386).
DECAY_LAMBDA = 0.5


def _recency_score(entry: MemoryEntry, now: datetime) -> float:
    """Exponential decay based on age."""
    age_hours = max((now - entry.timestamp).total_seconds() / 3600.0, 0.0)
    return math.exp(-DECAY_LAMBDA * age_hours)


def _eviction_score(entry: MemoryEntry, now: datetime) -> float:
    """Combined score used for eviction: importance * recency.

    Lower score -> evicted first.
    """
    return entry.importance * _recency_score(entry, now)


class WorkingMemory:
    """In-memory store for the agent's immediate cognitive context.

    Synchronisation to SAS is performed explicitly via ``sync_to_sas()``
    (called by MemoryManager after each tick). The memory is purely
    in-process (no external DB).
    """

    def __init__(self, capacity: int = _MAX_CAPACITY) -> None:
        self._entries: list[MemoryEntry] = []
        self._capacity = capacity
        self._sas: SharedAgentState | None = None

    # --- SAS binding ---

    def bind_sas(self, sas: SharedAgentState) -> None:
        """Attach a SAS instance for automatic synchronisation."""
        self._sas = sas

    # --- Core API ---

    def add(self, entry: MemoryEntry) -> MemoryEntry | None:
        """Add an entry to working memory.

        If capacity is exceeded the entry with the lowest eviction score
        is removed and returned. Returns ``None`` when no eviction occurs.
        """
        self._entries.append(entry)
        evicted: MemoryEntry | None = None
        if len(self._entries) > self._capacity:
            evicted = self._evict_one()
        return evicted

    def get_all(self) -> list[MemoryEntry]:
        """Return all working memory entries (newest first)."""
        return sorted(self._entries, key=lambda e: e.timestamp, reverse=True)

    def get_by_category(self, category: str) -> list[MemoryEntry]:
        """Return entries matching *category* (newest first)."""
        return sorted(
            [e for e in self._entries if e.category == category],
            key=lambda e: e.timestamp,
            reverse=True,
        )

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._entries)

    @property
    def capacity(self) -> int:
        """Maximum capacity."""
        return self._capacity

    def is_full(self) -> bool:
        """Return whether memory is at capacity."""
        return len(self._entries) >= self._capacity

    def remove(self, entry_id: str | UUID) -> bool:
        """Remove an entry by its UUID string or UUID. Returns True if found."""
        # Convert to UUID once for efficient comparison
        try:
            target_id = entry_id if isinstance(entry_id, UUID) else UUID(entry_id)
        except ValueError:
            # Invalid UUID string
            return False
        for i, e in enumerate(self._entries):
            if e.id == target_id:
                self._entries.pop(i)
                return True
        return False

    # --- SAS sync ---

    async def sync_to_sas(self) -> None:
        """Push current entries to the bound SAS."""
        if self._sas is not None:
            await self._sas.set_working_memory(list(self._entries))

    async def sync_from_sas(self) -> None:
        """Pull entries from SAS into local state."""
        if self._sas is not None:
            self._entries = list(await self._sas.get_working_memory())

    # --- Internal ---

    def _evict_one(self) -> MemoryEntry:
        """Remove and return the entry with the lowest eviction score."""
        now = datetime.now(UTC)
        worst_idx = 0
        worst_score = float("inf")
        for i, entry in enumerate(self._entries):
            score = _eviction_score(entry, now)
            if score < worst_score:
                worst_score = score
                worst_idx = i
        return self._entries.pop(worst_idx)
