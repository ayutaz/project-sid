"""Short-Term Memory (STM) for the PIANO architecture.

STM stores recent events as ``MemoryEntry`` objects in chronological order.
It supports category-based filtering and importance-based retrieval.

Capacity: latest 100 entries (FIFO; oldest dropped on overflow).

Reference: docs/implementation/04-memory-system.md Section 4.2.2
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from piano.core.sas import SharedAgentState
    from piano.core.types import MemoryEntry

_DEFAULT_CAPACITY = 100


class ShortTermMemory:
    """Time-ordered event store with category and importance filters.

    Entries are kept in insertion order (oldest first internally).
    Public query methods return newest-first unless stated otherwise.
    """

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._entries: collections.deque[MemoryEntry] = collections.deque(maxlen=capacity)
        self._capacity = capacity
        self._sas: SharedAgentState | None = None
        self._synced_ids: set[UUID] = set()

    # --- SAS binding ---

    def bind_sas(self, sas: SharedAgentState) -> None:
        """Attach a SAS instance for automatic synchronisation."""
        self._sas = sas

    # --- Core API ---

    def add(self, entry: MemoryEntry) -> MemoryEntry | None:
        """Append an entry.

        When the capacity is exceeded the oldest entry is removed and
        returned. Returns ``None`` when no eviction occurs.
        """
        evicted: MemoryEntry | None = None
        if len(self._entries) >= self._capacity:
            evicted = self._entries[0]  # Will be evicted by deque maxlen
        self._entries.append(entry)
        return evicted

    def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        """Return the *limit* most recent entries (newest first)."""
        entries = list(self._entries)
        return list(reversed(entries[-limit:]))

    def search_by_category(
        self,
        category: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Return entries matching *category* (newest first, up to *limit*)."""
        matches = [e for e in reversed(self._entries) if e.category == category]
        return matches[:limit]

    def get_by_importance(
        self,
        threshold: float = 0.5,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Return entries with importance >= *threshold* (newest first)."""
        matches = [e for e in reversed(self._entries) if e.importance >= threshold]
        return matches[:limit]

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
        self._synced_ids.clear()

    @property
    def size(self) -> int:
        """Current number of stored entries."""
        return len(self._entries)

    @property
    def capacity(self) -> int:
        """Maximum capacity."""
        return self._capacity

    def is_full(self) -> bool:
        """Return whether memory is at capacity."""
        return len(self._entries) >= self._capacity

    # --- SAS sync ---

    async def sync_to_sas(self) -> None:
        """Push only unsynced entries to the bound SAS.

        Tracks which entries have already been synced to prevent
        duplicate accumulation on repeated calls.
        """
        if self._sas is not None:
            for entry in self._entries:
                if entry.id not in self._synced_ids:
                    await self._sas.add_stm(entry)
                    self._synced_ids.add(entry.id)

    async def sync_from_sas(self) -> None:
        """Pull entries from SAS into local state."""
        if self._sas is not None:
            sas_entries = list(await self._sas.get_stm(limit=self._capacity))
            # SAS returns newest-first; we store oldest-first internally.
            sas_entries.reverse()
            self._entries.clear()
            self._synced_ids.clear()
            for entry in sas_entries:
                self._entries.append(entry)
                self._synced_ids.add(entry.id)
