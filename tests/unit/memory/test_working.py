"""Unit tests for WorkingMemory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from piano.core.types import MemoryEntry
from piano.memory.working import WorkingMemory

from .conftest import InMemorySAS

# --- helpers ---


def _entry(
    content: str = "test",
    category: str = "perception",
    importance: float = 0.5,
    age_minutes: float = 0.0,
) -> MemoryEntry:
    ts = datetime.now(UTC) - timedelta(minutes=age_minutes)
    return MemoryEntry(
        content=content,
        category=category,
        importance=importance,
        timestamp=ts,
        source_module="test",
    )


# --- basic operations ---


class TestWorkingMemoryBasic:
    def test_add_and_get_all(self) -> None:
        wm = WorkingMemory()
        wm.add(_entry("a"))
        wm.add(_entry("b"))
        assert wm.size == 2
        entries = wm.get_all()
        # newest first
        assert entries[0].content == "b"
        assert entries[1].content == "a"

    def test_clear(self) -> None:
        wm = WorkingMemory()
        wm.add(_entry("a"))
        wm.clear()
        assert wm.size == 0
        assert wm.get_all() == []

    def test_get_by_category(self) -> None:
        wm = WorkingMemory()
        wm.add(_entry("p1", category="perception"))
        wm.add(_entry("a1", category="action"))
        wm.add(_entry("p2", category="perception"))
        result = wm.get_by_category("perception")
        assert len(result) == 2
        assert all(e.category == "perception" for e in result)

    def test_get_by_category_empty(self) -> None:
        wm = WorkingMemory()
        assert wm.get_by_category("social") == []

    def test_remove_by_id(self) -> None:
        wm = WorkingMemory()
        entry = _entry("target")
        wm.add(entry)
        wm.add(_entry("other"))
        assert wm.remove(str(entry.id))
        assert wm.size == 1

    def test_remove_nonexistent(self) -> None:
        wm = WorkingMemory()
        assert wm.remove("does-not-exist") is False

    def test_is_full(self) -> None:
        wm = WorkingMemory(capacity=2)
        assert not wm.is_full()
        wm.add(_entry("a"))
        wm.add(_entry("b"))
        assert wm.is_full()

    def test_capacity_property(self) -> None:
        wm = WorkingMemory(capacity=5)
        assert wm.capacity == 5


# --- eviction ---


class TestWorkingMemoryEviction:
    def test_eviction_at_capacity(self) -> None:
        wm = WorkingMemory(capacity=3)
        wm.add(_entry("a", importance=0.8))
        wm.add(_entry("b", importance=0.5))
        wm.add(_entry("c", importance=0.9))
        # 4th add triggers eviction
        evicted = wm.add(_entry("d", importance=0.7))
        assert evicted is not None
        assert wm.size == 3

    def test_eviction_removes_lowest_score(self) -> None:
        """The entry with lowest importance*recency should be evicted."""
        wm = WorkingMemory(capacity=3)
        wm.add(_entry("high", importance=0.9))
        wm.add(_entry("low", importance=0.1))
        wm.add(_entry("mid", importance=0.5))
        evicted = wm.add(_entry("new", importance=0.6))
        assert evicted is not None
        assert evicted.content == "low"

    def test_eviction_prefers_old_low_importance(self) -> None:
        """Old + low importance entries should be evicted before recent ones."""
        wm = WorkingMemory(capacity=3)
        wm.add(_entry("old_low", importance=0.1, age_minutes=60))
        wm.add(_entry("recent_low", importance=0.1, age_minutes=0))
        wm.add(_entry("old_high", importance=0.9, age_minutes=60))
        evicted = wm.add(_entry("new", importance=0.5))
        assert evicted is not None
        assert evicted.content == "old_low"

    def test_no_eviction_under_capacity(self) -> None:
        wm = WorkingMemory(capacity=5)
        result = wm.add(_entry("first"))
        assert result is None
        assert wm.size == 1


# --- SAS sync ---


class TestWorkingMemorySAS:
    @pytest.mark.asyncio
    async def test_sync_to_sas(self) -> None:
        sas = InMemorySAS()
        wm = WorkingMemory()
        wm.bind_sas(sas)
        wm.add(_entry("a"))
        wm.add(_entry("b"))
        await wm.sync_to_sas()
        stored = await sas.get_working_memory()
        assert len(stored) == 2

    @pytest.mark.asyncio
    async def test_sync_from_sas(self) -> None:
        sas = InMemorySAS()
        # Pre-populate SAS
        await sas.set_working_memory([_entry("x"), _entry("y")])
        wm = WorkingMemory()
        wm.bind_sas(sas)
        await wm.sync_from_sas()
        assert wm.size == 2

    @pytest.mark.asyncio
    async def test_sync_without_sas_is_noop(self) -> None:
        wm = WorkingMemory()
        wm.add(_entry("a"))
        # Should not raise
        await wm.sync_to_sas()
        await wm.sync_from_sas()
        assert wm.size == 1
