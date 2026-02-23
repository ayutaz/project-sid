"""Unit tests for ShortTermMemory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from piano.core.types import MemoryEntry
from piano.memory.stm import ShortTermMemory

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


class TestSTMBasic:
    def test_add_and_get_recent(self) -> None:
        stm = ShortTermMemory()
        stm.add(_entry("a"))
        stm.add(_entry("b"))
        stm.add(_entry("c"))
        recent = stm.get_recent(limit=2)
        assert len(recent) == 2
        # newest first
        assert recent[0].content == "c"
        assert recent[1].content == "b"

    def test_get_recent_default_limit(self) -> None:
        stm = ShortTermMemory()
        for i in range(15):
            stm.add(_entry(f"e{i}"))
        recent = stm.get_recent()  # default limit=10
        assert len(recent) == 10
        assert recent[0].content == "e14"

    def test_clear(self) -> None:
        stm = ShortTermMemory()
        stm.add(_entry("a"))
        stm.clear()
        assert stm.size == 0

    def test_size_and_capacity(self) -> None:
        stm = ShortTermMemory(capacity=50)
        assert stm.capacity == 50
        assert stm.size == 0

    def test_is_full(self) -> None:
        stm = ShortTermMemory(capacity=2)
        stm.add(_entry("a"))
        assert not stm.is_full()
        stm.add(_entry("b"))
        assert stm.is_full()


# --- FIFO eviction ---


class TestSTMEviction:
    def test_fifo_eviction_at_capacity(self) -> None:
        stm = ShortTermMemory(capacity=3)
        stm.add(_entry("a"))
        stm.add(_entry("b"))
        stm.add(_entry("c"))
        evicted = stm.add(_entry("d"))
        assert evicted is not None
        assert evicted.content == "a"
        assert stm.size == 3

    def test_no_eviction_under_capacity(self) -> None:
        stm = ShortTermMemory(capacity=5)
        result = stm.add(_entry("a"))
        assert result is None

    def test_fifo_order_after_multiple_evictions(self) -> None:
        stm = ShortTermMemory(capacity=2)
        stm.add(_entry("a"))
        stm.add(_entry("b"))
        evicted1 = stm.add(_entry("c"))
        evicted2 = stm.add(_entry("d"))
        assert evicted1 is not None and evicted1.content == "a"
        assert evicted2 is not None and evicted2.content == "b"
        recent = stm.get_recent(limit=10)
        assert [e.content for e in recent] == ["d", "c"]

    def test_boundary_exact_capacity(self) -> None:
        """Filling exactly to capacity should not evict."""
        stm = ShortTermMemory(capacity=3)
        for label in ("a", "b", "c"):
            result = stm.add(_entry(label))
            assert result is None
        assert stm.size == 3


# --- search ---


class TestSTMSearch:
    def test_search_by_category(self) -> None:
        stm = ShortTermMemory()
        stm.add(_entry("p1", category="perception"))
        stm.add(_entry("a1", category="action"))
        stm.add(_entry("s1", category="social"))
        stm.add(_entry("p2", category="perception"))
        results = stm.search_by_category("perception")
        assert len(results) == 2
        assert all(e.category == "perception" for e in results)
        # newest first
        assert results[0].content == "p2"

    def test_search_by_category_limit(self) -> None:
        stm = ShortTermMemory()
        for i in range(10):
            stm.add(_entry(f"p{i}", category="perception"))
        results = stm.search_by_category("perception", limit=3)
        assert len(results) == 3

    def test_search_by_category_empty(self) -> None:
        stm = ShortTermMemory()
        stm.add(_entry("a", category="action"))
        assert stm.search_by_category("reflection") == []

    def test_get_by_importance(self) -> None:
        stm = ShortTermMemory()
        stm.add(_entry("low", importance=0.1))
        stm.add(_entry("mid", importance=0.5))
        stm.add(_entry("high", importance=0.9))
        results = stm.get_by_importance(threshold=0.5)
        assert len(results) == 2
        assert all(e.importance >= 0.5 for e in results)

    def test_get_by_importance_limit(self) -> None:
        stm = ShortTermMemory()
        for i in range(10):
            stm.add(_entry(f"e{i}", importance=0.8))
        results = stm.get_by_importance(threshold=0.5, limit=3)
        assert len(results) == 3

    def test_get_by_importance_none_match(self) -> None:
        stm = ShortTermMemory()
        stm.add(_entry("low", importance=0.1))
        assert stm.get_by_importance(threshold=0.9) == []


# --- SAS sync ---


class TestSTMSAS:
    @pytest.mark.asyncio
    async def test_sync_to_sas(self) -> None:
        sas = InMemorySAS()
        stm = ShortTermMemory()
        stm.bind_sas(sas)
        stm.add(_entry("a"))
        stm.add(_entry("b"))
        await stm.sync_to_sas()
        stored = await sas.get_stm()
        assert len(stored) == 2

    @pytest.mark.asyncio
    async def test_sync_from_sas(self) -> None:
        sas = InMemorySAS()
        await sas.add_stm(_entry("x"))
        await sas.add_stm(_entry("y"))
        stm = ShortTermMemory()
        stm.bind_sas(sas)
        await stm.sync_from_sas()
        assert stm.size == 2

    @pytest.mark.asyncio
    async def test_sync_without_sas_is_noop(self) -> None:
        stm = ShortTermMemory()
        stm.add(_entry("a"))
        await stm.sync_to_sas()
        await stm.sync_from_sas()
        assert stm.size == 1
