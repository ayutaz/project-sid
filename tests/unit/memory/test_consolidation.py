"""Unit tests for Memory Consolidation Module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from piano.core.types import LLMRequest, LLMResponse, MemoryEntry, ModuleTier
from piano.memory.consolidation import (
    ConsolidationPolicy,
    MemoryConsolidationModule,
)

from .conftest import InMemorySAS

# --- Test helpers ---


def _entry(
    content: str = "test",
    category: str = "perception",
    importance: float = 0.5,
    age_minutes: float = 0.0,
) -> MemoryEntry:
    """Create a test memory entry with specified age."""
    ts = datetime.now(UTC) - timedelta(minutes=age_minutes)
    return MemoryEntry(
        content=content,
        category=category,
        importance=importance,
        timestamp=ts,
        source_module="test",
    )


class MockLTMStore:
    """Mock LTM store for testing."""

    def __init__(self) -> None:
        self.stored_entries: list[MemoryEntry] = []

    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry."""
        self.stored_entries.append(entry)


class MockLLMProvider:
    """Mock LLM provider for testing summarization."""

    def __init__(self, summary: str = "Test summary") -> None:
        self.summary = summary
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a mock summary."""
        self.requests.append(request)
        return LLMResponse(
            content=self.summary,
            model="mock-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            latency_ms=100.0,
            cost_usd=0.001,
        )


# --- Basic consolidation tests ---


class TestConsolidationBasic:
    @pytest.mark.asyncio
    async def test_consolidation_selects_high_importance_entries(self) -> None:
        """High-importance entries above threshold should be consolidated."""
        sas = InMemorySAS()
        ltm_store = MockLTMStore()
        policy = ConsolidationPolicy(
            min_importance=0.6,
            min_age_seconds=0.0,
            max_stm_before_consolidation=3,
            batch_size=10,
        )
        module = MemoryConsolidationModule(ltm_store=ltm_store, policy=policy)

        # Add entries: 2 high-importance, 1 low-importance
        await sas.add_stm(_entry("high1", importance=0.8, age_minutes=2))
        await sas.add_stm(_entry("high2", importance=0.7, age_minutes=1))
        await sas.add_stm(_entry("low", importance=0.3, age_minutes=2))

        result = await module.tick(sas)

        assert result.success
        assert result.data["consolidated"] == 2
        assert len(ltm_store.stored_entries) == 2
        # Verify high-importance entries were stored
        contents = {e.content for e in ltm_store.stored_entries}
        assert "high1" in contents
        assert "high2" in contents
        assert "low" not in contents

    @pytest.mark.asyncio
    async def test_entries_below_min_importance_skipped(self) -> None:
        """Entries below importance threshold should not be consolidated."""
        sas = InMemorySAS()
        ltm_store = MockLTMStore()
        policy = ConsolidationPolicy(
            min_importance=0.7,
            min_age_seconds=0.0,
            max_stm_before_consolidation=2,
            batch_size=10,
        )
        module = MemoryConsolidationModule(ltm_store=ltm_store, policy=policy)

        # All entries below threshold
        await sas.add_stm(_entry("low1", importance=0.5, age_minutes=2))
        await sas.add_stm(_entry("low2", importance=0.6, age_minutes=2))

        result = await module.tick(sas)

        assert result.success
        assert result.data["consolidated"] == 0
        assert len(ltm_store.stored_entries) == 0

    @pytest.mark.asyncio
    async def test_entries_too_recent_skipped(self) -> None:
        """Entries younger than min_age should not be consolidated."""
        sas = InMemorySAS()
        ltm_store = MockLTMStore()
        policy = ConsolidationPolicy(
            min_importance=0.5,
            min_age_seconds=60.0,
            max_stm_before_consolidation=2,
            batch_size=10,
        )
        module = MemoryConsolidationModule(ltm_store=ltm_store, policy=policy)

        # Add entries: 1 old enough, 1 too recent
        await sas.add_stm(_entry("old", importance=0.7, age_minutes=2))  # 120s old
        await sas.add_stm(_entry("recent", importance=0.7, age_minutes=0.5))  # 30s old

        result = await module.tick(sas)

        assert result.success
        assert result.data["consolidated"] == 1
        assert len(ltm_store.stored_entries) == 1
        assert ltm_store.stored_entries[0].content == "old"

    @pytest.mark.asyncio
    async def test_batch_size_limiting(self) -> None:
        """Consolidation should respect batch_size limit."""
        sas = InMemorySAS()
        ltm_store = MockLTMStore()
        policy = ConsolidationPolicy(
            min_importance=0.5,
            min_age_seconds=0.0,
            max_stm_before_consolidation=5,
            batch_size=3,
        )
        module = MemoryConsolidationModule(ltm_store=ltm_store, policy=policy)

        # Add 6 eligible entries
        for i in range(6):
            await sas.add_stm(_entry(f"entry{i}", importance=0.8, age_minutes=2))

        result = await module.tick(sas)

        assert result.success
        assert result.data["consolidated"] == 3  # Limited by batch_size
        assert len(ltm_store.stored_entries) == 3

    @pytest.mark.asyncio
    async def test_empty_stm_no_consolidation(self) -> None:
        """Empty STM should not trigger consolidation."""
        sas = InMemorySAS()
        ltm_store = MockLTMStore()
        module = MemoryConsolidationModule(ltm_store=ltm_store)

        result = await module.tick(sas)

        assert result.success
        assert result.data["consolidated"] == 0
        assert len(ltm_store.stored_entries) == 0

    @pytest.mark.asyncio
    async def test_stm_below_threshold_no_consolidation(self) -> None:
        """STM below max_stm_before_consolidation should not trigger."""
        sas = InMemorySAS()
        ltm_store = MockLTMStore()
        policy = ConsolidationPolicy(
            min_importance=0.5,
            min_age_seconds=0.0,
            max_stm_before_consolidation=10,
            batch_size=10,
        )
        module = MemoryConsolidationModule(ltm_store=ltm_store, policy=policy)

        # Add only 5 entries (below threshold of 10)
        for i in range(5):
            await sas.add_stm(_entry(f"entry{i}", importance=0.8, age_minutes=2))

        result = await module.tick(sas)

        assert result.success
        assert result.data["consolidated"] == 0
        assert result.data["stm_size"] == 5


# --- Summarization tests ---


class TestConsolidationSummarization:
    @pytest.mark.asyncio
    async def test_summarization_with_llm(self) -> None:
        """LLM should be used to summarize related memories."""
        sas = InMemorySAS()
        ltm_store = MockLTMStore()
        llm = MockLLMProvider(summary="Agent cooperated with Lila on mining tasks")
        policy = ConsolidationPolicy(
            min_importance=0.5,
            min_age_seconds=0.0,
            max_stm_before_consolidation=3,
            batch_size=10,
        )
        module = MemoryConsolidationModule(
            ltm_store=ltm_store, policy=policy, llm_provider=llm
        )

        # Add multiple related entries
        await sas.add_stm(_entry("Talked with Lila", importance=0.7, age_minutes=5))
        await sas.add_stm(_entry("Mined iron with Lila", importance=0.8, age_minutes=3))
        await sas.add_stm(_entry("Shared food with Lila", importance=0.6, age_minutes=2))

        result = await module.tick(sas)

        assert result.success
        assert result.data["summaries_created"] == 1
        # LLM should have been called
        assert len(llm.requests) == 1
        # Summary entry should be stored
        summaries = [e for e in ltm_store.stored_entries if e.category == "reflection"]
        assert len(summaries) == 1
        assert summaries[0].content == "Agent cooperated with Lila on mining tasks"
        assert summaries[0].metadata.get("is_summary") is True

    @pytest.mark.asyncio
    async def test_no_summarization_without_llm(self) -> None:
        """Without LLM provider, no summaries should be created."""
        sas = InMemorySAS()
        ltm_store = MockLTMStore()
        policy = ConsolidationPolicy(
            min_importance=0.5,
            min_age_seconds=0.0,
            max_stm_before_consolidation=2,
            batch_size=10,
        )
        module = MemoryConsolidationModule(ltm_store=ltm_store, policy=policy)

        await sas.add_stm(_entry("entry1", importance=0.7, age_minutes=2))
        await sas.add_stm(_entry("entry2", importance=0.7, age_minutes=2))

        result = await module.tick(sas)

        assert result.success
        assert result.data["summaries_created"] == 0
        # Only individual entries should be stored
        assert len(ltm_store.stored_entries) == 2

    @pytest.mark.asyncio
    async def test_summarization_preserves_max_importance(self) -> None:
        """Summary entry should have the max importance of consolidated entries."""
        sas = InMemorySAS()
        ltm_store = MockLTMStore()
        llm = MockLLMProvider(summary="Summary of events")
        policy = ConsolidationPolicy(
            min_importance=0.5,
            min_age_seconds=0.0,
            max_stm_before_consolidation=2,
            batch_size=10,
        )
        module = MemoryConsolidationModule(
            ltm_store=ltm_store, policy=policy, llm_provider=llm
        )

        await sas.add_stm(_entry("low", importance=0.6, age_minutes=2))
        await sas.add_stm(_entry("high", importance=0.9, age_minutes=2))

        await module.tick(sas)

        summaries = [e for e in ltm_store.stored_entries if e.category == "reflection"]
        assert len(summaries) == 1
        assert summaries[0].importance == 0.9


# --- Consolidation result tests ---


class TestConsolidationResult:
    @pytest.mark.asyncio
    async def test_consolidation_stats_correct(self) -> None:
        """Result should contain accurate consolidation statistics."""
        sas = InMemorySAS()
        ltm_store = MockLTMStore()
        policy = ConsolidationPolicy(
            min_importance=0.6,
            min_age_seconds=0.0,
            max_stm_before_consolidation=3,
            batch_size=10,
        )
        module = MemoryConsolidationModule(ltm_store=ltm_store, policy=policy)

        # Add 3 eligible entries
        for i in range(3):
            await sas.add_stm(_entry(f"entry{i}", importance=0.8, age_minutes=2))

        result = await module.tick(sas)

        assert result.data["consolidated"] == 3
        assert result.data["summaries_created"] == 0
        assert result.data["stm_size"] == 3


# --- Embedding placeholder tests ---


class TestEmbeddingPlaceholder:
    @pytest.mark.asyncio
    async def test_embedding_placeholder_added(self) -> None:
        """Consolidated entries should have embedding placeholders."""
        sas = InMemorySAS()
        ltm_store = MockLTMStore()
        policy = ConsolidationPolicy(
            min_importance=0.5,
            min_age_seconds=0.0,
            max_stm_before_consolidation=1,
            batch_size=10,
        )
        module = MemoryConsolidationModule(ltm_store=ltm_store, policy=policy)

        await sas.add_stm(_entry("test entry", importance=0.8, age_minutes=2))

        result = await module.tick(sas)

        assert result.success
        assert len(ltm_store.stored_entries) == 1
        entry = ltm_store.stored_entries[0]
        assert "embedding" in entry.metadata
        assert isinstance(entry.metadata["embedding"], list)
        assert len(entry.metadata["embedding"]) == 1536  # text-embedding-3-small dim


# --- Module lifecycle tests ---


class TestModuleLifecycle:
    def test_module_name(self) -> None:
        """Module should have correct name."""
        ltm_store = MockLTMStore()
        module = MemoryConsolidationModule(ltm_store=ltm_store)
        assert module.name == "memory_consolidation"

    def test_module_tier(self) -> None:
        """Module should be in SLOW tier."""
        ltm_store = MockLTMStore()
        module = MemoryConsolidationModule(ltm_store=ltm_store)
        assert module.tier == ModuleTier.SLOW

    @pytest.mark.asyncio
    async def test_module_initialize(self) -> None:
        """Module initialization should succeed."""
        ltm_store = MockLTMStore()
        module = MemoryConsolidationModule(ltm_store=ltm_store)
        await module.initialize()

    @pytest.mark.asyncio
    async def test_module_shutdown(self) -> None:
        """Module shutdown should succeed."""
        ltm_store = MockLTMStore()
        module = MemoryConsolidationModule(ltm_store=ltm_store)
        await module.shutdown()


# --- Edge cases ---


class TestConsolidationEdgeCases:
    @pytest.mark.asyncio
    async def test_consolidation_with_single_entry(self) -> None:
        """Single entry should be consolidated without summarization."""
        sas = InMemorySAS()
        ltm_store = MockLTMStore()
        llm = MockLLMProvider()
        policy = ConsolidationPolicy(
            min_importance=0.5,
            min_age_seconds=0.0,
            max_stm_before_consolidation=1,
            batch_size=10,
        )
        module = MemoryConsolidationModule(
            ltm_store=ltm_store, policy=policy, llm_provider=llm
        )

        await sas.add_stm(_entry("single", importance=0.8, age_minutes=2))

        result = await module.tick(sas)

        assert result.success
        assert result.data["consolidated"] == 1
        # No summarization for single entry (requires > 1)
        assert result.data["summaries_created"] == 0
        assert len(llm.requests) == 0

    @pytest.mark.asyncio
    async def test_oldest_entries_consolidated_first(self) -> None:
        """Consolidation should prioritize oldest entries."""
        sas = InMemorySAS()
        ltm_store = MockLTMStore()
        policy = ConsolidationPolicy(
            min_importance=0.5,
            min_age_seconds=0.0,
            max_stm_before_consolidation=3,
            batch_size=2,
        )
        module = MemoryConsolidationModule(ltm_store=ltm_store, policy=policy)

        # Add entries with different ages
        await sas.add_stm(_entry("newest", importance=0.8, age_minutes=1))
        await sas.add_stm(_entry("middle", importance=0.8, age_minutes=3))
        await sas.add_stm(_entry("oldest", importance=0.8, age_minutes=5))

        result = await module.tick(sas)

        assert result.success
        assert len(ltm_store.stored_entries) == 2
        # Oldest entries should be consolidated first
        contents = [e.content for e in ltm_store.stored_entries]
        assert "oldest" in contents
        assert "middle" in contents
        assert "newest" not in contents

    @pytest.mark.asyncio
    async def test_error_handling_in_tick(self) -> None:
        """Module should handle errors gracefully."""

        class FailingLTMStore:
            async def store(self, entry: MemoryEntry) -> None:
                raise RuntimeError("Storage error")

        sas = InMemorySAS()
        ltm_store = FailingLTMStore()
        policy = ConsolidationPolicy(
            min_importance=0.5,
            min_age_seconds=0.0,
            max_stm_before_consolidation=1,
            batch_size=10,
        )
        module = MemoryConsolidationModule(ltm_store=ltm_store, policy=policy)

        await sas.add_stm(_entry("test", importance=0.8, age_minutes=2))

        result = await module.tick(sas)

        assert not result.success
        assert result.error is not None
        assert "Storage error" in result.error
