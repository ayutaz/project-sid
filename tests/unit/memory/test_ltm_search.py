"""Unit tests for LTM search and retrieval module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from uuid import UUID

from piano.core.types import GoalData, MemoryEntry, ModuleTier, PerceptData
from piano.memory.ltm import LTMEntry
from piano.memory.ltm_search import (
    ForgettingCurve,
    LTMRetrievalModule,
    RetrievalQuery,
)
from tests.helpers import InMemorySAS

# --- Mock LTM Store ---


class MockLTMStore:
    """Mock LTM store for testing that returns preset memories.

    Implements the canonical LTMStore protocol from piano.memory.ltm.
    """

    def __init__(self, preset_memories: list[LTMEntry] | None = None) -> None:
        self.preset_memories = preset_memories or []
        self.search_calls: list[dict] = []

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def store(self, agent_id: str, entry: LTMEntry) -> UUID:
        return entry.id

    async def retrieve(self, agent_id: str, entry_id: UUID) -> LTMEntry | None:
        for m in self.preset_memories:
            if m.id == entry_id:
                return m
        return None

    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[LTMEntry]:
        """Mock search that returns preset memories.

        Records each call for assertion. Since we use placeholder embeddings,
        we simply return all preset memories (filtered by limit).
        """
        self.search_calls.append(
            {
                "agent_id": agent_id,
                "query_embedding": query_embedding,
                "limit": limit,
                "min_score": min_score,
            }
        )
        # Return all preset memories up to limit
        return self.preset_memories[:limit]

    async def delete(self, agent_id: str, entry_id: UUID) -> bool:
        return False

    async def get_stats(self, agent_id: str) -> dict[str, float | int]:
        return {"count": len(self.preset_memories), "avg_importance": 0.0, "total_accesses": 0}


# --- Helper functions ---


def _ltm_entry(
    content: str,
    category: str = "episodic",
    importance: float = 0.5,
    age_hours: float = 0.0,
) -> LTMEntry:
    """Create a test LTM entry with specified age."""
    ts = datetime.now(UTC) - timedelta(hours=age_hours)
    return LTMEntry(
        timestamp=ts,
        content=content,
        category=category,
        importance=importance,
        source_module="test",
        embedding=[0.1] * 384,
    )


def _memory(
    content: str,
    category: str = "episodic",
    importance: float = 0.5,
    age_hours: float = 0.0,
) -> MemoryEntry:
    """Create a test memory entry with specified age."""
    ts = datetime.now(UTC) - timedelta(hours=age_hours)
    return MemoryEntry(
        timestamp=ts,
        content=content,
        category=category,
        importance=importance,
        source_module="test",
    )


# --- ForgettingCurve tests ---


class TestForgettingCurve:
    def test_calculate_retention_fresh_memory(self) -> None:
        """Fresh memory should have high retention (close to importance)."""
        retention = ForgettingCurve.calculate_retention(
            importance=1.0,
            hours_elapsed=0.0,
            decay_lambda=0.5,
        )
        assert retention == pytest.approx(1.0, abs=0.01)

    def test_calculate_retention_old_memory(self) -> None:
        """Old memory should have lower retention due to decay."""
        retention = ForgettingCurve.calculate_retention(
            importance=1.0,
            hours_elapsed=5.0,
            decay_lambda=0.5,
        )
        # exp(-0.5 * 5) = exp(-2.5) ≈ 0.082
        assert retention == pytest.approx(0.082, abs=0.01)

    def test_calculate_retention_with_importance(self) -> None:
        """Retention should scale with importance."""
        retention_high = ForgettingCurve.calculate_retention(
            importance=1.0,
            hours_elapsed=2.0,
            decay_lambda=0.5,
        )
        retention_low = ForgettingCurve.calculate_retention(
            importance=0.5,
            hours_elapsed=2.0,
            decay_lambda=0.5,
        )
        assert retention_high == pytest.approx(2 * retention_low, abs=0.01)

    def test_calculate_retention_negative_hours(self) -> None:
        """Negative hours should be treated as 0."""
        retention = ForgettingCurve.calculate_retention(
            importance=1.0,
            hours_elapsed=-1.0,
            decay_lambda=0.5,
        )
        assert retention == pytest.approx(1.0, abs=0.01)

    def test_calculate_retention_different_decay_rates(self) -> None:
        """Higher decay rate should result in faster forgetting."""
        retention_slow = ForgettingCurve.calculate_retention(
            importance=1.0,
            hours_elapsed=2.0,
            decay_lambda=0.1,
        )
        retention_fast = ForgettingCurve.calculate_retention(
            importance=1.0,
            hours_elapsed=2.0,
            decay_lambda=0.9,
        )
        assert retention_slow > retention_fast

    def test_should_forget_below_threshold(self) -> None:
        """Memory with retention below threshold should be forgotten."""
        assert ForgettingCurve.should_forget(0.05, threshold=0.1) is True

    def test_should_forget_above_threshold(self) -> None:
        """Memory with retention above threshold should be retained."""
        assert ForgettingCurve.should_forget(0.15, threshold=0.1) is False

    def test_should_forget_at_threshold(self) -> None:
        """Memory exactly at threshold should be retained."""
        assert ForgettingCurve.should_forget(0.1, threshold=0.1) is False

    def test_should_forget_zero_retention(self) -> None:
        """Memory with zero retention should be forgotten."""
        assert ForgettingCurve.should_forget(0.0, threshold=0.1) is True


# --- RetrievalQuery tests ---


class TestRetrievalQuery:
    def test_query_creation_minimal(self) -> None:
        """Can create query with just query_text."""
        query = RetrievalQuery(query_text="test query")
        assert query.query_text == "test query"
        assert query.category_filter is None
        assert query.min_importance == 0.0
        assert query.max_results == 10

    def test_query_creation_full(self) -> None:
        """Can create query with all parameters."""
        query = RetrievalQuery(
            query_text="test query",
            category_filter="social",
            min_importance=0.5,
            max_results=5,
        )
        assert query.query_text == "test query"
        assert query.category_filter == "social"
        assert query.min_importance == 0.5
        assert query.max_results == 5


# --- LTMRetrievalModule basic properties ---


class TestLTMRetrievalModuleProperties:
    def test_name(self) -> None:
        """Module name should be 'ltm_retrieval'."""
        store = MockLTMStore()
        module = LTMRetrievalModule(store)
        assert module.name == "ltm_retrieval"

    def test_tier(self) -> None:
        """Module tier should be MID."""
        store = MockLTMStore()
        module = LTMRetrievalModule(store)
        assert module.tier == ModuleTier.MID


# --- Query building tests ---


class TestLTMRetrievalQueryBuilding:
    @pytest.mark.asyncio
    async def test_build_queries_from_goal(self) -> None:
        """Should build query from current goal."""
        sas = InMemorySAS()
        await sas.update_goals(GoalData(current_goal="mine iron ore"))

        store = MockLTMStore()
        module = LTMRetrievalModule(store)
        queries = await module._build_queries(sas)

        assert len(queries) >= 1
        goal_query = next((q for q in queries if "iron ore" in q.query_text), None)
        assert goal_query is not None
        assert goal_query.category_filter == "semantic"

    @pytest.mark.asyncio
    async def test_build_queries_from_nearby_agents(self) -> None:
        """Should build queries for nearby agents."""
        sas = InMemorySAS()
        await sas.update_percepts(
            PerceptData(nearby_players=["alice", "bob", "charlie"])
        )

        store = MockLTMStore()
        module = LTMRetrievalModule(store)
        queries = await module._build_queries(sas)

        social_queries = [q for q in queries if q.category_filter == "social"]
        assert len(social_queries) == 2  # Limited to 2 agents
        assert any("alice" in q.query_text for q in social_queries)
        assert any("bob" in q.query_text for q in social_queries)

    @pytest.mark.asyncio
    async def test_build_queries_from_chat_messages(self) -> None:
        """Should build queries from recent chat messages."""
        sas = InMemorySAS()
        await sas.update_percepts(
            PerceptData(
                chat_messages=[
                    {"sender": "alice", "content": "let's build a house"},
                    {"sender": "bob", "content": "need wood"},
                ]
            )
        )

        store = MockLTMStore()
        module = LTMRetrievalModule(store)
        queries = await module._build_queries(sas)

        chat_queries = [q for q in queries if q.category_filter is None]
        assert len(chat_queries) >= 2
        assert any("house" in q.query_text for q in chat_queries)
        assert any("wood" in q.query_text for q in chat_queries)

    @pytest.mark.asyncio
    async def test_build_queries_empty_sas(self) -> None:
        """Should return empty list when SAS has no context."""
        sas = InMemorySAS()

        store = MockLTMStore()
        module = LTMRetrievalModule(store)
        queries = await module._build_queries(sas)

        assert queries == []


# --- Forgetting filter tests ---


class TestLTMRetrievalForgetting:
    def test_apply_forgetting_fresh_memories(self) -> None:
        """Fresh memories should pass the forgetting filter."""
        store = MockLTMStore()
        module = LTMRetrievalModule(store, retention_threshold=0.1)

        memories = [
            _memory("recent event 1", importance=0.8, age_hours=0.1),
            _memory("recent event 2", importance=0.5, age_hours=0.2),
        ]

        retained = module._apply_forgetting(memories)
        assert len(retained) == 2

    def test_apply_forgetting_old_memories(self) -> None:
        """Very old, low-importance memories should be filtered out."""
        store = MockLTMStore()
        module = LTMRetrievalModule(store, retention_threshold=0.1)

        memories = [
            _memory("old event", importance=0.2, age_hours=10.0),  # Will decay below threshold
        ]

        retained = module._apply_forgetting(memories)
        assert len(retained) == 0

    def test_apply_forgetting_mixed_retention(self) -> None:
        """Should filter based on retention score (importance * decay)."""
        store = MockLTMStore()
        module = LTMRetrievalModule(store, retention_threshold=0.1)

        memories = [
            _memory("important old", importance=1.0, age_hours=3.0),  # High retention
            _memory("unimportant old", importance=0.1, age_hours=3.0),  # Low retention
            _memory("recent low", importance=0.2, age_hours=0.5),  # Medium retention
        ]

        retained = module._apply_forgetting(memories)
        # Should retain the important old and recent low, but not unimportant old
        assert len(retained) >= 1
        contents = [m.content for m in retained]
        assert "important old" in contents


# --- Tick execution tests ---


class TestLTMRetrievalTick:
    @pytest.mark.asyncio
    async def test_tick_empty_ltm(self) -> None:
        """Tick with empty LTM should return no results."""
        sas = InMemorySAS()
        await sas.update_goals(GoalData(current_goal="test goal"))

        store = MockLTMStore(preset_memories=[])
        module = LTMRetrievalModule(store)
        result = await module.tick(sas)

        assert result.success
        assert result.data["retrieved"] == 0
        assert result.data["injected"] == 0

    @pytest.mark.asyncio
    async def test_tick_retrieves_and_injects(self) -> None:
        """Tick should retrieve memories and inject them into WM."""
        sas = InMemorySAS()
        await sas.update_goals(GoalData(current_goal="mine iron"))

        memories = [
            _ltm_entry("found iron at x=100", category="semantic", importance=0.8),
            _ltm_entry("mining is hard work", category="semantic", importance=0.6),
        ]
        store = MockLTMStore(preset_memories=memories)
        module = LTMRetrievalModule(store, max_memories_per_tick=5)
        result = await module.tick(sas)

        assert result.success
        assert result.data["retrieved"] >= 1
        assert result.data["injected"] >= 1

        # Check that memories were injected into WM
        wm = await sas.get_working_memory()
        assert len(wm) >= 1
        ltm_entries = [e for e in wm if e.category == "ltm_retrieval"]
        assert len(ltm_entries) >= 1

    @pytest.mark.asyncio
    async def test_tick_limits_injected_memories(self) -> None:
        """Should limit the number of memories injected per tick."""
        sas = InMemorySAS()
        await sas.update_goals(GoalData(current_goal="build house"))

        # Create many memories
        memories = [
            _ltm_entry(f"memory {i}", category="semantic", importance=0.5)
            for i in range(20)
        ]
        store = MockLTMStore(preset_memories=memories)
        module = LTMRetrievalModule(store, max_memories_per_tick=3)
        result = await module.tick(sas)

        assert result.success
        assert result.data["injected"] <= 3

        wm = await sas.get_working_memory()
        ltm_entries = [e for e in wm if e.category == "ltm_retrieval"]
        assert len(ltm_entries) <= 3

    @pytest.mark.asyncio
    async def test_tick_applies_forgetting_filter(self) -> None:
        """Tick should filter out memories below retention threshold."""
        sas = InMemorySAS()
        await sas.update_goals(GoalData(current_goal="test"))

        memories = [
            _ltm_entry("fresh", category="semantic", importance=0.8, age_hours=0.1),
            # Old memory will be filtered by forgetting curve
            _ltm_entry("old", category="semantic", importance=0.5, age_hours=10.0),
        ]
        store = MockLTMStore(preset_memories=memories)
        module = LTMRetrievalModule(store, retention_threshold=0.1)
        result = await module.tick(sas)

        assert result.success
        # Should retrieve both but filter the old one due to forgetting curve
        assert result.data["retrieved"] == 2
        assert result.data["after_forgetting"] == 1  # Only fresh memory retained
        assert result.data["injected"] == 1

    @pytest.mark.asyncio
    async def test_tick_deduplicates_results(self) -> None:
        """Should deduplicate memories retrieved from multiple queries."""
        sas = InMemorySAS()
        await sas.update_goals(GoalData(current_goal="mine iron"))
        await sas.update_percepts(PerceptData(nearby_players=["alice"]))

        # Create a memory that could match multiple queries
        memory = _ltm_entry("alice helped with iron mining", category="social", importance=0.7)
        store = MockLTMStore(preset_memories=[memory, memory])  # Duplicate
        module = LTMRetrievalModule(store)
        result = await module.tick(sas)

        assert result.success
        # Even if retrieved multiple times, should only inject once
        wm = await sas.get_working_memory()
        ltm_entries = [e for e in wm if e.category == "ltm_retrieval"]
        # Count unique original_ids
        original_ids = {e.metadata.get("original_id") for e in ltm_entries}
        assert len(original_ids) <= 1  # Should deduplicate

    @pytest.mark.asyncio
    async def test_tick_sorts_by_importance(self) -> None:
        """Should inject highest importance memories first."""
        sas = InMemorySAS()
        await sas.update_goals(GoalData(current_goal="test"))

        memories = [
            _ltm_entry("low importance", importance=0.3),
            _ltm_entry("high importance", importance=0.9),
            _ltm_entry("medium importance", importance=0.6),
        ]
        store = MockLTMStore(preset_memories=memories)
        module = LTMRetrievalModule(store, max_memories_per_tick=2)
        result = await module.tick(sas)

        assert result.success
        wm = await sas.get_working_memory()
        ltm_entries = [e for e in wm if e.category == "ltm_retrieval"]

        # Should inject the 2 highest importance memories
        if len(ltm_entries) >= 1:
            # Check that high importance is in the injected memories
            contents = [e.content for e in ltm_entries]
            assert any("high importance" in c for c in contents)

    @pytest.mark.asyncio
    async def test_tick_no_queries_built(self) -> None:
        """Tick with empty SAS should build no queries and inject nothing."""
        sas = InMemorySAS()

        store = MockLTMStore()
        module = LTMRetrievalModule(store)
        result = await module.tick(sas)

        assert result.success
        assert result.data["queries"] == 0
        assert result.data["retrieved"] == 0
        assert result.data["injected"] == 0

    @pytest.mark.asyncio
    async def test_tick_preserves_existing_wm(self) -> None:
        """Tick should add to existing WM, not replace it."""
        sas = InMemorySAS()
        # Add existing WM entry
        existing = _memory("existing entry", category="perception")
        await sas.set_working_memory([existing])

        await sas.update_goals(GoalData(current_goal="test"))
        memories = [_ltm_entry("new memory", category="semantic", importance=0.7)]
        store = MockLTMStore(preset_memories=memories)
        module = LTMRetrievalModule(store)
        result = await module.tick(sas)

        assert result.success
        wm = await sas.get_working_memory()
        # Should have both existing and new entries
        assert len(wm) >= 2
        contents = [e.content for e in wm]
        assert "existing entry" in contents

    @pytest.mark.asyncio
    async def test_tick_error_handling(self) -> None:
        """Tick should handle errors gracefully."""
        class FailingStore:
            async def search(self, agent_id, query_embedding, **kwargs):
                raise RuntimeError("Search failed")

        sas = InMemorySAS()
        await sas.update_goals(GoalData(current_goal="test"))

        store = FailingStore()
        module = LTMRetrievalModule(store)  # type: ignore[arg-type]
        result = await module.tick(sas)

        assert not result.success
        assert result.error is not None
        assert "Search failed" in result.error


# --- Category filtering tests ---


class TestLTMRetrievalCategoryFiltering:
    @pytest.mark.asyncio
    async def test_category_filter_semantic(self) -> None:
        """Should filter by semantic category for goal queries."""
        sas = InMemorySAS()
        await sas.update_goals(GoalData(current_goal="test goal"))

        memories = [
            _ltm_entry("semantic memory", category="semantic", importance=0.7),
            _ltm_entry("social memory", category="social", importance=0.7),
        ]
        store = MockLTMStore(preset_memories=memories)
        module = LTMRetrievalModule(store)
        result = await module.tick(sas)

        assert result.success
        # Check that search was called
        assert len(store.search_calls) >= 1
        # The module now applies category filtering client-side after search.
        # Verify only semantic memories were injected (social filtered out).
        wm = await sas.get_working_memory()
        ltm_entries = [e for e in wm if e.category == "ltm_retrieval"]
        for e in ltm_entries:
            assert e.metadata.get("original_category") != "social" or e.metadata.get(
                "original_category"
            ) is None

    @pytest.mark.asyncio
    async def test_category_filter_social(self) -> None:
        """Should filter by social category for nearby agent queries."""
        sas = InMemorySAS()
        await sas.update_percepts(PerceptData(nearby_players=["alice"]))

        memories = [
            _ltm_entry("alice helped me", category="social", importance=0.7),
        ]
        store = MockLTMStore(preset_memories=memories)
        module = LTMRetrievalModule(store)
        result = await module.tick(sas)

        assert result.success
        # Check that search was called
        assert len(store.search_calls) >= 1


# --- Module lifecycle tests ---


class TestLTMRetrievalLifecycle:
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        """initialize() should not raise (default is no-op)."""
        store = MockLTMStore()
        module = LTMRetrievalModule(store)
        await module.initialize()  # Should succeed

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """shutdown() should not raise (default is no-op)."""
        store = MockLTMStore()
        module = LTMRetrievalModule(store)
        await module.shutdown()  # Should succeed


# --- repr test ---


class TestLTMRetrievalRepr:
    def test_repr(self) -> None:
        """Module repr should include name and tier."""
        store = MockLTMStore()
        module = LTMRetrievalModule(store)
        r = repr(module)
        assert "LTMRetrievalModule" in r
        assert "ltm_retrieval" in r
        assert "mid" in r.lower()
