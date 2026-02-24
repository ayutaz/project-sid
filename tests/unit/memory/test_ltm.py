"""Tests for Long-Term Memory (LTM) store."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np
import pytest

from piano.memory.ltm import InMemoryLTMStore, LTMEntry

# --- Fixtures ---


@pytest.fixture
def store() -> InMemoryLTMStore:
    """Create an in-memory LTM store for testing."""
    return InMemoryLTMStore()


@pytest.fixture
def sample_entry() -> LTMEntry:
    """Create a sample LTM entry with embedding."""
    return LTMEntry(
        content="Agent met Lila near the farm",
        category="episodic",
        importance=0.7,
        source_module="social_awareness",
        metadata={"location": "farm", "other_agent": "Lila"},
        embedding=[0.1, 0.2, 0.3, 0.4],
    )


@pytest.fixture
def sample_entries() -> list[LTMEntry]:
    """Create multiple sample entries with different embeddings."""
    return [
        LTMEntry(
            content="Mining iron ore near the cave",
            category="episodic",
            importance=0.6,
            source_module="action_awareness",
            embedding=[1.0, 0.0, 0.0, 0.0],  # Orthogonal vectors
        ),
        LTMEntry(
            content="Talking to Noah about farming",
            category="episodic",
            importance=0.5,
            source_module="social_awareness",
            embedding=[0.0, 1.0, 0.0, 0.0],
        ),
        LTMEntry(
            content="Learned that iron needs coal for smelting",
            category="semantic",
            importance=0.8,
            source_module="self_reflection",
            embedding=[0.707, 0.707, 0.0, 0.0],  # Similar to first
        ),
        LTMEntry(
            content="Built a house with wooden planks",
            category="procedural",
            importance=0.4,
            source_module="action_awareness",
            embedding=[0.0, 0.0, 1.0, 0.0],
        ),
    ]


# --- Basic CRUD Tests ---


async def test_store_and_retrieve_single_entry(
    store: InMemoryLTMStore,
    sample_entry: LTMEntry,
) -> None:
    """Test storing and retrieving a single entry."""
    await store.initialize()

    agent_id = "agent_001"
    entry_id = await store.store(agent_id, sample_entry)

    assert entry_id == sample_entry.id

    retrieved = await store.retrieve(agent_id, entry_id)
    assert retrieved is not None
    assert retrieved.id == sample_entry.id
    assert retrieved.content == sample_entry.content
    assert retrieved.category == sample_entry.category
    assert retrieved.importance == sample_entry.importance
    assert retrieved.embedding == sample_entry.embedding


async def test_retrieve_nonexistent_entry(store: InMemoryLTMStore) -> None:
    """Test retrieving an entry that doesn't exist."""
    await store.initialize()

    agent_id = "agent_001"
    nonexistent_id = uuid4()

    retrieved = await store.retrieve(agent_id, nonexistent_id)
    assert retrieved is None


async def test_delete_entry(
    store: InMemoryLTMStore,
    sample_entry: LTMEntry,
) -> None:
    """Test deleting an entry."""
    await store.initialize()

    agent_id = "agent_001"
    entry_id = await store.store(agent_id, sample_entry)

    # Delete the entry
    deleted = await store.delete(agent_id, entry_id)
    assert deleted is True

    # Verify it's gone
    retrieved = await store.retrieve(agent_id, entry_id)
    assert retrieved is None


async def test_delete_nonexistent_entry(store: InMemoryLTMStore) -> None:
    """Test deleting an entry that doesn't exist."""
    await store.initialize()

    agent_id = "agent_001"
    nonexistent_id = uuid4()

    deleted = await store.delete(agent_id, nonexistent_id)
    assert deleted is False


# --- Search Tests ---


async def test_search_empty_store(store: InMemoryLTMStore) -> None:
    """Test searching in an empty store returns empty list."""
    await store.initialize()

    agent_id = "agent_001"
    query_embedding = [1.0, 0.0, 0.0, 0.0]

    results = await store.search(agent_id, query_embedding, limit=10)
    assert results == []


async def test_search_by_similarity(
    store: InMemoryLTMStore,
    sample_entries: list[LTMEntry],
) -> None:
    """Test searching by cosine similarity."""
    await store.initialize()

    agent_id = "agent_001"

    # Store all entries
    for entry in sample_entries:
        await store.store(agent_id, entry)

    # Query similar to first entry (mining)
    query_embedding = [1.0, 0.0, 0.0, 0.0]
    results = await store.search(agent_id, query_embedding, limit=5)

    # Should return results, with mining-related memories first
    assert len(results) > 0
    # First result should be the exact match
    assert results[0].content == "Mining iron ore near the cave"


async def test_cosine_similarity_returns_closest_matches(
    store: InMemoryLTMStore,
    sample_entries: list[LTMEntry],
) -> None:
    """Test that cosine similarity search returns closest matches in order."""
    await store.initialize()

    agent_id = "agent_001"

    # Store all entries
    for entry in sample_entries:
        await store.store(agent_id, entry)

    # Query similar to first entry [1.0, 0.0, 0.0, 0.0]
    query_embedding = [1.0, 0.0, 0.0, 0.0]
    results = await store.search(agent_id, query_embedding, limit=10)

    assert len(results) == 4  # All entries

    # First should be exact match (mining)
    assert results[0].content == "Mining iron ore near the cave"

    # Second should be the one with [0.707, 0.707, 0.0, 0.0] (similar)
    assert results[1].content == "Learned that iron needs coal for smelting"

    # Calculate expected similarities to verify ordering
    def cosine_sim(a: list[float], b: list[float]) -> float:
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10))

    query = [1.0, 0.0, 0.0, 0.0]

    # Verify results are sorted by similarity
    for i in range(len(results) - 1):
        result_emb_i = results[i].embedding or [0.0]
        result_emb_j = results[i + 1].embedding or [0.0]
        sim_i = cosine_sim(query, result_emb_i)
        sim_j = cosine_sim(query, result_emb_j)
        assert sim_i >= sim_j


async def test_search_with_min_score(
    store: InMemoryLTMStore,
    sample_entries: list[LTMEntry],
) -> None:
    """Test searching with minimum similarity threshold."""
    await store.initialize()

    agent_id = "agent_001"

    # Store all entries
    for entry in sample_entries:
        await store.store(agent_id, entry)

    # Query with high threshold
    query_embedding = [1.0, 0.0, 0.0, 0.0]
    results = await store.search(agent_id, query_embedding, limit=10, min_score=0.9)

    # Should only return exact match or very close matches
    assert len(results) >= 1
    assert all(
        np.dot(np.array(query_embedding), np.array(r.embedding or [0.0]))
        / (np.linalg.norm(query_embedding) * np.linalg.norm(r.embedding or [0.0]) + 1e-10)
        >= 0.9
        for r in results
    )


async def test_search_with_limit(
    store: InMemoryLTMStore,
    sample_entries: list[LTMEntry],
) -> None:
    """Test that search respects the limit parameter."""
    await store.initialize()

    agent_id = "agent_001"

    # Store all entries
    for entry in sample_entries:
        await store.store(agent_id, entry)

    # Search with limit=2
    query_embedding = [1.0, 0.0, 0.0, 0.0]
    results = await store.search(agent_id, query_embedding, limit=2)

    assert len(results) == 2


async def test_search_excludes_entries_without_embeddings(
    store: InMemoryLTMStore,
) -> None:
    """Test that search skips entries without embeddings."""
    await store.initialize()

    agent_id = "agent_001"

    # Store entry without embedding
    entry_no_emb = LTMEntry(
        content="Entry without embedding",
        category="episodic",
        importance=0.5,
        embedding=None,
    )
    await store.store(agent_id, entry_no_emb)

    # Store entry with embedding
    entry_with_emb = LTMEntry(
        content="Entry with embedding",
        category="episodic",
        importance=0.5,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    await store.store(agent_id, entry_with_emb)

    # Search
    query_embedding = [1.0, 0.0, 0.0, 0.0]
    results = await store.search(agent_id, query_embedding, limit=10)

    # Should only return the one with embedding
    assert len(results) == 1
    assert results[0].content == "Entry with embedding"


# --- Agent Isolation Tests ---


async def test_agent_isolation(
    store: InMemoryLTMStore,
    sample_entry: LTMEntry,
) -> None:
    """Test that different agents don't see each other's data."""
    await store.initialize()

    agent_1 = "agent_001"
    agent_2 = "agent_002"

    # Store entry for agent_1
    entry_id = await store.store(agent_1, sample_entry)

    # Agent_1 can retrieve it
    retrieved_1 = await store.retrieve(agent_1, entry_id)
    assert retrieved_1 is not None

    # Agent_2 cannot retrieve it
    retrieved_2 = await store.retrieve(agent_2, entry_id)
    assert retrieved_2 is None

    # Agent_2's search doesn't find agent_1's entry
    query_embedding = sample_entry.embedding or [0.0]
    results = await store.search(agent_2, query_embedding, limit=10)
    assert len(results) == 0


async def test_agent_isolated_stats(
    store: InMemoryLTMStore,
    sample_entries: list[LTMEntry],
) -> None:
    """Test that stats are agent-specific."""
    await store.initialize()

    agent_1 = "agent_001"
    agent_2 = "agent_002"

    # Store entries for agent_1
    for entry in sample_entries[:2]:
        await store.store(agent_1, entry)

    # Store entries for agent_2
    for entry in sample_entries[2:]:
        await store.store(agent_2, entry)

    stats_1 = await store.get_stats(agent_1)
    stats_2 = await store.get_stats(agent_2)

    assert stats_1["count"] == 2
    assert stats_2["count"] == 2


# --- Statistics Tests ---


async def test_get_stats_empty_agent(store: InMemoryLTMStore) -> None:
    """Test stats for agent with no memories."""
    await store.initialize()

    stats = await store.get_stats("agent_001")

    assert stats["count"] == 0
    assert stats["avg_importance"] == 0.0
    assert stats["total_accesses"] == 0


async def test_get_stats_returns_correct_counts(
    store: InMemoryLTMStore,
    sample_entries: list[LTMEntry],
) -> None:
    """Test that get_stats returns correct counts and averages."""
    await store.initialize()

    agent_id = "agent_001"

    # Store entries
    for entry in sample_entries:
        await store.store(agent_id, entry)

    stats = await store.get_stats(agent_id)

    assert stats["count"] == 4
    # Calculate expected average importance
    expected_avg = sum(e.importance for e in sample_entries) / len(sample_entries)
    assert stats["avg_importance"] == pytest.approx(expected_avg)
    assert stats["total_accesses"] == 0  # No accesses yet


async def test_get_stats_with_accesses(
    store: InMemoryLTMStore,
    sample_entry: LTMEntry,
) -> None:
    """Test that stats count access_count correctly."""
    await store.initialize()

    agent_id = "agent_001"

    # Update access count
    sample_entry.update_access()
    sample_entry.update_access()

    await store.store(agent_id, sample_entry)

    stats = await store.get_stats(agent_id)

    assert stats["total_accesses"] == 2


# --- Importance-based Filtering ---


async def test_importance_based_search(
    store: InMemoryLTMStore,
    sample_entries: list[LTMEntry],
) -> None:
    """Test filtering memories by importance (implicit via min_score)."""
    await store.initialize()

    agent_id = "agent_001"

    # Store entries with varying importance
    for entry in sample_entries:
        await store.store(agent_id, entry)

    # Verify we have entries with different importance levels
    high_importance = [e for e in sample_entries if e.importance >= 0.7]
    assert len(high_importance) > 0

    # Search with query that matches multiple entries
    # Using a query that's similar to multiple entries
    query_embedding = [0.5, 0.5, 0.5, 0.5]
    results = await store.search(agent_id, query_embedding, limit=10, min_score=0.0)

    # All stored entries should be returned
    assert len(results) == len(sample_entries)


# --- LTMEntry Methods Tests ---


def test_ltm_entry_update_access() -> None:
    """Test updating access statistics."""
    entry = LTMEntry(content="Test", category="episodic", importance=0.5)

    assert entry.access_count == 0
    assert entry.last_accessed is None

    before_update = datetime.now(UTC)
    entry.update_access()

    assert entry.access_count == 1
    assert entry.last_accessed is not None
    assert entry.last_accessed >= before_update

    entry.update_access()
    assert entry.access_count == 2


def test_ltm_entry_decay_factor() -> None:
    """Test decay factor calculation."""
    # Create entry with known timestamp
    past_time = datetime.now(UTC) - timedelta(hours=2)
    entry = LTMEntry(
        content="Test",
        category="episodic",
        importance=0.8,
        timestamp=past_time,
        access_count=5,
    )

    decay = entry.calculate_decay_factor()

    # Verify decay is less than importance due to time decay
    assert decay < entry.importance
    assert decay > 0

    # Verify recency component (exp(-0.5 * 2) ≈ 0.368)
    expected_recency = math.exp(-0.5 * 2)
    # Access weight: log(1+5) / log(1+100) ≈ 0.389
    expected_access = math.log(1 + 5) / math.log(1 + 100)
    expected_decay = 0.8 * expected_recency * (0.5 + 0.5 * expected_access)

    assert decay == pytest.approx(expected_decay, rel=1e-5)


def test_ltm_entry_decay_factor_fresh_entry() -> None:
    """Test decay factor for a fresh entry (just created)."""
    entry = LTMEntry(
        content="Test",
        category="episodic",
        importance=1.0,
        access_count=0,
    )

    decay = entry.calculate_decay_factor()

    # Fresh entry with no accesses: importance * 1.0 * 0.5
    assert decay == pytest.approx(0.5, rel=1e-5)


def test_ltm_entry_decay_factor_with_high_accesses() -> None:
    """Test decay factor increases with access count."""
    entry = LTMEntry(
        content="Test",
        category="episodic",
        importance=0.5,
        access_count=0,
    )

    decay_no_access = entry.calculate_decay_factor()

    entry.access_count = 50
    decay_with_access = entry.calculate_decay_factor()

    # Higher access count should increase decay factor
    assert decay_with_access > decay_no_access


# --- Multiple Entries and Complex Scenarios ---


async def test_store_multiple_entries_and_search(
    store: InMemoryLTMStore,
    sample_entries: list[LTMEntry],
) -> None:
    """Test storing multiple entries and searching across them."""
    await store.initialize()

    agent_id = "agent_001"

    # Store all entries
    stored_ids = []
    for entry in sample_entries:
        entry_id = await store.store(agent_id, entry)
        stored_ids.append(entry_id)

    assert len(stored_ids) == len(sample_entries)

    # Retrieve each one
    for entry_id in stored_ids:
        retrieved = await store.retrieve(agent_id, entry_id)
        assert retrieved is not None

    # Search returns results
    query_embedding = [1.0, 0.0, 0.0, 0.0]
    results = await store.search(agent_id, query_embedding, limit=10)
    assert len(results) > 0


async def test_update_existing_entry(
    store: InMemoryLTMStore,
    sample_entry: LTMEntry,
) -> None:
    """Test updating an existing entry (re-store with same ID)."""
    await store.initialize()

    agent_id = "agent_001"

    # Store original
    entry_id = await store.store(agent_id, sample_entry)

    # Modify and re-store
    sample_entry.content = "Updated content"
    sample_entry.importance = 0.9
    await store.store(agent_id, sample_entry)

    # Retrieve should get updated version
    retrieved = await store.retrieve(agent_id, entry_id)
    assert retrieved is not None
    assert retrieved.content == "Updated content"
    assert retrieved.importance == 0.9


# --- Edge Cases ---


async def test_search_with_zero_vector(store: InMemoryLTMStore) -> None:
    """Test searching with a zero query vector."""
    await store.initialize()

    agent_id = "agent_001"

    entry = LTMEntry(
        content="Test",
        category="episodic",
        importance=0.5,
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    await store.store(agent_id, entry)

    # Search with zero vector
    query_embedding = [0.0, 0.0, 0.0, 0.0]
    results = await store.search(agent_id, query_embedding, limit=10)

    # Should handle gracefully (division by zero protection)
    # Result depends on implementation - likely returns all or none
    assert isinstance(results, list)


async def test_empty_metadata(store: InMemoryLTMStore) -> None:
    """Test entry with empty metadata."""
    await store.initialize()

    agent_id = "agent_001"

    entry = LTMEntry(
        content="Test",
        category="episodic",
        importance=0.5,
        metadata={},
        embedding=[1.0, 0.0, 0.0, 0.0],
    )

    entry_id = await store.store(agent_id, entry)
    retrieved = await store.retrieve(agent_id, entry_id)

    assert retrieved is not None
    assert retrieved.metadata == {}


# --- InMemoryLTMStore Capacity Limit Tests ---


async def test_inmemory_store_capacity_limit() -> None:
    """Test that InMemoryLTMStore evicts oldest entries when at max_entries."""
    store = InMemoryLTMStore(max_entries=3)
    await store.initialize()

    agent_id = "agent_001"

    entries = []
    for i in range(5):
        ts = datetime.now(UTC) - timedelta(hours=5 - i)  # Oldest first
        entry = LTMEntry(
            content=f"Memory {i}",
            category="episodic",
            importance=0.5,
            timestamp=ts,
            embedding=[float(i), 0.0, 0.0, 0.0],
        )
        entries.append(entry)
        await store.store(agent_id, entry)

    stats = await store.get_stats(agent_id)
    assert stats["count"] == 3  # Capped at max_entries

    # Oldest entries (0, 1) should be evicted
    assert await store.retrieve(agent_id, entries[0].id) is None
    assert await store.retrieve(agent_id, entries[1].id) is None
    # Newest entries (2, 3, 4) should remain
    assert await store.retrieve(agent_id, entries[2].id) is not None
    assert await store.retrieve(agent_id, entries[3].id) is not None
    assert await store.retrieve(agent_id, entries[4].id) is not None


async def test_inmemory_store_update_does_not_evict() -> None:
    """Updating an existing entry should not trigger eviction."""
    store = InMemoryLTMStore(max_entries=2)
    await store.initialize()

    agent_id = "agent_001"

    e1 = LTMEntry(content="First", category="episodic", importance=0.5, embedding=[1.0, 0.0])
    e2 = LTMEntry(content="Second", category="episodic", importance=0.5, embedding=[0.0, 1.0])
    await store.store(agent_id, e1)
    await store.store(agent_id, e2)

    # Update e1 (same ID)
    e1.content = "First updated"
    await store.store(agent_id, e1)

    stats = await store.get_stats(agent_id)
    assert stats["count"] == 2  # No extra eviction
    retrieved = await store.retrieve(agent_id, e1.id)
    assert retrieved is not None
    assert retrieved.content == "First updated"


async def test_inmemory_store_default_capacity() -> None:
    """Default max_entries should be 10000."""
    store = InMemoryLTMStore()
    assert store._max_entries == 10000


# --- Async QdrantLTMStore Tests (mocked) ---


async def test_qdrant_store_uses_async_client() -> None:
    """QdrantLTMStore should import and use AsyncQdrantClient."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from piano.memory.ltm import QdrantLTMStore

    store = QdrantLTMStore(url="http://localhost:6333")

    mock_client = MagicMock()
    mock_client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    mock_client.create_collection = AsyncMock()
    mock_client.upsert = AsyncMock()
    mock_client.close = AsyncMock()

    # Patch inside qdrant_client module where it's imported from
    with patch("qdrant_client.AsyncQdrantClient", return_value=mock_client):
        await store.initialize()

    # Verify client was set
    assert store._client is mock_client


async def test_qdrant_ensure_collection_idempotent() -> None:
    """_ensure_collection should handle race conditions gracefully."""
    from unittest.mock import AsyncMock, MagicMock

    from piano.memory.ltm import QdrantLTMStore

    store = QdrantLTMStore(url="http://localhost:6333")

    # Mock async client
    mock_client = MagicMock()
    # First call: collection does not exist
    mock_collections = MagicMock()
    mock_collections.collections = []

    # Create a proper mock for the collection that exists after race
    existing_collection = MagicMock()
    existing_collection.name = "ltm_test_agent"  # Set .name as attribute, not mock name
    mock_collections_after = MagicMock()
    mock_collections_after.collections = [existing_collection]

    mock_client.get_collections = AsyncMock(
        side_effect=[
            mock_collections,  # First check: not exists
            mock_collections_after,  # After race: exists
        ]
    )
    # create_collection raises to simulate race condition
    mock_client.create_collection = AsyncMock(side_effect=Exception("Collection already exists"))

    store._client = mock_client

    # Should not raise due to race condition handling
    await store._ensure_collection("test_agent")

    # Verify get_collections was called twice (once to check, once to verify after exception)
    assert mock_client.get_collections.call_count == 2
