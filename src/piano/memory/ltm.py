"""Long-Term Memory (LTM) for the PIANO architecture.

LTM stores memories with vector embeddings for semantic search using Qdrant.
Supports agent-partitioned collections, importance scoring, and decay.

Reference: docs/implementation/04-memory-system.md Section 4.2.3
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

import numpy as np
import structlog
from pydantic import BaseModel, Field

__all__ = [
    "InMemoryLTMStore",
    "LTMEntry",
    "LTMStore",
    "QdrantLTMStore",
]

logger = structlog.get_logger(__name__)


# --- LTM Entry Model ---


class LTMEntry(BaseModel):
    """A single long-term memory entry with vector embedding."""

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content: str = ""
    category: str = ""  # "episodic", "semantic", "procedural"
    importance: float = 0.5  # 0.0 - 1.0
    source_module: str = ""
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
    embedding: list[float] | None = None
    access_count: int = 0
    last_accessed: datetime | None = None

    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now(UTC)

    def calculate_decay_factor(self, now: datetime | None = None) -> float:
        """Calculate decay factor based on recency and access count.

        decay_factor = importance * recency_weight * access_weight

        where:
        - recency_weight = exp(-0.5 * hours_since_creation)
        - access_weight = log(1 + access_count) / log(1 + max_access)
          (assuming max_access = 100 for normalization)
        """
        if now is None:
            now = datetime.now(UTC)

        # Recency weight (exponential decay, lambda=0.5, half-life ~1.4 hours)
        age_hours = max((now - self.timestamp).total_seconds() / 3600.0, 0.0)
        recency_weight = math.exp(-0.5 * age_hours)

        # Access weight (logarithmic scaling)
        max_access = 100  # normalization constant
        access_weight = (
            math.log(1 + self.access_count) / math.log(1 + max_access)
            if self.access_count > 0
            else 0.0
        )

        return self.importance * recency_weight * (0.5 + 0.5 * access_weight)


# --- LTM Store Protocol ---


class LTMStore(Protocol):
    """Protocol for long-term memory storage implementations."""

    async def initialize(self) -> None:
        """Initialize the store (create collections, etc.)."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the store and cleanup resources."""
        ...

    async def store(self, agent_id: str, entry: LTMEntry) -> UUID:
        """Store a memory entry for an agent.

        Returns:
            UUID of the stored entry.
        """
        ...

    async def retrieve(self, agent_id: str, entry_id: UUID) -> LTMEntry | None:
        """Retrieve a specific memory entry by ID.

        Returns:
            The entry if found, None otherwise.
        """
        ...

    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[LTMEntry]:
        """Search for similar memories using vector similarity.

        Args:
            agent_id: The agent whose memories to search.
            query_embedding: The query vector.
            limit: Maximum number of results to return.
            min_score: Minimum similarity score threshold.

        Returns:
            List of matching entries ordered by similarity (highest first).
        """
        ...

    async def delete(self, agent_id: str, entry_id: UUID) -> bool:
        """Delete a memory entry.

        Returns:
            True if the entry was deleted, False if not found.
        """
        ...

    async def get_stats(self, agent_id: str) -> dict[str, float | int]:
        """Get statistics for an agent's memories.

        Returns:
            Dictionary with keys: count, avg_importance, total_accesses, etc.
        """
        ...


# --- Qdrant Implementation ---


class QdrantLTMStore:
    """Qdrant-based LTM store with vector search capabilities.

    Uses agent-partitioned collections for isolation.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_prefix: str = "ltm",
        embedding_dim: int = 384,
        use_https: bool = False,
        api_key: str | None = None,
    ) -> None:
        """Initialize the Qdrant LTM store.

        Args:
            url: Qdrant server URL.
            collection_prefix: Prefix for collection names.
            embedding_dim: Dimension of embedding vectors (default: 384 for MiniLM).
            use_https: Whether to connect to Qdrant over HTTPS.
            api_key: Optional API key for Qdrant authentication.
        """
        self.url = url
        self.collection_prefix = collection_prefix
        self.embedding_dim = embedding_dim
        self.use_https = use_https
        self.api_key = api_key
        self._client: object | None = None  # qdrant_client.QdrantClient
        logger.info(
            "qdrant_ltm_init",
            url=url,
            prefix=collection_prefix,
            dim=embedding_dim,
        )

    async def initialize(self) -> None:
        """Initialize Qdrant client and create collections."""
        try:
            from qdrant_client import QdrantClient

            kwargs: dict[str, object] = {"url": self.url}
            if self.use_https:
                kwargs["https"] = True
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = QdrantClient(**kwargs)

            # Note: Collections are created on-demand per agent in store()
            logger.info("qdrant_initialized", url=self.url)
        except ImportError:
            logger.error("qdrant_import_error", msg="qdrant-client not installed")
            raise

    async def shutdown(self) -> None:
        """Shutdown the Qdrant client."""
        if self._client is not None:
            # Qdrant client doesn't require explicit shutdown
            self._client = None
            logger.info("qdrant_shutdown")

    def _collection_name(self, agent_id: str) -> str:
        """Generate collection name for an agent."""
        return f"{self.collection_prefix}_{agent_id}"

    async def _ensure_collection(self, agent_id: str) -> None:
        """Ensure collection exists for agent."""
        if self._client is None:
            raise RuntimeError("QdrantClient not initialized")

        from qdrant_client.models import Distance, VectorParams

        collection_name = self._collection_name(agent_id)
        collections = self._client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("qdrant_collection_created", collection=collection_name)

    async def store(self, agent_id: str, entry: LTMEntry) -> UUID:
        """Store a memory entry in Qdrant."""
        if self._client is None:
            raise RuntimeError("QdrantClient not initialized")

        await self._ensure_collection(agent_id)

        from qdrant_client.models import PointStruct

        if entry.embedding is None:
            raise ValueError("LTMEntry must have an embedding before storing")

        point = PointStruct(
            id=str(entry.id),
            vector=entry.embedding,
            payload={
                "timestamp": entry.timestamp.isoformat(),
                "content": entry.content,
                "category": entry.category,
                "importance": entry.importance,
                "source_module": entry.source_module,
                "metadata": entry.metadata,
                "access_count": entry.access_count,
                "last_accessed": (
                    entry.last_accessed.isoformat() if entry.last_accessed is not None else None
                ),
            },
        )

        self._client.upsert(
            collection_name=self._collection_name(agent_id),
            points=[point],
        )

        logger.debug(
            "ltm_stored",
            agent_id=agent_id,
            entry_id=str(entry.id),
            category=entry.category,
        )
        return entry.id

    async def retrieve(self, agent_id: str, entry_id: UUID) -> LTMEntry | None:
        """Retrieve a specific memory by ID."""
        if self._client is None:
            raise RuntimeError("QdrantClient not initialized")

        try:
            point = self._client.retrieve(
                collection_name=self._collection_name(agent_id),
                ids=[str(entry_id)],
            )

            if not point:
                return None

            p = point[0]
            return LTMEntry(
                id=UUID(p.id),
                timestamp=datetime.fromisoformat(p.payload["timestamp"]),
                content=p.payload["content"],
                category=p.payload["category"],
                importance=p.payload["importance"],
                source_module=p.payload["source_module"],
                metadata=p.payload["metadata"],
                embedding=p.vector,
                access_count=p.payload.get("access_count", 0),
                last_accessed=(
                    datetime.fromisoformat(p.payload["last_accessed"])
                    if p.payload.get("last_accessed")
                    else None
                ),
            )
        except Exception:
            # Collection doesn't exist or other error
            logger.warning(
                "ltm_retrieve_error",
                agent_id=agent_id,
                entry_id=str(entry_id),
            )
            return None

    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[LTMEntry]:
        """Search for similar memories."""
        if self._client is None:
            raise RuntimeError("QdrantClient not initialized")

        try:
            results = self._client.search(
                collection_name=self._collection_name(agent_id),
                query_vector=query_embedding,
                limit=limit,
                score_threshold=min_score,
            )

            entries = []
            for r in results:
                entry = LTMEntry(
                    id=UUID(r.id),
                    timestamp=datetime.fromisoformat(r.payload["timestamp"]),
                    content=r.payload["content"],
                    category=r.payload["category"],
                    importance=r.payload["importance"],
                    source_module=r.payload["source_module"],
                    metadata=r.payload["metadata"],
                    embedding=r.vector if hasattr(r, "vector") else None,
                    access_count=r.payload.get("access_count", 0),
                    last_accessed=(
                        datetime.fromisoformat(r.payload["last_accessed"])
                        if r.payload.get("last_accessed")
                        else None
                    ),
                )
                entries.append(entry)

            logger.debug(
                "ltm_search",
                agent_id=agent_id,
                results=len(entries),
                limit=limit,
            )
            return entries

        except Exception as e:
            logger.error("ltm_search_error", agent_id=agent_id, error=str(e))
            return []

    async def delete(self, agent_id: str, entry_id: UUID) -> bool:
        """Delete a memory entry."""
        if self._client is None:
            raise RuntimeError("QdrantClient not initialized")

        try:
            self._client.delete(
                collection_name=self._collection_name(agent_id),
                points_selector=[str(entry_id)],
            )
            logger.debug("ltm_deleted", agent_id=agent_id, entry_id=str(entry_id))
            return True
        except Exception:
            return False

    async def get_stats(self, agent_id: str) -> dict[str, float | int]:
        """Get statistics for an agent's memories."""
        if self._client is None:
            raise RuntimeError("QdrantClient not initialized")

        try:
            collection_info = self._client.get_collection(
                collection_name=self._collection_name(agent_id)
            )
            count = collection_info.points_count

            # For avg_importance, we'd need to fetch all points (expensive)
            # For now, return basic stats
            return {
                "count": count,
                "avg_importance": 0.0,  # Would require full scan
                "total_accesses": 0,  # Would require full scan
            }
        except Exception:
            return {
                "count": 0,
                "avg_importance": 0.0,
                "total_accesses": 0,
            }


# --- In-Memory Implementation (for testing) ---


class InMemoryLTMStore:
    """In-memory LTM store with cosine similarity search.

    Used for testing without Qdrant dependency.
    """

    def __init__(self) -> None:
        """Initialize the in-memory store."""
        # agent_id -> {entry_id -> LTMEntry}
        self._store: dict[str, dict[UUID, LTMEntry]] = {}
        logger.info("inmemory_ltm_init")

    async def initialize(self) -> None:
        """Initialize (no-op for in-memory)."""
        pass

    async def shutdown(self) -> None:
        """Shutdown (no-op for in-memory)."""
        pass

    async def store(self, agent_id: str, entry: LTMEntry) -> UUID:
        """Store a memory entry in memory."""
        if agent_id not in self._store:
            self._store[agent_id] = {}

        self._store[agent_id][entry.id] = entry
        logger.debug(
            "ltm_stored",
            agent_id=agent_id,
            entry_id=str(entry.id),
            category=entry.category,
        )
        return entry.id

    async def retrieve(self, agent_id: str, entry_id: UUID) -> LTMEntry | None:
        """Retrieve a specific memory by ID."""
        if agent_id not in self._store:
            return None
        return self._store[agent_id].get(entry_id)

    async def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[LTMEntry]:
        """Search for similar memories using cosine similarity."""
        if agent_id not in self._store:
            return []

        query_vec = np.array(query_embedding)
        results: list[tuple[float, LTMEntry]] = []

        for entry in self._store[agent_id].values():
            if entry.embedding is None:
                continue

            # Cosine similarity
            entry_vec = np.array(entry.embedding)
            similarity = float(
                np.dot(query_vec, entry_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(entry_vec) + 1e-10)
            )

            if similarity >= min_score:
                results.append((similarity, entry))

        # Sort by similarity (highest first) and return top-k
        results.sort(key=lambda x: x[0], reverse=True)
        entries = [entry for _, entry in results[:limit]]

        logger.debug(
            "ltm_search",
            agent_id=agent_id,
            results=len(entries),
            limit=limit,
        )
        return entries

    async def delete(self, agent_id: str, entry_id: UUID) -> bool:
        """Delete a memory entry."""
        if agent_id not in self._store:
            return False

        if entry_id in self._store[agent_id]:
            del self._store[agent_id][entry_id]
            logger.debug("ltm_deleted", agent_id=agent_id, entry_id=str(entry_id))
            return True

        return False

    async def get_stats(self, agent_id: str) -> dict[str, float | int]:
        """Get statistics for an agent's memories."""
        if agent_id not in self._store:
            return {
                "count": 0,
                "avg_importance": 0.0,
                "total_accesses": 0,
            }

        entries = list(self._store[agent_id].values())
        count = len(entries)

        if count == 0:
            return {
                "count": 0,
                "avg_importance": 0.0,
                "total_accesses": 0,
            }

        avg_importance = sum(e.importance for e in entries) / count
        total_accesses = sum(e.access_count for e in entries)

        return {
            "count": count,
            "avg_importance": avg_importance,
            "total_accesses": total_accesses,
        }
