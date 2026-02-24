"""LTM Retrieval Module for the PIANO architecture.

This module searches long-term memory for relevant context based on the agent's
current situation. It generates search queries from percepts, goals, and recent
interactions, retrieves relevant memories, and injects them into working memory.

Implements the forgetting curve (exponential decay) to filter out low-retention
memories based on time elapsed and importance.

Reference: docs/implementation/04-memory-system.md Section 4.6
"""

from __future__ import annotations

__all__ = [
    "ForgettingCurve",
    "LTMRetrievalModule",
    "RetrievalQuery",
]

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel

from piano.core.module import Module
from piano.core.types import MemoryEntry, ModuleResult, ModuleTier

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState
    from piano.memory.ltm import LTMEntry, LTMStore


class RetrievalQuery(BaseModel):
    """A query for retrieving memories from LTM."""

    query_text: str
    category_filter: str | None = None
    min_importance: float = 0.0
    max_results: int = 10


class ForgettingCurve:
    """Static methods for calculating memory retention based on time and importance.

    Implements exponential decay: retention = importance * exp(-lambda * hours_elapsed)
    """

    @staticmethod
    def calculate_retention(
        importance: float,
        hours_elapsed: float,
        decay_lambda: float = 0.5,
    ) -> float:
        """Calculate retention score using exponential decay.

        Args:
            importance: Memory importance score (0.0 - 1.0).
            hours_elapsed: Time since memory creation in hours.
            decay_lambda: Decay rate parameter (higher = faster decay).
                         Default 0.5 gives half-life ≈ 1.4 hours.

        Returns:
            Retention score (0.0 - 1.0).
        """
        if hours_elapsed < 0:
            hours_elapsed = 0.0
        recency_factor = math.exp(-decay_lambda * hours_elapsed)
        return importance * recency_factor

    @staticmethod
    def should_forget(retention_score: float, threshold: float = 0.1) -> bool:
        """Determine if a memory should be forgotten based on retention score.

        Args:
            retention_score: The calculated retention score.
            threshold: Minimum retention score to keep the memory.

        Returns:
            True if the memory should be forgotten (retention < threshold).
        """
        return retention_score < threshold


class LTMRetrievalModule(Module):
    """LTM retrieval module that searches long-term memory and injects results into WM.

    On each tick (MID tier):
    1. Builds retrieval queries from current SAS context (goals, nearby agents, etc.)
    2. Searches the LTM store for relevant memories
    3. Applies forgetting curve to filter low-retention entries
    4. Injects top relevant memories into working memory
    """

    def __init__(
        self,
        ltm_store: LTMStore,
        decay_lambda: float = 0.5,
        retention_threshold: float = 0.1,
        max_memories_per_tick: int = 5,
    ) -> None:
        """Initialize the LTM retrieval module.

        Args:
            ltm_store: The backend LTM store to search.
            decay_lambda: Decay rate for forgetting curve (default 0.5).
            retention_threshold: Minimum retention score to keep memories (default 0.1).
            max_memories_per_tick: Maximum memories to inject per tick (default 5).
        """
        self._store = ltm_store
        self._decay_lambda = decay_lambda
        self._retention_threshold = retention_threshold
        self._max_memories = max_memories_per_tick

    # --- Module ABC ---

    @property
    def name(self) -> str:
        return "ltm_retrieval"

    @property
    def tier(self) -> ModuleTier:
        return ModuleTier.MID

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        """Execute one tick: build queries, search LTM, apply forgetting, inject to WM.

        Args:
            sas: Shared Agent State to read from and write to.

        Returns:
            ModuleResult summarizing retrieved memories.
        """
        try:
            # 1. Build retrieval queries from current SAS context
            queries = await self._build_queries(sas)

            if not queries:
                return ModuleResult(
                    module_name=self.name,
                    tier=self.tier,
                    data={"queries": 0, "retrieved": 0, "injected": 0},
                )

            agent_id = sas.agent_id

            # 2. Execute queries and collect results (converted to MemoryEntry)
            all_results: list[MemoryEntry] = []
            for query in queries:
                query_embedding = self._text_to_embedding_placeholder(query.query_text)
                ltm_results: list[LTMEntry] = await self._store.search(
                    agent_id,
                    query_embedding,
                    limit=query.max_results,
                    min_score=query.min_importance,
                )
                # Convert LTMEntry -> MemoryEntry and apply category filter
                for ltm_entry in ltm_results:
                    if (
                        query.category_filter is not None
                        and ltm_entry.category != query.category_filter
                    ):
                        continue
                    all_results.append(self._ltm_to_memory(ltm_entry))

            # 3. Apply forgetting curve to filter out low-retention memories
            retained = self._apply_forgetting(all_results)

            # 4. Deduplicate by ID and sort by importance (highest first)
            seen_ids = set()
            unique_retained = []
            for entry in sorted(retained, key=lambda e: e.importance, reverse=True):
                if entry.id not in seen_ids:
                    seen_ids.add(entry.id)
                    unique_retained.append(entry)

            # 5. Take top N and inject into working memory
            top_memories = unique_retained[: self._max_memories]
            if top_memories:
                current_wm = await sas.get_working_memory()
                # Inject LTM memories with category="ltm_retrieval"
                ltm_entries = [
                    MemoryEntry(
                        timestamp=datetime.now(UTC),
                        content=f"[LTM] {mem.content}",
                        category="ltm_retrieval",
                        importance=mem.importance,
                        source_module=self.name,
                        metadata={"original_id": str(mem.id), "original_category": mem.category},
                    )
                    for mem in top_memories
                ]
                combined = current_wm + ltm_entries
                # Cap to WM capacity (10) keeping highest importance entries
                wm_capacity = 10
                if len(combined) > wm_capacity:
                    combined = sorted(combined, key=lambda e: e.importance, reverse=True)[
                        :wm_capacity
                    ]
                await sas.set_working_memory(combined)

            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                data={
                    "queries": len(queries),
                    "retrieved": len(all_results),
                    "after_forgetting": len(retained),
                    "injected": len(top_memories),
                },
            )

        except Exception as exc:
            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                error=str(exc),
            )

    # --- Internal methods ---

    async def _build_queries(self, sas: SharedAgentState) -> list[RetrievalQuery]:
        """Build retrieval queries from current SAS context.

        Generates queries based on:
        - Current goals
        - Nearby agents (for social context)
        - Recent chat messages (for conversation context)

        Args:
            sas: Shared Agent State to read context from.

        Returns:
            List of RetrievalQuery objects.
        """
        queries: list[RetrievalQuery] = []

        # Query 1: Goal-related memories
        goals = await sas.get_goals()
        if goals.current_goal:
            queries.append(
                RetrievalQuery(
                    query_text=goals.current_goal,
                    category_filter="semantic",
                    min_importance=0.3,
                    max_results=5,
                )
            )

        # Query 2: Social context (nearby agents)
        percepts = await sas.get_percepts()
        if percepts.nearby_players:
            for player in percepts.nearby_players[:2]:  # Limit to 2 most recent
                queries.append(
                    RetrievalQuery(
                        query_text=f"interactions with {player}",
                        category_filter="social",
                        min_importance=0.2,
                        max_results=3,
                    )
                )

        # Query 3: Recent conversation context
        if percepts.chat_messages:
            recent_chats = percepts.chat_messages[-3:]  # Last 3 messages
            for msg in recent_chats:
                if isinstance(msg, dict) and "content" in msg:
                    queries.append(
                        RetrievalQuery(
                            query_text=msg["content"],
                            category_filter=None,  # Any category
                            min_importance=0.1,
                            max_results=2,
                        )
                    )

        return queries

    def _apply_forgetting(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Apply forgetting curve to filter out low-retention memories.

        Args:
            entries: List of MemoryEntry objects to filter.

        Returns:
            List of entries that pass the retention threshold.
        """
        now = datetime.now(UTC)
        retained: list[MemoryEntry] = []

        for entry in entries:
            hours_elapsed = (now - entry.timestamp).total_seconds() / 3600.0
            retention = ForgettingCurve.calculate_retention(
                importance=entry.importance,
                hours_elapsed=hours_elapsed,
                decay_lambda=self._decay_lambda,
            )

            if not ForgettingCurve.should_forget(retention, self._retention_threshold):
                retained.append(entry)

        return retained

    @staticmethod
    def _ltm_to_memory(ltm_entry: LTMEntry) -> MemoryEntry:
        """Convert an LTMEntry to a MemoryEntry for use in working memory.

        Args:
            ltm_entry: The LTMEntry to convert.

        Returns:
            A MemoryEntry with the same content and metadata.
        """
        return MemoryEntry(
            id=ltm_entry.id,
            timestamp=ltm_entry.timestamp,
            content=ltm_entry.content,
            category=ltm_entry.category,
            importance=ltm_entry.importance,
            source_module=ltm_entry.source_module,
            metadata=dict(ltm_entry.metadata),
        )

    @staticmethod
    def _text_to_embedding_placeholder(text: str) -> list[float]:
        """Convert text to a placeholder embedding vector.

        In production, this would call an embedding model (e.g., OpenAI
        text-embedding-3-small). For now, returns a simple hash-based vector.

        Args:
            text: The text to embed.

        Returns:
            A placeholder embedding vector of dimension 384 (MiniLM default).
        """
        # Simple hash-based placeholder: deterministic but not semantically meaningful
        # In production, replace with actual embedding model call
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        # Use hash bytes to seed a simple vector
        vec = [float(b) / 255.0 for b in h]
        # Pad or truncate to 384 dimensions
        dim = 384
        if len(vec) < dim:
            vec = vec * (dim // len(vec) + 1)
        return vec[:dim]
