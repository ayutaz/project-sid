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
    "LTMStore",
    "RetrievalQuery",
]

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from piano.core.module import Module
from piano.core.types import MemoryEntry, ModuleResult, ModuleTier

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState


class RetrievalQuery(BaseModel):
    """A query for retrieving memories from LTM."""

    query_text: str
    category_filter: str | None = None
    min_importance: float = 0.0
    max_results: int = 10


class LTMStore(Protocol):
    """Protocol for a long-term memory store that supports search.

    This allows the LTMRetrievalModule to work with any backend that
    implements this interface (e.g., Qdrant, Chroma, mock stores).
    """

    async def search(
        self,
        query_text: str,
        category_filter: str | None = None,
        min_importance: float = 0.0,
        max_results: int = 10,
    ) -> list[MemoryEntry]:
        """Search for memories matching the query.

        Args:
            query_text: The search query string.
            category_filter: Optional category to filter by.
            min_importance: Minimum importance threshold.
            max_results: Maximum number of results to return.

        Returns:
            List of matching MemoryEntry objects.
        """
        ...


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

            # 2. Execute queries and collect results
            all_results: list[MemoryEntry] = []
            for query in queries:
                results = await self._store.search(
                    query_text=query.query_text,
                    category_filter=query.category_filter,
                    min_importance=query.min_importance,
                    max_results=query.max_results,
                )
                all_results.extend(results)

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
                await sas.set_working_memory(current_wm + ltm_entries)

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
