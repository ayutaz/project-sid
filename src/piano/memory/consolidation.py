"""Memory Consolidation Module - transfers important STM entries to LTM.

Implements periodic consolidation of short-term memory (STM) into long-term
memory (LTM) based on importance, recency, and access patterns. Optionally
summarizes related memories using LLM compression before storage.

Reference: docs/implementation/04-memory-system.md Section 4.3
"""

from __future__ import annotations

__all__ = [
    "ConsolidationPolicy",
    "ConsolidationResult",
    "MemoryConsolidationModule",
]

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from piano.core.module import Module
from piano.core.types import LLMRequest, MemoryEntry, ModuleResult, ModuleTier
from piano.memory.ltm import LTMEntry, LTMStore

if TYPE_CHECKING:
    from uuid import UUID

    from piano.core.sas import SharedAgentState
    from piano.llm.provider import LLMProvider

logger = structlog.get_logger(__name__)


class ConsolidationPolicy(BaseModel):
    """Policy parameters for STM → LTM consolidation."""

    min_importance: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum importance threshold for consolidation",
    )
    min_age_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description="Minimum age in seconds before entry is eligible",
    )
    max_stm_before_consolidation: int = Field(
        default=80,
        ge=1,
        description="Trigger consolidation when STM reaches this size",
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Maximum number of entries to consolidate per tick",
    )


class ConsolidationResult(BaseModel):
    """Result of a consolidation operation."""

    consolidated_count: int = Field(default=0, ge=0)
    entries_consolidated: list[UUID] = Field(default_factory=list)
    summaries_created: int = Field(default=0, ge=0)


class MemoryConsolidationModule(Module):
    """Consolidates important STM entries into LTM.

    This module runs on the SLOW tier and periodically scans STM for entries
    worthy of long-term storage. Entries are filtered by importance and age,
    optionally summarized with LLM, and stored in the LTM backend.
    """

    def __init__(
        self,
        ltm_store: LTMStore,
        policy: ConsolidationPolicy | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """Initialize the consolidation module.

        Args:
            ltm_store: Backend store for long-term memory.
            policy: Consolidation policy (uses defaults if None).
            llm_provider: Optional LLM for memory summarization.
        """
        self._ltm_store = ltm_store
        self._policy = policy or ConsolidationPolicy()
        self._llm_provider = llm_provider

    @property
    def name(self) -> str:
        return "memory_consolidation"

    @property
    def tier(self) -> ModuleTier:
        return ModuleTier.SLOW

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        """Execute consolidation: scan STM → filter → summarize → store in LTM.

        Returns:
            ModuleResult with consolidation statistics.
        """
        try:
            # 1. Get current STM entries
            stm_entries = await sas.get_stm(limit=100)

            # 2. Check if consolidation is needed
            if len(stm_entries) < self._policy.max_stm_before_consolidation:
                logger.debug(
                    "stm_below_threshold",
                    stm_size=len(stm_entries),
                    threshold=self._policy.max_stm_before_consolidation,
                )
                return ModuleResult(
                    module_name=self.name,
                    tier=self.tier,
                    data={"consolidated": 0, "stm_size": len(stm_entries)},
                )

            # 3. Select entries for consolidation
            candidates = self._select_for_consolidation(stm_entries, self._policy)

            # 4. Limit batch size
            to_consolidate = candidates[: self._policy.batch_size]

            if not to_consolidate:
                logger.debug("no_candidates_for_consolidation", stm_size=len(stm_entries))
                return ModuleResult(
                    module_name=self.name,
                    tier=self.tier,
                    data={"consolidated": 0, "stm_size": len(stm_entries)},
                )

            # Get agent_id from SAS for LTM store calls
            agent_id = sas.agent_id

            # 5. Optionally summarize related entries
            summaries_created = 0
            if self._llm_provider is not None and len(to_consolidate) > 1:
                summary = await self._summarize_related(to_consolidate)
                if summary:
                    # Create a summary entry
                    summary_entry = MemoryEntry(
                        timestamp=datetime.now(UTC),
                        content=summary,
                        category="reflection",
                        importance=max(e.importance for e in to_consolidate),
                        source_module=self.name,
                        metadata={
                            "consolidated_from": [str(e.id) for e in to_consolidate],
                            "is_summary": True,
                        },
                    )
                    ltm_summary = self._memory_to_ltm(summary_entry)
                    await self._ltm_store.store(agent_id, ltm_summary)
                    summaries_created = 1

            # 6. Store individual entries in LTM
            consolidated_ids: list[UUID] = []
            for entry in to_consolidate:
                ltm_entry = self._memory_to_ltm(entry)
                await self._ltm_store.store(agent_id, ltm_entry)
                consolidated_ids.append(entry.id)

            logger.info(
                "consolidation_complete",
                consolidated_count=len(consolidated_ids),
                summaries_created=summaries_created,
            )

            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                data={
                    "consolidated": len(consolidated_ids),
                    "summaries_created": summaries_created,
                    "stm_size": len(stm_entries),
                },
            )

        except Exception as exc:
            logger.error("consolidation_failed", error=str(exc))
            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                error=str(exc),
            )

    def _select_for_consolidation(
        self,
        stm_entries: list[MemoryEntry],
        policy: ConsolidationPolicy,
    ) -> list[MemoryEntry]:
        """Select STM entries eligible for consolidation.

        Filters by importance threshold and minimum age.

        Args:
            stm_entries: All current STM entries.
            policy: Consolidation policy with thresholds.

        Returns:
            List of entries that should be consolidated (oldest first).
        """
        now = datetime.now(UTC)
        candidates: list[MemoryEntry] = []

        for entry in stm_entries:
            # Check importance threshold
            if entry.importance < policy.min_importance:
                continue

            # Check age threshold
            age_seconds = (now - entry.timestamp).total_seconds()
            if age_seconds < policy.min_age_seconds:
                continue

            candidates.append(entry)

        # Sort by timestamp (oldest first) to prioritize older memories
        return sorted(candidates, key=lambda e: e.timestamp)

    async def _summarize_related(self, entries: list[MemoryEntry]) -> str:
        """Summarize related memories using LLM.

        Args:
            entries: List of related memory entries to summarize.

        Returns:
            Summary text, or empty string if summarization fails.
        """
        if self._llm_provider is None or not entries:
            return ""

        try:
            # Build prompt with all entry contents
            memories_text = "\n".join(
                f"- [{i+1}] {e.content} (importance: {e.importance:.2f})"
                for i, e in enumerate(entries)
            )

            prompt = f"""Summarize the following related memories into a single concise summary.
Preserve important relationships, emotions, and learned patterns.
Keep the summary to 1-3 sentences.

Memories:
{memories_text}

Summary:"""

            request = LLMRequest(
                prompt=prompt,
                system_prompt=(
                    "You are a memory consolidation system. "
                    "Create concise, meaningful summaries."
                ),
                tier=ModuleTier.SLOW,
                temperature=0.3,
                max_tokens=150,
            )

            response = await self._llm_provider.complete(request)
            return response.content.strip()

        except Exception as exc:
            logger.warning("summarization_failed", error=str(exc))
            return ""

    def _memory_to_ltm(self, entry: MemoryEntry) -> LTMEntry:
        """Convert a MemoryEntry (STM/WM) to an LTMEntry for long-term storage.

        Generates a placeholder embedding vector for the entry.

        Args:
            entry: The MemoryEntry to convert.

        Returns:
            A new LTMEntry with a placeholder embedding.
        """
        return LTMEntry(
            id=entry.id,
            timestamp=entry.timestamp,
            content=entry.content,
            category=entry.category,
            importance=entry.importance,
            source_module=entry.source_module,
            metadata={
                k: v
                for k, v in entry.metadata.items()
                if isinstance(v, (str, int, float, bool))
            },
            embedding=self._generate_embedding_placeholder(entry.content),
        )

    @staticmethod
    def _generate_embedding_placeholder(content: str) -> list[float]:
        """Generate a placeholder embedding vector.

        In production, this would call an embedding model. For now, we return
        a simple hash-based vector as a placeholder.

        Args:
            content: Text content to embed.

        Returns:
            A placeholder embedding vector (zero vector).
        """
        # Simple placeholder: return zero vector of dimension 1536 (text-embedding-3-small)
        # In production, this would call OpenAI's embedding API
        return [0.0] * 1536
