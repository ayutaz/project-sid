"""Phase 1 SAS extension mixin for PIANO architecture.

Provides typed access to Phase 1 SAS sections using the existing
get_section/update_section generic methods. This allows Phase 1 modules
to read/write personality, emotions, social graphs, LTM stats, talking state,
and checkpoint metadata without modifying the SharedAgentState ABC.

Usage:
    sas = RedisSAS(redis, agent_id="agent-001")
    phase1 = SASPhase1Mixin(sas)

    # Read/write personality traits
    traits = await phase1.get_personality()
    await phase1.update_personality({"openness": 0.8, "extraversion": 0.6})

    # Read/write emotion state
    emotions = await phase1.get_emotion_state()
    await phase1.update_emotion_state({"valence": 0.5, "arousal": 0.3})
"""

from __future__ import annotations

__all__ = ["SASPhase1Mixin"]

import contextlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState


class SASPhase1Mixin:
    """Mixin helper providing typed Phase 1 section access.

    Wraps a SharedAgentState and provides convenient typed methods
    for Phase 1 module sections using the existing get_section/update_section
    generic interface.

    Args:
        sas: The SharedAgentState instance to wrap.
    """

    def __init__(self, sas: SharedAgentState) -> None:
        self._sas = sas

    # --- Personality ---

    async def get_personality(self) -> dict[str, float]:
        """Get personality trait values (Big Five or similar).

        Returns:
            Dictionary mapping trait names to float values (typically 0-1).
            Returns empty dict if not set. Non-numeric values are skipped.
        """
        data = await self._sas.get_section("personality")
        result: dict[str, float] = {}
        for k, v in data.items():
            with contextlib.suppress(ValueError, TypeError):
                result[k] = float(v)
        return result

    async def update_personality(self, traits: dict[str, float]) -> None:
        """Update personality trait values.

        Args:
            traits: Dictionary mapping trait names to float values.
        """
        await self._sas.update_section("personality", traits)

    # --- Emotions ---

    async def get_emotion_state(self) -> dict[str, Any]:
        """Get current emotional state.

        Returns:
            Dictionary containing emotion dimensions (valence, arousal, etc.)
            and metadata. Returns empty dict if not set.
        """
        return await self._sas.get_section("emotions")

    async def update_emotion_state(self, state: dict[str, Any]) -> None:
        """Update emotional state.

        Args:
            state: Dictionary containing emotion dimensions and metadata.
        """
        await self._sas.update_section("emotions", state)

    # --- Social Graph ---

    async def get_social_graph_snapshot(self) -> dict[str, Any]:
        """Get social graph snapshot (relationships, trust scores, etc.).

        Returns:
            Dictionary containing social graph data (nodes, edges, scores).
            Returns empty dict if not set.
        """
        return await self._sas.get_section("social_graph")

    async def update_social_graph_snapshot(self, data: dict[str, Any]) -> None:
        """Update social graph snapshot.

        Args:
            data: Dictionary containing social graph data.
        """
        await self._sas.update_section("social_graph", data)

    # --- LTM Stats ---

    async def get_ltm_stats(self) -> dict[str, Any]:
        """Get long-term memory statistics and indices.

        Returns:
            Dictionary containing LTM stats (e.g., vector count, last sync time).
            Returns empty dict if not set.
        """
        return await self._sas.get_section("ltm_stats")

    async def update_ltm_stats(self, stats: dict[str, Any]) -> None:
        """Update LTM statistics.

        Args:
            stats: Dictionary containing LTM stats and metadata.
        """
        await self._sas.update_section("ltm_stats", stats)

    # --- Talking State ---

    async def get_talking_state(self) -> dict[str, Any]:
        """Get talking/conversation state.

        Returns:
            Dictionary containing pending utterances, conversation context, etc.
            Returns empty dict if not set.
        """
        return await self._sas.get_section("talking")

    async def update_talking_state(self, state: dict[str, Any]) -> None:
        """Update talking/conversation state.

        Args:
            state: Dictionary containing talking state data.
        """
        await self._sas.update_section("talking", state)

    # --- Checkpoint Metadata ---

    async def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Get checkpoint metadata (for save/load).

        Returns:
            Dictionary containing checkpoint metadata (timestamp, version, etc.).
            Returns empty dict if not set.
        """
        return await self._sas.get_section("checkpoint")

    async def update_checkpoint_metadata(self, meta: dict[str, Any]) -> None:
        """Update checkpoint metadata.

        Args:
            meta: Dictionary containing checkpoint metadata.
        """
        await self._sas.update_section("checkpoint", meta)

    # --- Access to underlying SAS ---

    @property
    def sas(self) -> SharedAgentState:
        """Get the underlying SharedAgentState instance."""
        return self._sas
