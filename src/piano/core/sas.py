"""Shared Agent State (SAS) interface and protocol.

SAS is the central data store for each agent in the PIANO architecture.
All modules read from and write to SAS - they maintain no internal state.
The default implementation uses Redis for persistence and cross-module sharing.

Reference: docs/implementation/01-system-architecture.md Section 2
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from piano.core.types import (
    ActionHistoryEntry,
    AgentId,
    GoalData,
    MemoryEntry,
    PerceptData,
    PlanData,
    SelfReflectionData,
    SocialData,
)


class SharedAgentState(ABC):
    """Abstract interface for the Shared Agent State.

    Each agent has its own SAS instance, partitioned by agent_id.
    SAS sections correspond to the PIANO cognitive modules.

    Capacity limits (from roadmap 0-2):
    - STM: latest 100 entries
    - action_history: latest 50 entries
    - chat_messages in percepts: latest 20 messages
    """

    @property
    @abstractmethod
    def agent_id(self) -> AgentId:
        """The agent this SAS belongs to."""
        ...

    # --- Percepts ---

    @abstractmethod
    async def get_percepts(self) -> PerceptData:
        """Get current perception data."""
        ...

    @abstractmethod
    async def update_percepts(self, percepts: PerceptData) -> None:
        """Update perception data (called by environment interface)."""
        ...

    # --- Goals ---

    @abstractmethod
    async def get_goals(self) -> GoalData:
        """Get current goal state."""
        ...

    @abstractmethod
    async def update_goals(self, goals: GoalData) -> None:
        """Update goal state."""
        ...

    # --- Social ---

    @abstractmethod
    async def get_social(self) -> SocialData:
        """Get social awareness data."""
        ...

    @abstractmethod
    async def update_social(self, social: SocialData) -> None:
        """Update social data."""
        ...

    # --- Plans ---

    @abstractmethod
    async def get_plans(self) -> PlanData:
        """Get current plan state."""
        ...

    @abstractmethod
    async def update_plans(self, plans: PlanData) -> None:
        """Update plan state."""
        ...

    # --- Action History ---

    @abstractmethod
    async def get_action_history(self, limit: int = 50) -> list[ActionHistoryEntry]:
        """Get recent action history (newest first, max 50)."""
        ...

    @abstractmethod
    async def add_action(self, entry: ActionHistoryEntry) -> None:
        """Add an action to history (auto-trims to capacity limit)."""
        ...

    # --- Memory (WM/STM) ---

    @abstractmethod
    async def get_working_memory(self) -> list[MemoryEntry]:
        """Get current working memory entries."""
        ...

    @abstractmethod
    async def set_working_memory(self, entries: list[MemoryEntry]) -> None:
        """Replace working memory contents."""
        ...

    @abstractmethod
    async def get_stm(self, limit: int = 100) -> list[MemoryEntry]:
        """Get short-term memory entries (newest first, max 100)."""
        ...

    @abstractmethod
    async def add_stm(self, entry: MemoryEntry) -> None:
        """Add entry to short-term memory (auto-trims to capacity limit)."""
        ...

    # --- Self Reflection ---

    @abstractmethod
    async def get_self_reflection(self) -> SelfReflectionData:
        """Get self-reflection state."""
        ...

    @abstractmethod
    async def update_self_reflection(self, reflection: SelfReflectionData) -> None:
        """Update self-reflection state."""
        ...

    # --- CC Decision (latest broadcast) ---

    @abstractmethod
    async def get_last_cc_decision(self) -> dict[str, Any] | None:
        """Get the last CC decision broadcast."""
        ...

    @abstractmethod
    async def set_cc_decision(self, decision: dict[str, Any]) -> None:
        """Store the latest CC decision."""
        ...

    # --- Generic Section Access ---

    @abstractmethod
    async def get_section(self, section: str) -> dict[str, Any]:
        """Get a raw SAS section by name."""
        ...

    @abstractmethod
    async def update_section(self, section: str, data: dict[str, Any]) -> None:
        """Update a raw SAS section."""
        ...

    # --- Snapshot ---

    @abstractmethod
    async def snapshot(self) -> dict[str, Any]:
        """Get a full snapshot of all SAS sections (for CC input / checkpointing)."""
        ...

    # --- Lifecycle ---

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the SAS (create Redis keys, set defaults)."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all SAS data for this agent."""
        ...
