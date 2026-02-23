"""Reusable test helpers for PIANO tests.

Provides InMemorySAS (dict-based SAS implementation) and DummyModule
for testing without external dependencies (Redis, LLM, etc.).
"""

from __future__ import annotations

from typing import Any

from piano.core.module import Module
from piano.core.sas import SharedAgentState
from piano.core.types import (
    ActionHistoryEntry,
    AgentId,
    GoalData,
    MemoryEntry,
    ModuleResult,
    ModuleTier,
    PerceptData,
    PlanData,
    SelfReflectionData,
    SocialData,
)

# --- Capacity limits (matching SAS docstring) ---

STM_CAPACITY = 100
ACTION_HISTORY_CAPACITY = 50
CHAT_MESSAGE_CAPACITY = 20


class InMemorySAS(SharedAgentState):
    """In-memory dict-based SharedAgentState for testing.

    All data is stored in plain Python dicts/lists with the same capacity
    limits as the production Redis implementation.
    """

    def __init__(self, agent_id: AgentId = "test-agent-001") -> None:
        self._agent_id = agent_id
        self._percepts = PerceptData()
        self._goals = GoalData()
        self._social = SocialData()
        self._plans = PlanData()
        self._action_history: list[ActionHistoryEntry] = []
        self._working_memory: list[MemoryEntry] = []
        self._stm: list[MemoryEntry] = []
        self._self_reflection = SelfReflectionData()
        self._cc_decision: dict[str, Any] | None = None
        self._sections: dict[str, dict[str, Any]] = {}

    @property
    def agent_id(self) -> AgentId:
        """The agent this SAS belongs to."""
        return self._agent_id

    # --- Percepts ---

    async def get_percepts(self) -> PerceptData:
        """Get current perception data."""
        return self._percepts

    async def update_percepts(self, percepts: PerceptData) -> None:
        """Update perception data."""
        self._percepts = percepts

    # --- Goals ---

    async def get_goals(self) -> GoalData:
        """Get current goal state."""
        return self._goals

    async def update_goals(self, goals: GoalData) -> None:
        """Update goal state."""
        self._goals = goals

    # --- Social ---

    async def get_social(self) -> SocialData:
        """Get social awareness data."""
        return self._social

    async def update_social(self, social: SocialData) -> None:
        """Update social data."""
        self._social = social

    # --- Plans ---

    async def get_plans(self) -> PlanData:
        """Get current plan state."""
        return self._plans

    async def update_plans(self, plans: PlanData) -> None:
        """Update plan state."""
        self._plans = plans

    # --- Action History ---

    async def get_action_history(self, limit: int = 50) -> list[ActionHistoryEntry]:
        """Get recent action history (newest first)."""
        return list(reversed(self._action_history[-limit:]))

    async def add_action(self, entry: ActionHistoryEntry) -> None:
        """Add an action to history (auto-trims to capacity)."""
        self._action_history.append(entry)
        if len(self._action_history) > ACTION_HISTORY_CAPACITY:
            self._action_history = self._action_history[-ACTION_HISTORY_CAPACITY:]

    # --- Memory (WM/STM) ---

    async def get_working_memory(self) -> list[MemoryEntry]:
        """Get current working memory entries."""
        return list(self._working_memory)

    async def set_working_memory(self, entries: list[MemoryEntry]) -> None:
        """Replace working memory contents."""
        self._working_memory = list(entries)

    async def get_stm(self, limit: int = 100) -> list[MemoryEntry]:
        """Get short-term memory entries (newest first)."""
        return list(reversed(self._stm[-limit:]))

    async def add_stm(self, entry: MemoryEntry) -> None:
        """Add entry to short-term memory (auto-trims to capacity)."""
        self._stm.append(entry)
        if len(self._stm) > STM_CAPACITY:
            self._stm = self._stm[-STM_CAPACITY:]

    # --- Self Reflection ---

    async def get_self_reflection(self) -> SelfReflectionData:
        """Get self-reflection state."""
        return self._self_reflection

    async def update_self_reflection(self, reflection: SelfReflectionData) -> None:
        """Update self-reflection state."""
        self._self_reflection = reflection

    # --- CC Decision ---

    async def get_last_cc_decision(self) -> dict[str, Any] | None:
        """Get the last CC decision broadcast."""
        return self._cc_decision

    async def set_cc_decision(self, decision: dict[str, Any]) -> None:
        """Store the latest CC decision."""
        self._cc_decision = decision

    # --- Generic Section Access ---

    async def get_section(self, section: str) -> dict[str, Any]:
        """Get a raw SAS section by name."""
        return dict(self._sections.get(section, {}))

    async def update_section(self, section: str, data: dict[str, Any]) -> None:
        """Update a raw SAS section."""
        self._sections[section] = dict(data)

    # --- Snapshot ---

    async def snapshot(self) -> dict[str, Any]:
        """Get a full snapshot of all SAS sections."""
        return {
            "percepts": self._percepts.model_dump(),
            "goals": self._goals.model_dump(),
            "social": self._social.model_dump(),
            "plans": self._plans.model_dump(),
            "action_history": [e.model_dump() for e in self._action_history],
            "working_memory": [e.model_dump() for e in self._working_memory],
            "stm": [e.model_dump() for e in self._stm],
            "self_reflection": self._self_reflection.model_dump(),
            "cc_decision": self._cc_decision,
        }

    # --- Lifecycle ---

    async def initialize(self) -> None:
        """Initialize (no-op for in-memory)."""

    async def clear(self) -> None:
        """Clear all SAS data."""
        self._percepts = PerceptData()
        self._goals = GoalData()
        self._social = SocialData()
        self._plans = PlanData()
        self._action_history = []
        self._working_memory = []
        self._stm = []
        self._self_reflection = SelfReflectionData()
        self._cc_decision = None
        self._sections = {}

    # --- Test assertion helpers ---

    def assert_has_action(self, action: str) -> ActionHistoryEntry:
        """Assert that action history contains an entry with the given action string."""
        for entry in self._action_history:
            if entry.action == action:
                return entry
        raise AssertionError(f"No action '{action}' found in action history")

    def assert_goal(self, expected_goal: str) -> None:
        """Assert that the current goal matches expected."""
        assert self._goals.current_goal == expected_goal, (
            f"Expected goal '{expected_goal}', got '{self._goals.current_goal}'"
        )

    @property
    def action_count(self) -> int:
        """Number of actions in history."""
        return len(self._action_history)

    @property
    def stm_count(self) -> int:
        """Number of entries in short-term memory."""
        return len(self._stm)


class DummyModule(Module):
    """A simple Module implementation for testing.

    Returns a fixed ModuleResult on each tick. Tracks tick count
    and lifecycle calls for assertions.
    """

    def __init__(
        self,
        module_name: str = "dummy",
        tier: ModuleTier = ModuleTier.FAST,
        result_data: dict[str, Any] | None = None,
    ) -> None:
        self._name = module_name
        self._tier = tier
        self._result_data = result_data or {}
        self.tick_count: int = 0
        self.initialized: bool = False
        self.shut_down: bool = False
        self.broadcast_decisions: list[Any] = []

    @property
    def name(self) -> str:
        """Module name."""
        return self._name

    @property
    def tier(self) -> ModuleTier:
        """Module tier."""
        return self._tier

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        """Execute a tick, returning a fixed result."""
        self.tick_count += 1
        return ModuleResult(
            module_name=self._name,
            tier=self._tier,
            data={"tick": self.tick_count, **self._result_data},
        )

    async def on_broadcast(self, decision: Any) -> None:
        """Record broadcast decisions for assertions."""
        self.broadcast_decisions.append(decision)

    async def initialize(self) -> None:
        """Mark as initialized."""
        self.initialized = True

    async def shutdown(self) -> None:
        """Mark as shut down."""
        self.shut_down = True
