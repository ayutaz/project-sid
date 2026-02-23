"""Shared fixtures for memory tests."""

from __future__ import annotations

from typing import Any

import pytest

from piano.core.sas import SharedAgentState
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


class InMemorySAS(SharedAgentState):
    """Minimal dict-backed SAS for unit tests (no Redis required)."""

    def __init__(self, agent_id: AgentId = "test-agent-001") -> None:
        self._agent_id = agent_id
        self._percepts = PerceptData()
        self._goals = GoalData()
        self._social = SocialData()
        self._plans = PlanData()
        self._actions: list[ActionHistoryEntry] = []
        self._wm: list[MemoryEntry] = []
        self._stm: list[MemoryEntry] = []
        self._reflection = SelfReflectionData()
        self._cc_decision: dict[str, Any] | None = None
        self._sections: dict[str, dict[str, Any]] = {}

    # --- identity ---

    @property
    def agent_id(self) -> AgentId:
        return self._agent_id

    # --- percepts ---

    async def get_percepts(self) -> PerceptData:
        return self._percepts

    async def update_percepts(self, percepts: PerceptData) -> None:
        self._percepts = percepts

    # --- goals ---

    async def get_goals(self) -> GoalData:
        return self._goals

    async def update_goals(self, goals: GoalData) -> None:
        self._goals = goals

    # --- social ---

    async def get_social(self) -> SocialData:
        return self._social

    async def update_social(self, social: SocialData) -> None:
        self._social = social

    # --- plans ---

    async def get_plans(self) -> PlanData:
        return self._plans

    async def update_plans(self, plans: PlanData) -> None:
        self._plans = plans

    # --- action history ---

    async def get_action_history(self, limit: int = 50) -> list[ActionHistoryEntry]:
        return sorted(self._actions, key=lambda a: a.timestamp, reverse=True)[:limit]

    async def add_action(self, entry: ActionHistoryEntry) -> None:
        self._actions.append(entry)
        if len(self._actions) > 50:
            self._actions = self._actions[-50:]

    # --- working memory ---

    async def get_working_memory(self) -> list[MemoryEntry]:
        return list(self._wm)

    async def set_working_memory(self, entries: list[MemoryEntry]) -> None:
        self._wm = list(entries)

    # --- STM ---

    async def get_stm(self, limit: int = 100) -> list[MemoryEntry]:
        return sorted(self._stm, key=lambda m: m.timestamp, reverse=True)[:limit]

    async def add_stm(self, entry: MemoryEntry) -> None:
        self._stm.append(entry)
        if len(self._stm) > 100:
            self._stm = self._stm[-100:]

    # --- reflection ---

    async def get_self_reflection(self) -> SelfReflectionData:
        return self._reflection

    async def update_self_reflection(self, reflection: SelfReflectionData) -> None:
        self._reflection = reflection

    # --- CC decision ---

    async def get_last_cc_decision(self) -> dict[str, Any] | None:
        return self._cc_decision

    async def set_cc_decision(self, decision: dict[str, Any]) -> None:
        self._cc_decision = decision

    # --- generic ---

    async def get_section(self, section: str) -> dict[str, Any]:
        return self._sections.get(section, {})

    async def update_section(self, section: str, data: dict[str, Any]) -> None:
        self._sections[section] = data

    # --- snapshot ---

    async def snapshot(self) -> dict[str, Any]:
        return {
            "agent_id": self._agent_id,
            "percepts": self._percepts.model_dump(),
            "goals": self._goals.model_dump(),
            "social": self._social.model_dump(),
            "plans": self._plans.model_dump(),
            "working_memory": [m.model_dump() for m in self._wm],
            "stm": [m.model_dump() for m in self._stm],
        }

    # --- lifecycle ---

    async def initialize(self) -> None:
        pass

    async def clear(self) -> None:
        self._percepts = PerceptData()
        self._goals = GoalData()
        self._social = SocialData()
        self._plans = PlanData()
        self._actions.clear()
        self._wm.clear()
        self._stm.clear()
        self._reflection = SelfReflectionData()
        self._cc_decision = None
        self._sections.clear()


@pytest.fixture
def sas() -> InMemorySAS:
    """A fresh InMemorySAS for each test."""
    return InMemorySAS()
