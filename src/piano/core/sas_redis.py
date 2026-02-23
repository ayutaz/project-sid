"""Redis-backed implementation of SharedAgentState.

Uses redis.asyncio for all operations. Each agent's data is partitioned
under the key prefix ``piano:{agent_id}:{section}``.

Capacity limits (enforced on write):
- STM: latest 100 entries
- action_history: latest 50 entries
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from piano.core.sas import SharedAgentState

if TYPE_CHECKING:
    from redis.asyncio import Redis
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

# Capacity limits
_STM_LIMIT = 100
_ACTION_HISTORY_LIMIT = 50


class RedisSAS(SharedAgentState):
    """Redis-backed Shared Agent State.

    All data is stored in Redis using the key scheme
    ``piano:{agent_id}:{section}``.  Simple sections (percepts, goals, etc.)
    are stored as JSON strings.  List sections (stm, action_history, working_memory)
    use Redis lists with newest entries on the left (LPUSH).

    Args:
        redis: An ``redis.asyncio.Redis`` connection instance.
        agent_id: The agent identifier that owns this SAS.
    """

    def __init__(self, redis: Redis, agent_id: AgentId) -> None:  # type: ignore[type-arg]
        self._redis = redis
        self._agent_id = agent_id

    # --- Key helpers ---

    def _key(self, section: str) -> str:
        """Build a Redis key for the given section."""
        return f"piano:{self._agent_id}:{section}"

    # --- Property ---

    @property
    def agent_id(self) -> AgentId:
        """The agent this SAS belongs to."""
        return self._agent_id

    # --- Percepts ---

    async def get_percepts(self) -> PerceptData:
        """Get current perception data."""
        raw = await self._redis.get(self._key("percepts"))
        if raw is None:
            return PerceptData()
        return PerceptData.model_validate_json(raw)

    async def update_percepts(self, percepts: PerceptData) -> None:
        """Update perception data (called by environment interface)."""
        await self._redis.set(self._key("percepts"), percepts.model_dump_json())

    # --- Goals ---

    async def get_goals(self) -> GoalData:
        """Get current goal state."""
        raw = await self._redis.get(self._key("goals"))
        if raw is None:
            return GoalData()
        return GoalData.model_validate_json(raw)

    async def update_goals(self, goals: GoalData) -> None:
        """Update goal state."""
        await self._redis.set(self._key("goals"), goals.model_dump_json())

    # --- Social ---

    async def get_social(self) -> SocialData:
        """Get social awareness data."""
        raw = await self._redis.get(self._key("social"))
        if raw is None:
            return SocialData()
        return SocialData.model_validate_json(raw)

    async def update_social(self, social: SocialData) -> None:
        """Update social data."""
        await self._redis.set(self._key("social"), social.model_dump_json())

    # --- Plans ---

    async def get_plans(self) -> PlanData:
        """Get current plan state."""
        raw = await self._redis.get(self._key("plans"))
        if raw is None:
            return PlanData()
        return PlanData.model_validate_json(raw)

    async def update_plans(self, plans: PlanData) -> None:
        """Update plan state."""
        await self._redis.set(self._key("plans"), plans.model_dump_json())

    # --- Action History ---

    async def get_action_history(self, limit: int = 50) -> list[ActionHistoryEntry]:
        """Get recent action history (newest first, max 50)."""
        effective_limit = min(limit, _ACTION_HISTORY_LIMIT)
        raw_list: list[bytes] = await self._redis.lrange(
            self._key("action_history"), 0, effective_limit - 1
        )
        return [ActionHistoryEntry.model_validate_json(item) for item in raw_list]

    async def add_action(self, entry: ActionHistoryEntry) -> None:
        """Add an action to history (auto-trims to capacity limit)."""
        key = self._key("action_history")
        await self._redis.lpush(key, entry.model_dump_json())
        await self._redis.ltrim(key, 0, _ACTION_HISTORY_LIMIT - 1)

    # --- Memory (WM/STM) ---

    async def get_working_memory(self) -> list[MemoryEntry]:
        """Get current working memory entries."""
        raw_list: list[bytes] = await self._redis.lrange(
            self._key("working_memory"), 0, -1
        )
        return [MemoryEntry.model_validate_json(item) for item in raw_list]

    async def set_working_memory(self, entries: list[MemoryEntry]) -> None:
        """Replace working memory contents."""
        key = self._key("working_memory")
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(key)
            for entry in entries:
                pipe.rpush(key, entry.model_dump_json())
            await pipe.execute()

    async def get_stm(self, limit: int = 100) -> list[MemoryEntry]:
        """Get short-term memory entries (newest first, max 100)."""
        effective_limit = min(limit, _STM_LIMIT)
        raw_list: list[bytes] = await self._redis.lrange(
            self._key("stm"), 0, effective_limit - 1
        )
        return [MemoryEntry.model_validate_json(item) for item in raw_list]

    async def add_stm(self, entry: MemoryEntry) -> None:
        """Add entry to short-term memory (auto-trims to capacity limit)."""
        key = self._key("stm")
        await self._redis.lpush(key, entry.model_dump_json())
        await self._redis.ltrim(key, 0, _STM_LIMIT - 1)

    # --- Self Reflection ---

    async def get_self_reflection(self) -> SelfReflectionData:
        """Get self-reflection state."""
        raw = await self._redis.get(self._key("self_reflection"))
        if raw is None:
            return SelfReflectionData()
        return SelfReflectionData.model_validate_json(raw)

    async def update_self_reflection(self, reflection: SelfReflectionData) -> None:
        """Update self-reflection state."""
        await self._redis.set(
            self._key("self_reflection"), reflection.model_dump_json()
        )

    # --- CC Decision ---

    async def get_last_cc_decision(self) -> dict[str, Any] | None:
        """Get the last CC decision broadcast."""
        raw = await self._redis.get(self._key("cc_decision"))
        if raw is None:
            return None
        return json.loads(raw)

    async def set_cc_decision(self, decision: dict[str, Any]) -> None:
        """Store the latest CC decision."""
        await self._redis.set(self._key("cc_decision"), json.dumps(decision))

    # --- Generic Section Access ---

    async def get_section(self, section: str) -> dict[str, Any]:
        """Get a raw SAS section by name."""
        raw = await self._redis.get(self._key(section))
        if raw is None:
            return {}
        return json.loads(raw)

    async def update_section(self, section: str, data: dict[str, Any]) -> None:
        """Update a raw SAS section."""
        await self._redis.set(self._key(section), json.dumps(data))

    # --- Snapshot ---

    async def snapshot(self) -> dict[str, Any]:
        """Get a full snapshot of all SAS sections (for CC input / checkpointing)."""
        percepts = await self.get_percepts()
        goals = await self.get_goals()
        social = await self.get_social()
        plans = await self.get_plans()
        action_history = await self.get_action_history()
        working_memory = await self.get_working_memory()
        stm = await self.get_stm()
        self_reflection = await self.get_self_reflection()
        cc_decision = await self.get_last_cc_decision()

        return {
            "agent_id": self._agent_id,
            "percepts": percepts.model_dump(),
            "goals": goals.model_dump(),
            "social": social.model_dump(),
            "plans": plans.model_dump(),
            "action_history": [e.model_dump() for e in action_history],
            "working_memory": [e.model_dump() for e in working_memory],
            "stm": [e.model_dump() for e in stm],
            "self_reflection": self_reflection.model_dump(),
            "cc_decision": cc_decision,
        }

    # --- Lifecycle ---

    async def initialize(self) -> None:
        """Initialize the SAS (ensure default values exist).

        Sets each section to its default (empty) value only if the key
        does not already exist, preserving any previously stored state.
        """
        defaults: list[tuple[str, str]] = [
            ("percepts", PerceptData().model_dump_json()),
            ("goals", GoalData().model_dump_json()),
            ("social", SocialData().model_dump_json()),
            ("plans", PlanData().model_dump_json()),
            ("self_reflection", SelfReflectionData().model_dump_json()),
            ("cc_decision", "null"),
        ]
        for section, value in defaults:
            await self._redis.set(self._key(section), value, nx=True)

    async def clear(self) -> None:
        """Clear all SAS data for this agent."""
        pattern = f"piano:{self._agent_id}:*"
        cursor: int | bytes = 0
        while True:
            cursor, keys = await self._redis.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                await self._redis.delete(*keys)
            if cursor == 0:
                break
