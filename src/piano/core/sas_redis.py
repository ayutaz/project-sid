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

import redis
import structlog

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

logger = structlog.get_logger(__name__)

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
        self._healthy = True

    # --- Key helpers ---

    def _key(self, section: str) -> str:
        """Build a Redis key for the given section."""
        return f"piano:{self._agent_id}:{section}"

    # --- Property ---

    @property
    def agent_id(self) -> AgentId:
        """The agent this SAS belongs to."""
        return self._agent_id

    @property
    def is_healthy(self) -> bool:
        """Check if the last Redis operation succeeded."""
        return self._healthy

    # --- Percepts ---

    async def get_percepts(self) -> PerceptData:
        """Get current perception data."""
        try:
            raw = await self._redis.get(self._key("percepts"))
            if raw is None:
                self._healthy = True
                return PerceptData()
            result = PerceptData.model_validate_json(raw)
            self._healthy = True
            return result
        except redis.RedisError as e:
            self._healthy = False
            logger.warning(
                "redis_get_percepts_failed",
                agent_id=self._agent_id,
                error=str(e),
            )
            return PerceptData()

    async def update_percepts(self, percepts: PerceptData) -> None:
        """Update perception data (called by environment interface)."""
        try:
            await self._redis.set(self._key("percepts"), percepts.model_dump_json())
            self._healthy = True
        except redis.RedisError as e:
            self._healthy = False
            logger.error(
                "redis_update_percepts_failed",
                agent_id=self._agent_id,
                error=str(e),
            )

    # --- Goals ---

    async def get_goals(self) -> GoalData:
        """Get current goal state."""
        try:
            raw = await self._redis.get(self._key("goals"))
            if raw is None:
                self._healthy = True
                return GoalData()
            result = GoalData.model_validate_json(raw)
            self._healthy = True
            return result
        except redis.RedisError as e:
            self._healthy = False
            logger.warning(
                "redis_get_goals_failed",
                agent_id=self._agent_id,
                error=str(e),
            )
            return GoalData()

    async def update_goals(self, goals: GoalData) -> None:
        """Update goal state."""
        try:
            await self._redis.set(self._key("goals"), goals.model_dump_json())
            self._healthy = True
        except redis.RedisError as e:
            self._healthy = False
            logger.error(
                "redis_update_goals_failed",
                agent_id=self._agent_id,
                error=str(e),
            )

    # --- Social ---

    async def get_social(self) -> SocialData:
        """Get social awareness data."""
        try:
            raw = await self._redis.get(self._key("social"))
            if raw is None:
                self._healthy = True
                return SocialData()
            result = SocialData.model_validate_json(raw)
            self._healthy = True
            return result
        except redis.RedisError as e:
            self._healthy = False
            logger.warning(
                "redis_get_social_failed",
                agent_id=self._agent_id,
                error=str(e),
            )
            return SocialData()

    async def update_social(self, social: SocialData) -> None:
        """Update social data."""
        try:
            await self._redis.set(self._key("social"), social.model_dump_json())
            self._healthy = True
        except redis.RedisError as e:
            self._healthy = False
            logger.error(
                "redis_update_social_failed",
                agent_id=self._agent_id,
                error=str(e),
            )

    # --- Plans ---

    async def get_plans(self) -> PlanData:
        """Get current plan state."""
        try:
            raw = await self._redis.get(self._key("plans"))
            if raw is None:
                self._healthy = True
                return PlanData()
            result = PlanData.model_validate_json(raw)
            self._healthy = True
            return result
        except redis.RedisError as e:
            self._healthy = False
            logger.warning(
                "redis_get_plans_failed",
                agent_id=self._agent_id,
                error=str(e),
            )
            return PlanData()

    async def update_plans(self, plans: PlanData) -> None:
        """Update plan state."""
        try:
            await self._redis.set(self._key("plans"), plans.model_dump_json())
            self._healthy = True
        except redis.RedisError as e:
            self._healthy = False
            logger.error(
                "redis_update_plans_failed",
                agent_id=self._agent_id,
                error=str(e),
            )

    # --- Action History ---

    async def get_action_history(self, limit: int = 50) -> list[ActionHistoryEntry]:
        """Get recent action history (newest first, max 50)."""
        try:
            effective_limit = min(limit, _ACTION_HISTORY_LIMIT)
            raw_list: list[bytes] = await self._redis.lrange(
                self._key("action_history"), 0, effective_limit - 1
            )
            result = [ActionHistoryEntry.model_validate_json(item) for item in raw_list]
            self._healthy = True
            return result
        except redis.RedisError as e:
            self._healthy = False
            logger.warning(
                "redis_get_action_history_failed",
                agent_id=self._agent_id,
                error=str(e),
            )
            return []

    async def add_action(self, entry: ActionHistoryEntry) -> None:
        """Add an action to history (auto-trims to capacity limit)."""
        try:
            key = self._key("action_history")
            await self._redis.lpush(key, entry.model_dump_json())
            await self._redis.ltrim(key, 0, _ACTION_HISTORY_LIMIT - 1)
            self._healthy = True
        except redis.RedisError as e:
            self._healthy = False
            logger.error(
                "redis_add_action_failed",
                agent_id=self._agent_id,
                error=str(e),
            )

    # --- Memory (WM/STM) ---

    async def get_working_memory(self) -> list[MemoryEntry]:
        """Get current working memory entries."""
        try:
            raw_list: list[bytes] = await self._redis.lrange(
                self._key("working_memory"), 0, -1
            )
            result = [MemoryEntry.model_validate_json(item) for item in raw_list]
            self._healthy = True
            return result
        except redis.RedisError as e:
            self._healthy = False
            logger.warning(
                "redis_get_working_memory_failed",
                agent_id=self._agent_id,
                error=str(e),
            )
            return []

    async def set_working_memory(self, entries: list[MemoryEntry]) -> None:
        """Replace working memory contents."""
        try:
            key = self._key("working_memory")
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.delete(key)
                for entry in entries:
                    pipe.rpush(key, entry.model_dump_json())
                await pipe.execute()
            self._healthy = True
        except redis.RedisError as e:
            self._healthy = False
            logger.error(
                "redis_set_working_memory_failed",
                agent_id=self._agent_id,
                error=str(e),
            )

    async def get_stm(self, limit: int = 100) -> list[MemoryEntry]:
        """Get short-term memory entries (newest first, max 100)."""
        try:
            effective_limit = min(limit, _STM_LIMIT)
            raw_list: list[bytes] = await self._redis.lrange(
                self._key("stm"), 0, effective_limit - 1
            )
            result = [MemoryEntry.model_validate_json(item) for item in raw_list]
            self._healthy = True
            return result
        except redis.RedisError as e:
            self._healthy = False
            logger.warning(
                "redis_get_stm_failed",
                agent_id=self._agent_id,
                error=str(e),
            )
            return []

    async def add_stm(self, entry: MemoryEntry) -> None:
        """Add entry to short-term memory (auto-trims to capacity limit)."""
        try:
            key = self._key("stm")
            await self._redis.lpush(key, entry.model_dump_json())
            await self._redis.ltrim(key, 0, _STM_LIMIT - 1)
            self._healthy = True
        except redis.RedisError as e:
            self._healthy = False
            logger.error(
                "redis_add_stm_failed",
                agent_id=self._agent_id,
                error=str(e),
            )

    # --- Self Reflection ---

    async def get_self_reflection(self) -> SelfReflectionData:
        """Get self-reflection state."""
        try:
            raw = await self._redis.get(self._key("self_reflection"))
            if raw is None:
                self._healthy = True
                return SelfReflectionData()
            result = SelfReflectionData.model_validate_json(raw)
            self._healthy = True
            return result
        except redis.RedisError as e:
            self._healthy = False
            logger.warning(
                "redis_get_self_reflection_failed",
                agent_id=self._agent_id,
                error=str(e),
            )
            return SelfReflectionData()

    async def update_self_reflection(self, reflection: SelfReflectionData) -> None:
        """Update self-reflection state."""
        try:
            await self._redis.set(
                self._key("self_reflection"), reflection.model_dump_json()
            )
            self._healthy = True
        except redis.RedisError as e:
            self._healthy = False
            logger.error(
                "redis_update_self_reflection_failed",
                agent_id=self._agent_id,
                error=str(e),
            )

    # --- CC Decision ---

    async def get_last_cc_decision(self) -> dict[str, Any] | None:
        """Get the last CC decision broadcast."""
        try:
            raw = await self._redis.get(self._key("cc_decision"))
            if raw is None:
                self._healthy = True
                return None
            result = json.loads(raw)
            self._healthy = True
            return result
        except redis.RedisError as e:
            self._healthy = False
            logger.warning(
                "redis_get_last_cc_decision_failed",
                agent_id=self._agent_id,
                error=str(e),
            )
            return None

    async def set_cc_decision(self, decision: dict[str, Any]) -> None:
        """Store the latest CC decision."""
        try:
            await self._redis.set(self._key("cc_decision"), json.dumps(decision))
            self._healthy = True
        except redis.RedisError as e:
            self._healthy = False
            logger.error(
                "redis_set_cc_decision_failed",
                agent_id=self._agent_id,
                error=str(e),
            )

    # --- Generic Section Access ---

    async def get_section(self, section: str) -> dict[str, Any]:
        """Get a raw SAS section by name."""
        try:
            raw = await self._redis.get(self._key(section))
            if raw is None:
                self._healthy = True
                return {}
            result = json.loads(raw)
            self._healthy = True
            return result
        except redis.RedisError as e:
            self._healthy = False
            logger.warning(
                "redis_get_section_failed",
                agent_id=self._agent_id,
                section=section,
                error=str(e),
            )
            return {}

    async def update_section(self, section: str, data: dict[str, Any]) -> None:
        """Update a raw SAS section."""
        try:
            await self._redis.set(self._key(section), json.dumps(data))
            self._healthy = True
        except redis.RedisError as e:
            self._healthy = False
            logger.error(
                "redis_update_section_failed",
                agent_id=self._agent_id,
                section=section,
                error=str(e),
            )

    # --- Snapshot ---

    async def snapshot(self) -> dict[str, Any]:
        """Get a full snapshot of all SAS sections (for CC input / checkpointing)."""
        try:
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
        except redis.RedisError as e:
            self._healthy = False
            logger.warning(
                "redis_snapshot_failed",
                agent_id=self._agent_id,
                error=str(e),
            )
            # Return snapshot with defaults
            return {
                "agent_id": self._agent_id,
                "percepts": PerceptData().model_dump(),
                "goals": GoalData().model_dump(),
                "social": SocialData().model_dump(),
                "plans": PlanData().model_dump(),
                "action_history": [],
                "working_memory": [],
                "stm": [],
                "self_reflection": SelfReflectionData().model_dump(),
                "cc_decision": None,
            }

    # --- Lifecycle ---

    async def initialize(self) -> None:
        """Initialize the SAS (ensure default values exist).

        Sets each section to its default (empty) value only if the key
        does not already exist, preserving any previously stored state.
        """
        try:
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
            self._healthy = True
        except redis.RedisError as e:
            self._healthy = False
            logger.error(
                "redis_initialize_failed",
                agent_id=self._agent_id,
                error=str(e),
            )
            raise

    async def clear(self) -> None:
        """Clear all SAS data for this agent."""
        try:
            pattern = f"piano:{self._agent_id}:*"
            cursor: int | bytes = 0
            while True:
                cursor, keys = await self._redis.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break
            self._healthy = True
        except redis.RedisError as e:
            self._healthy = False
            logger.error(
                "redis_clear_failed",
                agent_id=self._agent_id,
                error=str(e),
            )
            raise
