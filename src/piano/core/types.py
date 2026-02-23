"""Shared type definitions for the PIANO architecture."""

from __future__ import annotations

__all__ = [
    "ActionHistoryEntry",
    "AgentId",
    "BridgeCommand",
    "BridgeEvent",
    "CCDecision",
    "GoalData",
    "LLMRequest",
    "LLMResponse",
    "MemoryEntry",
    "ModuleResult",
    "ModuleTier",
    "PerceptData",
    "PlanData",
    "SelfReflectionData",
    "SocialData",
]

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# --- Agent Identity ---

AgentId = str  # e.g., "agent-001"


# --- Module Tiers ---


class ModuleTier(StrEnum):
    """Module execution speed tiers (from design doc 01)."""

    FAST = "fast"  # non-LLM, <100ms (action awareness, skill exec)
    MID = "mid"  # lightweight LLM or rule-based, 100ms-1s (CC, memory recall)
    SLOW = "slow"  # full LLM calls, 1-10s (goal gen, planning, social awareness)


# --- Module Result ---


class ModuleResult(BaseModel):
    """Result returned by a module after a tick execution."""

    module_name: str
    tier: ModuleTier
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


# --- CC Decision ---


class CCDecision(BaseModel):
    """Decision made by the Cognitive Controller after information bottleneck + ignition."""

    cycle_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    summary: str = ""  # compressed information summary
    action: str = ""  # decided action (e.g., "mine", "talk", "idle")
    action_params: dict[str, Any] = Field(default_factory=dict)
    speaking: str | None = None  # utterance if the agent should speak
    reasoning: str = ""  # chain-of-thought reasoning
    salience_scores: dict[str, float] = Field(default_factory=dict)
    raw_llm_response: str = ""


# --- SAS Sections ---


class PerceptData(BaseModel):
    """Current perception of the environment."""

    nearby_players: list[str] = Field(default_factory=list)
    nearby_blocks: list[dict[str, Any]] = Field(default_factory=list)
    inventory: dict[str, int] = Field(default_factory=dict)
    health: float = 20.0
    hunger: float = 20.0
    position: dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    time_of_day: int = 0
    weather: str = "clear"
    chat_messages: list[dict[str, Any]] = Field(default_factory=list)


class GoalData(BaseModel):
    """Current goals of the agent."""

    current_goal: str = ""
    goal_stack: list[str] = Field(default_factory=list)
    sub_goals: list[str] = Field(default_factory=list)
    completed_goals: list[str] = Field(default_factory=list)


class SocialData(BaseModel):
    """Social awareness data."""

    relationships: dict[str, float] = Field(default_factory=dict)  # agent_id -> affinity
    emotions: dict[str, float] = Field(default_factory=dict)  # emotion -> intensity (0-10)
    recent_interactions: list[dict[str, Any]] = Field(default_factory=list)


class PlanData(BaseModel):
    """Current plan state."""

    current_plan: list[str] = Field(default_factory=list)
    plan_status: str = "idle"  # idle, executing, replanning
    current_step: int = 0


class ActionHistoryEntry(BaseModel):
    """A single action history entry."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: str = ""
    expected_result: str = ""
    actual_result: str = ""
    success: bool = True


class SelfReflectionData(BaseModel):
    """Self-reflection state."""

    last_reflection: str = ""
    insights: list[str] = Field(default_factory=list)
    personality_traits: dict[str, float] = Field(default_factory=dict)  # Big Five


# --- Memory Types ---


class MemoryEntry(BaseModel):
    """A single memory entry (used for both WM and STM)."""

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    content: str = ""
    category: str = ""  # "perception", "action", "social", "reflection"
    importance: float = 0.5  # 0.0 - 1.0
    source_module: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- LLM Types ---


class LLMRequest(BaseModel):
    """A request to the LLM provider."""

    prompt: str
    system_prompt: str = ""
    model: str = ""  # empty = use default for tier
    tier: ModuleTier = ModuleTier.SLOW
    temperature: float = 0.0
    max_tokens: int = 1024
    json_mode: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Response from the LLM provider."""

    content: str = ""
    model: str = ""
    usage: dict[str, int] = Field(default_factory=dict)  # prompt_tokens, completion_tokens
    cached: bool = False
    latency_ms: float = 0.0
    cost_usd: float = 0.0


# --- Bridge Types ---


class BridgeCommand(BaseModel):
    """Command sent from Python to the Mineflayer bridge."""

    id: UUID = Field(default_factory=uuid4)
    action: str  # "move", "mine", "craft", "chat", "look", etc.
    params: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = 5000


class BridgeEvent(BaseModel):
    """Event received from the Mineflayer bridge."""

    event_type: str  # "perception", "chat", "error", "action_complete"
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
