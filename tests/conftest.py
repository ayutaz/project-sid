"""Shared test fixtures for PIANO tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from piano.core.types import (
    AgentId,
    CCDecision,
    GoalData,
    MemoryEntry,
    ModuleTier,
    PerceptData,
    SocialData,
)

from .helpers import DummyModule, InMemorySAS


@pytest.fixture
def agent_id() -> AgentId:
    """Default test agent ID."""
    return "test-agent-001"


@pytest.fixture
def agent_ids() -> list[AgentId]:
    """Multiple test agent IDs."""
    return [f"test-agent-{i:03d}" for i in range(1, 6)]


@pytest.fixture
def mock_sas(agent_id: AgentId) -> InMemorySAS:
    """In-memory SAS instance for unit tests."""
    return InMemorySAS(agent_id=agent_id)


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Mock LLM provider that returns canned responses."""
    return MockLLMProvider()


@pytest.fixture
def sample_percepts() -> PerceptData:
    """Sample perception data for tests."""
    return PerceptData(
        nearby_players=["Alice", "Bob"],
        nearby_blocks=[{"type": "stone", "x": 10, "y": 64, "z": 10}],
        inventory={"diamond_pickaxe": 1, "cobblestone": 64, "torch": 12},
        health=18.0,
        hunger=15.0,
        position={"x": 100.5, "y": 64.0, "z": -200.3},
        time_of_day=6000,
        weather="clear",
        chat_messages=[
            {"sender": "Alice", "message": "Hello!", "timestamp": "2024-01-01T00:00:00Z"},
        ],
    )


@pytest.fixture
def sample_goals() -> GoalData:
    """Sample goal data for tests."""
    return GoalData(
        current_goal="Build a shelter",
        goal_stack=["Build a shelter", "Gather wood"],
        sub_goals=["Find flat ground", "Collect 20 planks"],
        completed_goals=["Craft tools"],
    )


@pytest.fixture
def sample_social() -> SocialData:
    """Sample social data for tests."""
    return SocialData(
        relationships={"Alice": 0.8, "Bob": 0.3, "Eve": -0.2},
        emotions={"happiness": 6.0, "curiosity": 4.0},
        recent_interactions=[
            {"partner": "Alice", "action": "traded", "timestamp": "2024-01-01T00:00:00Z"},
        ],
    )


@pytest.fixture
def sample_cc_decision() -> CCDecision:
    """Sample CC decision for tests."""
    return CCDecision(
        summary="Near shelter site with Alice, inventory has tools.",
        action="mine",
        action_params={"block": "stone", "count": 10},
        reasoning="Need stone for shelter walls.",
        salience_scores={"percepts": 0.8, "goals": 0.9},
    )


@pytest.fixture
def sample_memory_entries() -> list[MemoryEntry]:
    """Sample memory entries for tests."""
    now = datetime.now(tz=timezone.utc)
    return [
        MemoryEntry(
            content="Found a village at (200, 64, 300)",
            category="perception",
            importance=0.7,
            source_module="action_awareness",
            timestamp=now,
        ),
        MemoryEntry(
            content="Alice offered to help build",
            category="social",
            importance=0.9,
            source_module="social_awareness",
            timestamp=now,
        ),
        MemoryEntry(
            content="Stone pickaxe broke",
            category="action",
            importance=0.5,
            source_module="skill_execution",
            timestamp=now,
        ),
    ]


@pytest.fixture
def dummy_module() -> DummyModule:
    """A simple dummy module for testing."""
    return DummyModule(module_name="test_module", tier=ModuleTier.FAST)


# --- Mock LLM Provider ---


class MockLLMProvider:
    """Mock LLM provider for testing (conforms to LLMProvider protocol)."""

    def __init__(self, default_response: str = "mock response") -> None:
        self.default_response = default_response
        self.call_count: int = 0
        self.last_request: object | None = None
        self._responses: list[str] = []

    def set_responses(self, responses: list[str]) -> None:
        """Pre-load a sequence of responses (consumed in order)."""
        self._responses = list(responses)

    async def complete(self, request: object) -> MockLLMResponse:
        """Return a mock LLM response."""
        self.call_count += 1
        self.last_request = request
        content = self._responses.pop(0) if self._responses else self.default_response
        return MockLLMResponse(content=content)


class MockLLMResponse:
    """Minimal LLM response object for testing."""

    def __init__(self, content: str = "") -> None:
        self.content = content
        self.model = "mock-model"
        self.usage: dict[str, int] = {"prompt_tokens": 10, "completion_tokens": 5}
        self.cached = False
        self.latency_ms = 1.0
        self.cost_usd = 0.0
