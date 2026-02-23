"""Shared test fixtures for PIANO tests."""

from __future__ import annotations

import pytest

from piano.core.types import AgentId


@pytest.fixture
def agent_id() -> AgentId:
    """Default test agent ID."""
    return "test-agent-001"


@pytest.fixture
def agent_ids() -> list[AgentId]:
    """Multiple test agent IDs."""
    return [f"test-agent-{i:03d}" for i in range(1, 6)]
