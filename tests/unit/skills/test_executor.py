"""Tests for SkillExecutor."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from piano.core.types import (
    ActionHistoryEntry,
    CCDecision,
    ModuleTier,
)
from piano.skills.basic import create_default_registry
from piano.skills.executor import SkillExecutor


class MockBridgeClient:
    """Mock bridge client for testing."""

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self._response = response or {"success": True, "data": {}}

    async def send_command(self, cmd: object) -> dict[str, Any]:
        return self._response


class SlowBridgeClient:
    """Bridge client that delays responses (for timeout tests)."""

    def __init__(self, delay: float) -> None:
        self._delay = delay

    async def send_command(self, cmd: object) -> dict[str, Any]:
        await asyncio.sleep(self._delay)
        return {"success": True}


class FailBridgeClient:
    """Bridge client that raises exceptions."""

    async def send_command(self, cmd: object) -> dict[str, Any]:
        raise RuntimeError("bridge connection failed")


class MockSAS:
    """Minimal SAS mock that records action history calls."""

    def __init__(self) -> None:
        self.actions: list[ActionHistoryEntry] = []

    async def add_action(self, entry: ActionHistoryEntry) -> None:
        self.actions.append(entry)


@pytest.fixture
def bridge() -> MockBridgeClient:
    return MockBridgeClient()


@pytest.fixture
def sas() -> MockSAS:
    return MockSAS()


@pytest.fixture
def registry():
    return create_default_registry()


@pytest.fixture
def executor(registry, bridge, sas) -> SkillExecutor:
    return SkillExecutor(registry=registry, bridge=bridge, sas=sas, timeout_s=5.0)


class TestSkillExecutorProperties:
    def test_name(self, executor: SkillExecutor) -> None:
        assert executor.name == "skill_executor"

    def test_tier_is_fast(self, executor: SkillExecutor) -> None:
        assert executor.tier == ModuleTier.FAST

    def test_timeout(self, executor: SkillExecutor) -> None:
        assert executor.timeout_s == 5.0


class TestSkillExecutorTick:
    async def test_tick_when_idle(self, executor: SkillExecutor, sas: MockSAS) -> None:
        result = await executor.tick(sas)
        assert result.module_name == "skill_executor"
        assert result.data["executing"] is False
        assert result.data["current_action"] == ""


class TestSkillExecutorBroadcast:
    async def test_execute_known_skill(
        self, executor: SkillExecutor, sas: MockSAS
    ) -> None:
        decision = CCDecision(action="move_to", action_params={"x": 1, "y": 2, "z": 3})
        await executor.on_broadcast(decision)
        # Wait for background task to complete
        if executor._current_task:
            await executor._current_task
        assert len(sas.actions) == 1
        assert sas.actions[0].action == "move_to"
        assert sas.actions[0].success is True

    async def test_idle_action_does_nothing(
        self, executor: SkillExecutor, sas: MockSAS
    ) -> None:
        decision = CCDecision(action="idle")
        await executor.on_broadcast(decision)
        assert len(sas.actions) == 0

    async def test_empty_action_does_nothing(
        self, executor: SkillExecutor, sas: MockSAS
    ) -> None:
        decision = CCDecision(action="")
        await executor.on_broadcast(decision)
        assert len(sas.actions) == 0

    async def test_unknown_skill_records_failure(
        self, executor: SkillExecutor, sas: MockSAS
    ) -> None:
        decision = CCDecision(action="fly_to_moon", action_params={})
        await executor.on_broadcast(decision)
        assert len(sas.actions) == 1
        assert sas.actions[0].success is False
        assert "not found" in sas.actions[0].actual_result

    async def test_timeout_records_failure(self, registry, sas: MockSAS) -> None:
        slow_bridge = SlowBridgeClient(delay=2.0)
        executor = SkillExecutor(
            registry=registry, bridge=slow_bridge, sas=sas, timeout_s=0.1
        )
        decision = CCDecision(action="move_to", action_params={"x": 0, "y": 0, "z": 0})
        await executor.on_broadcast(decision)
        if executor._current_task:
            await executor._current_task
        assert len(sas.actions) == 1
        assert sas.actions[0].success is False
        assert "timeout" in sas.actions[0].actual_result

    async def test_bridge_error_records_failure(self, registry, sas: MockSAS) -> None:
        fail_bridge = FailBridgeClient()
        executor = SkillExecutor(
            registry=registry, bridge=fail_bridge, sas=sas, timeout_s=5.0
        )
        decision = CCDecision(action="chat", action_params={"message": "hi"})
        await executor.on_broadcast(decision)
        if executor._current_task:
            await executor._current_task
        assert len(sas.actions) == 1
        assert sas.actions[0].success is False
        assert "bridge connection failed" in sas.actions[0].actual_result

    async def test_new_broadcast_cancels_previous(
        self, registry, sas: MockSAS
    ) -> None:
        slow_bridge = SlowBridgeClient(delay=10.0)
        executor = SkillExecutor(
            registry=registry, bridge=slow_bridge, sas=sas, timeout_s=30.0
        )
        # Start a slow skill
        decision1 = CCDecision(action="move_to", action_params={"x": 0, "y": 0, "z": 0})
        await executor.on_broadcast(decision1)
        first_task = executor._current_task

        # Send a new broadcast (should cancel the first)
        decision2 = CCDecision(action="chat", action_params={"message": "hello"})
        await executor.on_broadcast(decision2)

        # First task should be cancelled
        assert first_task is not None
        assert first_task.done()


class TestSkillExecutorShutdown:
    async def test_shutdown_cancels_running(self, registry, sas: MockSAS) -> None:
        slow_bridge = SlowBridgeClient(delay=10.0)
        executor = SkillExecutor(
            registry=registry, bridge=slow_bridge, sas=sas, timeout_s=30.0
        )
        decision = CCDecision(action="move_to", action_params={"x": 0, "y": 0, "z": 0})
        await executor.on_broadcast(decision)
        task = executor._current_task
        assert task is not None

        await executor.shutdown()
        assert task.done()
