"""Tests for SkillExecutor."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

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


class FailingSAS:
    """SAS mock that raises on add_action."""

    def __init__(self) -> None:
        self.actions: list[ActionHistoryEntry] = []

    async def add_action(self, entry: ActionHistoryEntry) -> None:
        raise RuntimeError("SAS connection lost")


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
    async def test_execute_known_skill(self, executor: SkillExecutor, sas: MockSAS) -> None:
        decision = CCDecision(action="move", action_params={"x": 1, "y": 2, "z": 3})
        await executor.on_broadcast(decision)
        # Wait for background task to complete
        if executor._current_task:
            await executor._current_task
        assert len(sas.actions) == 1
        assert sas.actions[0].action == "move_to"
        assert sas.actions[0].success is True

    async def test_idle_action_does_nothing(self, executor: SkillExecutor, sas: MockSAS) -> None:
        decision = CCDecision(action="idle")
        await executor.on_broadcast(decision)
        assert len(sas.actions) == 0

    async def test_empty_action_does_nothing(self, executor: SkillExecutor, sas: MockSAS) -> None:
        decision = CCDecision(action="")
        await executor.on_broadcast(decision)
        assert len(sas.actions) == 0

    async def test_unmapped_action_skipped(self, executor: SkillExecutor, sas: MockSAS) -> None:
        decision = CCDecision(action="fly_to_moon", action_params={})
        await executor.on_broadcast(decision)
        # Unmapped actions are skipped (no skill to execute)
        assert len(sas.actions) == 0

    async def test_timeout_records_failure(self, registry, sas: MockSAS) -> None:
        slow_bridge = SlowBridgeClient(delay=2.0)
        executor = SkillExecutor(registry=registry, bridge=slow_bridge, sas=sas, timeout_s=0.1)
        decision = CCDecision(action="move", action_params={"x": 0, "y": 0, "z": 0})
        await executor.on_broadcast(decision)
        if executor._current_task:
            await executor._current_task
        assert len(sas.actions) == 1
        assert sas.actions[0].success is False
        assert "timeout" in sas.actions[0].actual_result

    async def test_bridge_error_records_failure(self, registry, sas: MockSAS) -> None:
        fail_bridge = FailBridgeClient()
        executor = SkillExecutor(registry=registry, bridge=fail_bridge, sas=sas, timeout_s=5.0)
        decision = CCDecision(action="chat", action_params={"message": "hi"})
        await executor.on_broadcast(decision)
        if executor._current_task:
            await executor._current_task
        assert len(sas.actions) == 1
        assert sas.actions[0].success is False
        assert "bridge connection failed" in sas.actions[0].actual_result

    async def test_new_broadcast_cancels_previous(self, registry, sas: MockSAS) -> None:
        slow_bridge = SlowBridgeClient(delay=10.0)
        executor = SkillExecutor(registry=registry, bridge=slow_bridge, sas=sas, timeout_s=30.0)
        # Start a slow skill
        decision1 = CCDecision(action="move", action_params={"x": 0, "y": 0, "z": 0})
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
        executor = SkillExecutor(registry=registry, bridge=slow_bridge, sas=sas, timeout_s=30.0)
        decision = CCDecision(action="move", action_params={"x": 0, "y": 0, "z": 0})
        await executor.on_broadcast(decision)
        task = executor._current_task
        assert task is not None

        await executor.shutdown()
        assert task.done()


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestSkillExecutorEdgeCases:
    """Edge case tests for cancellation, rapid broadcasts, and unusual inputs."""

    async def test_cancel_during_execution(self, registry, sas: MockSAS) -> None:
        """Cancel an actively executing skill."""
        slow_bridge = SlowBridgeClient(delay=5.0)
        executor = SkillExecutor(registry=registry, bridge=slow_bridge, sas=sas, timeout_s=30.0)

        decision = CCDecision(action="move", action_params={"x": 1, "y": 2, "z": 3})
        await executor.on_broadcast(decision)
        task = executor._current_task
        assert task is not None

        # Give the task time to actually start executing
        await asyncio.sleep(0.1)
        assert not task.done()

        # Manually cancel and wait for cleanup
        await executor._cancel_current()

        assert task.done()
        assert executor._current_task is None

        # Should have recorded cancellation
        assert len(sas.actions) == 1
        assert sas.actions[0].success is False
        assert "cancelled" in sas.actions[0].actual_result

    async def test_broadcast_while_executing_cancels_current(self, registry, sas: MockSAS) -> None:
        """New broadcast interrupts and cancels current skill."""
        slow_bridge = SlowBridgeClient(delay=5.0)
        executor = SkillExecutor(registry=registry, bridge=slow_bridge, sas=sas, timeout_s=30.0)

        # Start first skill
        decision1 = CCDecision(action="move", action_params={"x": 0, "y": 0, "z": 0})
        await executor.on_broadcast(decision1)
        first_task = executor._current_task
        assert first_task is not None

        # Give it a moment to start
        await asyncio.sleep(0.05)

        # Immediately start second skill (should cancel first)
        decision2 = CCDecision(action="chat", action_params={"message": "hello"})
        await executor.on_broadcast(decision2)
        second_task = executor._current_task

        # First task should be cancelled
        assert first_task.done()
        assert second_task is not None
        assert second_task != first_task

        # Wait for second to complete
        await second_task

        # Give the cancelled task time to record
        await asyncio.sleep(0.1)

        # Should have two actions: first cancelled, second succeeded
        assert len(sas.actions) == 2
        assert sas.actions[0].action == "move_to"
        assert sas.actions[0].success is False
        assert "cancelled" in sas.actions[0].actual_result
        assert sas.actions[1].action == "chat"
        assert sas.actions[1].success is True

    async def test_execute_unmapped_action(self, registry, sas: MockSAS) -> None:
        """Attempting to execute an unmapped CC action is silently skipped."""
        bridge = MockBridgeClient()
        executor = SkillExecutor(registry=registry, bridge=bridge, sas=sas, timeout_s=5.0)

        decision = CCDecision(action="teleport", action_params={"x": 100, "y": 200, "z": 300})
        await executor.on_broadcast(decision)

        # Unmapped actions are skipped (no skill to execute)
        assert len(sas.actions) == 0

    async def test_multiple_rapid_broadcasts(self, registry, sas: MockSAS) -> None:
        """Rapid sequential broadcasts cancel previous tasks correctly."""
        slow_bridge = SlowBridgeClient(delay=2.0)
        executor = SkillExecutor(registry=registry, bridge=slow_bridge, sas=sas, timeout_s=30.0)

        # Send 5 rapid broadcasts (using CC action "move" which maps to "move_to")
        decisions = [
            CCDecision(action="move", action_params={"x": i, "y": i, "z": i}) for i in range(5)
        ]

        tasks = []
        for decision in decisions:
            await executor.on_broadcast(decision)
            if executor._current_task:
                tasks.append(executor._current_task)
            await asyncio.sleep(0.05)  # Small delay between broadcasts

        # Wait for the last task to complete
        if executor._current_task:
            await executor._current_task

        # All but the last task should be cancelled
        for task in tasks[:-1]:
            assert task.done()

        # Should have multiple cancelled actions and one success
        assert len(sas.actions) >= 5
        cancelled_count = sum(1 for action in sas.actions if "cancelled" in action.actual_result)
        assert cancelled_count >= 4

    async def test_on_broadcast_with_no_action(self, executor: SkillExecutor, sas: MockSAS) -> None:
        """CCDecision with action=None or empty string does nothing."""
        # Test with None (using empty string as None is not allowed by Pydantic)
        decision1 = CCDecision(action="")
        await executor.on_broadcast(decision1)
        assert len(sas.actions) == 0

        # Test with "idle"
        decision2 = CCDecision(action="idle")
        await executor.on_broadcast(decision2)
        assert len(sas.actions) == 0

    async def test_tick_reports_current_action(self, registry, sas: MockSAS) -> None:
        """Tick reports the currently executing action name."""
        slow_bridge = SlowBridgeClient(delay=2.0)
        executor = SkillExecutor(registry=registry, bridge=slow_bridge, sas=sas, timeout_s=30.0)

        # Start a skill (using CC action "move" which maps to "move_to")
        decision = CCDecision(action="move", action_params={"x": 10, "y": 20, "z": 30})
        await executor.on_broadcast(decision)

        # Tick should report it's executing (current_action is the CC action name)
        result = await executor.tick(sas)
        assert result.data["executing"] is True
        assert result.data["current_action"] == "move"

        # Wait for completion
        if executor._current_task:
            await executor._current_task

        # Tick should report idle
        result = await executor.tick(sas)
        assert result.data["executing"] is False

    async def test_execute_with_missing_params(self, registry, sas: MockSAS) -> None:
        """Executing a skill with missing required params raises an error."""
        bridge = MockBridgeClient()
        executor = SkillExecutor(registry=registry, bridge=bridge, sas=sas, timeout_s=5.0)

        # move_to requires x, y, z - provide only x (using CC action "move")
        decision = CCDecision(action="move", action_params={"x": 1})
        await executor.on_broadcast(decision)

        if executor._current_task:
            await executor._current_task

        # Should record an error
        assert len(sas.actions) == 1
        assert sas.actions[0].success is False
        assert "error" in sas.actions[0].actual_result

    async def test_cancel_when_no_task_running(self, executor: SkillExecutor) -> None:
        """Cancelling when no task is running is a no-op."""
        # Should not raise
        await executor._cancel_current()
        assert executor._current_task is None

    async def test_shutdown_when_idle(self, executor: SkillExecutor) -> None:
        """Shutting down when no task is running is safe."""
        await executor.shutdown()
        assert executor._current_task is None

    async def test_idle_and_wait_skip_with_debug_log(
        self, executor: SkillExecutor, sas: MockSAS, caplog: pytest.LogCaptureFixture
    ) -> None:
        """idle/wait actions are skipped gracefully with a debug log."""
        with caplog.at_level(logging.DEBUG, logger="piano.skills.executor"):
            for action in ("idle", "wait", "think", "observe"):
                caplog.clear()
                decision = CCDecision(action=action)
                await executor.on_broadcast(decision)

                # No skill should be executed (no action recorded)
                assert len(sas.actions) == 0
                # Debug log should mention skipping
                assert any("no skill mapping" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# New Tests for Fixes
# ---------------------------------------------------------------------------


class TestParameterValidation:
    """Tests for parameter validation in executor (Fix #4)."""

    async def test_unknown_params_are_filtered(self, registry, sas: MockSAS) -> None:
        """Unknown params are ignored and logged."""
        bridge = MockBridgeClient()
        executor = SkillExecutor(registry=registry, bridge=bridge, sas=sas, timeout_s=5.0)

        # move_to expects x, y, z - providing extra params
        decision = CCDecision(
            action="move",
            action_params={"x": 1.0, "y": 2.0, "z": 3.0, "speed": 10.0, "sneak": True},
        )
        await executor.on_broadcast(decision)
        if executor._current_task:
            await executor._current_task

        # Should succeed because unknown params are filtered out
        assert len(sas.actions) == 1
        assert sas.actions[0].success is True

    async def test_validate_params_logs_warning_for_unknown(
        self, executor: SkillExecutor, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_validate_params logs warning for unknown parameters."""
        with caplog.at_level(logging.WARNING, logger="piano.skills.executor"):
            result = executor._validate_params(
                "move_to",
                {"x": 1.0, "y": 2.0, "z": 3.0, "unknown_param": True},
                {"x": "float", "y": "float", "z": "float"},
            )

        assert "unknown_param" not in result
        assert result == {"x": 1.0, "y": 2.0, "z": 3.0}
        assert any("unknown params" in rec.message for rec in caplog.records)

    async def test_validate_params_empty_schema_passes_all(self, executor: SkillExecutor) -> None:
        """When schema is empty, all params are passed through."""
        result = executor._validate_params(
            "some_skill",
            {"a": 1, "b": 2},
            {},
        )
        assert result == {"a": 1, "b": 2}

    async def test_get_position_skill_execution(self, registry, sas: MockSAS) -> None:
        """get_position maps to the actual get_position skill and executes."""
        bridge = MockBridgeClient()
        executor = SkillExecutor(registry=registry, bridge=bridge, sas=sas, timeout_s=5.0)

        decision = CCDecision(action="get_position", action_params={})
        await executor.on_broadcast(decision)
        if executor._current_task:
            await executor._current_task

        assert len(sas.actions) == 1
        assert sas.actions[0].action == "get_position"
        assert sas.actions[0].success is True

    async def test_get_inventory_skill_execution(self, registry, sas: MockSAS) -> None:
        """get_inventory maps to the actual get_inventory skill and executes."""
        bridge = MockBridgeClient()
        executor = SkillExecutor(registry=registry, bridge=bridge, sas=sas, timeout_s=5.0)

        decision = CCDecision(action="get_inventory", action_params={})
        await executor.on_broadcast(decision)
        if executor._current_task:
            await executor._current_task

        assert len(sas.actions) == 1
        assert sas.actions[0].action == "get_inventory"
        assert sas.actions[0].success is True


class TestCancelCurrentCleanup:
    """Tests for _cancel_current race condition fix (Fix #6)."""

    async def test_cancel_current_sets_none_before_await(self, registry, sas: MockSAS) -> None:
        """_current_task is set to None BEFORE awaiting the cancelled task."""
        slow_bridge = SlowBridgeClient(delay=10.0)
        executor = SkillExecutor(registry=registry, bridge=slow_bridge, sas=sas, timeout_s=30.0)

        decision = CCDecision(action="move", action_params={"x": 0, "y": 0, "z": 0})
        await executor.on_broadcast(decision)
        task = executor._current_task
        assert task is not None
        await asyncio.sleep(0.05)

        await executor._cancel_current()
        # After cancel, _current_task should be None
        assert executor._current_task is None
        assert task.done()

    async def test_cancel_current_idempotent(self, executor: SkillExecutor) -> None:
        """Calling _cancel_current multiple times is safe."""
        await executor._cancel_current()
        await executor._cancel_current()
        assert executor._current_task is None


class TestSASFailureHandling:
    """Tests for SAS failure handling in _record_action (Fix #7)."""

    async def test_sas_failure_does_not_mask_skill_result(self, registry) -> None:
        """SAS failure in _record_action should not crash the executor."""
        bridge = MockBridgeClient()
        failing_sas = FailingSAS()
        executor = SkillExecutor(registry=registry, bridge=bridge, sas=failing_sas, timeout_s=5.0)

        decision = CCDecision(action="move", action_params={"x": 1, "y": 2, "z": 3})
        await executor.on_broadcast(decision)
        if executor._current_task:
            result = await executor._current_task

        # Skill execution should still complete despite SAS failure
        assert result["success"] is True

    async def test_sas_failure_logged(self, registry, caplog: pytest.LogCaptureFixture) -> None:
        """SAS failure should be logged."""
        bridge = MockBridgeClient()
        failing_sas = FailingSAS()
        executor = SkillExecutor(registry=registry, bridge=bridge, sas=failing_sas, timeout_s=5.0)

        with caplog.at_level(logging.ERROR, logger="piano.skills.executor"):
            decision = CCDecision(action="chat", action_params={"message": "hi"})
            await executor.on_broadcast(decision)
            if executor._current_task:
                await executor._current_task

        assert any("Failed to record action" in rec.message for rec in caplog.records)
