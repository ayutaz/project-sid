"""Fault injection integration tests.

Tests system resilience against various failure modes.
"""

from __future__ import annotations

import time

import pytest

from piano.cc.controller import CognitiveController
from piano.core.checkpoint import CheckpointManager
from piano.core.types import GoalData, LLMRequest, LLMResponse
from piano.llm.mock import MockLLMProvider
from piano.testing.chaos import (
    AgentCrashSimulator,
    BridgeFailureSimulator,
    LLMFailureSimulator,
    RedisFailureSimulator,
)
from tests.helpers import InMemorySAS


@pytest.mark.chaos
class TestRedisFailureSimulator:
    """Test Redis failure injection."""

    def test_failure_rate(self) -> None:
        """Simulator should fail at approximately the configured rate."""
        sim = RedisFailureSimulator(failure_rate=0.5)
        failures = sum(1 for _ in range(1000) if sim.should_fail())
        # Allow some variance: 35-65% failure rate
        assert 350 < failures < 650

    async def test_delay_injection(self) -> None:
        """Simulator should inject delays."""
        sim = RedisFailureSimulator(delay_ms=10)
        start = time.perf_counter()
        await sim.maybe_delay()
        elapsed = time.perf_counter() - start
        assert elapsed >= 0.005  # At least 5ms (allow some tolerance)

    def test_stats_tracking(self) -> None:
        """Simulator should track call and failure counts."""
        sim = RedisFailureSimulator(failure_rate=1.0)
        for _ in range(5):
            sim.should_fail()
        assert sim.stats["calls"] == 5
        assert sim.stats["failures"] == 5


@pytest.mark.chaos
class TestBridgeFailureSimulator:
    """Test bridge failure injection."""

    def test_disconnect_after_n_commands(self) -> None:
        """Bridge should disconnect after configured number of commands."""
        sim = BridgeFailureSimulator(disconnect_after=3)
        assert not sim.check_and_maybe_disconnect()  # 1
        assert not sim.check_and_maybe_disconnect()  # 2
        assert sim.check_and_maybe_disconnect()  # 3 -> disconnect
        assert sim.is_disconnected

    def test_reset(self) -> None:
        """Reset should restore simulator state."""
        sim = BridgeFailureSimulator(disconnect_after=1)
        sim.check_and_maybe_disconnect()
        assert sim.is_disconnected
        sim.reset()
        assert not sim.is_disconnected


@pytest.mark.chaos
class TestLLMFailureSimulator:
    """Test LLM failure injection."""

    async def test_error_injection(self) -> None:
        """Simulator should inject RuntimeError."""
        sim = LLMFailureSimulator(error_rate=1.0, timeout_rate=0.0)
        with pytest.raises(RuntimeError, match="injected fault"):
            await sim.maybe_fail()

    async def test_timeout_injection(self) -> None:
        """Simulator should inject TimeoutError."""
        sim = LLMFailureSimulator(error_rate=0.0, timeout_rate=1.0)
        with pytest.raises(TimeoutError, match="injected fault"):
            await sim.maybe_fail()

    async def test_no_failure(self) -> None:
        """When rates are 0, no failures should occur."""
        sim = LLMFailureSimulator(error_rate=0.0, timeout_rate=0.0)
        # Should not raise
        await sim.maybe_fail()

    async def test_stats(self) -> None:
        """Stats should track calls and failures."""
        sim = LLMFailureSimulator(error_rate=1.0, timeout_rate=0.0)
        with pytest.raises(RuntimeError):
            await sim.maybe_fail()
        assert sim.stats["calls"] == 1
        assert sim.stats["errors"] == 1


@pytest.mark.chaos
class TestAgentCrashSimulator:
    """Test agent crash injection."""

    def test_crash_at_tick(self) -> None:
        """Agent should crash at configured tick."""
        sim = AgentCrashSimulator(crash_at_tick=3)
        assert not sim.tick()  # 1
        assert not sim.tick()  # 2
        assert sim.tick()  # 3 -> crash
        assert sim.has_crashed

    def test_reset(self) -> None:
        """Reset should allow re-use."""
        sim = AgentCrashSimulator(crash_at_tick=1)
        sim.tick()
        assert sim.has_crashed
        sim.reset()
        assert not sim.has_crashed


@pytest.mark.chaos
class TestLLMFallbackBehavior:
    """Test that CC falls back to previous decision on LLM failure."""

    async def test_cc_fallback_on_llm_error(self) -> None:
        """CC should reuse last decision when LLM fails."""
        # First, create a successful CC cycle
        mock_llm = MockLLMProvider()
        cc = CognitiveController(llm=mock_llm)
        sas = InMemorySAS()
        await sas.initialize()

        # First tick should succeed (MockLLMProvider returns valid JSON)
        result1 = await cc.tick(sas)
        assert result1.error is None

        # Now make LLM fail
        async def failing_complete(request: LLMRequest) -> LLMResponse:
            raise RuntimeError("Simulated LLM failure")

        mock_llm.complete = failing_complete  # type: ignore[assignment]

        # Second tick should fall back
        result2 = await cc.tick(sas)
        # Should have fallback=True and reuse previous decision
        assert result2.data.get("fallback") is True


@pytest.mark.chaos
class TestCheckpointRecovery:
    """Test checkpoint/restore after simulated crash."""

    async def test_checkpoint_and_restore(self, tmp_path: object) -> None:
        """Agent state should be recoverable from checkpoint."""
        sas = InMemorySAS(agent_id="crash-test-001")
        await sas.initialize()

        # Write some state
        await sas.update_goals(GoalData(current_goal="gather wood"))

        # Save checkpoint
        mgr = CheckpointManager(checkpoint_dir=tmp_path, max_checkpoints=5)  # type: ignore[arg-type]
        await mgr.save("crash-test-001", sas, tick_count=10)

        # Simulate crash: create new SAS
        new_sas = InMemorySAS(agent_id="crash-test-001")
        await new_sas.initialize()

        # Verify state is default
        goals = await new_sas.get_goals()
        assert goals.current_goal == ""

        # Restore from checkpoint
        await mgr.restore("crash-test-001", new_sas)

        # Verify state is restored
        restored_goals = await new_sas.get_goals()
        assert restored_goals.current_goal == "gather wood"
