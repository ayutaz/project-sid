"""Tests for the resource limiter with backpressure."""

from __future__ import annotations

import asyncio

import pytest

from piano.scaling.resource_limiter import (
    BackpressurePolicy,
    ResourceLimiter,
    ResourceType,
    ResourceUsage,
)

# ---------------------------------------------------------------------------
# ResourceUsage model tests
# ---------------------------------------------------------------------------


class TestResourceUsage:
    """Tests for the ResourceUsage Pydantic model."""

    def test_defaults(self) -> None:
        usage = ResourceUsage(agent_id="agent-001")
        assert usage.agent_id == "agent-001"
        assert usage.concurrent_llm == 0
        assert usage.memory_mb == 0.0
        assert usage.throttled is False

    def test_custom_values(self) -> None:
        usage = ResourceUsage(
            agent_id="agent-002",
            concurrent_llm=3,
            memory_mb=48.5,
            throttled=True,
        )
        assert usage.concurrent_llm == 3
        assert usage.memory_mb == 48.5
        assert usage.throttled is True

    def test_serialization_round_trip(self) -> None:
        usage = ResourceUsage(agent_id="agent-001", concurrent_llm=2, memory_mb=32.0)
        data = usage.model_dump()
        restored = ResourceUsage(**data)
        assert restored == usage


# ---------------------------------------------------------------------------
# BackpressurePolicy tests
# ---------------------------------------------------------------------------


class TestBackpressurePolicy:
    """Tests for the BackpressurePolicy class."""

    def test_default_thresholds(self) -> None:
        policy = BackpressurePolicy()
        assert policy.warning_threshold == 0.8
        assert policy.throttle_threshold == 0.9

    def test_custom_thresholds(self) -> None:
        policy = BackpressurePolicy(warning_threshold=0.5, throttle_threshold=0.7)
        assert policy.warning_threshold == 0.5
        assert policy.throttle_threshold == 0.7

    def test_invalid_warning_threshold_zero(self) -> None:
        with pytest.raises(ValueError, match="warning_threshold"):
            BackpressurePolicy(warning_threshold=0.0)

    def test_invalid_throttle_threshold_over_one(self) -> None:
        with pytest.raises(ValueError, match="throttle_threshold"):
            BackpressurePolicy(throttle_threshold=1.5)

    def test_warning_greater_than_throttle_raises(self) -> None:
        with pytest.raises(ValueError, match="warning_threshold"):
            BackpressurePolicy(warning_threshold=0.95, throttle_threshold=0.8)

    def test_should_warn_llm(self) -> None:
        policy = BackpressurePolicy(warning_threshold=0.8, throttle_threshold=0.9)
        # 80% of 10 = 8 -> usage 9 (> 8) should warn
        usage = ResourceUsage(agent_id="a", concurrent_llm=9)
        assert policy.should_warn(usage, max_concurrent_llm=10, max_memory_mb=64) is True

    def test_should_not_warn_at_threshold(self) -> None:
        policy = BackpressurePolicy(warning_threshold=0.8, throttle_threshold=0.9)
        # Exactly at 80% boundary (8 == 10 * 0.8) should NOT warn with > comparison
        usage = ResourceUsage(agent_id="a", concurrent_llm=8)
        assert policy.should_warn(usage, max_concurrent_llm=10, max_memory_mb=64) is False

    def test_should_not_warn_below_threshold(self) -> None:
        policy = BackpressurePolicy(warning_threshold=0.8, throttle_threshold=0.9)
        usage = ResourceUsage(agent_id="a", concurrent_llm=7)
        assert policy.should_warn(usage, max_concurrent_llm=10, max_memory_mb=64) is False

    def test_should_throttle_llm(self) -> None:
        policy = BackpressurePolicy(warning_threshold=0.8, throttle_threshold=0.9)
        # 90% of 10 = 9 -> usage 10 (> 9) should throttle
        usage = ResourceUsage(agent_id="a", concurrent_llm=10)
        assert policy.should_throttle(usage, max_concurrent_llm=10, max_memory_mb=64) is True

    def test_should_not_throttle_at_threshold(self) -> None:
        policy = BackpressurePolicy(warning_threshold=0.8, throttle_threshold=0.9)
        # Exactly at 90% boundary (9 == 10 * 0.9) should NOT throttle with > comparison
        usage = ResourceUsage(agent_id="a", concurrent_llm=9)
        assert policy.should_throttle(usage, max_concurrent_llm=10, max_memory_mb=64) is False

    def test_should_not_throttle_below_threshold(self) -> None:
        policy = BackpressurePolicy(warning_threshold=0.8, throttle_threshold=0.9)
        usage = ResourceUsage(agent_id="a", concurrent_llm=8)
        assert policy.should_throttle(usage, max_concurrent_llm=10, max_memory_mb=64) is False

    def test_should_throttle_memory(self) -> None:
        policy = BackpressurePolicy(warning_threshold=0.8, throttle_threshold=0.9)
        # 90% of 64 = 57.6 -> memory 58 (> 57.6) should throttle
        usage = ResourceUsage(agent_id="a", memory_mb=58.0)
        assert policy.should_throttle(usage, max_concurrent_llm=3, max_memory_mb=64) is True

    def test_should_warn_memory(self) -> None:
        policy = BackpressurePolicy(warning_threshold=0.8, throttle_threshold=0.9)
        # 80% of 64 = 51.2 -> memory 52 (> 51.2) should warn
        usage = ResourceUsage(agent_id="a", memory_mb=52.0)
        assert policy.should_warn(usage, max_concurrent_llm=3, max_memory_mb=64) is True

    def test_no_warn_or_throttle_when_limits_zero(self) -> None:
        """Zero limits means no restriction -- never warn or throttle."""
        policy = BackpressurePolicy()
        usage = ResourceUsage(agent_id="a", concurrent_llm=100, memory_mb=1000.0)
        assert policy.should_warn(usage, max_concurrent_llm=0, max_memory_mb=0) is False
        assert policy.should_throttle(usage, max_concurrent_llm=0, max_memory_mb=0) is False


# ---------------------------------------------------------------------------
# ResourceLimiter tests
# ---------------------------------------------------------------------------


class TestResourceLimiter:
    """Tests for the ResourceLimiter class."""

    def test_default_config(self) -> None:
        limiter = ResourceLimiter()
        assert limiter.max_concurrent_llm_per_agent == 3
        assert limiter.max_concurrent_llm_per_worker == 50
        assert limiter.max_memory_per_agent_mb == 64

    def test_custom_config(self) -> None:
        limiter = ResourceLimiter(
            max_concurrent_llm_per_agent=5,
            max_concurrent_llm_per_worker=100,
            max_memory_per_agent_mb=128,
        )
        assert limiter.max_concurrent_llm_per_agent == 5
        assert limiter.max_concurrent_llm_per_worker == 100
        assert limiter.max_memory_per_agent_mb == 128

    async def test_acquire_and_release_llm(self) -> None:
        limiter = ResourceLimiter(max_concurrent_llm_per_agent=3)
        agent = "agent-001"

        ok = await limiter.acquire(agent, ResourceType.LLM)
        assert ok is True

        usage = limiter.get_usage(agent)
        assert usage.concurrent_llm == 1

        await limiter.release(agent, ResourceType.LLM)
        usage = limiter.get_usage(agent)
        assert usage.concurrent_llm == 0

    async def test_acquire_up_to_agent_limit(self) -> None:
        limiter = ResourceLimiter(max_concurrent_llm_per_agent=2)
        agent = "agent-001"

        assert await limiter.acquire(agent, ResourceType.LLM) is True
        assert await limiter.acquire(agent, ResourceType.LLM) is True
        # Third acquire should fail (agent limit = 2, throttle at 90% = 1.8 -> 2 hits)
        # With default policy 0.9 threshold, 2 / 2 = 100% >= 90% -> throttled
        assert await limiter.acquire(agent, ResourceType.LLM) is False

    async def test_acquire_respects_worker_limit(self) -> None:
        limiter = ResourceLimiter(
            max_concurrent_llm_per_agent=10,
            max_concurrent_llm_per_worker=2,
        )

        # Two different agents, worker limit is 2
        assert await limiter.acquire("a1", ResourceType.LLM) is True
        assert await limiter.acquire("a2", ResourceType.LLM) is True
        # Worker level is at capacity
        assert await limiter.acquire("a1", ResourceType.LLM) is False

    async def test_release_frees_slot(self) -> None:
        limiter = ResourceLimiter(max_concurrent_llm_per_agent=2)
        agent = "agent-001"

        await limiter.acquire(agent, ResourceType.LLM)
        await limiter.acquire(agent, ResourceType.LLM)
        # At limit
        assert await limiter.acquire(agent, ResourceType.LLM) is False

        # Release one
        await limiter.release(agent, ResourceType.LLM)
        # Should be able to acquire again
        assert await limiter.acquire(agent, ResourceType.LLM) is True

    async def test_get_usage_unregistered_agent(self) -> None:
        limiter = ResourceLimiter()
        usage = limiter.get_usage("nonexistent")
        assert usage.agent_id == "nonexistent"
        assert usage.concurrent_llm == 0
        assert usage.memory_mb == 0.0
        assert usage.throttled is False

    async def test_is_throttled_default_false(self) -> None:
        limiter = ResourceLimiter()
        assert limiter.is_throttled("agent-001") is False

    async def test_throttle_via_memory(self) -> None:
        limiter = ResourceLimiter(max_memory_per_agent_mb=100)
        agent = "agent-001"

        # Set memory above throttle threshold (90% of 100 = 90)
        await limiter.set_memory_usage(agent, 95.0)
        assert limiter.is_throttled(agent) is True
        assert limiter.get_usage(agent).throttled is True

    async def test_unthrottle_when_memory_drops(self) -> None:
        limiter = ResourceLimiter(max_memory_per_agent_mb=100)
        agent = "agent-001"

        await limiter.set_memory_usage(agent, 95.0)
        assert limiter.is_throttled(agent) is True

        # Drop memory below threshold
        await limiter.set_memory_usage(agent, 50.0)
        assert limiter.is_throttled(agent) is False

    async def test_memory_throttle_blocks_llm_acquire(self) -> None:
        limiter = ResourceLimiter(
            max_concurrent_llm_per_agent=10,
            max_memory_per_agent_mb=100,
        )
        agent = "agent-001"

        # Throttle via memory
        await limiter.set_memory_usage(agent, 95.0)
        # LLM acquire should be refused due to memory throttle
        assert await limiter.acquire(agent, ResourceType.LLM) is False

    async def test_memory_acquire_returns_true_when_not_throttled(self) -> None:
        limiter = ResourceLimiter(max_memory_per_agent_mb=100)
        agent = "agent-001"
        ok = await limiter.acquire(agent, ResourceType.MEMORY)
        assert ok is True

    async def test_memory_acquire_returns_false_when_throttled(self) -> None:
        limiter = ResourceLimiter(max_memory_per_agent_mb=100)
        agent = "agent-001"
        await limiter.set_memory_usage(agent, 95.0)
        ok = await limiter.acquire(agent, ResourceType.MEMORY)
        assert ok is False

    async def test_unknown_resource_type_returns_false(self) -> None:
        limiter = ResourceLimiter()
        ok = await limiter.acquire("agent-001", "unknown_resource")
        assert ok is False

    async def test_release_without_acquire_is_safe(self) -> None:
        limiter = ResourceLimiter()
        # Should not raise
        await limiter.release("agent-001", ResourceType.LLM)

    async def test_release_unknown_type_is_safe(self) -> None:
        limiter = ResourceLimiter()
        # Should not raise
        await limiter.release("agent-001", "unknown")

    async def test_worker_llm_in_use_property(self) -> None:
        limiter = ResourceLimiter(max_concurrent_llm_per_agent=5)

        await limiter.acquire("a1", ResourceType.LLM)
        await limiter.acquire("a2", ResourceType.LLM)
        assert limiter.worker_llm_in_use == 2

        await limiter.release("a1", ResourceType.LLM)
        assert limiter.worker_llm_in_use == 1

    async def test_custom_policy(self) -> None:
        """Use a stricter throttle policy (50% / 60%)."""
        policy = BackpressurePolicy(warning_threshold=0.5, throttle_threshold=0.6)
        limiter = ResourceLimiter(
            max_concurrent_llm_per_agent=10,
            policy=policy,
        )
        agent = "agent-001"

        # Acquire 7 out of 10 -> 70% > 60% threshold -> should hit throttle
        for _ in range(7):
            await limiter.acquire(agent, ResourceType.LLM)

        # Next acquire should be throttled (7/10 = 70% > 60%)
        assert await limiter.acquire(agent, ResourceType.LLM) is False
        assert limiter.is_throttled(agent) is True

    async def test_multiple_agents_independent(self) -> None:
        limiter = ResourceLimiter(
            max_concurrent_llm_per_agent=2,
            max_concurrent_llm_per_worker=10,
        )

        await limiter.acquire("a1", ResourceType.LLM)
        await limiter.acquire("a1", ResourceType.LLM)
        await limiter.acquire("a2", ResourceType.LLM)

        usage_a1 = limiter.get_usage("a1")
        usage_a2 = limiter.get_usage("a2")

        assert usage_a1.concurrent_llm == 2
        assert usage_a2.concurrent_llm == 1

    async def test_concurrent_acquire_release(self) -> None:
        """Multiple coroutines acquire and release without corruption."""
        limiter = ResourceLimiter(
            max_concurrent_llm_per_agent=100,
            max_concurrent_llm_per_worker=100,
        )
        agent = "agent-001"
        n_tasks = 20

        async def worker() -> None:
            ok = await limiter.acquire(agent, ResourceType.LLM)
            if ok:
                await asyncio.sleep(0.001)
                await limiter.release(agent, ResourceType.LLM)

        await asyncio.gather(*[worker() for _ in range(n_tasks)])

        # All slots should be released
        usage = limiter.get_usage(agent)
        assert usage.concurrent_llm == 0
        assert limiter.worker_llm_in_use == 0

    async def test_throttle_clears_after_release(self) -> None:
        """After releasing enough slots, throttle should clear."""
        limiter = ResourceLimiter(max_concurrent_llm_per_agent=3)
        agent = "agent-001"

        # Fill up -- acquire 3
        await limiter.acquire(agent, ResourceType.LLM)
        await limiter.acquire(agent, ResourceType.LLM)
        await limiter.acquire(agent, ResourceType.LLM)
        # 3/3 = 100% > 90% -- now throttled on next attempt
        assert await limiter.acquire(agent, ResourceType.LLM) is False
        assert limiter.is_throttled(agent) is True

        # Release two
        await limiter.release(agent, ResourceType.LLM)
        await limiter.release(agent, ResourceType.LLM)

        # Should no longer be throttled (1/3 = 33%)
        assert limiter.is_throttled(agent) is False
        # Can acquire again
        assert await limiter.acquire(agent, ResourceType.LLM) is True

    async def test_deregister_agent(self) -> None:
        """deregister_agent removes all tracking state for the agent."""
        limiter = ResourceLimiter(
            max_concurrent_llm_per_agent=5,
            max_concurrent_llm_per_worker=10,
        )
        agent = "agent-001"

        # Acquire some slots
        await limiter.acquire(agent, ResourceType.LLM)
        await limiter.acquire(agent, ResourceType.LLM)
        assert limiter.worker_llm_in_use == 2

        # Deregister
        await limiter.deregister_agent(agent)

        # Agent state should be gone
        usage = limiter.get_usage(agent)
        assert usage.concurrent_llm == 0
        assert limiter.is_throttled(agent) is False

        # Worker slots should be freed
        assert limiter.worker_llm_in_use == 0

    async def test_deregister_unknown_agent(self) -> None:
        """deregister_agent on unknown agent is a no-op."""
        limiter = ResourceLimiter()
        # Should not raise
        await limiter.deregister_agent("nonexistent")

    async def test_atomic_acquire_under_contention(self) -> None:
        """Multiple coroutines acquiring concurrently should not exceed limits."""
        limiter = ResourceLimiter(
            max_concurrent_llm_per_agent=5,
            max_concurrent_llm_per_worker=5,
        )
        agent = "agent-001"
        acquired_count = 0

        async def try_acquire() -> None:
            nonlocal acquired_count
            ok = await limiter.acquire(agent, ResourceType.LLM)
            if ok:
                acquired_count += 1

        # Run many concurrent acquires
        await asyncio.gather(*[try_acquire() for _ in range(20)])

        # Should never exceed the limit
        assert acquired_count <= 5
        assert limiter.worker_llm_in_use <= 5
        usage = limiter.get_usage(agent)
        assert usage.concurrent_llm <= 5
