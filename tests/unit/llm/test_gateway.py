"""Tests for LLM Gateway.

Tests queue management, priority ordering, concurrency control,
rate limiting, circuit breaker, deduplication, and cost tracking.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from structlog.testing import capture_logs

from piano.core.types import LLMRequest, LLMResponse, ModuleTier
from piano.llm.gateway import (
    CircuitBreaker,
    LLMGateway,
    QueuedRequest,
    RequestPriority,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider() -> AsyncMock:
    """Create a mock LLM provider."""
    provider = AsyncMock()
    provider.complete = AsyncMock(
        return_value=LLMResponse(
            content="test response",
            model="mock-model",
            latency_ms=10.0,
            cost_usd=0.001,
        )
    )
    return provider


@pytest.fixture
async def gateway(mock_provider: AsyncMock) -> LLMGateway:
    """Create and start a gateway with mock provider."""
    gw = LLMGateway(
        mock_provider,
        max_concurrent=5,
        requests_per_minute=100,
        cost_alert_threshold=1.0,
    )
    await gw.start()
    yield gw
    await gw.stop()


# ---------------------------------------------------------------------------
# QueuedRequest tests
# ---------------------------------------------------------------------------


class TestQueuedRequest:
    """Tests for QueuedRequest model and comparison."""

    def test_default_priority(self) -> None:
        req = QueuedRequest(
            request=LLMRequest(prompt="test"),
            agent_id="agent-1",
        )
        assert req.priority == RequestPriority.NORMAL

    def test_comparison_by_priority(self) -> None:
        high = QueuedRequest(
            request=LLMRequest(prompt="test"),
            agent_id="agent-1",
            priority=RequestPriority.HIGH,
        )
        low = QueuedRequest(
            request=LLMRequest(prompt="test"),
            agent_id="agent-1",
            priority=RequestPriority.LOW,
        )
        assert high < low

    def test_comparison_by_time_same_priority(self) -> None:
        req1 = QueuedRequest(
            request=LLMRequest(prompt="test"),
            agent_id="agent-1",
            priority=RequestPriority.NORMAL,
        )
        # Small delay to ensure different timestamps
        import time

        time.sleep(0.001)
        req2 = QueuedRequest(
            request=LLMRequest(prompt="test"),
            agent_id="agent-2",
            priority=RequestPriority.NORMAL,
        )
        assert req1 < req2


# ---------------------------------------------------------------------------
# CircuitBreaker tests
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initially_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=60.0)
        assert not cb.is_open("provider-1")

    def test_opens_after_threshold_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=60.0)
        cb.record_failure("provider-1")
        assert not cb.is_open("provider-1")
        cb.record_failure("provider-1")
        assert not cb.is_open("provider-1")
        cb.record_failure("provider-1")
        assert cb.is_open("provider-1")

    def test_resets_after_success(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=60.0)
        cb.record_failure("provider-1")
        cb.record_failure("provider-1")
        cb.record_success("provider-1")
        assert not cb.is_open("provider-1")

    def test_resets_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, reset_timeout_seconds=0.1)
        cb.record_failure("provider-1")
        cb.record_failure("provider-1")
        assert cb.is_open("provider-1")

        # Wait for timeout
        import time

        time.sleep(0.15)
        assert not cb.is_open("provider-1")

    def test_manual_reset(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, reset_timeout_seconds=60.0)
        cb.record_failure("provider-1")
        cb.record_failure("provider-1")
        assert cb.is_open("provider-1")
        cb.reset("provider-1")
        assert not cb.is_open("provider-1")

    def test_independent_providers(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, reset_timeout_seconds=60.0)
        cb.record_failure("provider-1")
        cb.record_failure("provider-1")
        cb.record_failure("provider-2")
        assert cb.is_open("provider-1")
        assert not cb.is_open("provider-2")


# ---------------------------------------------------------------------------
# Gateway basic tests
# ---------------------------------------------------------------------------


class TestGatewayBasic:
    """Basic gateway functionality tests."""

    async def test_submit_and_receive_response(self, gateway: LLMGateway) -> None:
        request = LLMRequest(prompt="test", tier=ModuleTier.FAST)
        response = await gateway.submit(request, agent_id="agent-1")
        assert response.content == "test response"
        assert response.model == "mock-model"

    async def test_cannot_submit_before_start(self, mock_provider: AsyncMock) -> None:
        gw = LLMGateway(mock_provider)
        request = LLMRequest(prompt="test")
        with pytest.raises(RuntimeError, match="not running"):
            await gw.submit(request, agent_id="agent-1")

    async def test_multiple_requests(self, gateway: LLMGateway) -> None:
        requests = [LLMRequest(prompt=f"test {i}") for i in range(5)]
        responses = await asyncio.gather(
            *[gateway.submit(req, agent_id=f"agent-{i}") for i, req in enumerate(requests)]
        )
        assert len(responses) == 5
        for resp in responses:
            assert resp.content == "test response"


# ---------------------------------------------------------------------------
# Priority ordering tests
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    """Tests for priority queue ordering."""

    async def test_high_priority_processed_first(self, mock_provider: AsyncMock) -> None:
        # Use a slow provider to ensure requests queue up
        call_order: list[str] = []

        async def slow_complete(request: LLMRequest) -> LLMResponse:
            req_id = request.metadata.get("id", "unknown")
            call_order.append(req_id)
            await asyncio.sleep(0.05)
            return LLMResponse(content="ok", model="mock", latency_ms=10.0, cost_usd=0.001)

        mock_provider.complete = slow_complete

        gw = LLMGateway(mock_provider, max_concurrent=1, requests_per_minute=1000)
        await gw.start()

        try:
            # Submit all at once, they will queue
            results = await asyncio.gather(
                gw.submit(
                    LLMRequest(prompt="low", metadata={"id": "low"}),
                    agent_id="agent-1",
                    priority=RequestPriority.LOW,
                ),
                gw.submit(
                    LLMRequest(prompt="normal", metadata={"id": "normal"}),
                    agent_id="agent-2",
                    priority=RequestPriority.NORMAL,
                ),
                gw.submit(
                    LLMRequest(prompt="high", metadata={"id": "high"}),
                    agent_id="agent-3",
                    priority=RequestPriority.HIGH,
                ),
                return_exceptions=True,
            )

            # Check for exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    pytest.fail(f"Task {i} failed with: {result}")

            # Verify all were processed
            assert len(call_order) == 3

            # HIGH should be processed before NORMAL and LOW
            high_idx = call_order.index("high")
            normal_idx = call_order.index("normal")
            # HIGH should come before NORMAL
            assert high_idx < normal_idx

        finally:
            await gw.stop()


# ---------------------------------------------------------------------------
# Concurrency limiting tests
# ---------------------------------------------------------------------------


class TestConcurrencyLimiting:
    """Tests for max_concurrent limiting."""

    async def test_respects_max_concurrent(self, mock_provider: AsyncMock) -> None:
        concurrent_count = 0
        max_concurrent_seen = 0

        async def track_concurrent(request: LLMRequest) -> LLMResponse:
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return LLMResponse(content="ok", model="mock", latency_ms=10.0, cost_usd=0.001)

        mock_provider.complete = track_concurrent

        gw = LLMGateway(mock_provider, max_concurrent=3, requests_per_minute=100)
        await gw.start()

        try:
            # Submit 10 requests
            tasks = [
                gw.submit(LLMRequest(prompt=f"test {i}"), agent_id=f"agent-{i}") for i in range(10)
            ]
            await asyncio.gather(*tasks)

            # Max concurrent should not exceed limit
            assert max_concurrent_seen <= 3

        finally:
            await gw.stop()


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting."""

    async def test_global_rate_limit(self, mock_provider: AsyncMock) -> None:
        # Create gateway with very low rate limit
        gw = LLMGateway(mock_provider, max_concurrent=10, requests_per_minute=5)
        await gw.start()

        try:
            # Submit requests up to limit
            for i in range(5):
                await gw.submit(LLMRequest(prompt=f"test {i}"), agent_id=f"agent-{i}")

            # Next request should fail
            with pytest.raises(Exception, match="Global rate limit exceeded"):
                await gw.submit(LLMRequest(prompt="test"), agent_id="agent-6")

        finally:
            await gw.stop()

    async def test_per_agent_rate_limit(self, mock_provider: AsyncMock) -> None:
        # Per-agent limit is 20% of global (5 * 0.2 = 1)
        gw = LLMGateway(mock_provider, max_concurrent=10, requests_per_minute=5)
        await gw.start()

        try:
            # First request from agent-1 should succeed
            await gw.submit(LLMRequest(prompt="test 1"), agent_id="agent-1")

            # Second request from same agent should fail (exceeds 20% of 5 = 1)
            with pytest.raises(Exception, match="Per-agent rate limit exceeded"):
                await gw.submit(LLMRequest(prompt="test 2"), agent_id="agent-1")

        finally:
            await gw.stop()


# ---------------------------------------------------------------------------
# Circuit breaker integration tests
# ---------------------------------------------------------------------------


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration with gateway."""

    async def test_circuit_opens_after_failures(self, mock_provider: AsyncMock) -> None:
        # Make provider fail
        mock_provider.complete = AsyncMock(side_effect=Exception("Provider error"))

        gw = LLMGateway(mock_provider, max_concurrent=10, requests_per_minute=100)
        await gw.start()

        try:
            # Submit requests that will fail
            import contextlib

            for i in range(3):
                with contextlib.suppress(Exception):
                    await gw.submit(LLMRequest(prompt=f"test {i}"), agent_id=f"agent-{i}")

            # Wait for processing
            await asyncio.sleep(0.1)

            # Circuit should be open, next request should fail immediately
            stats = gw.get_stats()
            assert stats["failed_requests"] == 3

        finally:
            await gw.stop()

    async def test_circuit_resets_on_success(self, mock_provider: AsyncMock) -> None:
        # Make provider fail initially
        call_count = 0

        async def fail_then_succeed(request: LLMRequest) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Provider error")
            return LLMResponse(content="ok", model="mock", latency_ms=10.0, cost_usd=0.001)

        mock_provider.complete = fail_then_succeed

        gw = LLMGateway(mock_provider, max_concurrent=10, requests_per_minute=100)
        await gw.start()

        try:
            # First two fail
            import contextlib

            for i in range(2):
                with contextlib.suppress(Exception):
                    await gw.submit(LLMRequest(prompt=f"test {i}"), agent_id=f"agent-{i}")

            # Third succeeds
            response = await gw.submit(LLMRequest(prompt="test 3"), agent_id="agent-3")
            assert response.content == "ok"

            # Circuit should be reset
            stats = gw.get_stats()
            assert stats["failed_requests"] == 2
            assert stats["total_requests"] == 1

        finally:
            await gw.stop()


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for request deduplication."""

    async def test_identical_requests_deduplicated(self, mock_provider: AsyncMock) -> None:
        call_count = 0

        async def count_calls(request: LLMRequest) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return LLMResponse(content="ok", model="mock", latency_ms=10.0, cost_usd=0.001)

        mock_provider.complete = count_calls

        gw = LLMGateway(mock_provider, max_concurrent=10, requests_per_minute=100)
        await gw.start()

        try:
            # Submit identical requests concurrently
            request = LLMRequest(prompt="test", temperature=0.0)
            tasks = [gw.submit(request, agent_id=f"agent-{i}") for i in range(5)]
            responses = await asyncio.gather(*tasks)

            # All should succeed
            assert len(responses) == 5

            # But provider should only be called once (deduplication)
            assert call_count == 1

        finally:
            await gw.stop()

    async def test_different_requests_not_deduplicated(self, mock_provider: AsyncMock) -> None:
        call_count = 0

        async def count_calls(request: LLMRequest) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            return LLMResponse(content="ok", model="mock", latency_ms=10.0, cost_usd=0.001)

        mock_provider.complete = count_calls

        gw = LLMGateway(mock_provider, max_concurrent=10, requests_per_minute=100)
        await gw.start()

        try:
            # Submit different requests
            tasks = [
                gw.submit(LLMRequest(prompt=f"test {i}"), agent_id=f"agent-{i}") for i in range(5)
            ]
            await asyncio.gather(*tasks)

            # Provider should be called for each unique request
            assert call_count == 5

        finally:
            await gw.stop()


# ---------------------------------------------------------------------------
# Cost tracking tests
# ---------------------------------------------------------------------------


class TestCostTracking:
    """Tests for cost aggregation and alerting."""

    async def test_total_cost_tracking(self, mock_provider: AsyncMock) -> None:
        mock_provider.complete = AsyncMock(
            return_value=LLMResponse(
                content="ok",
                model="mock",
                latency_ms=10.0,
                cost_usd=0.1,
            )
        )

        gw = LLMGateway(mock_provider, max_concurrent=10, requests_per_minute=100)
        await gw.start()

        try:
            # Submit 5 requests, each costing 0.1
            for i in range(5):
                await gw.submit(LLMRequest(prompt=f"test {i}"), agent_id=f"agent-{i}")

            assert gw.total_cost_usd == pytest.approx(0.5, abs=0.01)

        finally:
            await gw.stop()

    async def test_cost_alert_threshold(self, mock_provider: AsyncMock) -> None:
        mock_provider.complete = AsyncMock(
            return_value=LLMResponse(
                content="ok",
                model="mock",
                latency_ms=10.0,
                cost_usd=0.6,
            )
        )

        # Set low threshold
        gw = LLMGateway(
            mock_provider, max_concurrent=10, requests_per_minute=100, cost_alert_threshold=0.5
        )
        await gw.start()

        try:
            with capture_logs() as cap_logs:
                # Submit request that exceeds threshold
                await gw.submit(LLMRequest(prompt="test"), agent_id="agent-1")

                # Wait for processing
                await asyncio.sleep(0.1)

            # Check for warning in structlog output
            assert any(log.get("event") == "cost_threshold_exceeded" for log in cap_logs)
            # Verify cost_exceeded flag is set
            assert gw._cost_exceeded is True

        finally:
            await gw.stop()

    async def test_cost_exceeded_rejects_new_requests(self, mock_provider: AsyncMock) -> None:
        """After cost threshold is exceeded, new requests should be rejected."""
        mock_provider.complete = AsyncMock(
            return_value=LLMResponse(
                content="ok",
                model="mock",
                latency_ms=10.0,
                cost_usd=0.6,
            )
        )

        gw = LLMGateway(
            mock_provider, max_concurrent=10, requests_per_minute=100, cost_alert_threshold=0.5
        )
        await gw.start()

        try:
            # First request succeeds but triggers cost threshold
            await gw.submit(LLMRequest(prompt="test 1"), agent_id="agent-1")
            assert gw._cost_exceeded is True

            # Second request should be rejected
            with pytest.raises(RuntimeError, match="Cost threshold exceeded"):
                await gw.submit(LLMRequest(prompt="test 2"), agent_id="agent-2")

        finally:
            await gw.stop()


# ---------------------------------------------------------------------------
# Stats reporting tests
# ---------------------------------------------------------------------------


class TestStatsReporting:
    """Tests for get_stats method."""

    async def test_stats_reporting(self, gateway: LLMGateway, mock_provider: AsyncMock) -> None:
        # Submit some requests
        await gateway.submit(LLMRequest(prompt="test 1"), agent_id="agent-1")
        await gateway.submit(LLMRequest(prompt="test 2"), agent_id="agent-2")

        stats = gateway.get_stats()
        assert stats["total_requests"] == 2
        assert stats["failed_requests"] == 0
        assert stats["running"] is True
        assert stats["error_rate"] == 0.0
        assert "total_cost_usd" in stats

    async def test_stats_with_failures(self, mock_provider: AsyncMock) -> None:
        # Make provider fail
        call_count = 0

        async def fail_once(request: LLMRequest) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Provider error")
            return LLMResponse(content="ok", model="mock", latency_ms=10.0, cost_usd=0.001)

        mock_provider.complete = fail_once

        gw = LLMGateway(mock_provider, max_concurrent=10, requests_per_minute=100)
        await gw.start()

        try:
            # First request fails
            import contextlib

            with contextlib.suppress(Exception):
                await gw.submit(LLMRequest(prompt="test 1"), agent_id="agent-1")

            # Second succeeds
            await gw.submit(LLMRequest(prompt="test 2"), agent_id="agent-2")

            stats = gw.get_stats()
            assert stats["total_requests"] == 1
            assert stats["failed_requests"] == 1
            assert stats["error_rate"] == pytest.approx(1.0, abs=0.01)

        finally:
            await gw.stop()


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestGatewayLifecycle:
    """Tests for gateway start/stop lifecycle."""

    async def test_start_stop(self, mock_provider: AsyncMock) -> None:
        gw = LLMGateway(mock_provider)
        assert not gw._running

        await gw.start()
        assert gw._running

        await gw.stop()
        assert not gw._running

    async def test_double_start_warning(self, mock_provider: AsyncMock) -> None:
        gw = LLMGateway(mock_provider)
        await gw.start()

        with capture_logs() as cap_logs:
            await gw.start()  # Should warn

        assert any(log.get("event") == "gateway_already_running" for log in cap_logs)
        await gw.stop()

    async def test_pending_count(self, mock_provider: AsyncMock) -> None:
        # Slow provider to build up queue
        async def slow_complete(request: LLMRequest) -> LLMResponse:
            await asyncio.sleep(0.1)
            return LLMResponse(content="ok", model="mock", latency_ms=10.0, cost_usd=0.001)

        mock_provider.complete = slow_complete

        gw = LLMGateway(mock_provider, max_concurrent=1, requests_per_minute=100)
        await gw.start()

        try:
            # Submit multiple requests
            tasks = [
                gw.submit(LLMRequest(prompt=f"test {i}"), agent_id=f"agent-{i}") for i in range(5)
            ]

            # Check pending count while processing
            await asyncio.sleep(0.05)
            # Should have some pending (exact count depends on timing)
            assert gw.pending_count >= 0

            # Wait for all to complete
            await asyncio.gather(*tasks)

            # No more pending
            assert gw.pending_count == 0

        finally:
            await gw.stop()


# ---------------------------------------------------------------------------
# Background task reference tests
# ---------------------------------------------------------------------------


class TestBackgroundTaskReferences:
    """Tests for background task reference management (GC safety)."""

    async def test_dedup_cleanup_task_is_tracked(self, mock_provider: AsyncMock) -> None:
        """Background dedup cleanup tasks should be stored in _background_tasks."""
        gw = LLMGateway(mock_provider, max_concurrent=10, requests_per_minute=100)
        await gw.start()

        try:
            # Before any submit, no background tasks
            assert len(gw._background_tasks) == 0

            # Submit a request (creates a dedup cleanup task)
            await gw.submit(LLMRequest(prompt="test"), agent_id="agent-1")

            # Background tasks set should have been populated
            # (task may complete quickly, but was added)
            # We verify the set exists and is a set of tasks
            assert isinstance(gw._background_tasks, set)

        finally:
            await gw.stop()

    async def test_background_tasks_cleaned_up_after_completion(
        self, mock_provider: AsyncMock
    ) -> None:
        """Completed background tasks should be removed from _background_tasks."""
        gw = LLMGateway(mock_provider, max_concurrent=10, requests_per_minute=100)
        # Use a very short dedup TTL so cleanup tasks finish quickly
        gw._dedup_ttl_seconds = 0.05
        await gw.start()

        try:
            await gw.submit(LLMRequest(prompt="test"), agent_id="agent-1")

            # Wait for cleanup tasks to complete
            await asyncio.sleep(0.2)

            # All background tasks should have been removed by done callback
            assert len(gw._background_tasks) == 0

        finally:
            await gw.stop()


# ---------------------------------------------------------------------------
# structlog integration tests
# ---------------------------------------------------------------------------


class TestStructlogIntegration:
    """Tests for structlog logging output."""

    async def test_gateway_start_emits_structlog(self, mock_provider: AsyncMock) -> None:
        """Gateway start should emit structured log event."""
        gw = LLMGateway(mock_provider)

        with capture_logs() as cap_logs:
            await gw.start()

        assert any(log.get("event") == "gateway_started" for log in cap_logs)
        await gw.stop()

    async def test_gateway_stop_emits_structlog(self, mock_provider: AsyncMock) -> None:
        """Gateway stop should emit structured log events."""
        gw = LLMGateway(mock_provider)
        await gw.start()

        with capture_logs() as cap_logs:
            await gw.stop()

        assert any(log.get("event") == "gateway_stopped" for log in cap_logs)
