"""Tests for MultiProviderRouter.

Tests routing strategies (weighted round-robin, least-cost, least-latency,
failover), fallback behaviour, last-response caching, provider registration
and removal, and statistics tracking.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from piano.core.types import LLMRequest, LLMResponse, ModuleTier
from piano.llm.multi_provider import (
    AllProvidersFailedError,
    MultiProviderRouter,
    NoProvidersError,
    ProviderConfig,
    ProviderStats,
    RoutingStrategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(
    *,
    content: str = "ok",
    model: str = "mock",
    latency_ms: float = 10.0,
    cost_usd: float = 0.001,
    side_effect: Exception | None = None,
) -> AsyncMock:
    """Create a mock LLMProvider with a configured complete method."""
    provider = AsyncMock()
    if side_effect is not None:
        provider.complete = AsyncMock(side_effect=side_effect)
    else:
        provider.complete = AsyncMock(
            return_value=LLMResponse(
                content=content,
                model=model,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
            )
        )
    return provider


def _make_request(prompt: str = "test", tier: ModuleTier = ModuleTier.FAST) -> LLMRequest:
    return LLMRequest(prompt=prompt, tier=tier)


# ---------------------------------------------------------------------------
# ProviderStats / ProviderConfig model tests
# ---------------------------------------------------------------------------


class TestProviderModels:
    """Tests for ProviderStats and ProviderConfig pydantic models."""

    def test_provider_stats_defaults(self) -> None:
        stats = ProviderStats(name="test")
        assert stats.total_requests == 0
        assert stats.errors == 0
        assert stats.avg_latency_ms == 0.0
        assert stats.total_cost_usd == 0.0
        assert stats.is_healthy is True

    def test_provider_config_defaults(self) -> None:
        provider = _make_provider()
        config = ProviderConfig(name="test", provider=provider)
        assert config.weight == 1.0
        assert config.priority == 0
        assert config.max_rpm == 60

    def test_provider_config_custom(self) -> None:
        provider = _make_provider()
        config = ProviderConfig(
            name="custom",
            provider=provider,
            weight=2.5,
            priority=1,
            max_rpm=120,
        )
        assert config.weight == 2.5
        assert config.priority == 1
        assert config.max_rpm == 120


# ---------------------------------------------------------------------------
# Registration / removal
# ---------------------------------------------------------------------------


class TestProviderRegistration:
    """Tests for adding and removing providers."""

    def test_add_provider(self) -> None:
        router = MultiProviderRouter()
        provider = _make_provider()
        router.add_provider("openai", provider, weight=2.0, priority=0)
        stats = router.get_provider_stats()
        assert "openai" in stats
        assert stats["openai"].name == "openai"

    def test_add_multiple_providers(self) -> None:
        router = MultiProviderRouter()
        router.add_provider("a", _make_provider())
        router.add_provider("b", _make_provider())
        router.add_provider("c", _make_provider())
        assert len(router.get_provider_stats()) == 3

    def test_remove_provider(self) -> None:
        router = MultiProviderRouter()
        router.add_provider("openai", _make_provider())
        router.remove_provider("openai")
        assert "openai" not in router.get_provider_stats()

    def test_remove_nonexistent_provider_raises(self) -> None:
        router = MultiProviderRouter()
        with pytest.raises(KeyError, match="not found"):
            router.remove_provider("nope")

    def test_overwrite_provider(self) -> None:
        router = MultiProviderRouter()
        router.add_provider("openai", _make_provider(), weight=1.0)
        router.add_provider("openai", _make_provider(), weight=5.0)
        # Should overwrite
        assert len(router.get_provider_stats()) == 1


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------


class TestBasicRouting:
    """Tests for basic route functionality."""

    async def test_route_single_provider(self) -> None:
        router = MultiProviderRouter()
        router.add_provider("p1", _make_provider(content="hello"))
        resp = await router.route(_make_request())
        assert resp.content == "hello"

    async def test_route_no_providers_raises(self) -> None:
        router = MultiProviderRouter()
        with pytest.raises(NoProvidersError, match="No providers registered"):
            await router.route(_make_request())

    async def test_route_updates_stats(self) -> None:
        router = MultiProviderRouter()
        router.add_provider("p1", _make_provider(cost_usd=0.01))
        await router.route(_make_request())
        stats = router.get_provider_stats()["p1"]
        assert stats.total_requests == 1
        assert stats.errors == 0
        assert stats.total_cost_usd == pytest.approx(0.01, abs=0.001)
        assert stats.avg_latency_ms > 0


# ---------------------------------------------------------------------------
# Failover strategy
# ---------------------------------------------------------------------------


class TestFailoverStrategy:
    """Tests for the failover routing strategy."""

    async def test_failover_uses_priority_order(self) -> None:
        router = MultiProviderRouter(strategy=RoutingStrategy.FAILOVER)
        # p2 has higher priority (lower value)
        p1 = _make_provider(content="primary")
        p2 = _make_provider(content="secondary")
        router.add_provider("secondary", p2, priority=1)
        router.add_provider("primary", p1, priority=0)

        resp = await router.route(_make_request())
        assert resp.content == "primary"

    async def test_failover_to_secondary(self) -> None:
        router = MultiProviderRouter(strategy=RoutingStrategy.FAILOVER)
        failing = _make_provider(side_effect=Exception("down"))
        healthy = _make_provider(content="backup")
        router.add_provider("primary", failing, priority=0)
        router.add_provider("backup", healthy, priority=1)

        resp = await router.route(_make_request())
        assert resp.content == "backup"

    async def test_failover_all_fail_raises(self) -> None:
        router = MultiProviderRouter(strategy=RoutingStrategy.FAILOVER)
        router.add_provider("a", _make_provider(side_effect=Exception("fail a")), priority=0)
        router.add_provider("b", _make_provider(side_effect=Exception("fail b")), priority=1)

        with pytest.raises(AllProvidersFailedError, match="All 2 providers failed"):
            await router.route(_make_request())


# ---------------------------------------------------------------------------
# Least-cost strategy
# ---------------------------------------------------------------------------


class TestLeastCostStrategy:
    """Tests for the least-cost routing strategy."""

    async def test_least_cost_selects_cheapest(self) -> None:
        router = MultiProviderRouter(strategy=RoutingStrategy.LEAST_COST)
        expensive = _make_provider(content="expensive", cost_usd=0.1)
        cheap = _make_provider(content="cheap", cost_usd=0.001)
        router.add_provider("expensive", expensive)
        router.add_provider("cheap", cheap)

        # Both start at zero cost, first call goes to one of them.
        # After a few calls, cheapest provider should be preferred.
        # Let's route once to each to set stats.
        router._stats["expensive"].total_cost_usd = 1.0
        router._stats["expensive"].total_requests = 10

        resp = await router.route(_make_request())
        assert resp.content == "cheap"

    async def test_least_cost_prefers_healthy(self) -> None:
        router = MultiProviderRouter(strategy=RoutingStrategy.LEAST_COST)
        router.add_provider("a", _make_provider(content="a"))
        router.add_provider("b", _make_provider(content="b"))
        # Mark 'a' as unhealthy with lower cost
        router._stats["a"].is_healthy = False
        router._stats["a"].total_cost_usd = 0.0

        resp = await router.route(_make_request())
        assert resp.content == "b"


# ---------------------------------------------------------------------------
# Least-latency strategy
# ---------------------------------------------------------------------------


class TestLeastLatencyStrategy:
    """Tests for the least-latency routing strategy."""

    async def test_least_latency_selects_fastest(self) -> None:
        router = MultiProviderRouter(strategy=RoutingStrategy.LEAST_LATENCY)
        slow = _make_provider(content="slow")
        fast = _make_provider(content="fast")
        router.add_provider("slow", slow)
        router.add_provider("fast", fast)

        # Set historical latency stats
        router._stats["slow"].avg_latency_ms = 500.0
        router._stats["slow"].total_requests = 10
        router._stats["fast"].avg_latency_ms = 50.0
        router._stats["fast"].total_requests = 10

        resp = await router.route(_make_request())
        assert resp.content == "fast"

    async def test_least_latency_prefers_healthy(self) -> None:
        router = MultiProviderRouter(strategy=RoutingStrategy.LEAST_LATENCY)
        router.add_provider("a", _make_provider(content="a"))
        router.add_provider("b", _make_provider(content="b"))
        # a is fastest but unhealthy
        router._stats["a"].avg_latency_ms = 10.0
        router._stats["a"].is_healthy = False
        router._stats["b"].avg_latency_ms = 100.0

        resp = await router.route(_make_request())
        assert resp.content == "b"


# ---------------------------------------------------------------------------
# Weighted round-robin strategy
# ---------------------------------------------------------------------------


class TestWeightedRoundRobinStrategy:
    """Tests for the weighted round-robin routing strategy."""

    async def test_wrr_distributes_requests(self) -> None:
        router = MultiProviderRouter(strategy=RoutingStrategy.WEIGHTED_ROUND_ROBIN)
        router.add_provider("a", _make_provider(content="a"), weight=3.0)
        router.add_provider("b", _make_provider(content="b"), weight=1.0)

        results: dict[str, int] = {"a": 0, "b": 0}
        for _ in range(20):
            resp = await router.route(_make_request())
            results[resp.content] += 1

        # With weight 3:1, 'a' should get more requests than 'b'
        assert results["a"] > results["b"]

    async def test_wrr_single_provider(self) -> None:
        router = MultiProviderRouter(strategy=RoutingStrategy.WEIGHTED_ROUND_ROBIN)
        router.add_provider("only", _make_provider(content="only"))
        resp = await router.route(_make_request())
        assert resp.content == "only"


# ---------------------------------------------------------------------------
# Fallback and caching
# ---------------------------------------------------------------------------


class TestFallbackCaching:
    """Tests for last-response cache on total failure."""

    async def test_cache_used_when_all_fail(self) -> None:
        router = MultiProviderRouter(strategy=RoutingStrategy.FAILOVER)
        good = _make_provider(content="cached-value")
        router.add_provider("good", good, priority=0)

        # First call succeeds and caches
        resp1 = await router.route(_make_request("hello"))
        assert resp1.content == "cached-value"

        # Now remove good provider and add failing one
        router.remove_provider("good")
        router.add_provider("bad", _make_provider(side_effect=Exception("down")), priority=0)

        # Should return cached response
        resp2 = await router.route(_make_request("hello"))
        assert resp2.content == "cached-value"
        assert resp2.cached is True
        assert resp2.cost_usd == 0.0

    async def test_no_cache_raises(self) -> None:
        router = MultiProviderRouter(strategy=RoutingStrategy.FAILOVER)
        router.add_provider("bad", _make_provider(side_effect=Exception("down")))

        with pytest.raises(AllProvidersFailedError):
            await router.route(_make_request("never-seen"))

    async def test_cache_per_prompt(self) -> None:
        router = MultiProviderRouter()
        router.add_provider("p1", _make_provider(content="resp-a"))

        await router.route(_make_request("prompt-a"))
        # Cache key is a SHA-256 hash of the prompt
        import hashlib

        expected_key = hashlib.sha256(b"prompt-a").hexdigest()
        assert expected_key in router._last_response_cache


# ---------------------------------------------------------------------------
# Strategy switching
# ---------------------------------------------------------------------------


class TestStrategySwitching:
    """Tests for changing routing strategy at runtime."""

    def test_default_strategy(self) -> None:
        router = MultiProviderRouter()
        assert router.strategy == RoutingStrategy.WEIGHTED_ROUND_ROBIN

    def test_set_strategy(self) -> None:
        router = MultiProviderRouter()
        router.strategy = RoutingStrategy.FAILOVER
        assert router.strategy == RoutingStrategy.FAILOVER

    async def test_switch_strategy_at_runtime(self) -> None:
        router = MultiProviderRouter(strategy=RoutingStrategy.FAILOVER)
        p1 = _make_provider(content="p1")
        p2 = _make_provider(content="p2")
        router.add_provider("p1", p1, priority=0, weight=1.0)
        router.add_provider("p2", p2, priority=1, weight=10.0)

        # With failover, p1 (priority 0) should be selected
        resp = await router.route(_make_request())
        assert resp.content == "p1"

        # Switch to least-latency; p2 has lower latency
        router.strategy = RoutingStrategy.LEAST_LATENCY
        router._stats["p1"].avg_latency_ms = 500.0
        router._stats["p1"].total_requests = 5
        router._stats["p2"].avg_latency_ms = 10.0
        router._stats["p2"].total_requests = 5

        resp = await router.route(_make_request())
        assert resp.content == "p2"


# ---------------------------------------------------------------------------
# Health tracking
# ---------------------------------------------------------------------------


class TestHealthTracking:
    """Tests for provider health status tracking."""

    async def test_provider_marked_unhealthy_after_errors(self) -> None:
        router = MultiProviderRouter()
        failing = _make_provider(side_effect=Exception("fail"))
        backup = _make_provider(content="backup")
        router.add_provider("failing", failing)
        router.add_provider("backup", backup)

        # Route many times so failing provider accumulates >= 3 errors
        for _ in range(20):
            await router.route(_make_request())

        stats = router.get_provider_stats()
        assert stats["failing"].errors >= 3
        assert stats["failing"].is_healthy is False
        assert stats["backup"].is_healthy is True

    async def test_healthy_provider_stats(self) -> None:
        router = MultiProviderRouter()
        router.add_provider("p1", _make_provider(cost_usd=0.005))

        for _ in range(3):
            await router.route(_make_request())

        stats = router.get_provider_stats()["p1"]
        assert stats.total_requests == 3
        assert stats.errors == 0
        assert stats.is_healthy is True
        assert stats.total_cost_usd == pytest.approx(0.015, abs=0.001)


# ---------------------------------------------------------------------------
# RoutingStrategy enum
# ---------------------------------------------------------------------------


class TestRoutingStrategyEnum:
    """Tests for RoutingStrategy string enum."""

    def test_strategy_values(self) -> None:
        assert RoutingStrategy.WEIGHTED_ROUND_ROBIN == "weighted_round_robin"
        assert RoutingStrategy.LEAST_COST == "least_cost"
        assert RoutingStrategy.LEAST_LATENCY == "least_latency"
        assert RoutingStrategy.FAILOVER == "failover"

    def test_strategy_from_string(self) -> None:
        assert RoutingStrategy("failover") == RoutingStrategy.FAILOVER


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and boundary tests."""

    async def test_route_with_zero_weight_providers(self) -> None:
        router = MultiProviderRouter(strategy=RoutingStrategy.WEIGHTED_ROUND_ROBIN)
        router.add_provider("zero", _make_provider(content="zero"), weight=0.0)
        router.add_provider("nonzero", _make_provider(content="nonzero"), weight=1.0)

        resp = await router.route(_make_request())
        # Should still work (nonzero should be picked)
        assert resp.content in ("zero", "nonzero")

    async def test_multiple_sequential_requests(self) -> None:
        router = MultiProviderRouter()
        router.add_provider("p1", _make_provider(content="ok", cost_usd=0.01))

        for i in range(10):
            resp = await router.route(_make_request(f"request-{i}"))
            assert resp.content == "ok"

        stats = router.get_provider_stats()["p1"]
        assert stats.total_requests == 10
        assert stats.total_cost_usd == pytest.approx(0.1, abs=0.01)

    async def test_get_provider_stats_returns_copy(self) -> None:
        router = MultiProviderRouter()
        router.add_provider("p1", _make_provider())
        stats1 = router.get_provider_stats()
        stats2 = router.get_provider_stats()
        assert stats1 is not stats2
