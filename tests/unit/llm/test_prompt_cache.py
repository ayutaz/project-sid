"""Tests for PromptCacheManager, PrefixCacheOptimizer, and SemanticCache."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from piano.core.types import LLMRequest, LLMResponse
from piano.llm.prompt_cache import (
    CacheBackend,
    CacheStats,
    PrefixCacheOptimizer,
    PromptCacheManager,
    SemanticCache,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req(prompt: str = "hello", **kwargs: object) -> LLMRequest:
    return LLMRequest(prompt=prompt, **kwargs)  # type: ignore[arg-type]


def _resp(content: str = "world", cost_usd: float = 0.01) -> LLMResponse:
    return LLMResponse(content=content, model="test", cost_usd=cost_usd)


class InMemoryCacheBackend:
    """In-memory implementation of CacheBackend for testing."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, ttl_seconds: float | None = None) -> None:
        self._store[key] = value

    async def delete_pattern(self, pattern: str) -> int:
        # Simple glob matching: only handle "*" prefix/suffix
        to_delete = []
        search = pattern.replace("*", "")
        for k in self._store:
            if search in k:
                to_delete.append(k)
        for k in to_delete:
            del self._store[k]
        return len(to_delete)

    async def size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# CacheStats tests
# ---------------------------------------------------------------------------


class TestCacheStats:
    def test_default_values(self) -> None:
        stats = CacheStats()
        assert stats.total_requests == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        assert stats.estimated_cost_saved == 0.0

    def test_custom_values(self) -> None:
        stats = CacheStats(
            total_requests=100,
            hits=75,
            misses=25,
            hit_rate=0.75,
            estimated_cost_saved=1.5,
        )
        assert stats.hit_rate == 0.75
        assert stats.estimated_cost_saved == 1.5

    def test_is_pydantic_model(self) -> None:
        stats = CacheStats(total_requests=10, hits=5, misses=5, hit_rate=0.5)
        dumped = stats.model_dump()
        assert dumped["total_requests"] == 10
        assert dumped["hit_rate"] == 0.5


# ---------------------------------------------------------------------------
# PrefixCacheOptimizer tests
# ---------------------------------------------------------------------------


class TestPrefixCacheOptimizer:
    def test_normalize_whitespace(self) -> None:
        optimizer = PrefixCacheOptimizer()
        result = optimizer.optimize_prompt(_req(prompt="  hello   world  \n\n  foo  "))
        assert result.prompt == "hello world foo"

    def test_normalize_system_prompt(self) -> None:
        optimizer = PrefixCacheOptimizer()
        result = optimizer.optimize_prompt(
            _req(prompt="test", system_prompt="  You  are  a  helper.  ")
        )
        assert result.system_prompt == "You are a helper."

    def test_prefix_hash_tracked(self) -> None:
        optimizer = PrefixCacheOptimizer()
        optimizer.optimize_prompt(_req(prompt="test", system_prompt="You are a Minecraft agent."))
        stats = optimizer.get_prefix_stats()
        assert stats["unique_prefixes"] == 1

    def test_same_prefix_not_duplicated(self) -> None:
        optimizer = PrefixCacheOptimizer()
        optimizer.optimize_prompt(_req(prompt="test1", system_prompt="You are a helper."))
        optimizer.optimize_prompt(_req(prompt="test2", system_prompt="You are a helper."))
        stats = optimizer.get_prefix_stats()
        assert stats["unique_prefixes"] == 1

    def test_different_prefixes_tracked(self) -> None:
        optimizer = PrefixCacheOptimizer()
        optimizer.optimize_prompt(_req(prompt="test1", system_prompt="System A"))
        optimizer.optimize_prompt(_req(prompt="test2", system_prompt="System B"))
        stats = optimizer.get_prefix_stats()
        assert stats["unique_prefixes"] == 2

    def test_compute_prefix_hash_deterministic(self) -> None:
        h1 = PrefixCacheOptimizer.compute_prefix_hash("hello world")
        h2 = PrefixCacheOptimizer.compute_prefix_hash("hello world")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_compute_prefix_hash_different_inputs(self) -> None:
        h1 = PrefixCacheOptimizer.compute_prefix_hash("hello")
        h2 = PrefixCacheOptimizer.compute_prefix_hash("world")
        assert h1 != h2

    def test_empty_system_prompt_not_tracked(self) -> None:
        optimizer = PrefixCacheOptimizer()
        optimizer.optimize_prompt(_req(prompt="test"))
        stats = optimizer.get_prefix_stats()
        assert stats["unique_prefixes"] == 0

    def test_clear(self) -> None:
        optimizer = PrefixCacheOptimizer()
        optimizer.optimize_prompt(_req(prompt="test", system_prompt="sys"))
        optimizer.clear()
        assert optimizer.get_prefix_stats()["unique_prefixes"] == 0

    def test_original_request_unchanged(self) -> None:
        optimizer = PrefixCacheOptimizer()
        original = _req(prompt="  hello  ", system_prompt="  sys  ")
        optimized = optimizer.optimize_prompt(original)
        # Original should be unchanged
        assert original.prompt == "  hello  "
        assert original.system_prompt == "  sys  "
        # Optimized should be normalized
        assert optimized.prompt == "hello"
        assert optimized.system_prompt == "sys"


# ---------------------------------------------------------------------------
# SemanticCache tests
# ---------------------------------------------------------------------------


class TestSemanticCache:
    def test_exact_match(self) -> None:
        cache = SemanticCache()
        resp = _resp("answer")
        cache.put("What is the weather?", resp)
        result = cache.find_similar("What is the weather?")
        assert result is not None
        assert result.content == "answer"
        assert result.cached is True

    def test_miss_returns_none(self) -> None:
        cache = SemanticCache()
        assert cache.find_similar("hello") is None

    def test_similar_prompt_detected(self) -> None:
        cache = SemanticCache(default_threshold=0.8)
        resp = _resp("answer about weather")
        cache.put("What is the current weather in Tokyo?", resp)
        # Very similar prompt
        result = cache.find_similar("What is the current weather in Tokyo today?")
        assert result is not None
        assert result.content == "answer about weather"

    def test_dissimilar_prompt_not_matched(self) -> None:
        cache = SemanticCache(default_threshold=0.95)
        cache.put("What is the weather?", _resp("sunny"))
        result = cache.find_similar("Tell me about quantum physics")
        assert result is None

    def test_ttl_expiry(self) -> None:
        cache = SemanticCache(ttl_seconds=1.0)
        cache.put("hello", _resp("world"))

        with patch(
            "piano.llm.prompt_cache.time.monotonic",
            return_value=time.monotonic() + 2.0,
        ):
            result = cache.find_similar("hello")
            assert result is None

    def test_lru_eviction(self) -> None:
        cache = SemanticCache(max_entries=2)
        cache.put("first prompt", _resp("first"))
        cache.put("second prompt", _resp("second"))
        cache.put("third prompt", _resp("third"))
        # "first prompt" should be evicted
        assert cache.size == 2

    def test_size_property(self) -> None:
        cache = SemanticCache()
        assert cache.size == 0
        cache.put("hello", _resp("world"))
        assert cache.size == 1

    def test_hit_count(self) -> None:
        cache = SemanticCache()
        cache.put("hello", _resp("world"))
        cache.find_similar("hello")
        cache.find_similar("hello")
        assert cache.hit_count == 2

    def test_clear(self) -> None:
        cache = SemanticCache()
        cache.put("hello", _resp("world"))
        cache.find_similar("hello")
        cache.clear()
        assert cache.size == 0
        assert cache.hit_count == 0

    def test_custom_threshold(self) -> None:
        cache = SemanticCache(default_threshold=0.5)
        cache.put("What is the weather today in Tokyo Japan", _resp("sunny"))
        # With low threshold, somewhat similar prompts should match
        result = cache.find_similar(
            "What is the weather today in Osaka Japan",
            threshold=0.5,
        )
        assert result is not None

    def test_threshold_override(self) -> None:
        cache = SemanticCache(default_threshold=0.5)
        cache.put("What is the weather?", _resp("sunny"))
        # With very high threshold, only exact-ish matches work
        result = cache.find_similar("What is the temperature?", threshold=0.99)
        assert result is None

    def test_case_insensitive_matching(self) -> None:
        cache = SemanticCache()
        cache.put("What is the Weather?", _resp("sunny"))
        result = cache.find_similar("what is the weather?")
        assert result is not None


# ---------------------------------------------------------------------------
# PromptCacheManager L1 tests
# ---------------------------------------------------------------------------


class TestPromptCacheManagerL1:
    def test_miss_returns_none(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        assert manager.get(_req()) is None

    def test_put_and_get(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        req = _req("hello")
        resp = _resp("world")
        manager.put(req, resp)
        result = manager.get(req)
        assert result is not None
        assert result.content == "world"
        assert result.cached is True

    def test_l1_size(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        assert manager.l1_size == 0
        manager.put(_req("a"), _resp("1"))
        assert manager.l1_size == 1

    def test_lru_eviction(self) -> None:
        manager = PromptCacheManager(
            max_l1_entries=2,
            enable_semantic=False,
        )
        manager.put(_req("a"), _resp("1"))
        manager.put(_req("b"), _resp("2"))
        manager.put(_req("c"), _resp("3"))
        # "a" should be evicted
        assert manager.get(_req("a")) is None
        assert manager.get(_req("b")) is not None
        assert manager.get(_req("c")) is not None

    def test_ttl_expiry(self) -> None:
        manager = PromptCacheManager(
            ttl_seconds=1.0,
            enable_semantic=False,
        )
        manager.put(_req("a"), _resp("1"))
        with patch(
            "piano.llm.prompt_cache.time.monotonic",
            return_value=time.monotonic() + 2.0,
        ):
            result = manager.get(_req("a"))
            assert result is None

    def test_different_models_different_keys(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        manager.put(_req("hello", model="gpt-4o"), _resp("from-gpt4"))
        manager.put(_req("hello", model="gpt-4o-mini"), _resp("from-mini"))
        r1 = manager.get(_req("hello", model="gpt-4o"))
        r2 = manager.get(_req("hello", model="gpt-4o-mini"))
        assert r1 is not None
        assert r2 is not None
        assert r1.content == "from-gpt4"
        assert r2.content == "from-mini"

    def test_update_existing_entry(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        manager.put(_req("a"), _resp("old"))
        manager.put(_req("a"), _resp("new"))
        result = manager.get(_req("a"))
        assert result is not None
        assert result.content == "new"
        assert manager.l1_size == 1

    def test_clear(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        manager.put(_req("a"), _resp("1"))
        manager.put(_req("b"), _resp("2"))
        manager.clear()
        assert manager.l1_size == 0


# ---------------------------------------------------------------------------
# PromptCacheManager statistics tests
# ---------------------------------------------------------------------------


class TestPromptCacheManagerStats:
    def test_initial_stats(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        stats = manager.get_stats()
        assert stats.total_requests == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_miss_increments_stats(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        manager.get(_req("hello"))
        stats = manager.get_stats()
        assert stats.total_requests == 1
        assert stats.misses == 1
        assert stats.hits == 0

    def test_hit_increments_stats(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        manager.put(_req("hello"), _resp("world", cost_usd=0.05))
        manager.get(_req("hello"))
        stats = manager.get_stats()
        assert stats.total_requests == 1
        assert stats.hits == 1
        assert stats.misses == 0
        assert stats.hit_rate == 1.0
        assert stats.estimated_cost_saved == 0.05

    def test_hit_rate_calculation(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        manager.put(_req("a"), _resp("1"))
        manager.get(_req("a"))  # hit
        manager.get(_req("b"))  # miss
        manager.get(_req("a"))  # hit
        stats = manager.get_stats()
        assert stats.total_requests == 3
        assert stats.hits == 2
        assert stats.misses == 1
        assert abs(stats.hit_rate - 2 / 3) < 0.001

    def test_invalidation_counted(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        manager.put(_req("a"), _resp("1"))
        manager.invalidate("*")
        stats = manager.get_stats()
        assert stats.invalidations == 1

    def test_stats_is_cache_stats_model(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        stats = manager.get_stats()
        assert isinstance(stats, CacheStats)


# ---------------------------------------------------------------------------
# PromptCacheManager invalidation tests
# ---------------------------------------------------------------------------


class TestPromptCacheManagerInvalidation:
    def test_invalidate_clears_l1(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        manager.put(_req("a"), _resp("1"))
        manager.put(_req("b"), _resp("2"))
        count = manager.invalidate("*")
        assert count == 2
        assert manager.l1_size == 0

    def test_invalidate_clears_semantic(self) -> None:
        manager = PromptCacheManager(enable_semantic=True)
        manager.put(_req("a"), _resp("1"))
        count = manager.invalidate("*")
        assert count > 0  # L1 + semantic entries

    def test_invalidate_returns_count(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        manager.put(_req("a"), _resp("1"))
        manager.put(_req("b"), _resp("2"))
        manager.put(_req("c"), _resp("3"))
        count = manager.invalidate("test")
        assert count == 3  # All L1 entries cleared

    def test_get_after_invalidation_misses(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        manager.put(_req("hello"), _resp("world"))
        manager.invalidate("*")
        assert manager.get(_req("hello")) is None


# ---------------------------------------------------------------------------
# PromptCacheManager with prefix optimization
# ---------------------------------------------------------------------------


class TestPromptCacheManagerPrefixOptimization:
    def test_whitespace_normalized_for_cache_key(self) -> None:
        manager = PromptCacheManager(
            enable_semantic=False,
            enable_prefix_optimization=True,
        )
        manager.put(
            _req(prompt="  hello   world  "),
            _resp("answer"),
        )
        # Same prompt with different whitespace should hit cache
        result = manager.get(_req(prompt="hello world"))
        assert result is not None
        assert result.content == "answer"

    def test_without_prefix_optimization_whitespace_matters(self) -> None:
        manager = PromptCacheManager(
            enable_semantic=False,
            enable_prefix_optimization=False,
        )
        manager.put(
            _req(prompt="  hello   world  "),
            _resp("answer"),
        )
        # Without optimization, different whitespace = different key
        result = manager.get(_req(prompt="hello world"))
        assert result is None


# ---------------------------------------------------------------------------
# PromptCacheManager with semantic cache
# ---------------------------------------------------------------------------


class TestPromptCacheManagerSemantic:
    def test_semantic_hit_populates_l1(self) -> None:
        manager = PromptCacheManager(
            enable_semantic=True,
            semantic_threshold=0.8,
            enable_prefix_optimization=False,
        )
        prompt = "What is the current weather in Tokyo Japan today?"
        manager.put(_req(prompt=prompt), _resp("sunny"))

        # Clear L1 to force semantic lookup
        manager._l1.clear()

        # Similar prompt should match semantically
        result = manager.get(_req(prompt="What is the current weather in Tokyo Japan now?"))
        assert result is not None
        assert result.content == "sunny"
        # Should have populated L1
        assert manager.l1_size == 1


# ---------------------------------------------------------------------------
# PromptCacheManager async L2 tests
# ---------------------------------------------------------------------------


class TestPromptCacheManagerL2:
    @pytest.mark.asyncio
    async def test_put_async_stores_in_l2(self) -> None:
        backend = InMemoryCacheBackend()
        manager = PromptCacheManager(
            l2_backend=backend,
            enable_semantic=False,
        )
        await manager.put_async(_req("hello"), _resp("world"))
        assert manager.l1_size == 1
        assert await backend.size() == 1

    @pytest.mark.asyncio
    async def test_get_async_l2_hit(self) -> None:
        backend = InMemoryCacheBackend()
        manager = PromptCacheManager(
            l2_backend=backend,
            enable_semantic=False,
        )
        await manager.put_async(_req("hello"), _resp("world"))
        # Clear L1 to force L2 lookup
        manager._l1.clear()

        result = await manager.get_async(_req("hello"))
        assert result is not None
        assert result.content == "world"
        assert result.cached is True
        # Should populate L1
        assert manager.l1_size == 1

    @pytest.mark.asyncio
    async def test_get_async_miss(self) -> None:
        backend = InMemoryCacheBackend()
        manager = PromptCacheManager(
            l2_backend=backend,
            enable_semantic=False,
        )
        result = await manager.get_async(_req("hello"))
        assert result is None

    @pytest.mark.asyncio
    async def test_get_async_l1_hit_skips_l2(self) -> None:
        backend = InMemoryCacheBackend()
        manager = PromptCacheManager(
            l2_backend=backend,
            enable_semantic=False,
        )
        manager.put(_req("hello"), _resp("from-l1"))
        result = await manager.get_async(_req("hello"))
        assert result is not None
        assert result.content == "from-l1"

    @pytest.mark.asyncio
    async def test_invalidate_async_clears_l2(self) -> None:
        backend = InMemoryCacheBackend()
        manager = PromptCacheManager(
            l2_backend=backend,
            enable_semantic=False,
        )
        await manager.put_async(_req("hello"), _resp("world"))
        count = await manager.invalidate_async("*")
        assert count > 0
        assert await backend.size() == 0

    @pytest.mark.asyncio
    async def test_get_stats_async_includes_l2_size(self) -> None:
        backend = InMemoryCacheBackend()
        manager = PromptCacheManager(
            l2_backend=backend,
            enable_semantic=False,
        )
        await manager.put_async(_req("hello"), _resp("world"))
        stats = await manager.get_stats_async()
        assert stats.l2_size == 1

    @pytest.mark.asyncio
    async def test_backend_conforms_to_protocol(self) -> None:
        backend = InMemoryCacheBackend()
        assert isinstance(backend, CacheBackend)


# ---------------------------------------------------------------------------
# Integration: all layers together
# ---------------------------------------------------------------------------


class TestPromptCacheManagerIntegration:
    @pytest.mark.asyncio
    async def test_full_flow_l1_l2_semantic(self) -> None:
        backend = InMemoryCacheBackend()
        manager = PromptCacheManager(
            l2_backend=backend,
            enable_semantic=True,
            semantic_threshold=0.8,
            enable_prefix_optimization=True,
        )

        # Store a response
        req = _req(prompt="What is the weather in Tokyo?", system_prompt="You are a helper.")
        resp = _resp("It is sunny", cost_usd=0.02)
        await manager.put_async(req, resp)

        # L1 hit
        result = await manager.get_async(req)
        assert result is not None
        assert result.content == "It is sunny"
        assert result.cached is True

        # Clear L1, should fall through to L2
        manager._l1.clear()
        result = await manager.get_async(req)
        assert result is not None
        assert result.content == "It is sunny"

        # Check stats
        stats = await manager.get_stats_async()
        assert stats.hits >= 2
        assert stats.estimated_cost_saved > 0
        assert stats.l2_size == 1

    def test_cost_savings_tracked(self) -> None:
        manager = PromptCacheManager(enable_semantic=False)
        manager.put(_req("a"), _resp("1", cost_usd=0.05))
        manager.put(_req("b"), _resp("2", cost_usd=0.10))
        manager.get(_req("a"))
        manager.get(_req("b"))
        stats = manager.get_stats()
        assert stats.estimated_cost_saved == 0.15
