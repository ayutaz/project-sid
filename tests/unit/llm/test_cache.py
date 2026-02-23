"""Tests for LLMCache and CachedLLMProvider."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from piano.core.types import LLMRequest, LLMResponse
from piano.llm.cache import CachedLLMProvider, LLMCache
from piano.llm.mock import MockLLMProvider
from piano.llm.provider import LLMProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(prompt: str = "hello", **kwargs: object) -> LLMRequest:
    return LLMRequest(prompt=prompt, **kwargs)  # type: ignore[arg-type]


def _resp(content: str = "world") -> LLMResponse:
    return LLMResponse(content=content, model="test")


# ---------------------------------------------------------------------------
# LLMCache tests
# ---------------------------------------------------------------------------

class TestLLMCacheBasic:
    def test_miss_returns_none(self) -> None:
        cache = LLMCache()
        assert cache.get(_req()) is None

    def test_put_and_get(self) -> None:
        cache = LLMCache()
        req = _req("hello")
        resp = _resp("world")
        cache.put(req, resp)
        result = cache.get(req)
        assert result is not None
        assert result.content == "world"
        assert result.cached is True

    def test_cache_size(self) -> None:
        cache = LLMCache()
        assert cache.size == 0
        cache.put(_req("a"), _resp("1"))
        assert cache.size == 1
        cache.put(_req("b"), _resp("2"))
        assert cache.size == 2

    def test_clear(self) -> None:
        cache = LLMCache()
        cache.put(_req("a"), _resp("1"))
        cache.put(_req("b"), _resp("2"))
        cache.clear()
        assert cache.size == 0

    def test_different_prompts_different_keys(self) -> None:
        cache = LLMCache()
        cache.put(_req("alpha"), _resp("A"))
        cache.put(_req("beta"), _resp("B"))
        assert cache.get(_req("alpha")).content == "A"  # type: ignore[union-attr]
        assert cache.get(_req("beta")).content == "B"  # type: ignore[union-attr]

    def test_different_temperature_different_keys(self) -> None:
        cache = LLMCache()
        cache.put(_req("hi", temperature=0.0), _resp("cold"))
        cache.put(_req("hi", temperature=0.7), _resp("warm"))
        assert cache.get(_req("hi", temperature=0.0)).content == "cold"  # type: ignore[union-attr]
        assert cache.get(_req("hi", temperature=0.7)).content == "warm"  # type: ignore[union-attr]


class TestLLMCacheLRU:
    def test_evicts_oldest_when_full(self) -> None:
        cache = LLMCache(max_entries=2)
        cache.put(_req("a"), _resp("1"))
        cache.put(_req("b"), _resp("2"))
        cache.put(_req("c"), _resp("3"))
        # "a" should be evicted
        assert cache.get(_req("a")) is None
        assert cache.get(_req("b")) is not None
        assert cache.get(_req("c")) is not None
        assert cache.size == 2

    def test_access_refreshes_lru_order(self) -> None:
        cache = LLMCache(max_entries=2)
        cache.put(_req("a"), _resp("1"))
        cache.put(_req("b"), _resp("2"))
        # Access "a" to make it recently used
        cache.get(_req("a"))
        # Now "b" is the oldest
        cache.put(_req("c"), _resp("3"))
        assert cache.get(_req("a")) is not None
        assert cache.get(_req("b")) is None
        assert cache.get(_req("c")) is not None

    def test_update_existing_key(self) -> None:
        cache = LLMCache(max_entries=2)
        cache.put(_req("a"), _resp("old"))
        cache.put(_req("a"), _resp("new"))
        assert cache.size == 1
        assert cache.get(_req("a")).content == "new"  # type: ignore[union-attr]


class TestLLMCacheTTL:
    def test_expired_entry_returns_none(self) -> None:
        cache = LLMCache(ttl_seconds=1.0)
        cache.put(_req("a"), _resp("1"))

        # Advance time past TTL
        with patch("piano.llm.cache.time.monotonic", return_value=time.monotonic() + 2.0):
            assert cache.get(_req("a")) is None
        # Expired entry should be removed
        assert cache.size == 0

    def test_non_expired_entry_is_returned(self) -> None:
        cache = LLMCache(ttl_seconds=10.0)
        cache.put(_req("a"), _resp("1"))
        # Should still be valid
        result = cache.get(_req("a"))
        assert result is not None
        assert result.content == "1"


# ---------------------------------------------------------------------------
# CachedLLMProvider tests
# ---------------------------------------------------------------------------

class TestCachedLLMProvider:
    def test_implements_protocol(self) -> None:
        mock = MockLLMProvider()
        cached = CachedLLMProvider(mock)
        assert isinstance(cached, LLMProvider)

    @pytest.mark.asyncio
    async def test_cache_miss_calls_provider(self) -> None:
        mock = MockLLMProvider()
        mock.set_default_response("from provider")
        cached = CachedLLMProvider(mock)

        result = await cached.complete(_req("hello"))
        assert result.content == "from provider"
        assert len(mock.call_history) == 1

    @pytest.mark.asyncio
    async def test_cache_hit_skips_provider(self) -> None:
        mock = MockLLMProvider()
        mock.set_default_response("from provider")
        cached = CachedLLMProvider(mock)

        await cached.complete(_req("hello"))
        result = await cached.complete(_req("hello"))
        assert result.content == "from provider"
        assert result.cached is True
        assert len(mock.call_history) == 1  # only 1 call, second was cached

    @pytest.mark.asyncio
    async def test_different_requests_both_call_provider(self) -> None:
        mock = MockLLMProvider()
        mock.add_response("alpha", "A")
        mock.add_response("beta", "B")
        cached = CachedLLMProvider(mock)

        r1 = await cached.complete(_req("alpha"))
        r2 = await cached.complete(_req("beta"))
        assert r1.content == "A"
        assert r2.content == "B"
        assert len(mock.call_history) == 2

    @pytest.mark.asyncio
    async def test_uses_provided_cache_instance(self) -> None:
        cache = LLMCache(max_entries=5)
        mock = MockLLMProvider()
        cached = CachedLLMProvider(mock, cache=cache)

        await cached.complete(_req("x"))
        assert cached.cache.size == 1
        assert cached.cache is cache
