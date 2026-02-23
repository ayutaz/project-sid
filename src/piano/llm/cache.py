"""LLM response cache with LRU eviction and TTL expiry.

Provides an in-memory cache for Phase 0 and a ``CachedLLMProvider`` wrapper
that transparently caches responses while conforming to the LLMProvider protocol.

Reference: docs/implementation/02-llm-integration.md Section 4
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from piano.core.types import LLMRequest, LLMResponse

if TYPE_CHECKING:
    from piano.llm.provider import LLMProvider


class LLMCache:
    """In-memory LRU cache for LLM responses with TTL expiry.

    The cache key is derived from the prompt, model, and temperature fields
    of the request to ensure semantically equivalent requests share a cache entry.
    """

    def __init__(
        self,
        *,
        max_entries: int = 1000,
        ttl_seconds: float = 3600.0,
    ) -> None:
        """Initialise the cache.

        Args:
            max_entries: Maximum number of cached entries (LRU eviction).
            ttl_seconds: Time-to-live for each entry in seconds.
        """
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, LLMResponse]] = OrderedDict()

    @staticmethod
    def _make_key(request: LLMRequest) -> str:
        """Create a deterministic hash key from request fields."""
        payload = json.dumps(
            {
                "prompt": request.prompt,
                "system_prompt": request.system_prompt,
                "model": request.model,
                "temperature": request.temperature,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, request: LLMRequest) -> LLMResponse | None:
        """Look up a cached response.

        Returns ``None`` on cache miss or if the entry has expired.

        Args:
            request: The LLM request to look up.

        Returns:
            A cached LLMResponse with ``cached=True``, or None.
        """
        key = self._make_key(request)
        entry = self._store.get(key)
        if entry is None:
            return None

        created_at, response = entry
        if (time.monotonic() - created_at) > self.ttl_seconds:
            del self._store[key]
            return None

        # Move to end for LRU
        self._store.move_to_end(key)
        return response.model_copy(update={"cached": True})

    def put(self, request: LLMRequest, response: LLMResponse) -> None:
        """Store a response in the cache.

        If the cache is at capacity, the least recently used entry is evicted.

        Args:
            request: The original request (used to derive the cache key).
            response: The response to cache.
        """
        key = self._make_key(request)

        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = (time.monotonic(), response)
            return

        if len(self._store) >= self.max_entries:
            self._store.popitem(last=False)

        self._store[key] = (time.monotonic(), response)

    @property
    def size(self) -> int:
        """Return the current number of cached entries."""
        return len(self._store)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()


class CachedLLMProvider:
    """LLM provider wrapper that adds transparent caching.

    Conforms to the ``LLMProvider`` protocol. On cache hit the wrapped
    provider is not called.
    """

    def __init__(
        self,
        provider: LLMProvider,
        cache: LLMCache | None = None,
    ) -> None:
        """Initialise the cached provider.

        Args:
            provider: The underlying LLM provider to delegate to on cache miss.
            cache: An LLMCache instance. A default cache is created if omitted.
        """
        self._provider = provider
        self._cache = cache or LLMCache()

    @property
    def cache(self) -> LLMCache:
        """Access the underlying cache instance."""
        return self._cache

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return a cached response or delegate to the wrapped provider.

        Args:
            request: The LLM request to execute.

        Returns:
            An LLMResponse, with ``cached=True`` if served from cache.
        """
        cached = self._cache.get(request)
        if cached is not None:
            return cached

        response = await self._provider.complete(request)
        self._cache.put(request, response)
        return response
