"""Prompt caching with prefix optimization and semantic similarity.

Provides multi-layer prompt caching (L1 in-memory LRU, L2 Redis-backed),
prefix-based optimization for cache-friendly prompt structures, and
lightweight semantic cache for near-duplicate prompt detection.

Reference: docs/implementation/02-llm-integration.md
"""

from __future__ import annotations

__all__ = [
    "CacheStats",
    "PrefixCacheOptimizer",
    "PromptCacheManager",
    "RedisCacheBackend",
    "SemanticCache",
]

import hashlib
import json
import re
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel

from piano.core.types import LLMRequest, LLMResponse

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Cache Statistics
# ---------------------------------------------------------------------------


class CacheStats(BaseModel):
    """Statistics for cache performance monitoring."""

    total_requests: int = 0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    estimated_cost_saved: float = 0.0
    l1_size: int = 0
    l2_size: int = 0
    semantic_hits: int = 0
    invalidations: int = 0


# ---------------------------------------------------------------------------
# L2 Cache Backend Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CacheBackend(Protocol):
    """Protocol for L2 cache backends (e.g., Redis)."""

    async def get(self, key: str) -> str | None:
        """Get a value by key.

        Args:
            key: The cache key.

        Returns:
            The cached JSON string, or None on miss.
        """
        ...

    async def set(self, key: str, value: str, ttl_seconds: float | None = None) -> None:
        """Set a value with optional TTL.

        Args:
            key: The cache key.
            value: The JSON string to cache.
            ttl_seconds: Optional time-to-live in seconds.
        """
        ...

    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching a glob pattern.

        Args:
            pattern: Glob pattern for key matching.

        Returns:
            Number of keys deleted.
        """
        ...

    async def size(self) -> int:
        """Return the number of entries in the backend."""
        ...


# ---------------------------------------------------------------------------
# Redis Cache Backend
# ---------------------------------------------------------------------------


class RedisCacheBackend:
    """Redis-backed L2 cache backend.

    Stores cached LLM responses in Redis with optional TTL expiry.
    All keys are prefixed with ``piano:prompt_cache:`` for namespace isolation.

    Args:
        redis: An ``redis.asyncio.Redis`` connection instance.
        key_prefix: Key prefix for namespace isolation.
    """

    def __init__(
        self,
        redis: Redis,  # type: ignore[type-arg]
        *,
        key_prefix: str = "piano:prompt_cache:",
    ) -> None:
        self._redis = redis
        self._key_prefix = key_prefix

    def _full_key(self, key: str) -> str:
        """Build the full Redis key."""
        return f"{self._key_prefix}{key}"

    async def get(self, key: str) -> str | None:
        """Get a cached value from Redis.

        Args:
            key: The cache key.

        Returns:
            The cached JSON string, or None on miss.
        """
        try:
            result = await self._redis.get(self._full_key(key))
            if result is None:
                return None
            return result if isinstance(result, str) else result.decode("utf-8")
        except Exception:
            logger.warning("redis_cache_get_failed", key=key[:16])
            return None

    async def set(self, key: str, value: str, ttl_seconds: float | None = None) -> None:
        """Store a value in Redis with optional TTL.

        Args:
            key: The cache key.
            value: The JSON string to cache.
            ttl_seconds: Optional time-to-live in seconds.
        """
        try:
            full_key = self._full_key(key)
            if ttl_seconds is not None:
                await self._redis.set(full_key, value, ex=int(ttl_seconds))
            else:
                await self._redis.set(full_key, value)
        except Exception:
            logger.warning("redis_cache_set_failed", key=key[:16])

    async def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching a glob pattern.

        Args:
            pattern: Glob pattern for key matching (applied after prefix).

        Returns:
            Number of keys deleted.
        """
        try:
            full_pattern = self._full_key(pattern)
            count = 0
            cursor: int | bytes = 0
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor, match=full_pattern, count=100
                )
                if keys:
                    await self._redis.delete(*keys)
                    count += len(keys)
                if cursor == 0:
                    break
            return count
        except Exception:
            logger.warning("redis_cache_delete_pattern_failed", pattern=pattern)
            return 0

    async def size(self) -> int:
        """Return approximate number of cached entries.

        Returns:
            Number of keys matching the cache prefix.
        """
        try:
            count = 0
            full_pattern = self._full_key("*")
            cursor: int | bytes = 0
            while True:
                cursor, keys = await self._redis.scan(
                    cursor=cursor, match=full_pattern, count=100
                )
                count += len(keys)
                if cursor == 0:
                    break
            return count
        except Exception:
            logger.warning("redis_cache_size_failed")
            return 0


# ---------------------------------------------------------------------------
# Prefix Cache Optimizer
# ---------------------------------------------------------------------------


class PrefixCacheOptimizer:
    """Optimizes prompts for better cache hit rates with provider prefix caching.

    OpenAI and Anthropic both support prefix caching where shared prompt
    prefixes are cached server-side. This optimizer separates system prompts
    and user prompts, and normalizes whitespace for consistent hashing.
    """

    def __init__(self) -> None:
        """Initialize the prefix cache optimizer."""
        self._prefix_hashes: dict[str, str] = {}

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace for consistent hashing.

        Collapses multiple spaces/newlines into single spaces and strips
        leading/trailing whitespace.

        Args:
            text: The text to normalize.

        Returns:
            Normalized text.
        """
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def compute_prefix_hash(text: str) -> str:
        """Compute a SHA-256 hash of the given text.

        Args:
            text: The text to hash.

        Returns:
            Hex digest of the SHA-256 hash.
        """
        return hashlib.sha256(text.encode()).hexdigest()

    def optimize_prompt(self, request: LLMRequest) -> LLMRequest:
        """Optimize a prompt for better cache-friendliness.

        Normalizes whitespace in both system and user prompts to increase
        the likelihood of cache hits when prompts differ only in formatting.
        Tracks prefix hashes for deduplication analysis.

        Args:
            request: The original LLM request.

        Returns:
            A new LLMRequest with normalized prompts.
        """
        normalized_system = self._normalize_whitespace(request.system_prompt)
        normalized_prompt = self._normalize_whitespace(request.prompt)

        # Track the system prompt prefix hash for analysis
        if normalized_system:
            prefix_hash = self.compute_prefix_hash(normalized_system)
            self._prefix_hashes[prefix_hash] = normalized_system

        optimized = request.model_copy(
            update={
                "system_prompt": normalized_system,
                "prompt": normalized_prompt,
            }
        )

        logger.debug(
            "prompt_optimized",
            system_prompt_len=len(normalized_system),
            prompt_len=len(normalized_prompt),
            has_system_prompt=bool(normalized_system),
        )

        return optimized

    def get_prefix_stats(self) -> dict[str, int]:
        """Get statistics about tracked prefix hashes.

        Returns:
            Dictionary with prefix hash count.
        """
        return {"unique_prefixes": len(self._prefix_hashes)}

    def clear(self) -> None:
        """Clear tracked prefix hashes."""
        self._prefix_hashes.clear()


# ---------------------------------------------------------------------------
# Semantic Cache
# ---------------------------------------------------------------------------


class SemanticCache:
    """Lightweight semantic cache using text hash similarity.

    Instead of full embedding-based similarity, this uses n-gram based
    fingerprinting to detect near-duplicate prompts. This is much cheaper
    than vector similarity but still catches common reformulations.

    Each entry has a TTL and is evicted when expired.
    """

    def __init__(
        self,
        *,
        max_entries: int = 500,
        ttl_seconds: float = 1800.0,
        default_threshold: float = 0.95,
        ngram_size: int = 3,
    ) -> None:
        """Initialize the semantic cache.

        Args:
            max_entries: Maximum number of cached entries.
            ttl_seconds: Time-to-live for each entry in seconds.
            default_threshold: Default similarity threshold (0.0-1.0).
            ngram_size: Size of character n-grams for fingerprinting.
        """
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.default_threshold = default_threshold
        self._ngram_size = ngram_size
        # Maps fingerprint -> (created_at, normalized_prompt, response)
        self._store: OrderedDict[str, tuple[float, str, LLMResponse]] = OrderedDict()
        self._hits: int = 0

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison.

        Lowercases, collapses whitespace, and strips punctuation variations.

        Args:
            text: The text to normalize.

        Returns:
            Normalized text.
        """
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _compute_ngrams(self, text: str) -> set[str]:
        """Compute character-level n-grams from text.

        Args:
            text: The normalized text.

        Returns:
            Set of n-gram strings.
        """
        if len(text) < self._ngram_size:
            return {text} if text else set()
        return {text[i : i + self._ngram_size] for i in range(len(text) - self._ngram_size + 1)}

    def _jaccard_similarity(self, set_a: set[str], set_b: set[str]) -> float:
        """Compute Jaccard similarity between two sets.

        Args:
            set_a: First set of n-grams.
            set_b: Second set of n-grams.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union

    def _make_fingerprint(self, text: str) -> str:
        """Create a fingerprint hash for a normalized text.

        Args:
            text: The normalized text.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(text.encode()).hexdigest()

    def _evict_expired(self) -> None:
        """Remove expired entries from the store."""
        now = time.monotonic()
        expired_keys = [
            key
            for key, (created_at, _, _) in self._store.items()
            if (now - created_at) > self.ttl_seconds
        ]
        for key in expired_keys:
            del self._store[key]

    def put(self, prompt: str, response: LLMResponse) -> None:
        """Store a prompt-response pair in the semantic cache.

        Args:
            prompt: The original prompt text.
            response: The LLM response to cache.
        """
        self._evict_expired()

        normalized = self._normalize(prompt)
        fingerprint = self._make_fingerprint(normalized)

        if fingerprint in self._store:
            self._store.move_to_end(fingerprint)
            self._store[fingerprint] = (time.monotonic(), normalized, response)
            return

        if len(self._store) >= self.max_entries:
            self._store.popitem(last=False)

        self._store[fingerprint] = (time.monotonic(), normalized, response)

    def find_similar(
        self,
        prompt: str,
        threshold: float | None = None,
    ) -> LLMResponse | None:
        """Find a cached response for a semantically similar prompt.

        Uses n-gram Jaccard similarity to detect near-duplicate prompts.

        Args:
            prompt: The prompt to search for.
            threshold: Similarity threshold (0.0-1.0). Uses default if None.

        Returns:
            A cached LLMResponse with ``cached=True`` if a similar prompt
            is found above the threshold, or None.
        """
        self._evict_expired()

        effective_threshold = threshold if threshold is not None else self.default_threshold
        normalized = self._normalize(prompt)
        query_ngrams = self._compute_ngrams(normalized)

        # First check for exact match (fast path)
        fingerprint = self._make_fingerprint(normalized)
        if fingerprint in self._store:
            created_at, _, response = self._store[fingerprint]
            if (time.monotonic() - created_at) <= self.ttl_seconds:
                self._store.move_to_end(fingerprint)
                self._hits += 1
                return response.model_copy(update={"cached": True})

        # Scan for similar entries
        best_score = 0.0
        best_response: LLMResponse | None = None
        now = time.monotonic()

        for _key, (created_at, stored_prompt, response) in self._store.items():
            if (now - created_at) > self.ttl_seconds:
                continue
            stored_ngrams = self._compute_ngrams(stored_prompt)
            score = self._jaccard_similarity(query_ngrams, stored_ngrams)
            if score >= effective_threshold and score > best_score:
                best_score = score
                best_response = response

        if best_response is not None:
            self._hits += 1
            logger.debug(
                "semantic_cache_hit",
                similarity=round(best_score, 3),
                threshold=effective_threshold,
            )
            return best_response.model_copy(update={"cached": True})

        return None

    @property
    def size(self) -> int:
        """Return the current number of cached entries."""
        return len(self._store)

    @property
    def hit_count(self) -> int:
        """Return the total number of semantic cache hits."""
        return self._hits

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()
        self._hits = 0


# ---------------------------------------------------------------------------
# Prompt Cache Manager
# ---------------------------------------------------------------------------


class PromptCacheManager:
    """Multi-layer prompt cache with prefix optimization and semantic matching.

    Provides a unified caching interface that checks:
    1. L1: In-memory LRU cache (exact match, fast)
    2. Semantic cache (near-duplicate detection)
    3. L2: Redis-backed cache (optional, for distributed setups)

    Also integrates prefix optimization for provider-side cache benefits.

    Args:
        max_l1_entries: Maximum entries in the L1 in-memory cache.
        ttl_seconds: Default TTL for cache entries.
        l2_backend: Optional L2 cache backend (e.g., Redis).
        enable_semantic: Enable semantic cache layer.
        semantic_threshold: Similarity threshold for semantic matching.
        enable_prefix_optimization: Enable prefix optimization.
    """

    def __init__(
        self,
        *,
        max_l1_entries: int = 1000,
        ttl_seconds: float = 3600.0,
        l2_backend: CacheBackend | None = None,
        enable_semantic: bool = True,
        semantic_threshold: float = 0.95,
        enable_prefix_optimization: bool = True,
    ) -> None:
        self._ttl_seconds = ttl_seconds

        # L1: In-memory LRU cache
        self._l1_max = max_l1_entries
        self._l1: OrderedDict[str, tuple[float, LLMResponse]] = OrderedDict()

        # L2: Optional Redis backend
        self._l2 = l2_backend

        # Semantic cache
        self._semantic: SemanticCache | None = None
        if enable_semantic:
            self._semantic = SemanticCache(
                ttl_seconds=ttl_seconds,
                default_threshold=semantic_threshold,
            )

        # Prefix optimizer
        self._optimizer: PrefixCacheOptimizer | None = None
        if enable_prefix_optimization:
            self._optimizer = PrefixCacheOptimizer()

        # Statistics
        self._total_requests: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._estimated_cost_saved: float = 0.0
        self._invalidations: int = 0

    # --- Key generation ---

    @staticmethod
    def _make_key(request: LLMRequest) -> str:
        """Create a deterministic hash key from request fields.

        Args:
            request: The LLM request.

        Returns:
            SHA-256 hex digest of the serialized request fields.
        """
        payload = json.dumps(
            {
                "prompt": request.prompt,
                "system_prompt": request.system_prompt,
                "model": request.model,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "json_mode": request.json_mode,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    # --- L1 operations ---

    def _l1_get(self, key: str) -> LLMResponse | None:
        """Look up a value in the L1 cache.

        Args:
            key: The cache key.

        Returns:
            The cached response, or None on miss/expiry.
        """
        entry = self._l1.get(key)
        if entry is None:
            return None

        created_at, response = entry
        if (time.monotonic() - created_at) > self._ttl_seconds:
            del self._l1[key]
            return None

        self._l1.move_to_end(key)
        return response

    def _l1_put(self, key: str, response: LLMResponse) -> None:
        """Store a value in the L1 cache with LRU eviction.

        Args:
            key: The cache key.
            response: The response to cache.
        """
        if key in self._l1:
            self._l1.move_to_end(key)
            self._l1[key] = (time.monotonic(), response)
            return

        if len(self._l1) >= self._l1_max:
            self._l1.popitem(last=False)

        self._l1[key] = (time.monotonic(), response)

    # --- Public API ---

    def get(self, request: LLMRequest) -> LLMResponse | None:
        """Look up a cached response (L1 only, synchronous).

        For L2 lookups use ``get_async``.

        Args:
            request: The LLM request to look up.

        Returns:
            A cached LLMResponse with ``cached=True``, or None.
        """
        self._total_requests += 1

        # Apply prefix optimization for consistent key generation
        effective_request = request
        if self._optimizer:
            effective_request = self._optimizer.optimize_prompt(request)

        key = self._make_key(effective_request)

        # L1 lookup
        result = self._l1_get(key)
        if result is not None:
            self._hits += 1
            self._estimated_cost_saved += result.cost_usd
            logger.debug("prompt_cache_l1_hit", key=key[:16])
            return result.model_copy(update={"cached": True})

        # Semantic cache lookup
        if self._semantic is not None:
            semantic_result = self._semantic.find_similar(effective_request.prompt)
            if semantic_result is not None:
                self._hits += 1
                self._estimated_cost_saved += semantic_result.cost_usd
                # Also populate L1 for future exact hits
                self._l1_put(key, semantic_result)
                logger.debug("prompt_cache_semantic_hit", key=key[:16])
                return semantic_result

        self._misses += 1
        return None

    async def get_async(self, request: LLMRequest) -> LLMResponse | None:
        """Look up a cached response across all layers (L1 + L2).

        Args:
            request: The LLM request to look up.

        Returns:
            A cached LLMResponse with ``cached=True``, or None.
        """
        # First try synchronous layers (L1 + semantic)
        # Note: we handle statistics here to avoid double-counting
        effective_request = request
        if self._optimizer:
            effective_request = self._optimizer.optimize_prompt(request)

        key = self._make_key(effective_request)

        # L1 lookup (don't increment total_requests yet for L1 check)
        result = self._l1_get(key)
        if result is not None:
            self._total_requests += 1
            self._hits += 1
            self._estimated_cost_saved += result.cost_usd
            return result.model_copy(update={"cached": True})

        # Semantic cache lookup
        if self._semantic is not None:
            semantic_result = self._semantic.find_similar(effective_request.prompt)
            if semantic_result is not None:
                self._total_requests += 1
                self._hits += 1
                self._estimated_cost_saved += semantic_result.cost_usd
                self._l1_put(key, semantic_result)
                return semantic_result

        # L2 lookup
        if self._l2 is not None:
            raw = await self._l2.get(key)
            if raw is not None:
                self._total_requests += 1
                self._hits += 1
                response = LLMResponse.model_validate_json(raw)
                # Populate L1
                self._l1_put(key, response)
                self._estimated_cost_saved += response.cost_usd
                logger.debug("prompt_cache_l2_hit", key=key[:16])
                return response.model_copy(update={"cached": True})

        self._total_requests += 1
        self._misses += 1
        return None

    def put(self, request: LLMRequest, response: LLMResponse) -> None:
        """Store a response in the cache (L1 only, synchronous).

        For L2 storage use ``put_async``.

        Args:
            request: The original request (used to derive the cache key).
            response: The response to cache.
        """
        effective_request = request
        if self._optimizer:
            effective_request = self._optimizer.optimize_prompt(request)

        key = self._make_key(effective_request)
        self._l1_put(key, response)

        # Also store in semantic cache
        if self._semantic is not None:
            self._semantic.put(effective_request.prompt, response)

        logger.debug("prompt_cache_put", key=key[:16], layer="l1")

    async def put_async(self, request: LLMRequest, response: LLMResponse) -> None:
        """Store a response in all cache layers (L1 + L2).

        Args:
            request: The original request.
            response: The response to cache.
        """
        # Store in L1 + semantic
        self.put(request, response)

        # Store in L2
        if self._l2 is not None:
            effective_request = request
            if self._optimizer:
                effective_request = self._optimizer.optimize_prompt(request)
            key = self._make_key(effective_request)
            await self._l2.set(
                key,
                response.model_dump_json(),
                ttl_seconds=self._ttl_seconds,
            )
            logger.debug("prompt_cache_put", key=key[:16], layer="l2")

    def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern (L1 + semantic).

        The pattern is matched against the original prompt text (substring match).
        For L2 invalidation use ``invalidate_async``.

        Args:
            pattern: Substring pattern to match against cached prompts.

        Returns:
            Number of entries invalidated.
        """
        count = 0

        # Invalidate L1 entries where we need to re-check via full scan
        # Since L1 stores by hash key, we need to clear all (or track prompts)
        # For simplicity, clear all L1 entries (pattern-based invalidation is rare)
        if pattern == "*":
            count += len(self._l1)
            self._l1.clear()
        else:
            # Store prompts alongside for pattern matching is expensive;
            # for targeted invalidation, clear all L1
            count += len(self._l1)
            self._l1.clear()

        # Clear semantic cache as well (pattern-based invalidation)
        if self._semantic is not None:
            count += self._semantic.size
            self._semantic.clear()

        self._invalidations += 1
        logger.info("prompt_cache_invalidated", pattern=pattern, count=count)
        return count

    async def invalidate_async(self, pattern: str) -> int:
        """Invalidate cache entries across all layers.

        Args:
            pattern: Glob pattern for matching (L2) or substring (L1/semantic).

        Returns:
            Total number of entries invalidated.
        """
        count = self.invalidate(pattern)

        if self._l2 is not None:
            l2_count = await self._l2.delete_pattern(f"*{pattern}*")
            count += l2_count

        return count

    def get_stats(self) -> CacheStats:
        """Get cache performance statistics.

        Returns:
            CacheStats with hit rate, sizes, and cost savings.
        """
        l2_size = 0  # Synchronous; use get_stats_async for L2 size
        semantic_hits = self._semantic.hit_count if self._semantic else 0

        return CacheStats(
            total_requests=self._total_requests,
            hits=self._hits,
            misses=self._misses,
            hit_rate=(self._hits / self._total_requests) if self._total_requests > 0 else 0.0,
            estimated_cost_saved=round(self._estimated_cost_saved, 6),
            l1_size=len(self._l1),
            l2_size=l2_size,
            semantic_hits=semantic_hits,
            invalidations=self._invalidations,
        )

    async def get_stats_async(self) -> CacheStats:
        """Get cache performance statistics including L2 size.

        Returns:
            CacheStats with L2 backend size included.
        """
        stats = self.get_stats()
        if self._l2 is not None:
            stats.l2_size = await self._l2.size()
        return stats

    def clear(self) -> None:
        """Clear all cache layers (synchronous, L1 + semantic only)."""
        self._l1.clear()
        if self._semantic is not None:
            self._semantic.clear()

    @property
    def l1_size(self) -> int:
        """Return the current number of L1 cache entries."""
        return len(self._l1)
