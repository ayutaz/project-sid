"""Multi-provider LLM router with load balancing, failover, and statistics.

Routes LLM requests across multiple providers using configurable strategies
(weighted round-robin, least-cost, least-latency, failover) with automatic
fallback and last-response caching for resilience.

Reference: docs/implementation/02-llm-integration.md
"""

from __future__ import annotations

__all__ = [
    "AllProvidersFailedError",
    "MultiProviderRouter",
    "NoProvidersError",
    "ProviderConfig",
    "ProviderStats",
    "RoutingStrategy",
]

import hashlib
import time
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from piano.core.types import LLMResponse

if TYPE_CHECKING:
    from piano.core.types import LLMRequest
    from piano.llm.provider import LLMProvider

logger = structlog.get_logger(__name__)


# --- Custom Exceptions ---


class NoProvidersError(Exception):
    """Raised when no providers are registered."""


class AllProvidersFailedError(Exception):
    """Raised when all providers have failed and no cached response is available."""


# --- Routing Strategy ---


class RoutingStrategy(StrEnum):
    """Strategy for selecting a provider."""

    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_COST = "least_cost"
    LEAST_LATENCY = "least_latency"
    FAILOVER = "failover"


# --- Provider Stats ---


class ProviderStats(BaseModel):
    """Statistics for a single provider."""

    name: str
    total_requests: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    is_healthy: bool = True


# --- Provider Config ---


class ProviderConfig(BaseModel):
    """Configuration for a registered provider."""

    name: str
    provider: object = Field(..., description="LLMProvider instance")
    weight: float = Field(default=1.0, ge=0.0)
    priority: int = Field(default=0, ge=0)
    max_rpm: int = Field(default=60, ge=1)


# --- Multi-Provider Router ---


class MultiProviderRouter:
    """Routes LLM requests across multiple providers with load balancing and failover.

    Supports weighted round-robin, least-cost, least-latency, and failover
    routing strategies. When all providers fail, falls back to the last
    cached response if available.
    """

    def __init__(
        self,
        *,
        strategy: RoutingStrategy = RoutingStrategy.WEIGHTED_ROUND_ROBIN,
    ) -> None:
        """Initialize the router.

        Args:
            strategy: The routing strategy to use for provider selection.
        """
        self._strategy = strategy
        self._providers: dict[str, ProviderConfig] = {}
        self._stats: dict[str, ProviderStats] = {}
        self._last_response_cache: dict[str, LLMResponse] = {}

        # Weighted round-robin state
        self._wrr_index: int = 0
        self._wrr_counter: float = 0.0

    @property
    def strategy(self) -> RoutingStrategy:
        """Return the current routing strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, value: RoutingStrategy) -> None:
        """Set the routing strategy."""
        self._strategy = value

    def add_provider(
        self,
        name: str,
        provider: LLMProvider,
        weight: float = 1.0,
        priority: int = 0,
        max_rpm: int = 60,
    ) -> None:
        """Register a provider.

        Args:
            name: Unique name for the provider.
            provider: The LLMProvider instance.
            weight: Weight for weighted round-robin (higher = more traffic).
            priority: Priority for failover (lower = higher priority).
            max_rpm: Maximum requests per minute for this provider.
        """
        self._providers[name] = ProviderConfig(
            name=name,
            provider=provider,
            weight=weight,
            priority=priority,
            max_rpm=max_rpm,
        )
        self._stats[name] = ProviderStats(name=name)
        logger.info(
            "provider_added",
            name=name,
            weight=weight,
            priority=priority,
            max_rpm=max_rpm,
        )

    def remove_provider(self, name: str) -> None:
        """Remove a provider.

        Args:
            name: Name of the provider to remove.

        Raises:
            KeyError: If the provider does not exist.
        """
        if name not in self._providers:
            raise KeyError(f"Provider {name!r} not found")
        del self._providers[name]
        del self._stats[name]
        logger.info("provider_removed", name=name)

    async def route(self, request: LLMRequest) -> LLMResponse:
        """Route a request to a provider based on the current strategy.

        Attempts providers in order determined by the strategy. On failure,
        falls back to the next provider. If all providers fail, returns
        the last cached response if available.

        Args:
            request: The LLM request to route.

        Returns:
            The LLM response from the selected provider.

        Raises:
            NoProvidersError: If no providers are registered.
            AllProvidersFailedError: If all providers fail and no cache is available.
        """
        if not self._providers:
            raise NoProvidersError("No providers registered")

        ordered = self._select_order()
        cache_key = hashlib.sha256(request.prompt.encode()).hexdigest()
        errors: list[tuple[str, Exception]] = []

        for name in ordered:
            config = self._providers[name]
            provider: LLMProvider = config.provider  # type: ignore[assignment]
            stats = self._stats[name]

            try:
                start = time.monotonic()
                response = await provider.complete(request)
                latency_ms = (time.monotonic() - start) * 1000

                # Update stats
                stats.total_requests += 1
                stats.total_cost_usd += response.cost_usd
                # Running average for latency
                prev_total = stats.avg_latency_ms * (stats.total_requests - 1)
                stats.avg_latency_ms = (prev_total + latency_ms) / stats.total_requests
                stats.is_healthy = True

                # Cache the response
                self._last_response_cache[cache_key] = response

                logger.debug(
                    "request_routed",
                    provider=name,
                    strategy=self._strategy,
                    latency_ms=round(latency_ms, 1),
                    cost_usd=response.cost_usd,
                )
                return response

            except Exception as exc:
                stats.errors += 1
                stats.total_requests += 1
                if stats.errors >= 3 and stats.errors > stats.total_requests * 0.5:
                    stats.is_healthy = False
                errors.append((name, exc))
                logger.warning(
                    "provider_failed",
                    provider=name,
                    error=str(exc),
                    remaining=len(ordered) - len(errors),
                )

        # All providers failed -- try cache
        if cache_key in self._last_response_cache:
            cached = self._last_response_cache[cache_key]
            logger.warning(
                "all_providers_failed_using_cache",
                providers_tried=len(errors),
                cache_key=cache_key[:32],
            )
            return LLMResponse(
                content=cached.content,
                model=cached.model,
                usage=cached.usage,
                cached=True,
                latency_ms=0.0,
                cost_usd=0.0,
            )

        error_details = "; ".join(f"{n}: {e}" for n, e in errors)
        raise AllProvidersFailedError(
            f"All {len(errors)} providers failed: {error_details}"
        )

    def get_provider_stats(self) -> dict[str, ProviderStats]:
        """Return statistics for all registered providers.

        Returns:
            Dictionary mapping provider name to ProviderStats.
        """
        return dict(self._stats)

    def _select_order(self) -> list[str]:
        """Determine provider ordering based on the current strategy.

        Returns:
            Ordered list of provider names.
        """
        if self._strategy == RoutingStrategy.FAILOVER:
            return self._order_failover()
        if self._strategy == RoutingStrategy.LEAST_COST:
            return self._order_least_cost()
        if self._strategy == RoutingStrategy.LEAST_LATENCY:
            return self._order_least_latency()
        # Default: weighted round-robin
        return self._order_weighted_round_robin()

    def _order_failover(self) -> list[str]:
        """Order providers by priority (lower value = higher priority)."""
        configs = sorted(self._providers.values(), key=lambda c: c.priority)
        return [c.name for c in configs]

    def _order_least_cost(self) -> list[str]:
        """Order providers by total cost (lowest first), healthy first."""
        names = list(self._providers.keys())
        names.sort(
            key=lambda n: (
                not self._stats[n].is_healthy,
                self._stats[n].total_cost_usd / max(self._stats[n].total_requests, 1),
            )
        )
        return names

    def _order_least_latency(self) -> list[str]:
        """Order providers by average latency (lowest first), healthy first."""
        names = list(self._providers.keys())
        names.sort(
            key=lambda n: (
                not self._stats[n].is_healthy,
                self._stats[n].avg_latency_ms,
            )
        )
        return names

    def _order_weighted_round_robin(self) -> list[str]:
        """Order providers using weighted round-robin.

        The primary provider is rotated based on weights; remaining providers
        serve as fallbacks ordered by weight descending.
        """
        healthy = [
            name
            for name, stats in self._stats.items()
            if stats.is_healthy
        ]
        if not healthy:
            # All unhealthy -- fall back to all providers by weight
            healthy = list(self._providers.keys())

        # Sort by weight descending for stable fallback ordering
        healthy.sort(key=lambda n: self._providers[n].weight, reverse=True)

        if len(healthy) <= 1:
            return healthy

        # Advance round-robin counter
        total_weight = sum(self._providers[n].weight for n in healthy)
        if total_weight <= 0:
            return healthy

        self._wrr_counter += 1.0
        # Determine which provider is "primary" this round
        accumulated = 0.0
        primary_idx = 0
        threshold = self._wrr_counter % total_weight
        for i, name in enumerate(healthy):
            accumulated += self._providers[name].weight
            if accumulated > threshold:
                primary_idx = i
                break

        # Put primary first, then the rest in weight order
        result = [healthy[primary_idx]]
        for i, name in enumerate(healthy):
            if i != primary_idx:
                result.append(name)
        return result
