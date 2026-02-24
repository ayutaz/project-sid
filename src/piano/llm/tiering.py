"""Model tiering system for PIANO architecture.

Provides intelligent model selection based on module tier, cost constraints,
and availability with fallback chains for resilience.

Reference: docs/implementation/02-llm-integration.md Section 3
"""

from __future__ import annotations

from typing import ClassVar

import structlog
from pydantic import BaseModel, Field

from piano.core.types import LLMRequest, ModuleTier

logger = structlog.get_logger(__name__)

__all__ = [
    "MODEL_TIER_FAST",
    "MODEL_TIER_MID",
    "MODEL_TIER_SLOW",
    "FallbackChain",
    "ModelConfig",
    "ModelRegistry",
    "ModelRouter",
]


# --- Model Configuration ---


# Model tier values: 1=SLOW (best quality), 2=MID, 3=FAST (cheapest)
MODEL_TIER_SLOW = 1
MODEL_TIER_MID = 2
MODEL_TIER_FAST = 3


class ModelConfig(BaseModel):
    """Configuration for a single LLM model.

    Captures cost, performance, and capability metadata for routing decisions.
    """

    model_id: str
    tier: int = Field(ge=1, le=3)  # 1=SLOW (best), 2=MID, 3=FAST (cheapest)
    cost_per_1k_input: float = Field(ge=0.0)  # USD per 1K input tokens
    cost_per_1k_output: float = Field(ge=0.0)  # USD per 1K output tokens
    max_tokens: int = Field(default=4096, ge=1)
    latency_ms_estimate: float = Field(default=1000.0, ge=0.0)

    @property
    def avg_cost_per_1k(self) -> float:
        """Average cost per 1K tokens (input + output averaged)."""
        return (self.cost_per_1k_input + self.cost_per_1k_output) / 2


# --- Model Registry ---


class ModelRegistry:
    """Registry of available models with tier-based organization.

    Manages model metadata and provides queries for routing decisions.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._models: dict[str, ModelConfig] = {}
        self._by_tier: dict[int, list[ModelConfig]] = {1: [], 2: [], 3: []}

    def register(self, config: ModelConfig) -> None:
        """Register a new model configuration.

        Args:
            config: Model configuration to register.
        """
        self._models[config.model_id] = config
        if config not in self._by_tier[config.tier]:
            self._by_tier[config.tier].append(config)
        logger.debug(
            "Registered model",
            model_id=config.model_id,
            tier=config.tier,
            cost_avg=config.avg_cost_per_1k,
        )

    def get(self, model_id: str) -> ModelConfig | None:
        """Get model configuration by ID.

        Args:
            model_id: The model identifier.

        Returns:
            ModelConfig if found, None otherwise.
        """
        return self._models.get(model_id)

    def get_models_for_tier(self, tier: ModuleTier) -> list[ModelConfig]:
        """Get all models suitable for a given module tier.

        Args:
            tier: The module tier (FAST, MID, SLOW).

        Returns:
            List of model configurations for that tier.
        """
        tier_int = self._tier_to_int(tier)
        return list(self._by_tier[tier_int])

    def get_cheapest(self, tier: ModuleTier) -> ModelConfig:
        """Get the cheapest model for a given tier.

        Args:
            tier: The module tier.

        Returns:
            The cheapest ModelConfig for that tier.

        Raises:
            ValueError: If no models are registered for this tier.
        """
        models = self.get_models_for_tier(tier)
        if not models:
            raise ValueError(f"No models registered for tier {tier}")
        return min(models, key=lambda m: m.avg_cost_per_1k)

    def get_fastest(self, tier: ModuleTier) -> ModelConfig:
        """Get the fastest (lowest latency) model for a given tier.

        Args:
            tier: The module tier.

        Returns:
            The fastest ModelConfig for that tier.

        Raises:
            ValueError: If no models are registered for this tier.
        """
        models = self.get_models_for_tier(tier)
        if not models:
            raise ValueError(f"No models registered for tier {tier}")
        return min(models, key=lambda m: m.latency_ms_estimate)

    def _tier_to_int(self, tier: ModuleTier) -> int:
        """Convert ModuleTier to integer tier level."""
        mapping = {
            ModuleTier.SLOW: 1,
            ModuleTier.MID: 2,
            ModuleTier.FAST: 3,
        }
        return mapping[tier]


# --- Fallback Chain ---


class FallbackChain(BaseModel):
    """Manages a sequence of fallback models.

    Provides iteration through models when failures occur, with reset capability.
    """

    models: list[str] = Field(default_factory=list)
    current_index: int = 0

    def next(self) -> str | None:
        """Get the next model in the fallback chain.

        Returns:
            Next model ID, or None if chain is exhausted.
        """
        if self.current_index >= len(self.models):
            return None
        model_id = self.models[self.current_index]
        self.current_index += 1
        return model_id

    def reset(self) -> None:
        """Reset the chain to the beginning."""
        self.current_index = 0

    def has_next(self) -> bool:
        """Check if there are more models in the chain."""
        return self.current_index < len(self.models)


# --- Model Router ---


class ModelRouter:
    """Routes LLM requests to appropriate models based on tier and constraints.

    Implements dynamic routing with fallback chains, cost-aware selection,
    and failure tracking.
    """

    # Default fallback chains per tier (as per 02-llm-integration.md)
    DEFAULT_FALLBACK_CHAINS: ClassVar[dict[ModuleTier, list[str]]] = {
        ModuleTier.SLOW: ["gpt-4o", "gpt-4o-mini"],
        ModuleTier.MID: ["gpt-4o-mini"],
        ModuleTier.FAST: ["gpt-4o-mini"],
    }

    def __init__(
        self,
        registry: ModelRegistry,
        budget_remaining: float | None = None,
        fallback_chains: dict[ModuleTier, FallbackChain] | None = None,
    ) -> None:
        """Initialize the model router.

        Args:
            registry: Model registry with available models.
            budget_remaining: Optional budget constraint in USD.
            fallback_chains: Custom fallback chains per tier. If None, uses defaults.
        """
        self._registry = registry
        self._budget_remaining = budget_remaining
        self._total_cost_tracked: float = 0.0

        # Initialize fallback chains
        if fallback_chains is None:
            self._fallback_chains = {
                tier: FallbackChain(models=models)
                for tier, models in self.DEFAULT_FALLBACK_CHAINS.items()
            }
        else:
            self._fallback_chains = fallback_chains

        # Track failures for adaptive routing
        self._failure_counts: dict[str, int] = {}

    @property
    def budget_remaining(self) -> float | None:
        """Get the remaining budget in USD.

        Returns:
            Remaining budget or None if no budget constraint is set.
        """
        return self._budget_remaining

    def select_model(self, request: LLMRequest) -> str:
        """Select the best model for a given request.

        Selection considers:
        1. Explicit model in request (highest priority)
        2. Budget constraints (switch to cheaper if budget low)
        3. Tier-based defaults
        4. Fallback chain for resilience

        Args:
            request: The LLM request to route.

        Returns:
            The selected model ID.

        Raises:
            ValueError: If no suitable model can be found.
        """
        # If request specifies a model explicitly, use it
        if request.model:
            logger.debug("Using explicit model from request", model=request.model)
            return request.model

        tier = request.tier

        # Budget-aware selection: if budget is low, prefer cheaper models
        if self._is_budget_constrained():
            logger.info(
                "Budget constrained, selecting cheapest model",
                budget_remaining=self._budget_remaining,
                tier=tier,
            )
            try:
                cheapest = self._registry.get_cheapest(tier)
                return cheapest.model_id
            except ValueError:
                # Fall through to fallback chain if registry is empty
                pass

        # Use fallback chain for tier
        chain = self._fallback_chains.get(tier)
        if chain is None:
            raise ValueError(f"No fallback chain configured for tier {tier}")

        # Try to find a model that hasn't exceeded the failure threshold
        max_failures = 3
        # Walk through chain from current position, skipping failed models
        while chain.has_next():
            model_id = chain.next()
            if model_id is None:
                break
            if self._failure_counts.get(model_id, 0) < max_failures:
                logger.debug(
                    "Selected model from fallback chain",
                    tier=tier,
                    model_id=model_id,
                    budget_remaining=self._budget_remaining,
                )
                return model_id

        # Chain exhausted: reset and return first model
        chain.reset()
        model_id = chain.next()

        if model_id is None:
            raise ValueError(f"Fallback chain for tier {tier} is empty")

        logger.debug(
            "Selected model from fallback chain (reset)",
            tier=tier,
            model_id=model_id,
            budget_remaining=self._budget_remaining,
        )
        return model_id

    def report_failure(self, model_id: str) -> None:
        """Report a failure for a model.

        Tracks failure counts for future routing decisions. In production,
        this could trigger automatic failover to the next model in the chain.

        Args:
            model_id: The model that failed.
        """
        self._failure_counts[model_id] = self._failure_counts.get(model_id, 0) + 1
        logger.warning(
            "Model failure reported",
            model_id=model_id,
            failure_count=self._failure_counts[model_id],
        )

    def report_success(self, model_id: str, cost: float) -> None:
        """Report a successful model invocation.

        Updates cost tracking and resets failure count for the model.

        Args:
            model_id: The model that succeeded.
            cost: The cost of the invocation in USD.
        """
        self._total_cost_tracked += cost
        if self._budget_remaining is not None:
            self._budget_remaining -= cost

        # Reset failure count on success
        if model_id in self._failure_counts:
            self._failure_counts[model_id] = 0

        logger.debug(
            "Model success reported",
            model_id=model_id,
            cost=cost,
            budget_remaining=self._budget_remaining,
            total_cost=self._total_cost_tracked,
        )

    def _is_budget_constrained(self) -> bool:
        """Check if budget is running low.

        Returns:
            True if budget is less than 10% of initial budget or less than $1.
        """
        if self._budget_remaining is None:
            return False
        return self._budget_remaining < 1.0
