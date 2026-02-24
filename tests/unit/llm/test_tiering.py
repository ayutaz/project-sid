"""Tests for LLM model tiering system.

Tests model registry, fallback chains, and intelligent routing with
cost constraints and failure handling.
"""

from __future__ import annotations

import pytest

from piano.core.types import LLMRequest, ModuleTier
from piano.llm.tiering import FallbackChain, ModelConfig, ModelRegistry, ModelRouter

# --- ModelConfig Tests ---


def test_model_config_creation() -> None:
    """Test creating a model configuration."""
    config = ModelConfig(
        model_id="gpt-4o",
        tier=1,
        cost_per_1k_input=2.5,
        cost_per_1k_output=10.0,
        max_tokens=4096,
        latency_ms_estimate=800.0,
    )

    assert config.model_id == "gpt-4o"
    assert config.tier == 1
    assert config.cost_per_1k_input == 2.5
    assert config.cost_per_1k_output == 10.0
    assert config.max_tokens == 4096
    assert config.latency_ms_estimate == 800.0


def test_model_config_avg_cost() -> None:
    """Test average cost calculation."""
    config = ModelConfig(
        model_id="gpt-4o",
        tier=1,
        cost_per_1k_input=2.5,
        cost_per_1k_output=10.0,
    )
    # Average of 2.5 and 10.0
    assert config.avg_cost_per_1k == 6.25


def test_model_config_zero_cost() -> None:
    """Test model configuration with zero cost."""
    config = ModelConfig(
        model_id="cheap-model",
        tier=3,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
    )
    assert config.avg_cost_per_1k == 0.0


# --- ModelRegistry Tests ---


def test_registry_register_and_get() -> None:
    """Test registering and retrieving models."""
    registry = ModelRegistry()

    config = ModelConfig(
        model_id="gpt-4o",
        tier=1,
        cost_per_1k_input=2.5,
        cost_per_1k_output=10.0,
    )
    registry.register(config)

    retrieved = registry.get("gpt-4o")
    assert retrieved is not None
    assert retrieved.model_id == "gpt-4o"
    assert retrieved.tier == 1


def test_registry_get_nonexistent() -> None:
    """Test getting a non-existent model returns None."""
    registry = ModelRegistry()
    assert registry.get("nonexistent") is None


def test_registry_get_models_for_tier() -> None:
    """Test retrieving all models for a tier."""
    registry = ModelRegistry()

    # Register tier 1 models
    registry.register(
        ModelConfig(
            model_id="gpt-4o",
            tier=1,
            cost_per_1k_input=2.5,
            cost_per_1k_output=10.0,
        )
    )
    registry.register(
        ModelConfig(
            model_id="gpt-4.1",
            tier=1,
            cost_per_1k_input=2.0,
            cost_per_1k_output=8.0,
        )
    )

    # Register tier 2 model
    registry.register(
        ModelConfig(
            model_id="gpt-4o-mini",
            tier=2,
            cost_per_1k_input=0.15,
            cost_per_1k_output=0.60,
        )
    )

    tier1_models = registry.get_models_for_tier(ModuleTier.SLOW)
    assert len(tier1_models) == 2
    model_ids = {m.model_id for m in tier1_models}
    assert "gpt-4o" in model_ids
    assert "gpt-4.1" in model_ids

    tier2_models = registry.get_models_for_tier(ModuleTier.MID)
    assert len(tier2_models) == 1
    assert tier2_models[0].model_id == "gpt-4o-mini"


def test_registry_get_cheapest() -> None:
    """Test getting the cheapest model for a tier."""
    registry = ModelRegistry()

    registry.register(
        ModelConfig(
            model_id="gpt-4o",
            tier=1,
            cost_per_1k_input=2.5,
            cost_per_1k_output=10.0,  # avg = 6.25
        )
    )
    registry.register(
        ModelConfig(
            model_id="gpt-4.1",
            tier=1,
            cost_per_1k_input=2.0,
            cost_per_1k_output=8.0,  # avg = 5.0
        )
    )
    registry.register(
        ModelConfig(
            model_id="gpt-4.1-mini",
            tier=1,
            cost_per_1k_input=0.4,
            cost_per_1k_output=1.6,  # avg = 1.0 (cheapest)
        )
    )

    cheapest = registry.get_cheapest(ModuleTier.SLOW)
    assert cheapest.model_id == "gpt-4.1-mini"


def test_registry_get_fastest() -> None:
    """Test getting the fastest model for a tier."""
    registry = ModelRegistry()

    registry.register(
        ModelConfig(
            model_id="gpt-4o",
            tier=2,
            cost_per_1k_input=0.15,
            cost_per_1k_output=0.60,
            latency_ms_estimate=500.0,
        )
    )
    registry.register(
        ModelConfig(
            model_id="gpt-4.1-nano",
            tier=2,
            cost_per_1k_input=0.1,
            cost_per_1k_output=0.4,
            latency_ms_estimate=200.0,  # fastest
        )
    )
    registry.register(
        ModelConfig(
            model_id="gpt-4.1-mini",
            tier=2,
            cost_per_1k_input=0.4,
            cost_per_1k_output=1.6,
            latency_ms_estimate=300.0,
        )
    )

    fastest = registry.get_fastest(ModuleTier.MID)
    assert fastest.model_id == "gpt-4.1-nano"


def test_registry_empty_tier_raises_error() -> None:
    """Test that querying an empty tier raises ValueError."""
    registry = ModelRegistry()

    with pytest.raises(ValueError, match="No models registered for tier"):
        registry.get_cheapest(ModuleTier.SLOW)

    with pytest.raises(ValueError, match="No models registered for tier"):
        registry.get_fastest(ModuleTier.MID)


# --- FallbackChain Tests ---


def test_fallback_chain_next() -> None:
    """Test iterating through fallback chain."""
    chain = FallbackChain(models=["gpt-4o", "gpt-4.1", "gpt-4o-mini"])

    assert chain.next() == "gpt-4o"
    assert chain.next() == "gpt-4.1"
    assert chain.next() == "gpt-4o-mini"
    assert chain.next() is None  # Exhausted


def test_fallback_chain_reset() -> None:
    """Test resetting the fallback chain."""
    chain = FallbackChain(models=["model-a", "model-b"])

    chain.next()
    chain.next()
    assert chain.next() is None

    chain.reset()
    assert chain.next() == "model-a"


def test_fallback_chain_has_next() -> None:
    """Test checking if chain has more models."""
    chain = FallbackChain(models=["model-a", "model-b"])

    assert chain.has_next() is True
    chain.next()
    assert chain.has_next() is True
    chain.next()
    assert chain.has_next() is False


def test_fallback_chain_empty() -> None:
    """Test empty fallback chain."""
    chain = FallbackChain(models=[])
    assert chain.has_next() is False
    assert chain.next() is None


# --- ModelRouter Tests ---


def test_router_select_explicit_model() -> None:
    """Test that explicit model in request is always used."""
    registry = ModelRegistry()
    router = ModelRouter(registry=registry)

    request = LLMRequest(
        prompt="test",
        model="explicit-model",
        tier=ModuleTier.SLOW,
    )

    selected = router.select_model(request)
    assert selected == "explicit-model"


def test_router_select_from_fallback_chain() -> None:
    """Test selecting from default fallback chain."""
    registry = ModelRegistry()
    router = ModelRouter(registry=registry)

    request = LLMRequest(
        prompt="test",
        tier=ModuleTier.SLOW,  # Should use SLOW fallback chain
    )

    selected = router.select_model(request)
    # Should get first model from SLOW chain: gpt-4o
    assert selected == "gpt-4o"


def test_router_budget_aware_selection() -> None:
    """Test that router switches to cheapest model when budget is low."""
    registry = ModelRegistry()

    # Register expensive and cheap models for tier 1
    registry.register(
        ModelConfig(
            model_id="expensive",
            tier=1,
            cost_per_1k_input=10.0,
            cost_per_1k_output=20.0,
        )
    )
    registry.register(
        ModelConfig(
            model_id="cheap",
            tier=1,
            cost_per_1k_input=0.1,
            cost_per_1k_output=0.2,
        )
    )

    # Low budget (< $1)
    router = ModelRouter(registry=registry, budget_remaining=0.5)

    request = LLMRequest(prompt="test", tier=ModuleTier.SLOW)
    selected = router.select_model(request)

    # Should select cheapest model
    assert selected == "cheap"


def test_router_report_success_updates_budget() -> None:
    """Test that successful calls update budget."""
    registry = ModelRegistry()
    router = ModelRouter(registry=registry, budget_remaining=10.0)

    router.report_success(model_id="gpt-4o", cost=2.5)

    assert router.budget_remaining == 7.5


def test_router_report_success_no_budget_constraint() -> None:
    """Test that success reporting works without budget constraint."""
    registry = ModelRegistry()
    router = ModelRouter(registry=registry, budget_remaining=None)

    router.report_success(model_id="gpt-4o", cost=2.5)

    assert router.budget_remaining is None


def test_router_report_failure_tracks_count() -> None:
    """Test that failures are tracked."""
    registry = ModelRegistry()
    router = ModelRouter(registry=registry)

    router.report_failure(model_id="gpt-4o")
    router.report_failure(model_id="gpt-4o")

    # Check internal tracking (via inspection)
    assert router._failure_counts["gpt-4o"] == 2


def test_router_report_success_resets_failure_count() -> None:
    """Test that success resets failure count."""
    registry = ModelRegistry()
    router = ModelRouter(registry=registry)

    router.report_failure(model_id="gpt-4o")
    assert router._failure_counts["gpt-4o"] == 1

    router.report_success(model_id="gpt-4o", cost=0.5)
    assert router._failure_counts["gpt-4o"] == 0


def test_router_custom_fallback_chains() -> None:
    """Test router with custom fallback chains."""
    registry = ModelRegistry()

    custom_chains = {
        ModuleTier.SLOW: FallbackChain(models=["custom-slow"]),
        ModuleTier.MID: FallbackChain(models=["custom-mid"]),
        ModuleTier.FAST: FallbackChain(models=["custom-fast"]),
    }

    router = ModelRouter(registry=registry, fallback_chains=custom_chains)

    request_slow = LLMRequest(prompt="test", tier=ModuleTier.SLOW)
    assert router.select_model(request_slow) == "custom-slow"

    request_mid = LLMRequest(prompt="test", tier=ModuleTier.MID)
    assert router.select_model(request_mid) == "custom-mid"


def test_router_default_fallback_chains_configured() -> None:
    """Test that default fallback chains match specification."""
    registry = ModelRegistry()
    router = ModelRouter(registry=registry)

    # SLOW tier chain
    slow_chain = router._fallback_chains[ModuleTier.SLOW]
    assert slow_chain.models == ["gpt-4o", "gpt-4o-mini"]

    # MID tier chain
    mid_chain = router._fallback_chains[ModuleTier.MID]
    assert mid_chain.models == ["gpt-4o-mini"]

    # FAST tier chain
    fast_chain = router._fallback_chains[ModuleTier.FAST]
    assert fast_chain.models == ["gpt-4o-mini"]


def test_router_tier_based_selection() -> None:
    """Test that different tiers select different models."""
    registry = ModelRegistry()
    router = ModelRouter(registry=registry)

    # SLOW tier
    slow_request = LLMRequest(prompt="test", tier=ModuleTier.SLOW)
    slow_model = router.select_model(slow_request)
    assert slow_model == "gpt-4o"

    # MID tier
    mid_request = LLMRequest(prompt="test", tier=ModuleTier.MID)
    mid_model = router.select_model(mid_request)
    assert mid_model == "gpt-4o-mini"

    # FAST tier
    fast_request = LLMRequest(prompt="test", tier=ModuleTier.FAST)
    fast_model = router.select_model(fast_request)
    assert fast_model == "gpt-4o-mini"


def test_router_cost_tracking() -> None:
    """Test that router tracks total cost correctly."""
    registry = ModelRegistry()
    router = ModelRouter(registry=registry, budget_remaining=100.0)

    router.report_success(model_id="gpt-4o", cost=5.0)
    router.report_success(model_id="gpt-4o-mini", cost=0.5)
    router.report_success(model_id="claude-sonnet", cost=3.0)

    # Total cost should be tracked
    assert router._total_cost_tracked == 8.5
    # Budget should be reduced
    assert router.budget_remaining == 91.5


def test_router_fallback_after_failures() -> None:
    """Test that fallback chain skips models with too many failures."""
    registry = ModelRegistry()
    # SLOW chain has ["gpt-4o", "gpt-4o-mini"]
    router = ModelRouter(registry=registry)

    # Report 3 failures for gpt-4o (threshold is 3)
    router.report_failure("gpt-4o")
    router.report_failure("gpt-4o")
    router.report_failure("gpt-4o")

    request = LLMRequest(prompt="test", tier=ModuleTier.SLOW)
    selected = router.select_model(request)
    # gpt-4o should be skipped, fall back to gpt-4o-mini
    assert selected == "gpt-4o-mini"


def test_router_fallback_chain_resets_when_all_failed() -> None:
    """Test that chain resets and returns first model when all models have failures."""
    registry = ModelRegistry()
    router = ModelRouter(registry=registry)

    # Fail all models in the SLOW chain
    for _ in range(3):
        router.report_failure("gpt-4o")
        router.report_failure("gpt-4o-mini")

    request = LLMRequest(prompt="test", tier=ModuleTier.SLOW)
    selected = router.select_model(request)
    # Chain should reset, return first model
    assert selected == "gpt-4o"


def test_router_fallback_chain_success_resets_failures() -> None:
    """Test that success resets failure count, allowing model to be selected again."""
    registry = ModelRegistry()
    router = ModelRouter(registry=registry)

    # Fail gpt-4o 3 times
    for _ in range(3):
        router.report_failure("gpt-4o")

    # Select should skip gpt-4o
    request = LLMRequest(prompt="test", tier=ModuleTier.SLOW)
    selected = router.select_model(request)
    assert selected == "gpt-4o-mini"

    # Report success, resetting failure count
    router.report_success("gpt-4o", cost=0.01)

    # Now gpt-4o should be selectable again
    selected = router.select_model(request)
    assert selected == "gpt-4o"
