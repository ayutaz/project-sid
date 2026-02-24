"""Phase 1 smoke tests -- instantiation and integration checks.

Verifies that Phase 1 classes can be instantiated with minimal arguments
and that key integration points (module registration, etc.) work correctly.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
async def test_goals_generator_instantiation():
    """Test that GoalGenerator can be instantiated with mock dependencies."""
    try:
        from piano.goals.generator import GoalGenerator
        from piano.llm.mock import MockLLMProvider

        llm = MockLLMProvider()
        generator = GoalGenerator(llm_provider=llm)
        assert generator is not None
        assert hasattr(generator, "name")
        assert hasattr(generator, "tier")
        assert hasattr(generator, "tick")
    except ImportError:
        pytest.skip("GoalGenerator not yet implemented")
    except Exception as e:
        pytest.fail(f"GoalGenerator instantiation failed: {e}")


@pytest.mark.integration
async def test_planner_instantiation():
    """Test that Planner can be instantiated with mock dependencies."""
    try:
        from piano.llm.mock import MockLLMProvider
        from piano.planning.planner import Planner

        llm = MockLLMProvider()
        planner = Planner(llm_provider=llm)
        assert planner is not None
        assert hasattr(planner, "name")
        assert hasattr(planner, "tier")
        assert hasattr(planner, "tick")
    except ImportError:
        pytest.skip("Planner not yet implemented")
    except Exception as e:
        pytest.fail(f"Planner instantiation failed: {e}")


@pytest.mark.integration
async def test_ltm_instantiation():
    """Test that LongTermMemory can be instantiated."""
    try:
        from piano.memory.ltm import LongTermMemory

        ltm = LongTermMemory(collection_name="test_collection")
        assert ltm is not None
    except ImportError:
        pytest.skip("LongTermMemory not yet implemented")
    except Exception as e:
        pytest.fail(f"LongTermMemory instantiation failed: {e}")


@pytest.mark.integration
async def test_checkpoint_manager_instantiation():
    """Test that CheckpointManager can be instantiated."""
    try:
        from piano.core.checkpoint import CheckpointManager

        manager = CheckpointManager(checkpoint_dir="/tmp/test_checkpoints")
        assert manager is not None
    except ImportError:
        pytest.skip("CheckpointManager not yet implemented")
    except Exception as e:
        pytest.fail(f"CheckpointManager instantiation failed: {e}")


@pytest.mark.integration
async def test_module_registration_goals():
    """Test that GoalGenerator can be registered with ModuleScheduler."""
    try:
        from piano.core.scheduler import ModuleScheduler
        from piano.core.types import ExecutionTier
        from piano.goals.generator import GoalGenerator
        from piano.llm.mock import MockLLMProvider

        llm = MockLLMProvider()
        generator = GoalGenerator(llm_provider=llm)
        scheduler = ModuleScheduler()

        # Check module has required attributes
        assert hasattr(generator, "name")
        assert hasattr(generator, "tier")
        assert generator.tier in [ExecutionTier.FAST, ExecutionTier.MID, ExecutionTier.SLOW]

        # Test registration
        scheduler.register(generator)
        assert generator.name in [m.name for m in scheduler.get_modules()]
    except ImportError:
        pytest.skip("GoalGenerator or ModuleScheduler not yet implemented")
    except Exception as e:
        pytest.fail(f"Module registration failed: {e}")


@pytest.mark.integration
async def test_module_registration_planner():
    """Test that Planner can be registered with ModuleScheduler."""
    try:
        from piano.core.scheduler import ModuleScheduler
        from piano.core.types import ExecutionTier
        from piano.llm.mock import MockLLMProvider
        from piano.planning.planner import Planner

        llm = MockLLMProvider()
        planner = Planner(llm_provider=llm)
        scheduler = ModuleScheduler()

        # Check module has required attributes
        assert hasattr(planner, "name")
        assert hasattr(planner, "tier")
        assert planner.tier in [ExecutionTier.FAST, ExecutionTier.MID, ExecutionTier.SLOW]

        # Test registration
        scheduler.register(planner)
        assert planner.name in [m.name for m in scheduler.get_modules()]
    except ImportError:
        pytest.skip("Planner or ModuleScheduler not yet implemented")
    except Exception as e:
        pytest.fail(f"Module registration failed: {e}")


@pytest.mark.integration
async def test_llm_tiering_configuration():
    """Test that LLM tiering configuration can be loaded."""
    try:
        from piano.llm.tiering import TieringConfig

        config = TieringConfig()
        assert config is not None
        assert hasattr(config, "tier1_model")
        assert hasattr(config, "tier2_model")
        assert hasattr(config, "tier3_model")
    except ImportError:
        pytest.skip("TieringConfig not yet implemented")
    except Exception as e:
        pytest.fail(f"TieringConfig instantiation failed: {e}")


@pytest.mark.integration
async def test_action_awareness_nn_instantiation():
    """Test that ActionAwarenessNN can be instantiated."""
    try:
        from piano.awareness.nn_model import ActionAwarenessNN

        model = ActionAwarenessNN(input_dim=128, hidden_dim=64, output_dim=32)
        assert model is not None
    except ImportError:
        pytest.skip("ActionAwarenessNN not yet implemented")
    except Exception as e:
        pytest.fail(f"ActionAwarenessNN instantiation failed: {e}")
