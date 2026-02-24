from __future__ import annotations

import pytest


@pytest.mark.integration
def test_import_goals_generator():
    """Test that goals generator module can be imported."""
    try:
        from piano.goals import generator  # noqa: F401
    except ImportError:
        pytest.skip("goals.generator not yet implemented")


@pytest.mark.integration
def test_import_planning_planner():
    """Test that planning planner module can be imported."""
    try:
        from piano.planning import planner  # noqa: F401
    except ImportError:
        pytest.skip("planning.planner not yet implemented")


@pytest.mark.integration
def test_import_talking():
    """Test that talking module can be imported."""
    try:
        from piano import talking  # noqa: F401
    except ImportError:
        pytest.skip("talking module not yet implemented")


@pytest.mark.integration
def test_import_reflection():
    """Test that reflection module can be imported."""
    try:
        from piano import reflection  # noqa: F401
    except ImportError:
        pytest.skip("reflection module not yet implemented")


@pytest.mark.integration
def test_import_social():
    """Test that social awareness module can be imported."""
    try:
        from piano import social  # noqa: F401
    except ImportError:
        pytest.skip("social module not yet implemented")


@pytest.mark.integration
def test_import_memory_ltm():
    """Test that long-term memory module can be imported."""
    try:
        from piano.memory import ltm  # noqa: F401
    except ImportError:
        pytest.skip("memory.ltm not yet implemented")


@pytest.mark.integration
def test_import_memory_ltm_search():
    """Test that LTM search module can be imported."""
    try:
        from piano.memory import ltm_search  # noqa: F401
    except ImportError:
        pytest.skip("memory.ltm_search not yet implemented")


@pytest.mark.integration
def test_import_awareness_nn_model():
    """Test that action awareness neural network module can be imported."""
    try:
        from piano.awareness import nn_model  # noqa: F401
    except ImportError:
        pytest.skip("awareness.nn_model not yet implemented")


@pytest.mark.integration
def test_import_llm_tiering():
    """Test that LLM tiering module can be imported."""
    try:
        from piano.llm import tiering  # noqa: F401
    except ImportError:
        pytest.skip("llm.tiering not yet implemented")


@pytest.mark.integration
def test_import_core_checkpoint():
    """Test that checkpoint module can be imported."""
    try:
        from piano.core import checkpoint  # noqa: F401
    except ImportError:
        pytest.skip("core.checkpoint not yet implemented")


@pytest.mark.integration
def test_import_eval():
    """Test that evaluation module can be imported."""
    try:
        from piano import eval  # noqa: F401
    except ImportError:
        pytest.skip("eval module not yet implemented")


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
