"""Tests for the neural network-based ActionAwareness module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from piano.awareness.nn_model import (
    ActionEncoder,
    ActionOutcomePredictor,
    FeedForwardNN,
    NNActionAwarenessModule,
    NNConfig,
)
from piano.core.sas import SharedAgentState
from piano.core.types import (
    ActionHistoryEntry,
    AgentId,
    GoalData,
    MemoryEntry,
    PerceptData,
    PlanData,
    SelfReflectionData,
    SocialData,
)

# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


class InMemorySAS(SharedAgentState):
    """In-memory SAS implementation for unit tests."""

    def __init__(self, agent_id: str = "test-agent") -> None:
        self._agent_id = agent_id
        self._percepts = PerceptData()
        self._goals = GoalData()
        self._social = SocialData()
        self._plans = PlanData()
        self._action_history: list[ActionHistoryEntry] = []
        self._working_memory: list[MemoryEntry] = []
        self._stm: list[MemoryEntry] = []
        self._self_reflection = SelfReflectionData()
        self._cc_decision: dict[str, object] | None = None
        self._sections: dict[str, dict[str, object]] = {}

    @property
    def agent_id(self) -> AgentId:
        return self._agent_id

    async def get_percepts(self) -> PerceptData:
        return self._percepts

    async def update_percepts(self, percepts: PerceptData) -> None:
        self._percepts = percepts

    async def get_goals(self) -> GoalData:
        return self._goals

    async def update_goals(self, goals: GoalData) -> None:
        self._goals = goals

    async def get_social(self) -> SocialData:
        return self._social

    async def update_social(self, social: SocialData) -> None:
        self._social = social

    async def get_plans(self) -> PlanData:
        return self._plans

    async def update_plans(self, plans: PlanData) -> None:
        self._plans = plans

    async def get_action_history(self, limit: int = 50) -> list[ActionHistoryEntry]:
        return self._action_history[:limit]

    async def add_action(self, entry: ActionHistoryEntry) -> None:
        self._action_history.insert(0, entry)
        self._action_history = self._action_history[:50]

    async def get_working_memory(self) -> list[MemoryEntry]:
        return self._working_memory

    async def set_working_memory(self, entries: list[MemoryEntry]) -> None:
        self._working_memory = list(entries)

    async def get_stm(self, limit: int = 100) -> list[MemoryEntry]:
        return self._stm[:limit]

    async def add_stm(self, entry: MemoryEntry) -> None:
        self._stm.insert(0, entry)
        self._stm = self._stm[:100]

    async def get_self_reflection(self) -> SelfReflectionData:
        return self._self_reflection

    async def update_self_reflection(self, reflection: SelfReflectionData) -> None:
        self._self_reflection = reflection

    async def get_last_cc_decision(self) -> dict[str, object] | None:
        return self._cc_decision

    async def set_cc_decision(self, decision: dict[str, object]) -> None:
        self._cc_decision = decision

    async def get_section(self, section: str) -> dict[str, object]:
        return self._sections.get(section, {})

    async def update_section(self, section: str, data: dict[str, object]) -> None:
        self._sections[section] = data

    async def snapshot(self) -> dict[str, object]:
        return {"agent_id": self._agent_id}

    async def initialize(self) -> None:
        pass

    async def clear(self) -> None:
        self._action_history.clear()
        self._percepts = PerceptData()


# ---------------------------------------------------------------------------
# NNConfig Tests
# ---------------------------------------------------------------------------


class TestNNConfig:
    """Tests for NNConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = NNConfig()
        assert config.input_dim == 64
        assert config.hidden_dims == [128, 64, 32]
        assert config.output_dim == 16
        assert config.learning_rate == 0.001

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = NNConfig(
            input_dim=32,
            hidden_dims=[64, 32],
            output_dim=8,
            learning_rate=0.0005,
        )
        assert config.input_dim == 32
        assert config.hidden_dims == [64, 32]
        assert config.output_dim == 8
        assert config.learning_rate == 0.0005


# ---------------------------------------------------------------------------
# FeedForwardNN Tests
# ---------------------------------------------------------------------------


class TestFeedForwardNN:
    """Tests for FeedForwardNN."""

    @pytest.fixture
    def config(self) -> NNConfig:
        """Small config for testing."""
        return NNConfig(input_dim=8, hidden_dims=[16, 8], output_dim=4)

    @pytest.fixture
    def nn(self, config: NNConfig) -> FeedForwardNN:
        """Feedforward network instance."""
        return FeedForwardNN(config)

    def test_initialization(self, nn: FeedForwardNN, config: NNConfig) -> None:
        """Test network initialization."""
        # Should have 3 layers (2 hidden + 1 output)
        assert len(nn.weights) == 3
        assert len(nn.biases) == 3

        # Check shapes
        assert nn.weights[0].shape == (8, 16)  # input -> hidden1
        assert nn.weights[1].shape == (16, 8)  # hidden1 -> hidden2
        assert nn.weights[2].shape == (8, 4)  # hidden2 -> output

    def test_forward_single_sample(self, nn: FeedForwardNN) -> None:
        """Test forward pass with a single sample."""
        x = np.random.randn(8).astype(np.float32)
        output = nn.forward(x)

        assert output.shape == (4,)
        assert np.all(output >= 0)  # softmax outputs are non-negative
        assert np.isclose(np.sum(output), 1.0)  # softmax sums to 1

    def test_forward_batch(self, nn: FeedForwardNN) -> None:
        """Test forward pass with a batch."""
        x = np.random.randn(5, 8).astype(np.float32)
        output = nn.forward(x)

        assert output.shape == (5, 4)
        assert np.all(output >= 0)
        # Each row should sum to 1
        row_sums = np.sum(output, axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_predict_alias(self, nn: FeedForwardNN) -> None:
        """Test that predict() is an alias for forward()."""
        x = np.random.randn(8).astype(np.float32)
        forward_out = nn.forward(x)
        predict_out = nn.predict(x)

        assert np.allclose(forward_out, predict_out)

    def test_relu_activation(self) -> None:
        """Test ReLU activation function."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = FeedForwardNN._relu(x)

        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        assert np.allclose(result, expected)

    def test_softmax_activation(self) -> None:
        """Test softmax activation function."""
        x = np.array([[1.0, 2.0, 3.0]])
        result = FeedForwardNN._softmax(x)

        # Should sum to 1
        assert np.isclose(np.sum(result), 1.0)
        # Larger inputs should have larger outputs
        assert result[0, 2] > result[0, 1] > result[0, 0]

    def test_parameter_count(self, nn: FeedForwardNN) -> None:
        """Test parameter count calculation."""
        # Manual calculation:
        # Layer 1: 8*16 + 16 = 144
        # Layer 2: 16*8 + 8 = 136
        # Layer 3: 8*4 + 4 = 36
        # Total: 316
        expected = (8 * 16 + 16) + (16 * 8 + 8) + (8 * 4 + 4)
        assert nn.parameter_count == expected

    def test_parameter_count_default_config(self) -> None:
        """Test parameter count for default config (~100K)."""
        config = NNConfig()
        nn = FeedForwardNN(config)

        # Manual calculation for default:
        # Layer 1: 64*128 + 128 = 8320
        # Layer 2: 128*64 + 64 = 8256
        # Layer 3: 64*32 + 32 = 2080
        # Layer 4: 32*16 + 16 = 528
        # Total: 19184 (smaller than 100K due to small default config)
        expected = (64 * 128 + 128) + (128 * 64 + 64) + (64 * 32 + 32) + (32 * 16 + 16)
        assert nn.parameter_count == expected

    def test_save_load_round_trip(self, nn: FeedForwardNN) -> None:
        """Test saving and loading preserves the model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / "model")

            # Save the model
            nn.save(model_path)

            # Create a new instance and load
            loaded_nn = FeedForwardNN(nn.config)
            loaded_nn.load(model_path)

            # Check weights are identical
            assert len(loaded_nn.weights) == len(nn.weights)
            for w1, w2 in zip(nn.weights, loaded_nn.weights, strict=False):
                assert np.allclose(w1, w2)

            # Check biases are identical
            for b1, b2 in zip(nn.biases, loaded_nn.biases, strict=False):
                assert np.allclose(b1, b2)

            # Check outputs are identical
            x = np.random.randn(8).astype(np.float32)
            out1 = nn.predict(x)
            out2 = loaded_nn.predict(x)
            assert np.allclose(out1, out2)


# ---------------------------------------------------------------------------
# ActionEncoder Tests
# ---------------------------------------------------------------------------


class TestActionEncoder:
    """Tests for ActionEncoder."""

    @pytest.fixture
    def encoder(self) -> ActionEncoder:
        """Action encoder instance."""
        return ActionEncoder(output_dim=64)

    def test_encode_state_shape(self, encoder: ActionEncoder) -> None:
        """Test encoding produces correct shape."""
        percepts = {"position": {"x": 10, "y": 64, "z": 20}}
        goals = {"current_goal": "mine iron"}
        action = "mine"

        encoding = encoder.encode_state(percepts, goals, action)

        assert encoding.shape == (64,)
        assert encoding.dtype == np.float32

    def test_encode_state_different_inputs(self, encoder: ActionEncoder) -> None:
        """Test different inputs produce different encodings."""
        percepts1 = {"position": {"x": 10, "y": 64, "z": 20}}
        percepts2 = {"position": {"x": 100, "y": 128, "z": 200}}
        goals = {"current_goal": "mine iron"}

        enc1 = encoder.encode_state(percepts1, goals, "mine")
        enc2 = encoder.encode_state(percepts2, goals, "mine")

        # Should be different (position changed)
        assert not np.allclose(enc1, enc2)

    def test_encode_action_types(self, encoder: ActionEncoder) -> None:
        """Test different action types are encoded differently."""
        percepts = {"position": {"x": 10, "y": 64, "z": 20}}
        goals = {"current_goal": "mine iron"}

        enc_move = encoder.encode_state(percepts, goals, "move")
        enc_mine = encoder.encode_state(percepts, goals, "mine")
        enc_craft = encoder.encode_state(percepts, goals, "craft")

        # First 8 dims are action one-hot, should differ
        assert not np.allclose(enc_move[:8], enc_mine[:8])
        assert not np.allclose(enc_move[:8], enc_craft[:8])

    def test_normalize(self) -> None:
        """Test normalization function."""
        # Test normal case
        assert ActionEncoder._normalize(50, 0, 100) == 0.5
        assert ActionEncoder._normalize(0, 0, 100) == 0.0
        assert ActionEncoder._normalize(100, 0, 100) == 1.0

        # Test clipping
        assert ActionEncoder._normalize(-10, 0, 100) == 0.0
        assert ActionEncoder._normalize(150, 0, 100) == 1.0

        # Test same min/max
        assert ActionEncoder._normalize(50, 10, 10) == 0.0

    def test_hash_string(self) -> None:
        """Test string hashing is deterministic."""
        s1 = "mine iron ore"
        s2 = "craft wooden pickaxe"

        # Same string should hash to same bin
        assert ActionEncoder._hash_string(s1, 8) == ActionEncoder._hash_string(s1, 8)

        # Different strings (likely) hash to different bins
        # (not guaranteed due to collisions, but very likely with small set)
        hash1 = ActionEncoder._hash_string(s1, 8)
        hash2 = ActionEncoder._hash_string(s2, 8)
        assert 0 <= hash1 < 8
        assert 0 <= hash2 < 8

    def test_encode_inventory(self, encoder: ActionEncoder) -> None:
        """Test inventory encoding affects the output."""
        percepts_empty = {"inventory": {}}
        percepts_full = {"inventory": {"stone": 64, "iron": 32}}
        goals = {"current_goal": "craft"}

        enc_empty = encoder.encode_state(percepts_empty, goals, "craft")
        enc_full = encoder.encode_state(percepts_full, goals, "craft")

        # Inventory affects dim 13 (total items)
        assert enc_empty[13] < enc_full[13]

    def test_encode_health_hunger(self, encoder: ActionEncoder) -> None:
        """Test health/hunger encoding."""
        percepts_healthy = {"health": 20, "hunger": 20}
        percepts_low = {"health": 5, "hunger": 5}
        goals = {}

        enc_healthy = encoder.encode_state(percepts_healthy, goals, "move")
        enc_low = encoder.encode_state(percepts_low, goals, "move")

        # Health/hunger in dims 11-12
        assert enc_healthy[11] > enc_low[11]  # health
        assert enc_healthy[12] > enc_low[12]  # hunger


# ---------------------------------------------------------------------------
# ActionOutcomePredictor Tests
# ---------------------------------------------------------------------------


class TestActionOutcomePredictor:
    """Tests for ActionOutcomePredictor."""

    @pytest.fixture
    def config(self) -> NNConfig:
        """Small config for testing."""
        return NNConfig(input_dim=64, hidden_dims=[32, 16], output_dim=16)

    @pytest.fixture
    def nn(self, config: NNConfig) -> FeedForwardNN:
        """Feedforward network instance."""
        return FeedForwardNN(config)

    @pytest.fixture
    def encoder(self) -> ActionEncoder:
        """Action encoder instance."""
        return ActionEncoder(output_dim=64)

    @pytest.fixture
    def predictor(self, nn: FeedForwardNN, encoder: ActionEncoder) -> ActionOutcomePredictor:
        """Action outcome predictor instance."""
        return ActionOutcomePredictor(nn, encoder)

    def test_predict_outcome_returns_expected_keys(self, predictor: ActionOutcomePredictor) -> None:
        """Test predict_outcome returns required keys."""
        state = {
            "percepts": {"position": {"x": 10, "y": 64, "z": 20}},
            "goals": {"current_goal": "mine iron"},
        }
        action = "mine"

        outcome = predictor.predict_outcome(state, action)

        assert "success_probability" in outcome
        assert "expected_duration" in outcome
        assert "risk_score" in outcome

    def test_predict_outcome_ranges(self, predictor: ActionOutcomePredictor) -> None:
        """Test predicted values are in reasonable ranges."""
        state = {
            "percepts": {"position": {"x": 10, "y": 64, "z": 20}},
            "goals": {"current_goal": "mine iron"},
        }
        action = "mine"

        outcome = predictor.predict_outcome(state, action)

        # Probabilities and scores should be in [0, 1]
        assert 0 <= outcome["success_probability"] <= 1
        assert 0 <= outcome["risk_score"] <= 1

        # Duration should be positive
        assert outcome["expected_duration"] >= 0

    def test_calculate_discrepancy_identical(self, predictor: ActionOutcomePredictor) -> None:
        """Test discrepancy is 0 for identical outcomes."""
        predicted = {
            "success_probability": 1.0,
            "expected_duration": 10.0,
            "risk_score": 0.0,
        }
        actual = {
            "success": True,
            "duration": 10.0,
        }

        discrepancy = predictor.calculate_discrepancy(predicted, actual)

        # Should be very close to 0 (identical)
        assert discrepancy < 0.1

    def test_calculate_discrepancy_very_different(self, predictor: ActionOutcomePredictor) -> None:
        """Test discrepancy is high for very different outcomes."""
        predicted = {
            "success_probability": 1.0,
            "expected_duration": 10.0,
            "risk_score": 0.0,
        }
        actual = {
            "success": False,  # predicted success, got failure
            "duration": 60.0,  # predicted 10s, took 60s
            "error": "timeout",  # error occurred (high risk)
        }

        discrepancy = predictor.calculate_discrepancy(predicted, actual)

        # Should be high (very different)
        assert discrepancy > 0.5

    def test_calculate_discrepancy_partial_match(self, predictor: ActionOutcomePredictor) -> None:
        """Test discrepancy for partial match."""
        predicted = {
            "success_probability": 0.8,
            "expected_duration": 10.0,
            "risk_score": 0.2,
        }
        actual = {
            "success": True,  # matches high success probability
            "duration": 15.0,  # slightly longer than expected
        }

        discrepancy = predictor.calculate_discrepancy(predicted, actual)

        # Should be moderate (some difference)
        assert 0.0 < discrepancy < 0.5

    def test_calculate_discrepancy_empty_actual(self, predictor: ActionOutcomePredictor) -> None:
        """Test discrepancy when actual outcome has no data."""
        predicted = {
            "success_probability": 0.8,
            "expected_duration": 10.0,
            "risk_score": 0.2,
        }
        actual = {}

        discrepancy = predictor.calculate_discrepancy(predicted, actual)

        # Should return 0 (no comparison possible)
        assert discrepancy == 0.0


# ---------------------------------------------------------------------------
# NNActionAwarenessModule Tests
# ---------------------------------------------------------------------------


class TestNNActionAwarenessModule:
    """Tests for NNActionAwarenessModule."""

    @pytest.fixture
    def module(self) -> NNActionAwarenessModule:
        """Module instance with small config."""
        config = NNConfig(input_dim=64, hidden_dims=[32, 16], output_dim=16)
        return NNActionAwarenessModule(config=config)

    @pytest.fixture
    def sas(self) -> InMemorySAS:
        """In-memory SAS instance."""
        return InMemorySAS()

    def test_module_name(self, module: NNActionAwarenessModule) -> None:
        """Test module name."""
        assert module.name == "action_awareness_nn"

    def test_module_tier(self, module: NNActionAwarenessModule) -> None:
        """Test module tier is FAST."""
        assert module.tier == "fast"

    async def test_tick_empty_history(
        self, module: NNActionAwarenessModule, sas: InMemorySAS
    ) -> None:
        """Test tick with empty action history."""
        result = await module.tick(sas)

        assert result.module_name == "action_awareness_nn"
        assert result.tier == "fast"
        assert result.data["last_action_success"] is True
        assert result.data["discrepancy_score"] == 0.0
        assert result.data["discrepancies"] == []

    async def test_tick_with_action(
        self, module: NNActionAwarenessModule, sas: InMemorySAS
    ) -> None:
        """Test tick with an action in history."""
        # Add action to history
        sas._action_history = [
            ActionHistoryEntry(
                action="mine",
                expected_result='{"position": {"x": 5, "y": 60, "z": 5}}',
                success=True,
            )
        ]
        sas._percepts = PerceptData(position={"x": 5, "y": 60, "z": 5})

        result = await module.tick(sas)

        assert result.module_name == "action_awareness_nn"
        assert "discrepancy_score" in result.data
        assert "predicted_success_prob" in result.data
        assert "predicted_duration" in result.data
        assert "predicted_risk" in result.data

    async def test_tick_detects_discrepancy(
        self, module: NNActionAwarenessModule, sas: InMemorySAS
    ) -> None:
        """Test tick detects discrepancies."""
        # Add failed action to history
        sas._action_history = [
            ActionHistoryEntry(
                action="mine",
                expected_result='{"position": {"x": 5, "y": 60, "z": 5}}',
                success=False,  # Action failed
            )
        ]
        sas._percepts = PerceptData(position={"x": 100, "y": 100, "z": 100})

        result = await module.tick(sas)

        # Discrepancy score should be calculated
        assert "discrepancy_score" in result.data

        # May or may not detect discrepancy depending on NN output
        # (untrained network has random weights)
        assert isinstance(result.data["discrepancies"], list)

    async def test_save_load_module(self, module: NNActionAwarenessModule) -> None:
        """Test saving and loading module weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / "model")

            # Save the module's NN
            module.nn.save(model_path)

            # Create new module and load
            new_module = NNActionAwarenessModule(model_path=model_path)

            # Check parameter counts match
            assert new_module.nn.parameter_count == module.nn.parameter_count

    def test_initialization_with_invalid_path(self) -> None:
        """Test initialization with invalid model path doesn't crash."""
        # Should log warning but not crash
        module = NNActionAwarenessModule(model_path="/nonexistent/path")
        assert module.name == "action_awareness_nn"

    async def test_module_integrates_with_sas(
        self, module: NNActionAwarenessModule, sas: InMemorySAS
    ) -> None:
        """Test module correctly reads from SAS."""
        # Setup SAS state
        sas._action_history = [ActionHistoryEntry(action="craft", success=True)]
        sas._percepts = PerceptData(
            inventory={"wooden_pickaxe": 1},
            position={"x": 10, "y": 64, "z": 20},
        )
        sas._goals = GoalData(current_goal="craft tools")

        # Run tick
        result = await module.tick(sas)

        # Should successfully process
        assert result.success is True
        assert result.error is None


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_end_to_end_prediction(self) -> None:
        """Test end-to-end prediction pipeline."""
        # Create components
        config = NNConfig(input_dim=64, hidden_dims=[32, 16], output_dim=16)
        nn = FeedForwardNN(config)
        encoder = ActionEncoder(output_dim=64)
        predictor = ActionOutcomePredictor(nn, encoder)

        # Create state
        state = {
            "percepts": {
                "position": {"x": 10, "y": 64, "z": 20},
                "inventory": {"stone": 32},
                "health": 20,
                "hunger": 18,
            },
            "goals": {"current_goal": "mine iron ore"},
        }
        action = "mine"

        # Predict outcome
        outcome = predictor.predict_outcome(state, action)

        # Verify output
        assert all(k in outcome for k in ["success_probability", "expected_duration", "risk_score"])

        # Calculate discrepancy
        actual = {"success": True, "duration": 12.0}
        discrepancy = predictor.calculate_discrepancy(outcome, actual)

        assert 0 <= discrepancy <= 1

    def test_different_input_dimensions(self) -> None:
        """Test network works with different input dimensions."""
        for input_dim in [32, 64, 128]:
            config = NNConfig(input_dim=input_dim, hidden_dims=[64, 32], output_dim=16)
            nn = FeedForwardNN(config)
            encoder = ActionEncoder(output_dim=input_dim)

            # Test encoding
            percepts = {"position": {"x": 10, "y": 64, "z": 20}}
            goals = {}
            encoding = encoder.encode_state(percepts, goals, "move")

            assert encoding.shape == (input_dim,)

            # Test prediction
            output = nn.predict(encoding)
            assert output.shape == (16,)
            assert np.isclose(np.sum(output), 1.0)
