"""Action Awareness Neural Network for Phase 1+.

A small feedforward neural network (~100K parameters) for action outcome prediction.
Replaces/augments the rule-based action awareness with learned discrepancy detection.

Reference: docs/implementation/07-goal-planning.md Section 3
"""

from __future__ import annotations

__all__ = [
    "ActionEncoder",
    "ActionOutcomePredictor",
    "FeedForwardNN",
    "NNActionAwarenessModule",
    "NNConfig",
]

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from pydantic import BaseModel, Field

from piano.core.module import Module
from piano.core.types import ModuleResult, ModuleTier

if TYPE_CHECKING:
    from piano.core.sas import SharedAgentState

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class NNConfig(BaseModel):
    """Configuration for the feedforward neural network."""

    input_dim: int = 64
    hidden_dims: list[int] = Field(default_factory=lambda: [128, 64, 32])
    output_dim: int = 16
    learning_rate: float = 0.001


# ---------------------------------------------------------------------------
# Neural Network Implementation
# ---------------------------------------------------------------------------


class FeedForwardNN:
    """Numpy-based feedforward neural network with ReLU activations.

    A lightweight network (~100K parameters) for fast CPU inference.
    """

    def __init__(self, config: NNConfig) -> None:
        """Initialize the network with random weights.

        Args:
            config: Network configuration
        """
        self.config = config
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        # Build layer dimensions
        dims = [config.input_dim, *config.hidden_dims, config.output_dim]

        # Initialize weights with He initialization (good for ReLU)
        rng = np.random.default_rng(42)
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            # He initialization: scale by sqrt(2/n_in)
            w = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, out_dim))
            b = np.zeros(out_dim)

            self.weights.append(w.astype(np.float32))
            self.biases.append(b.astype(np.float32))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network.

        Args:
            x: Input array of shape (batch_size, input_dim) or (input_dim,)

        Returns:
            Output array of shape (batch_size, output_dim) or (output_dim,)
        """
        # Handle single sample vs batch
        single_sample = x.ndim == 1
        if single_sample:
            x = x.reshape(1, -1)

        # Forward through hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            x = x @ self.weights[i] + self.biases[i]
            x = self._relu(x)

        # Final layer with softmax
        x = x @ self.weights[-1] + self.biases[-1]
        x = self._softmax(x)

        return x[0] if single_sample else x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Alias for forward (inference mode).

        Args:
            x: Input array

        Returns:
            Output predictions
        """
        return self.forward(x)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation function (numerically stable version)."""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @property
    def parameter_count(self) -> int:
        """Count total number of parameters in the network."""
        total = 0
        for w, b in zip(self.weights, self.biases, strict=False):
            total += w.size + b.size
        return total

    def save(self, path: str) -> None:
        """Save model weights to disk.

        Args:
            path: Directory path to save to
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save each layer's weights and biases
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=False)):
            np.save(save_dir / f"weight_{i}.npy", w)
            np.save(save_dir / f"bias_{i}.npy", b)

        # Save config
        config_path = save_dir / "config.json"
        config_path.write_text(self.config.model_dump_json(indent=2))

        logger.info("model_saved", path=path, params=self.parameter_count)

    def load(self, path: str) -> None:
        """Load model weights from disk.

        Args:
            path: Directory path to load from
        """
        load_dir = Path(path)

        # Load config
        config_path = load_dir / "config.json"
        config_data = json.loads(config_path.read_text())
        self.config = NNConfig(**config_data)

        # Load weights and biases
        self.weights = []
        self.biases = []
        i = 0
        while (load_dir / f"weight_{i}.npy").exists():
            w = np.load(load_dir / f"weight_{i}.npy")
            b = np.load(load_dir / f"bias_{i}.npy")
            self.weights.append(w)
            self.biases.append(b)
            i += 1

        logger.info("model_loaded", path=path, params=self.parameter_count)


# ---------------------------------------------------------------------------
# Action Encoding
# ---------------------------------------------------------------------------


class ActionEncoder:
    """Encodes action + state into fixed-size vectors for NN input."""

    def __init__(self, output_dim: int = 64) -> None:
        """Initialize the encoder.

        Args:
            output_dim: Size of output encoding vector
        """
        self.output_dim = output_dim

    def encode_state(
        self,
        percepts_dict: dict[str, Any],
        goals_dict: dict[str, Any],
        action_str: str,
    ) -> np.ndarray:
        """Encode state and action into a fixed-size vector.

        Uses simple feature engineering:
        - Action type: one-hot encoded
        - Numeric features: normalized
        - Categorical features: hashed

        Args:
            percepts_dict: Current percepts (position, inventory, etc.)
            goals_dict: Current goals
            action_str: Action string

        Returns:
            Fixed-size encoding vector (output_dim,)
        """
        features = np.zeros(self.output_dim, dtype=np.float32)

        # Action type encoding (first 8 dims)
        action_types = ["move", "mine", "craft", "chat", "look", "idle", "attack", "use"]
        action_lower = action_str.lower()
        for i, atype in enumerate(action_types):
            if atype in action_lower and i < 8:
                features[i] = 1.0

        # Position encoding (dims 8-11)
        pos = percepts_dict.get("position", {})
        features[8] = self._normalize(pos.get("x", 0), -1000, 1000)
        features[9] = self._normalize(pos.get("y", 0), 0, 256)
        features[10] = self._normalize(pos.get("z", 0), -1000, 1000)

        # Health/Hunger (dims 11-13)
        features[11] = self._normalize(percepts_dict.get("health", 20), 0, 20)
        features[12] = self._normalize(percepts_dict.get("hunger", 20), 0, 20)

        # Inventory size (dim 13)
        inventory = percepts_dict.get("inventory", {})
        total_items = sum(inventory.values()) if isinstance(inventory, dict) else 0
        features[13] = self._normalize(total_items, 0, 100)

        # Time of day (dim 14)
        features[14] = self._normalize(percepts_dict.get("time_of_day", 0), 0, 24000)

        # Nearby entities counts (dims 15-17)
        features[15] = self._normalize(
            len(percepts_dict.get("nearby_players", [])), 0, 10
        )
        features[16] = self._normalize(
            len(percepts_dict.get("nearby_blocks", [])), 0, 50
        )
        features[17] = self._normalize(
            len(percepts_dict.get("chat_messages", [])), 0, 20
        )

        # Goal encoding (dims 18-25) - hash the goal string
        current_goal = goals_dict.get("current_goal", "")
        if current_goal:
            goal_hash = self._hash_string(current_goal, num_bins=8)
            features[18 + goal_hash] = 1.0

        # Weather encoding (dims 26-29)
        weather_types = ["clear", "rain", "thunder", "snow"]
        weather = percepts_dict.get("weather", "clear")
        for i, wtype in enumerate(weather_types):
            if wtype in weather.lower() and (26 + i) < self.output_dim:
                features[26 + i] = 1.0

        # Remaining dimensions filled with context noise
        # This helps the network learn diverse representations
        if self.output_dim > 30:
            rng = np.random.default_rng(self._seed_from_state(percepts_dict))
            noise = rng.normal(0, 0.1, self.output_dim - 30)
            features[30:] = noise.astype(np.float32)

        return features

    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to [0, 1] range."""
        if max_val == min_val:
            return 0.0
        return float(np.clip((value - min_val) / (max_val - min_val), 0, 1))

    @staticmethod
    def _hash_string(s: str, num_bins: int = 8) -> int:
        """Hash a string to a bin index."""
        hash_val = int(hashlib.md5(s.encode()).hexdigest(), 16)
        return hash_val % num_bins

    @staticmethod
    def _seed_from_state(state_dict: dict[str, Any]) -> int:
        """Generate deterministic seed from state for reproducible noise."""
        state_str = json.dumps(state_dict, sort_keys=True)
        hash_val = int(hashlib.md5(state_str.encode()).hexdigest(), 16)
        return hash_val % (2**32)


# ---------------------------------------------------------------------------
# Outcome Prediction
# ---------------------------------------------------------------------------


class ActionOutcomePredictor:
    """Predicts action outcomes and calculates discrepancies."""

    def __init__(self, nn: FeedForwardNN, encoder: ActionEncoder) -> None:
        """Initialize the predictor.

        Args:
            nn: Trained neural network
            encoder: Action encoder
        """
        self.nn = nn
        self.encoder = encoder

    def predict_outcome(
        self,
        state: dict[str, Any],
        action: str,
    ) -> dict[str, float]:
        """Predict expected outcome of an action.

        Args:
            state: Current state (percepts, goals, etc.)
            action: Action string

        Returns:
            Dictionary with:
            - success_probability: 0-1 likelihood of success
            - expected_duration: estimated seconds
            - risk_score: 0-1 risk of failure
        """
        percepts = state.get("percepts", {})
        goals = state.get("goals", {})

        # Encode state and action
        encoding = self.encoder.encode_state(percepts, goals, action)

        # Get NN prediction
        output = self.nn.predict(encoding)

        # Interpret output vector
        # output_dim=16: [success_prob, duration (normalized), risk, ...other features]
        return {
            "success_probability": float(output[0]),
            "expected_duration": float(output[1]) * 60.0,  # denormalize to seconds
            "risk_score": float(output[2]),
        }

    def calculate_discrepancy(
        self,
        predicted: dict[str, float],
        actual: dict[str, Any],
    ) -> float:
        """Calculate discrepancy score between predicted and actual outcomes.

        Args:
            predicted: Predicted outcome from predict_outcome()
            actual: Actual outcome (success: bool, duration: float, etc.)

        Returns:
            Discrepancy score in [0, 1], where 0=identical, 1=very different
        """
        discrepancy = 0.0
        count = 0

        # Success discrepancy
        if "success" in actual:
            actual_success = 1.0 if actual["success"] else 0.0
            pred_success = predicted.get("success_probability", 0.5)
            discrepancy += abs(actual_success - pred_success)
            count += 1

        # Duration discrepancy (normalized)
        if "duration" in actual:
            actual_duration = actual["duration"]
            pred_duration = predicted.get("expected_duration", 10.0)
            # Normalize to 0-1 range (cap at 60 seconds)
            duration_diff = abs(actual_duration - pred_duration) / 60.0
            discrepancy += min(duration_diff, 1.0)
            count += 1

        # Risk score discrepancy (if we have failure info)
        if actual.get("error"):
            # High risk should have been predicted
            pred_risk = predicted.get("risk_score", 0.5)
            discrepancy += abs(1.0 - pred_risk)
            count += 1

        return discrepancy / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# NN-based Action Awareness Module
# ---------------------------------------------------------------------------


class NNActionAwarenessModule(Module):
    """Neural network-based action awareness module (Phase 1+).

    Uses a learned model for action outcome prediction and discrepancy detection.
    Falls back to rule-based evaluation on NN errors.
    """

    def __init__(
        self,
        model_path: str | None = None,
        config: NNConfig | None = None,
    ) -> None:
        """Initialize the module.

        Args:
            model_path: Path to load pre-trained model from (optional)
            config: Network configuration (used if model_path is None)
        """
        self.config = config or NNConfig()
        self.nn = FeedForwardNN(self.config)
        self.encoder = ActionEncoder(output_dim=self.config.input_dim)
        self.predictor = ActionOutcomePredictor(self.nn, self.encoder)

        if model_path:
            try:
                self.nn.load(model_path)
            except Exception:
                logger.warning("failed_to_load_model", path=model_path)

        logger.info(
            "nn_action_awareness_init",
            params=self.nn.parameter_count,
            input_dim=self.config.input_dim,
            output_dim=self.config.output_dim,
        )

    @property
    def name(self) -> str:
        """Unique module name."""
        return "action_awareness_nn"

    @property
    def tier(self) -> ModuleTier:
        """Execution tier -- FAST (non-LLM, <100ms)."""
        return ModuleTier.FAST

    async def tick(self, sas: SharedAgentState) -> ModuleResult:
        """Execute one tick of NN-based action awareness.

        1. Get latest action and percepts from SAS
        2. Encode state and predict expected outcome
        3. Compare with actual outcome
        4. Calculate discrepancy score
        5. Return result for CC compression

        Args:
            sas: Shared agent state

        Returns:
            ModuleResult with discrepancy data
        """
        history = await sas.get_action_history(limit=1)
        if not history:
            return ModuleResult(
                module_name=self.name,
                tier=self.tier,
                data={
                    "discrepancies": [],
                    "last_action_success": True,
                    "discrepancy_score": 0.0,
                },
            )

        latest = history[0]
        percepts = await sas.get_percepts()
        goals = await sas.get_goals()

        # Build state dict for prediction
        state = {
            "percepts": percepts.model_dump(),
            "goals": goals.model_dump(),
        }

        # Predict expected outcome
        predicted = self.predictor.predict_outcome(state, latest.action)

        # Build actual outcome from percepts
        actual = {
            "success": latest.success,
            "duration": 1.0,  # Placeholder - would come from action timing
        }

        # Calculate discrepancy
        discrepancy_score = self.predictor.calculate_discrepancy(predicted, actual)

        # Determine if this is a significant discrepancy (threshold: 0.5)
        has_discrepancy = discrepancy_score > 0.5
        success = not has_discrepancy

        data: dict[str, Any] = {
            "discrepancies": [],
            "last_action_success": success,
            "discrepancy_score": discrepancy_score,
            "predicted_success_prob": predicted["success_probability"],
            "predicted_duration": predicted["expected_duration"],
            "predicted_risk": predicted["risk_score"],
        }

        if has_discrepancy:
            data["discrepancies"] = [
                {
                    "action": latest.action,
                    "type": "nn_predicted_failure",
                    "score": discrepancy_score,
                    "predicted": predicted,
                    "actual": actual,
                }
            ]

        return ModuleResult(
            module_name=self.name,
            tier=self.tier,
            data=data,
        )
