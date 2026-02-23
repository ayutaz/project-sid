"""Training pipeline for action awareness neural network.

Provides online and batch training for the NN-based action awareness model.
This training infrastructure collects experiences (state, action, outcome) and
incrementally updates the NN weights during simulation.

Reference: docs/implementation/07-goal-planning.md Section 3.6
"""

from __future__ import annotations

__all__ = [
    "ExperienceBuffer",
    "NNProtocol",
    "NNTrainer",
    "SGDOptimizer",
    "TrainingExample",
]

import random
from collections import deque
from datetime import datetime
from typing import Protocol

import numpy as np
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


# --- NN Protocol ---


class NNProtocol(Protocol):
    """Protocol for NN models compatible with the trainer."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        ...

    @property
    def weights(self) -> list[np.ndarray]:
        """Network weights."""
        ...

    @property
    def biases(self) -> list[np.ndarray]:
        """Network biases."""
        ...


# --- Training Example ---


class TrainingExample(BaseModel):
    """A single training example for the action awareness NN.

    Captures the state, action, and observed outcome for supervised learning.
    """

    state_vector: list[float] = Field(description="Encoded agent state")
    action_vector: list[float] = Field(description="Encoded action representation")
    outcome_vector: list[float] = Field(description="Encoded actual outcome")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    weight: float = Field(default=1.0, ge=0.0, description="Sample weight for training")

    def to_input(self) -> np.ndarray:
        """Concatenate state and action vectors into a single input vector."""
        return np.concatenate([self.state_vector, self.action_vector], dtype=np.float32)

    def to_target(self) -> np.ndarray:
        """Convert outcome vector to target for training."""
        return np.array(self.outcome_vector, dtype=np.float32)


# --- Experience Buffer ---


class ExperienceBuffer:
    """FIFO buffer for experience replay with random sampling.

    Maintains a fixed-size buffer of training examples, evicting oldest
    when capacity is reached. Supports random batch sampling for training.
    """

    def __init__(self, max_size: int = 10000) -> None:
        """Initialize experience buffer.

        Args:
            max_size: Maximum number of examples to store (FIFO eviction)
        """
        self._buffer: deque[TrainingExample] = deque(maxlen=max_size)
        self._max_size = max_size

    def add(self, example: TrainingExample) -> None:
        """Add a training example to the buffer.

        If buffer is at capacity, oldest example is evicted (FIFO).

        Args:
            example: Training example to add
        """
        self._buffer.append(example)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        """Randomly sample a batch of examples.

        Args:
            batch_size: Number of examples to sample

        Returns:
            List of randomly sampled training examples (may be smaller than
            batch_size if buffer doesn't contain enough examples)
        """
        actual_size = min(batch_size, len(self._buffer))
        if actual_size == 0:
            return []
        return random.sample(list(self._buffer), actual_size)

    @property
    def size(self) -> int:
        """Current number of examples in the buffer."""
        return len(self._buffer)

    def clear(self) -> None:
        """Remove all examples from the buffer."""
        self._buffer.clear()


# --- SGD Optimizer ---


class SGDOptimizer:
    """Simple SGD optimizer with momentum for NN training."""

    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.9) -> None:
        """Initialize SGD optimizer.

        Args:
            learning_rate: Step size for weight updates
            momentum: Momentum coefficient (0.0 = no momentum)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self._velocity_w: list[np.ndarray] = []
        self._velocity_b: list[np.ndarray] = []

    def step(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        grad_weights: list[np.ndarray],
        grad_biases: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Perform one optimization step.

        Updates weights and biases using momentum-based SGD.

        Args:
            weights: Current network weights
            biases: Current network biases
            grad_weights: Gradients w.r.t. weights
            grad_biases: Gradients w.r.t. biases

        Returns:
            Tuple of (updated_weights, updated_biases)
        """
        # Initialize velocity on first call
        if not self._velocity_w:
            self._velocity_w = [np.zeros_like(w) for w in weights]
            self._velocity_b = [np.zeros_like(b) for b in biases]

        new_weights = []
        new_biases = []

        for i, (w, b, gw, gb) in enumerate(
            zip(weights, biases, grad_weights, grad_biases, strict=False)
        ):
            # Update velocity with momentum
            self._velocity_w[i] = self.momentum * self._velocity_w[i] - self.learning_rate * gw
            self._velocity_b[i] = self.momentum * self._velocity_b[i] - self.learning_rate * gb

            # Apply velocity to weights/biases
            new_weights.append(w + self._velocity_w[i])
            new_biases.append(b + self._velocity_b[i])

        return new_weights, new_biases


# --- NN Trainer ---


class NNTrainer:
    """Training pipeline for action awareness neural network.

    Manages experience collection, batch sampling, gradient computation,
    and weight updates for the NN model.
    """

    def __init__(
        self,
        model: NNProtocol,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        buffer_size: int = 10000,
        momentum: float = 0.9,
    ) -> None:
        """Initialize NN trainer.

        Args:
            model: NN model to train (must implement NNProtocol)
            learning_rate: Learning rate for SGD optimizer
            batch_size: Number of examples per training batch
            buffer_size: Maximum size of experience replay buffer
            momentum: Momentum coefficient for SGD
        """
        self.model = model
        self.batch_size = batch_size
        self.buffer = ExperienceBuffer(max_size=buffer_size)
        self.optimizer = SGDOptimizer(learning_rate=learning_rate, momentum=momentum)
        self._loss_history: list[float] = []
        self._total_examples: int = 0
        self._epochs_completed: int = 0
        self._epsilon = 1e-6  # For numerical gradient computation

    def record_experience(
        self,
        state_vector: list[float],
        action_vector: list[float],
        outcome_vector: list[float],
        weight: float = 1.0,
    ) -> None:
        """Record a new training example from agent experience.

        Args:
            state_vector: Encoded agent state
            action_vector: Encoded action representation
            outcome_vector: Actual outcome observed
            weight: Sample weight (default 1.0)
        """
        example = TrainingExample(
            state_vector=state_vector,
            action_vector=action_vector,
            outcome_vector=outcome_vector,
            weight=weight,
        )
        self.buffer.add(example)
        self._total_examples += 1
        logger.debug(
            "experience_recorded",
            buffer_size=self.buffer.size,
            total_examples=self._total_examples,
        )

    def train_step(self) -> float:
        """Perform one training step.

        Samples a batch, computes gradients, updates weights, and returns loss.

        Returns:
            MSE loss on the batch (0.0 if buffer is empty)
        """
        # Sample batch
        batch = self.buffer.sample(self.batch_size)
        if not batch:
            logger.warning("train_step_empty_buffer", buffer_size=self.buffer.size)
            return 0.0

        # Prepare input/target arrays
        x_batch = np.array([ex.to_input() for ex in batch], dtype=np.float32)
        y_batch = np.array([ex.to_target() for ex in batch], dtype=np.float32)

        # Forward pass to compute loss
        predictions = np.array([self.model.forward(x) for x in x_batch])
        loss = self._compute_loss(predictions, y_batch)

        # Compute gradients
        grad_weights, grad_biases = self._compute_gradients(x_batch, y_batch)

        # Update weights
        new_weights, new_biases = self.optimizer.step(
            list(self.model.weights),
            list(self.model.biases),
            grad_weights,
            grad_biases,
        )

        # Apply updated weights back to model (via mutation)
        for i, (w, b) in enumerate(zip(new_weights, new_biases, strict=False)):
            self.model.weights[i][:] = w
            self.model.biases[i][:] = b

        self._loss_history.append(loss)
        logger.debug("train_step_completed", loss=loss, batch_size=len(batch))
        return loss

    def train_epoch(self, num_steps: int = 100) -> list[float]:
        """Train for multiple steps (one epoch).

        Args:
            num_steps: Number of training steps to perform

        Returns:
            List of loss values for each step
        """
        losses = []
        for _ in range(num_steps):
            loss = self.train_step()
            losses.append(loss)

        self._epochs_completed += 1
        logger.info(
            "epoch_completed",
            epochs=self._epochs_completed,
            steps=num_steps,
            avg_loss=np.mean(losses) if losses else 0.0,
        )
        return losses

    def get_training_stats(self) -> dict[str, int | float | list[float]]:
        """Get current training statistics.

        Returns:
            Dictionary with total_examples, loss_history, epochs_completed,
            buffer_size, and recent_avg_loss (last 10 steps)
        """
        recent_losses = self._loss_history[-10:] if self._loss_history else []
        return {
            "total_examples": self._total_examples,
            "loss_history": self._loss_history.copy(),
            "epochs_completed": self._epochs_completed,
            "buffer_size": self.buffer.size,
            "recent_avg_loss": float(np.mean(recent_losses)) if recent_losses else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute mean squared error loss.

        Args:
            predictions: Model predictions (shape: [batch_size, output_dim])
            targets: Target values (shape: [batch_size, output_dim])

        Returns:
            MSE loss value
        """
        mse = np.mean((predictions - targets) ** 2)
        return float(mse)

    def _compute_gradients(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute gradients using numerical approximation.

        Uses finite differences to approximate gradients w.r.t. weights/biases.

        Args:
            x_batch: Input batch (shape: [batch_size, input_dim])
            y_batch: Target batch (shape: [batch_size, output_dim])

        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        grad_weights = []
        grad_biases = []

        # Numerical gradient for weights
        for _i, w in enumerate(self.model.weights):
            grad_w = np.zeros_like(w)
            for idx in np.ndindex(w.shape):
                # Perturb weight
                original = w[idx]
                w[idx] = original + self._epsilon
                loss_plus = self._batch_loss(x_batch, y_batch)

                w[idx] = original - self._epsilon
                loss_minus = self._batch_loss(x_batch, y_batch)

                # Restore original
                w[idx] = original

                # Finite difference
                grad_w[idx] = (loss_plus - loss_minus) / (2 * self._epsilon)

            grad_weights.append(grad_w)

        # Numerical gradient for biases
        for _i, b in enumerate(self.model.biases):
            grad_b = np.zeros_like(b)
            for idx in np.ndindex(b.shape):
                # Perturb bias
                original = b[idx]
                b[idx] = original + self._epsilon
                loss_plus = self._batch_loss(x_batch, y_batch)

                b[idx] = original - self._epsilon
                loss_minus = self._batch_loss(x_batch, y_batch)

                # Restore original
                b[idx] = original

                # Finite difference
                grad_b[idx] = (loss_plus - loss_minus) / (2 * self._epsilon)

            grad_biases.append(grad_b)

        return grad_weights, grad_biases

    def _batch_loss(self, x_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Compute loss for a batch (helper for gradient computation).

        Args:
            x_batch: Input batch
            y_batch: Target batch

        Returns:
            MSE loss
        """
        predictions = np.array([self.model.forward(x) for x in x_batch])
        return self._compute_loss(predictions, y_batch)
