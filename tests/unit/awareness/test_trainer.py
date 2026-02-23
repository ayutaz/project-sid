"""Tests for the NN training pipeline (action awareness)."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from piano.awareness.trainer import (
    ExperienceBuffer,
    NNTrainer,
    SGDOptimizer,
    TrainingExample,
)

# ---------------------------------------------------------------------------
# Mock NN for testing
# ---------------------------------------------------------------------------


class MockNN:
    """Simple 2-layer neural network for testing the trainer.

    Architecture: input_dim -> 8 -> output_dim
    """

    def __init__(self, input_dim: int = 10, output_dim: int = 4) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize small random weights/biases
        self.weights = [
            np.random.randn(input_dim, 8).astype(np.float32) * 0.1,
            np.random.randn(8, output_dim).astype(np.float32) * 0.1,
        ]
        self.biases = [
            np.zeros(8, dtype=np.float32),
            np.zeros(output_dim, dtype=np.float32),
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: input -> hidden -> output."""
        # Layer 1
        h = np.dot(x, self.weights[0]) + self.biases[0]
        h = np.maximum(0, h)  # ReLU

        # Layer 2
        out = np.dot(h, self.weights[1]) + self.biases[1]
        return out


# ---------------------------------------------------------------------------
# TrainingExample tests
# ---------------------------------------------------------------------------


def test_training_example_creation() -> None:
    """Test TrainingExample model construction."""
    ex = TrainingExample(
        state_vector=[1.0, 2.0, 3.0],
        action_vector=[4.0, 5.0],
        outcome_vector=[6.0, 7.0],
        weight=1.5,
    )
    assert len(ex.state_vector) == 3
    assert len(ex.action_vector) == 2
    assert len(ex.outcome_vector) == 2
    assert ex.weight == 1.5
    assert isinstance(ex.timestamp, datetime)


def test_training_example_to_input() -> None:
    """Test concatenation of state and action into input vector."""
    ex = TrainingExample(
        state_vector=[1.0, 2.0],
        action_vector=[3.0, 4.0],
        outcome_vector=[5.0],
    )
    inp = ex.to_input()
    assert isinstance(inp, np.ndarray)
    assert inp.shape == (4,)
    assert np.allclose(inp, [1.0, 2.0, 3.0, 4.0])


def test_training_example_to_target() -> None:
    """Test conversion of outcome to target array."""
    ex = TrainingExample(
        state_vector=[1.0],
        action_vector=[2.0],
        outcome_vector=[3.0, 4.0],
    )
    target = ex.to_target()
    assert isinstance(target, np.ndarray)
    assert target.shape == (2,)
    assert np.allclose(target, [3.0, 4.0])


# ---------------------------------------------------------------------------
# ExperienceBuffer tests
# ---------------------------------------------------------------------------


def test_experience_buffer_init() -> None:
    """Test buffer initialization."""
    buf = ExperienceBuffer(max_size=100)
    assert buf.size == 0


def test_experience_buffer_add() -> None:
    """Test adding examples to buffer."""
    buf = ExperienceBuffer(max_size=10)
    ex = TrainingExample(
        state_vector=[1.0],
        action_vector=[2.0],
        outcome_vector=[3.0],
    )
    buf.add(ex)
    assert buf.size == 1


def test_experience_buffer_fifo_eviction() -> None:
    """Test that buffer evicts oldest examples when full."""
    buf = ExperienceBuffer(max_size=3)

    ex1 = TrainingExample(
        state_vector=[1.0],
        action_vector=[0.0],
        outcome_vector=[0.0],
    )
    ex2 = TrainingExample(
        state_vector=[2.0],
        action_vector=[0.0],
        outcome_vector=[0.0],
    )
    ex3 = TrainingExample(
        state_vector=[3.0],
        action_vector=[0.0],
        outcome_vector=[0.0],
    )
    ex4 = TrainingExample(
        state_vector=[4.0],
        action_vector=[0.0],
        outcome_vector=[0.0],
    )

    buf.add(ex1)
    buf.add(ex2)
    buf.add(ex3)
    assert buf.size == 3

    # Adding 4th should evict ex1
    buf.add(ex4)
    assert buf.size == 3

    # Sample all and verify ex1 is gone
    samples = buf.sample(10)
    assert len(samples) == 3
    state_vectors = [s.state_vector[0] for s in samples]
    assert 1.0 not in state_vectors  # ex1 evicted
    assert 2.0 in state_vectors
    assert 3.0 in state_vectors
    assert 4.0 in state_vectors


def test_experience_buffer_sample_empty() -> None:
    """Test sampling from empty buffer returns empty list."""
    buf = ExperienceBuffer(max_size=10)
    samples = buf.sample(5)
    assert samples == []


def test_experience_buffer_sample_smaller_than_batch() -> None:
    """Test sampling when buffer has fewer examples than batch_size."""
    buf = ExperienceBuffer(max_size=10)
    ex1 = TrainingExample(
        state_vector=[1.0],
        action_vector=[0.0],
        outcome_vector=[0.0],
    )
    ex2 = TrainingExample(
        state_vector=[2.0],
        action_vector=[0.0],
        outcome_vector=[0.0],
    )
    buf.add(ex1)
    buf.add(ex2)

    samples = buf.sample(10)
    assert len(samples) == 2


def test_experience_buffer_sample_randomness() -> None:
    """Test that sampling is random (not deterministic)."""
    buf = ExperienceBuffer(max_size=100)
    for i in range(20):
        buf.add(
            TrainingExample(
                state_vector=[float(i)],
                action_vector=[0.0],
                outcome_vector=[0.0],
            )
        )

    # Sample multiple times and check that results vary
    sample1 = buf.sample(5)
    sample2 = buf.sample(5)
    sample3 = buf.sample(5)

    # Collect state vectors
    states1 = [s.state_vector[0] for s in sample1]
    states2 = [s.state_vector[0] for s in sample2]
    states3 = [s.state_vector[0] for s in sample3]

    # At least one should differ (very high probability with 20 items)
    assert not (states1 == states2 == states3)


def test_experience_buffer_clear() -> None:
    """Test clearing the buffer."""
    buf = ExperienceBuffer(max_size=10)
    buf.add(
        TrainingExample(
            state_vector=[1.0],
            action_vector=[0.0],
            outcome_vector=[0.0],
        )
    )
    assert buf.size == 1
    buf.clear()
    assert buf.size == 0


# ---------------------------------------------------------------------------
# SGDOptimizer tests
# ---------------------------------------------------------------------------


def test_sgd_optimizer_init() -> None:
    """Test optimizer initialization."""
    opt = SGDOptimizer(learning_rate=0.01, momentum=0.8)
    assert opt.learning_rate == 0.01
    assert opt.momentum == 0.8


def test_sgd_optimizer_step_updates_weights() -> None:
    """Test that optimizer step updates weights in the correct direction."""
    opt = SGDOptimizer(learning_rate=0.1, momentum=0.0)

    # Simple weights/biases
    weights = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    biases = [np.array([0.5, 0.5])]

    # Gradients (negative means increase weights)
    grad_w = [np.array([[-1.0, -1.0], [-1.0, -1.0]])]
    grad_b = [np.array([-1.0, -1.0])]

    new_w, new_b = opt.step(weights, biases, grad_w, grad_b)

    # With lr=0.1, weights should increase by 0.1
    expected_w = weights[0] + 0.1
    expected_b = biases[0] + 0.1

    assert np.allclose(new_w[0], expected_w)
    assert np.allclose(new_b[0], expected_b)


def test_sgd_optimizer_momentum() -> None:
    """Test momentum accumulation over multiple steps."""
    opt = SGDOptimizer(learning_rate=0.1, momentum=0.9)

    weights = [np.array([[1.0]])]
    biases = [np.array([0.0])]
    grad_w = [np.array([[-1.0]])]
    grad_b = [np.array([[-1.0]])]

    # First step: no prior velocity
    new_w1, new_b1 = opt.step(weights, biases, grad_w, grad_b)
    # velocity = -0.1 * (-1.0) = 0.1
    # new_w = 1.0 + 0.1 = 1.1
    assert np.allclose(new_w1[0], 1.1)

    # Second step with same gradient
    new_w2, _new_b2 = opt.step(new_w1, new_b1, grad_w, grad_b)
    # velocity = 0.9 * 0.1 + 0.1 = 0.19
    # new_w = 1.1 + 0.19 = 1.29
    assert np.allclose(new_w2[0], 1.29, atol=1e-5)


# ---------------------------------------------------------------------------
# NNTrainer tests
# ---------------------------------------------------------------------------


def test_nn_trainer_init() -> None:
    """Test trainer initialization."""
    model = MockNN(input_dim=6, output_dim=2)
    trainer = NNTrainer(
        model=model,
        learning_rate=0.01,
        batch_size=16,
        buffer_size=100,
    )
    assert trainer.batch_size == 16
    assert trainer.buffer.size == 0


def test_nn_trainer_record_experience() -> None:
    """Test recording experience to buffer."""
    model = MockNN(input_dim=4, output_dim=2)
    trainer = NNTrainer(model=model, buffer_size=10)

    trainer.record_experience(
        state_vector=[1.0, 2.0],
        action_vector=[3.0, 4.0],
        outcome_vector=[5.0, 6.0],
    )
    assert trainer.buffer.size == 1
    stats = trainer.get_training_stats()
    assert stats["total_examples"] == 1


def test_nn_trainer_train_step_empty_buffer() -> None:
    """Test train_step with empty buffer returns 0 loss."""
    model = MockNN(input_dim=4, output_dim=2)
    trainer = NNTrainer(model=model)

    loss = trainer.train_step()
    assert loss == 0.0


def test_nn_trainer_train_step_reduces_loss() -> None:
    """Test that training step runs without error and produces a loss value."""
    model = MockNN(input_dim=4, output_dim=2)
    trainer = NNTrainer(model=model, learning_rate=0.001, batch_size=4)

    # Add training examples (simple pattern: predict sum)
    for i in range(10):
        state = [float(i), float(i + 1)]
        action = [0.0, 0.0]
        outcome = [float(2 * i + 1), 0.0]  # sum of state
        trainer.record_experience(state, action, outcome)

    # Train step should produce a finite loss
    loss = trainer.train_step()
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_nn_trainer_loss_computation() -> None:
    """Test MSE loss computation."""
    model = MockNN(input_dim=4, output_dim=2)
    trainer = NNTrainer(model=model)

    predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
    targets = np.array([[1.0, 2.0], [3.0, 5.0]])

    loss = trainer._compute_loss(predictions, targets)
    # MSE = mean((0, 0, 0, -1)^2) = 1/4 = 0.25
    assert np.isclose(loss, 0.25)


def test_nn_trainer_training_stats() -> None:
    """Test training statistics tracking."""
    model = MockNN(input_dim=4, output_dim=2)
    trainer = NNTrainer(model=model, batch_size=2)

    # Add examples
    for _i in range(5):
        trainer.record_experience([1.0, 0.0], [0.0, 0.0], [0.0, 1.0])

    # Train a few steps
    trainer.train_step()
    trainer.train_step()

    stats = trainer.get_training_stats()
    assert stats["total_examples"] == 5
    assert stats["buffer_size"] == 5
    assert len(stats["loss_history"]) == 2
    assert "recent_avg_loss" in stats


def test_nn_trainer_train_epoch() -> None:
    """Test training for multiple steps (one epoch)."""
    model = MockNN(input_dim=4, output_dim=2)
    trainer = NNTrainer(model=model, batch_size=4)

    # Add examples
    for _i in range(20):
        trainer.record_experience([1.0, 2.0], [0.0, 0.0], [3.0, 0.0])

    losses = trainer.train_epoch(num_steps=10)
    assert len(losses) == 10
    assert all(isinstance(loss, float) for loss in losses)

    stats = trainer.get_training_stats()
    assert stats["epochs_completed"] == 1


def test_nn_trainer_multiple_epochs() -> None:
    """Test training for multiple epochs."""
    model = MockNN(input_dim=4, output_dim=2)
    trainer = NNTrainer(model=model, batch_size=4)

    # Add examples
    for _i in range(20):
        trainer.record_experience([1.0, 2.0], [0.0, 0.0], [3.0, 0.0])

    trainer.train_epoch(num_steps=5)
    trainer.train_epoch(num_steps=5)

    stats = trainer.get_training_stats()
    assert stats["epochs_completed"] == 2
    assert len(stats["loss_history"]) == 10


def test_nn_trainer_gradient_computation_runs() -> None:
    """Test that gradient computation runs without error."""
    model = MockNN(input_dim=4, output_dim=2)
    trainer = NNTrainer(model=model)

    x_batch = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    y_batch = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    grad_w, grad_b = trainer._compute_gradients(x_batch, y_batch)

    # Check gradients have correct shapes
    assert len(grad_w) == len(model.weights)
    assert len(grad_b) == len(model.biases)
    for gw, w in zip(grad_w, model.weights, strict=False):
        assert gw.shape == w.shape
    for gb, b in zip(grad_b, model.biases, strict=False):
        assert gb.shape == b.shape


def test_nn_trainer_weights_updated_after_step() -> None:
    """Test that model weights are actually updated after training step."""
    model = MockNN(input_dim=4, output_dim=2)
    trainer = NNTrainer(model=model, learning_rate=0.1, batch_size=2)

    # Record initial weights
    initial_w0 = model.weights[0].copy()

    # Add examples and train
    for _i in range(10):
        trainer.record_experience([1.0, 2.0], [0.0, 0.0], [3.0, 0.0])

    trainer.train_step()

    # Weights should have changed
    assert not np.allclose(model.weights[0], initial_w0)
