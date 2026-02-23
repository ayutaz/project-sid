"""Demonstration of the Action Awareness Neural Network.

This script shows how to:
1. Create and configure the neural network
2. Encode state and actions
3. Predict action outcomes
4. Calculate discrepancies
5. Save and load trained models
"""

from __future__ import annotations

import numpy as np

from piano.awareness.nn_model import (
    ActionEncoder,
    ActionOutcomePredictor,
    FeedForwardNN,
    NNConfig,
)


def main() -> None:
    """Run the demonstration."""
    print("=" * 70)
    print("Action Awareness Neural Network Demonstration")
    print("=" * 70)
    print()

    # Step 1: Create the neural network
    print("1. Creating neural network...")
    config = NNConfig(
        input_dim=64,
        hidden_dims=[128, 64, 32],
        output_dim=16,
        learning_rate=0.001,
    )
    nn = FeedForwardNN(config)
    print(f"   Network created with {nn.parameter_count:,} parameters")
    print(f"   Architecture: {config.input_dim} -> {' -> '.join(map(str, config.hidden_dims))} -> {config.output_dim}")
    print()

    # Step 2: Create the encoder
    print("2. Creating action encoder...")
    encoder = ActionEncoder(output_dim=config.input_dim)
    print(f"   Encoder output dimension: {encoder.output_dim}")
    print()

    # Step 3: Encode some example states
    print("3. Encoding example states...")

    state_mining = {
        "position": {"x": 10, "y": 64, "z": 20},
        "inventory": {"wooden_pickaxe": 1, "coal": 16},
        "health": 20,
        "hunger": 18,
        "nearby_blocks": [{"type": "stone", "position": {"x": 11, "y": 64, "z": 20}}],
    }
    goals_mining = {"current_goal": "mine iron ore"}

    encoding_mining = encoder.encode_state(state_mining, goals_mining, "mine")
    print(f"   Mining action encoding shape: {encoding_mining.shape}")
    print(f"   First 10 values: {encoding_mining[:10]}")
    print()

    # Step 4: Make predictions
    print("4. Predicting action outcomes...")
    predictor = ActionOutcomePredictor(nn, encoder)

    state_dict = {
        "percepts": state_mining,
        "goals": goals_mining,
    }

    outcome = predictor.predict_outcome(state_dict, "mine")
    print(f"   Predicted success probability: {outcome['success_probability']:.3f}")
    print(f"   Predicted duration: {outcome['expected_duration']:.2f} seconds")
    print(f"   Predicted risk score: {outcome['risk_score']:.3f}")
    print()

    # Step 5: Calculate discrepancies
    print("5. Calculating discrepancies...")

    # Scenario 1: Successful action
    actual_success = {
        "success": True,
        "duration": 5.0,
    }
    discrepancy_success = predictor.calculate_discrepancy(outcome, actual_success)
    print(f"   Successful action discrepancy: {discrepancy_success:.3f}")

    # Scenario 2: Failed action
    actual_failure = {
        "success": False,
        "duration": 15.0,
        "error": "tool_broke",
    }
    discrepancy_failure = predictor.calculate_discrepancy(outcome, actual_failure)
    print(f"   Failed action discrepancy: {discrepancy_failure:.3f}")
    print()

    # Step 6: Test different action types
    print("6. Testing different action types...")
    action_types = ["move", "mine", "craft", "chat"]

    for action in action_types:
        encoding = encoder.encode_state(state_mining, goals_mining, action)
        pred = nn.predict(encoding)
        print(f"   {action:8s}: success_prob={pred[0]:.3f}, duration={pred[1]*60:.1f}s, risk={pred[2]:.3f}")
    print()

    # Step 7: Save and load the model
    print("7. Testing model persistence...")
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir) / "model")

        # Save
        nn.save(model_path)
        print(f"   Model saved to {model_path}")

        # Load
        loaded_nn = FeedForwardNN(config)
        loaded_nn.load(model_path)
        print(f"   Model loaded successfully")

        # Verify identical predictions
        original_pred = nn.predict(encoding_mining)
        loaded_pred = loaded_nn.predict(encoding_mining)

        if np.allclose(original_pred, loaded_pred):
            print("   [OK] Loaded model produces identical predictions")
        else:
            print("   [WARN] Warning: Predictions differ!")
    print()

    # Step 8: Demonstrate batch processing
    print("8. Batch processing demonstration...")
    batch_size = 10
    batch_encodings = np.array([
        encoder.encode_state(state_mining, goals_mining, action_types[i % len(action_types)])
        for i in range(batch_size)
    ])

    batch_predictions = nn.predict(batch_encodings)
    print(f"   Processed batch of {batch_size} samples")
    print(f"   Output shape: {batch_predictions.shape}")
    print(f"   Average success probability: {batch_predictions[:, 0].mean():.3f}")
    print()

    print("=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
