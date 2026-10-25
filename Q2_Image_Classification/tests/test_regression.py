"""
Regression Tests to Ensure Model Stability, Weights Integrity, and Invariance
"""

import os
import sys
import pytest
import numpy as np

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from app.service import model_service


def test_model_weights_file_integrity():
    """Verify saved weights file exists, is readable, and contains all expected layer arrays."""
    weights_path = os.path.join(PARENT_DIR, "best_model.npz")
    assert os.path.exists(weights_path), f"Weights file missing: {weights_path}"
    data = np.load(weights_path)
    expected_keys = [
        "W1", "b1", "gamma1", "beta1", "rmean1", "rvar1",
        "W2", "b2", "gamma2", "beta2", "rmean2", "rvar2",
        "W3", "b3", "gamma3", "beta3", "rmean3", "rvar3",
        "W4", "b4"
    ]
    for key in expected_keys:
        assert key in data, f"Key {key} missing from best_model.npz"
        assert not np.isnan(data[key]).any(), f"NaN detected in {key}"
        assert not np.isinf(data[key]).any(), f"Inf detected in {key}"


def test_deterministic_inference():
    """Verify inference is deterministic in eval mode across multiple calls on the same input."""
    # Synthetic fixed input
    rng = np.random.RandomState(1234)
    dummy_input = rng.rand(1, 3072).astype(np.float32)

    probs1 = model_service.model.predict_proba(dummy_input)
    probs2 = model_service.model.predict_proba(dummy_input)
    probs3 = model_service.model.predict_proba(dummy_input)

    assert np.allclose(probs1, probs2, atol=1e-7)
    assert np.allclose(probs2, probs3, atol=1e-7)


def test_model_parameter_count_regression():
    """Verify model parameter count has not accidentally drifted from 1,739,779."""
    trainable_params = sum(p.size for p in model_service.model.get_trainable_params())
    assert trainable_params == 1_739_779, f"Unexpected parameter count: {trainable_params}"


def test_output_bounds_regression():
    """Ensure all probabilities are strictly between 0 and 1 and sum to 1."""
    rng = np.random.RandomState(999)
    random_batch = rng.rand(50, 3072).astype(np.float32)
    probs = model_service.model.predict_proba(random_batch)

    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    sums = np.sum(probs, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5)
