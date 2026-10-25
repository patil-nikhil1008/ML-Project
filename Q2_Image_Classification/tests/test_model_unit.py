"""
Unit Tests for Pure NumPy Neural Network Layers and Functions
"""

import os
import sys
import pytest
import numpy as np

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from neural_network import (
    DenseLayer,
    BatchNormLayer,
    ReLULayer,
    DropoutLayer,
    SoftmaxActivation,
    AdamOptimizer,
    NeuralNetwork
)


def test_dense_layer_forward_and_backward():
    layer = DenseLayer(in_features=10, out_features=5, init_method='he', seed=42)
    assert layer.W.shape == (10, 5)
    assert layer.b.shape == (1, 5)

    X = np.random.randn(8, 10).astype(np.float32)
    out = layer.forward(X)
    assert out.shape == (8, 5)

    dOut = np.ones((8, 5), dtype=np.float32)
    dX = layer.backward(dOut, l2_reg=1e-4)
    assert dX.shape == (8, 10)
    assert layer.dW.shape == (10, 5)
    assert layer.db.shape == (1, 5)
    assert not np.isnan(dX).any()


def test_batch_norm_train_vs_eval():
    bn = BatchNormLayer(num_features=6, momentum=0.9)
    X_train = np.random.randn(16, 6).astype(np.float32) * 5.0 + 3.0

    # Train mode
    out_train = bn.forward(X_train, mode='train')
    assert out_train.shape == (16, 6)
    # Output should have mean ~0 and variance ~1 per feature
    assert np.allclose(np.mean(out_train, axis=0), 0.0, atol=1e-3)

    # Eval mode using updated running stats
    X_eval = np.random.randn(4, 6).astype(np.float32)
    out_eval = bn.forward(X_eval, mode='eval')
    assert out_eval.shape == (4, 6)

    # Backward pass
    dOut = np.ones((16, 6), dtype=np.float32)
    dX = bn.backward(dOut)
    assert dX.shape == (16, 6)
    assert not np.isnan(dX).any()


def test_relu_layer():
    relu = ReLULayer()
    X = np.array([[-2.0, 0.0, 3.5], [1.0, -0.5, 4.0]], dtype=np.float32)
    out = relu.forward(X)
    expected = np.array([[0.0, 0.0, 3.5], [1.0, 0.0, 4.0]], dtype=np.float32)
    assert np.allclose(out, expected)

    dOut = np.ones_like(X)
    dX = relu.backward(dOut)
    expected_grad = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    assert np.allclose(dX, expected_grad)


def test_dropout_inverted_scaling():
    np.random.seed(42)
    drop = DropoutLayer(p=0.4)
    X = np.ones((100, 100), dtype=np.float32)

    out_train = drop.forward(X, mode='train')
    # Mean should stay close to 1.0 because of inverted dropout scaling 1/(1-p)
    assert 0.90 <= np.mean(out_train) <= 1.10

    out_eval = drop.forward(X, mode='eval')
    # Eval mode must be identical to input
    assert np.allclose(out_eval, X)


def test_softmax_numerical_stability():
    # Extremely large logits that would overflow standard exp()
    extreme_logits = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    probs = SoftmaxActivation.forward(extreme_logits)
    assert not np.isnan(probs).any()
    assert not np.isinf(probs).any()
    assert np.allclose(np.sum(probs, axis=-1), 1.0)
    # The largest logit should have highest probability
    assert probs[0, 2] > probs[0, 1] > probs[0, 0]


def test_adam_optimizer_step():
    params = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    opt = AdamOptimizer(params=params, lr=0.01)
    grads = [np.array([0.1, -0.2, 0.5], dtype=np.float32)]

    initial_val = params[0].copy()
    opt.step(grads)

    # Weights must update in opposite direction of gradient
    assert params[0][0] < initial_val[0]  # pos grad -> decrease
    assert params[0][1] > initial_val[1]  # neg grad -> increase
    assert params[0][2] < initial_val[2]  # pos grad -> decrease


def test_neural_network_end_to_end_forward():
    net = NeuralNetwork(input_dim=3072, hidden_dims=[64, 32, 16], num_classes=3, seed=42)
    X = np.random.rand(5, 3072).astype(np.float32)
    probs = net.forward(X, mode='eval')
    assert probs.shape == (5, 3)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
