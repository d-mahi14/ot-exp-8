"""
test_pqc_qnn.py
Independent unit tests for Parameterized Quantum Circuits (PQCs),
Quantum Neural Networks (QNNs), and the Classical MLP baseline.
"""

import numpy as np
import pytest

from track_C_quantum.src.pqc_qnn import (
    generate_concentric_circles_dataset,
    ZZFeatureMap,
    QuantumClassifierPQC,
    ClassicalMLPClassifier,
    explain_pqc_and_qnn,
)


class TestQNNComponents:
    """Tests for dataset generation, feature maps, and circuit construction."""

    def test_concentric_circles_generation(self):
        X, y = generate_concentric_circles_dataset(n_samples=60, noise=0.03, seed=42)
        assert X.shape == (60, 2)
        assert len(y) == 60
        assert set(y).issubset({0.0, 1.0})
        assert np.sum(y == 0.0) == 30
        assert np.sum(y == 1.0) == 30

    def test_zz_feature_map_structure(self):
        fmap = ZZFeatureMap(num_qubits=2)
        qc = fmap.build_circuit()
        assert qc.num_qubits == 2
        ops = qc.count_ops()
        assert ops.get("h", 0) == 2
        assert ops.get("rz", 0) == 2
        assert ops.get("rzz", 0) == 1


class TestQuantumClassifier:
    """Tests for QNN predictions, gradient computation, and training."""

    def test_qnn_forward_and_probabilities(self):
        qnn = QuantumClassifierPQC(num_qubits=2, reps=1, seed=42)
        assert qnn.num_weights == 4  # 2 * (1 + 1) = 4

        X_sample = np.array([[0.5, 0.2], [-0.8, 0.6]])
        probs = qnn.predict_proba(X_sample)
        assert probs.shape == (2,)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

        preds = qnn.predict(X_sample)
        assert preds.shape == (2,)
        assert set(preds).issubset({0, 1})

    def test_qnn_parameter_shift_gradient(self):
        qnn = QuantumClassifierPQC(num_qubits=2, reps=1, seed=42)
        X_sample = np.array([[0.3, -0.4], [0.7, 0.1]])
        y_sample = np.array([0.0, 1.0])

        grad = qnn.compute_gradient(X_sample, y_sample)
        assert grad.shape == (qnn.num_weights,)
        assert np.all(np.isfinite(grad))

    def test_qnn_training_reduces_loss(self):
        X, y = generate_concentric_circles_dataset(n_samples=20, noise=0.02, seed=42)
        qnn = QuantumClassifierPQC(num_qubits=2, reps=1, lr=0.3, seed=42)

        initial_loss = qnn.compute_loss(X, y)
        res = qnn.fit(X, y, epochs=10)
        final_loss = res["final_loss"]

        assert final_loss < initial_loss
        assert len(res["loss_history"]) == 10


class TestClassicalMLPBaseline:
    """Tests for the Classical MLP baseline."""

    def test_mlp_forward_and_training(self):
        mlp = ClassicalMLPClassifier(in_dim=2, hidden_dim=6, lr=0.5, seed=42)
        assert mlp.num_parameters == 25

        X, y = generate_concentric_circles_dataset(n_samples=50, noise=0.03, seed=42)
        initial_loss = mlp.compute_loss(X, y)

        res = mlp.fit(X, y, epochs=100)
        assert res["final_loss"] < initial_loss
        assert res["final_accuracy"] >= 0.80

    def test_qnn_explanation_text(self):
        text = explain_pqc_and_qnn()
        assert "Quantum Feature Embedding" in text
        assert "Parameter-Shift Rule" in text
        assert "No Quantum Supremacy Claim" in text
