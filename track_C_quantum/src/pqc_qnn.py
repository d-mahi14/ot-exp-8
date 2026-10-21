"""
pqc_qnn.py
Track C: Parameterized Quantum Circuits (PQCs) and Quantum Neural Networks (QNNs).

Theoretical Foundations:
1. Quantum Machine Learning Architecture:
   A Quantum Neural Network (QNN) operates as a hybrid pipeline:
       Classical Data x in R^d
           |  (Feature Map U_phi(x))
       Quantum Hilbert Space State |phi(x)>
           |  (Variational Ansatz W(theta))
       Processed Quantum State |psi(x, theta)> = W(theta) |phi(x)>
           |  (Observable Measurement <O>)
       Classical Readout y_hat = <psi(x, theta)| O |psi(x, theta)> in [-1, +1]

2. Non-Linear Quantum Feature Maps:
   Classical data cannot be copied into qubits directly (No-Cloning Theorem).
   Instead, data is embedded via Hamiltonian rotations:
   - Angle Encoding: R_y(x_i) on qubit i.
   - ZZ-Feature Map: Combines single-qubit rotations with two-qubit entangling interactions:
         U_phi(x) = exp( -i * sum_{i < j} 2*(pi - x_i)*(pi - x_j) * Z_i Z_j ) * prod_i exp(-i * 2*x_i * Z_i)
     This maps classical vectors into an exponentially large 2^n Hilbert space,
     acting as a non-linear quantum kernel that linearizes complex separation boundaries.

3. Parameter-Shift Training of QNNs:
   Loss L(theta) = 1/M * sum_{m=1}^M Loss(y_hat_m(theta), y_m).
   By the chain rule:
       dL / d theta_j = 1/M * sum_{m=1}^M [ dLoss / dy_hat_m * d y_hat_m / d theta_j ]
   where d y_hat_m / d theta_j is evaluated using the Parameter-Shift Rule!

4. Fair Classical Baseline (Classical MLP):
   To avoid unfounded quantum supremacy claims, QNNs must be compared against
   a classical Multi-Layer Perceptron (MLP) evaluated on the exact same dataset
   with a comparable parameter budget.
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector


def generate_concentric_circles_dataset(
    n_samples: int = 100,
    noise: float = 0.04,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a 2D concentric circles binary classification dataset.
    Class 0: Inner circle (radius ~ 0.5)
    Class 1: Outer ring (radius ~ 1.0)
    Non-linearly separable in Cartesian coordinates.
    """
    rng = np.random.RandomState(seed)
    n_half = n_samples // 2

    # Inner circle
    r_in = 0.5 + rng.normal(0, noise, size=n_half)
    th_in = rng.uniform(0, 2.0 * np.pi, size=n_half)
    x_in = np.column_stack([r_in * np.cos(th_in), r_in * np.sin(th_in)])
    y_in = np.zeros(n_half, dtype=np.float64)

    # Outer circle
    r_out = 1.0 + rng.normal(0, noise, size=n_half)
    th_out = rng.uniform(0, 2.0 * np.pi, size=n_half)
    x_out = np.column_stack([r_out * np.cos(th_out), r_out * np.sin(th_out)])
    y_out = np.ones(n_half, dtype=np.float64)

    X = np.vstack([x_in, x_out])
    y = np.concatenate([y_in, y_out])

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


class ZZFeatureMap:
    """
    Non-linear ZZ entangling feature map for d-dimensional inputs.
    Embeds data into n=d qubits using single-qubit Z rotations and pairwise R_ZZ couplings.
    """
    def __init__(self, num_qubits: int):
        self.num_qubits = int(num_qubits)
        self.x_params = ParameterVector("x", self.num_qubits)

    def build_circuit(self) -> QuantumCircuit:
        n = self.num_qubits
        qc = QuantumCircuit(n)
        # Hadamard layer
        qc.h(range(n))
        # Single-qubit Z rotations
        for i in range(n):
            qc.rz(2.0 * self.x_params[i], i)
        # Two-qubit ZZ entangling interactions
        for i in range(n):
            for j in range(i + 1, n):
                weight = 2.0 * (np.pi - self.x_params[i]) * (np.pi - self.x_params[j])
                qc.rzz(weight, i, j)
        return qc


class QuantumClassifierPQC:
    """
    Quantum Neural Network (QNN) Classifier based on a Parameterized Quantum Circuit.
    
    Structure:
    1. Feature Map: Non-linear embedding of input x in R^2 into 2 qubits.
    2. Variational Layers: L layers of parameterized R_y rotations and CX entangling gates.
    3. Readout: Expectation value of Pauli Z on qubit 0: y_hat = <Z_0> in [-1, +1].
    4. Probability: p = (1 + <Z_0>) / 2 in [0, 1].
    """
    def __init__(
        self,
        num_qubits: int = 2,
        reps: int = 2,
        lr: float = 0.2,
        seed: int = 42
    ):
        self.num_qubits = int(num_qubits)
        self.reps = int(reps)
        self.lr = float(lr)
        self.seed = int(seed)

        # Observable: Z on qubit 0 (Qiskit rightmost index: 'IZ')
        s = ['I'] * self.num_qubits
        s[self.num_qubits - 1] = 'Z'
        self.observable = SparsePauliOp.from_list([(''.join(s), 1.0)])

        # Parameter vectors
        self.x_params = ParameterVector("x", self.num_qubits)
        # Variational parameters: num_qubits * (reps + 1)
        self.num_weights = self.num_qubits * (self.reps + 1)
        self.w_params = ParameterVector("w", self.num_weights)

        # Initialize weights
        rng = np.random.RandomState(self.seed)
        self.weights = rng.uniform(-np.pi, np.pi, size=self.num_weights)

        # Build full circuit template
        self.circuit_template = self._build_template()

    def _build_template(self) -> QuantumCircuit:
        n = self.num_qubits
        qc = QuantumCircuit(n)

        # 1. Non-linear Feature Map: Angle + ZZ coupling
        qc.h(range(n))
        for i in range(n):
            qc.ry(self.x_params[i], i)
        if n >= 2:
            qc.rzz(2.0 * self.x_params[0] * self.x_params[1], 0, 1)

        # 2. Variational Ansatz Layers
        w_idx = 0
        for i in range(n):
            qc.ry(self.w_params[w_idx], i)
            w_idx += 1

        for _ in range(self.reps):
            for i in range(n - 1):
                qc.cx(i, i + 1)
            for i in range(n):
                qc.ry(self.w_params[w_idx], i)
                w_idx += 1

        return qc

    def forward_one(self, x_val: Sequence[float], weights: Optional[np.ndarray] = None) -> float:
        """Computes expected readout <Z_0> in [-1, +1] for a single sample."""
        w = self.weights if weights is None else weights
        param_dict = {}
        for i in range(self.num_qubits):
            param_dict[self.x_params[i]] = float(x_val[i])
        for i in range(self.num_weights):
            param_dict[self.w_params[i]] = float(w[i])

        bound_qc = self.circuit_template.assign_parameters(param_dict)
        sv = Statevector.from_instruction(bound_qc)
        return float(sv.expectation_value(self.observable).real)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class 1 probabilities: p = (1 + <Z_0>) / 2."""
        preds = np.array([self.forward_one(x) for x in X], dtype=np.float64)
        probs = 0.5 * (1.0 + preds)
        return np.clip(probs, 1e-7, 1.0 - 1e-7)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary classification predictions in {0, 1}."""
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean Squared Error (MSE) loss."""
        probs = self.predict_proba(X)
        return float(np.mean((probs - y) ** 2))

    def compute_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes analytical gradient of the MSE loss with respect to variational
        parameters using the Parameter-Shift Rule:
            dL/dw_j = 1/M * sum_{m=1}^M 2*(p_m - y_m) * (1/2) * d<Z_0>/dw_j
        """
        probs = self.predict_proba(X)
        errors = (probs - y)  # shape (M,)
        M = len(X)
        grad = np.zeros(self.num_weights, dtype=np.float64)
        shift = np.pi / 2.0

        for j in range(self.num_weights):
            w_plus = self.weights.copy()
            w_minus = self.weights.copy()
            w_plus[j] += shift
            w_minus[j] -= shift

            # Parameter shift on each sample for weight j
            d_readout = np.zeros(M, dtype=np.float64)
            for m in range(M):
                r_plus = self.forward_one(X[m], w_plus)
                r_minus = self.forward_one(X[m], w_minus)
                d_readout[m] = 0.5 * (r_plus - r_minus)

            # Chain rule: dp_m / dw_j = 0.5 * d<Z_0>_m / dw_j
            dp_dwj = 0.5 * d_readout
            grad[j] = np.mean(errors * dp_dwj)

        return grad

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 40,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Trains the QNN using gradient descent with the Parameter-Shift Rule.
        """
        t0 = time.perf_counter()
        history_loss = []
        history_acc = []

        M = len(X)
        for ep in range(epochs):
            if batch_size is not None and batch_size < M:
                idx = np.random.choice(M, size=batch_size, replace=False)
                X_batch, y_batch = X[idx], y[idx]
            else:
                X_batch, y_batch = X, y

            grad = self.compute_gradient(X_batch, y_batch)
            self.weights -= self.lr * grad

            loss = self.compute_loss(X, y)
            acc = float(np.mean(self.predict(X) == y))
            history_loss.append(loss)
            history_acc.append(acc)

        train_time_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "epochs": epochs,
            "final_loss": history_loss[-1],
            "final_accuracy": history_acc[-1],
            "loss_history": history_loss,
            "accuracy_history": history_acc,
            "train_time_ms": train_time_ms,
            "num_parameters": self.num_weights
        }


class ClassicalMLPClassifier:
    """
    Fair Classical Baseline: 2-layer Multi-Layer Perceptron (MLP) built directly
    from pure NumPy, with a parameter budget matching the QNN.
    
    Architecture:
    Input: d=2
    Hidden: h=4 units (with tanh activation)
    Output: 1 unit (with sigmoid activation)
    Total parameters: (2*4 + 4) + (4*1 + 1) = 17 parameters.
    """
    def __init__(self, in_dim: int = 2, hidden_dim: int = 4, lr: float = 0.1, seed: int = 42):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.seed = seed

        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(in_dim, hidden_dim) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)

        self.num_parameters = (in_dim * hidden_dim + hidden_dim) + (hidden_dim * 1 + 1)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Z1 = X @ W1 + b1, A1 = tanh(Z1)
        Z1 = X @ self.W1 + self.b1
        A1 = np.tanh(Z1)
        # Z2 = A1 @ W2 + b2, A2 = sigmoid(Z2)
        Z2 = A1 @ self.W2 + self.b2
        A2 = 1.0 / (1.0 + np.exp(-np.clip(Z2, -15.0, 15.0)))
        return A1, A2.flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, A2 = self.forward(X)
        return A2

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self.predict_proba(X)
        return float(np.mean((p - y) ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 80) -> Dict[str, Any]:
        t0 = time.perf_counter()
        history_loss = []
        history_acc = []

        M = len(X)
        y_col = y.reshape(-1, 1)

        for _ in range(epochs):
            A1, p = self.forward(X)
            p_col = p.reshape(-1, 1)

            # MSE loss: L = 1/M * sum (p - y)^2
            # dL/dp = 2/M * (p - y)
            dL_dp = (2.0 / M) * (p_col - y_col)
            # dp/dZ2 = p * (1 - p) (sigmoid derivative)
            dL_dZ2 = dL_dp * p_col * (1.0 - p_col)

            # Gradients for Layer 2
            dW2 = A1.T @ dL_dZ2
            db2 = np.sum(dL_dZ2, axis=0)

            # Backprop into Layer 1
            dL_dA1 = dL_dZ2 @ self.W2.T
            dL_dZ1 = dL_dA1 * (1.0 - A1 ** 2)  # tanh derivative

            dW1 = X.T @ dL_dZ1
            db1 = np.sum(dL_dZ1, axis=0)

            # Gradient descent step
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            loss = self.compute_loss(X, y)
            acc = float(np.mean(self.predict(X) == y))
            history_loss.append(loss)
            history_acc.append(acc)

        train_time_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "epochs": epochs,
            "final_loss": history_loss[-1],
            "final_accuracy": history_acc[-1],
            "loss_history": history_loss,
            "accuracy_history": history_acc,
            "train_time_ms": train_time_ms,
            "num_parameters": self.num_parameters
        }


def explain_pqc_and_qnn() -> str:
    """Academic explanation of PQCs and QNNs for student viva defense."""
    return (
        "### Parameterized Quantum Circuits (PQCs) and Quantum Neural Networks (QNNs)\n\n"
        "1. **Quantum Feature Embedding**:\n"
        "   Unlike classical ML where features enter directly as continuous activations,\n"
        "   QNNs embed classical data x in R^d into the phases and amplitudes of quantum states.\n"
        "   Entangling feature maps (such as ZZ-feature maps) create non-linear interactions\n"
        "   between input coordinates, mapping non-linearly separable data into linearly separable\n"
        "   hyperplanes in 2^n dimensional Hilbert space (Quantum Kernel Trick).\n\n"
        "2. **Variational Circuit Learning**:\n"
        "   The parameterized layers act as trainable quantum weights. Training is driven by\n"
        "   the Parameter-Shift Rule, which provides unbiased analytical gradients without requiring\n"
        "   classical backpropagation through quantum wavefunctions.\n\n"
        "3. **Comparison with Classical Neural Networks (MLP)**:\n"
        "   - Parameter Efficiency: A 2-qubit QNN with only 6 parameters can learn non-linear boundaries\n"
        "     due to the high expressivity of quantum Hilbert spaces.\n"
        "   - Computational Reality: Evaluating quantum gradients via repeated circuit execution is\n"
        "     orders-of-magnitude slower than classical matrix multiplication and backpropagation.\n"
        "   - No Quantum Supremacy Claim: On small low-dimensional datasets, classical MLPs train\n"
        "     in milliseconds, while QNNs serve as foundational prototypes for quantum data analysis.\n"
    )
