"""
mlp_backpropagation.py
Standalone Two-Layer Multi-Layer Perceptron (MLP) with decoupled
Forward Propagation, Loss Computation, and Backpropagation Gradient Engine.

Architecture:
    X in R^(N x d_in)
    Z1 = X W1 + b1          (Hidden Linear Transformation)
    A1 = tanh(Z1)           (C^2 Smooth Non-linear Activation)
    Z2 = A1 W2 + b2         (Output Linear Transformation)
    y_hat = Z2              (Regression Output)
    Loss = 0.5/N * ||y_hat - Y||^2 + 0.5 * reg * (||W1||^2 + ||W2||^2)

Provides rigorous numerical finite-difference gradient verification.
"""

from typing import Dict, Tuple, Any
import numpy as np


class TwoLayerMLPBackprop:
    """
    Decoupled Two-Layer MLP showing Backpropagation as a Gradient Computation Engine,
    strictly distinct from any downstream optimizer (GD, Adam, Newton, etc.).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, reg: float = 1e-4, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.reg = float(reg)
        
        # Xavier/He initialization
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim, dtype=np.float64)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward Pass:
            Computes network predictions and caches intermediate activations
            needed for the chain-rule backward pass.
        """
        X = np.asarray(X, dtype=np.float64)
        Z1 = X @ self.W1 + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1 @ self.W2 + self.b2
        y_hat = Z2

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "y_hat": y_hat
        }
        return y_hat, cache

    def compute_loss(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Mean Squared Error Loss with L2 (Tikhonov) Weight Regularization:
            L = (1 / 2N) * ||y_hat - y_true||_F^2 + (reg / 2) * (||W1||_F^2 + ||W2||_F^2)
        """
        N = y_true.shape[0]
        data_loss = 0.5 * np.sum((y_hat - y_true)**2) / N
        reg_loss = 0.5 * self.reg * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return float(data_loss + reg_loss)

    def backward(self, y_true: np.ndarray, cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Backward Pass (Backpropagation Engine):
            Applies the multivariate calculus chain rule to compute exact analytical
            partial derivatives of the loss with respect to every weight and bias.
        """
        X = cache["X"]
        A1 = cache["A1"]
        y_hat = cache["y_hat"]
        N = X.shape[0]

        # Output gradient: dL / dZ2
        dZ2 = (y_hat - y_true) / N  # shape (N, output_dim)

        # Layer 2 parameter gradients
        dW2 = A1.T @ dZ2 + self.reg * self.W2
        db2 = np.sum(dZ2, axis=0)

        # Backpropagate through hidden layer: dL / dA1 = dZ2 @ W2^T
        dA1 = dZ2 @ self.W2.T
        # Activation gradient: d(tanh(z)) / dz = 1 - tanh^2(z) = 1 - A1^2
        dZ1 = dA1 * (1.0 - A1**2)

        # Layer 1 parameter gradients
        dW1 = X.T @ dZ1 + self.reg * self.W1
        db1 = np.sum(dZ1, axis=0)

        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2
        }
        return grads

    def get_flat_params(self) -> np.ndarray:
        """Flattens all network parameters into a single 1D vector theta."""
        return np.concatenate([
            self.W1.ravel(),
            self.b1.ravel(),
            self.W2.ravel(),
            self.b2.ravel()
        ])

    def set_flat_params(self, theta: np.ndarray):
        """Restores model parameter matrices from a flat 1D vector theta."""
        idx = 0
        w1_size = self.input_dim * self.hidden_dim
        self.W1 = theta[idx:idx + w1_size].reshape(self.input_dim, self.hidden_dim).copy()
        idx += w1_size

        b1_size = self.hidden_dim
        self.b1 = theta[idx:idx + b1_size].copy()
        idx += b1_size

        w2_size = self.hidden_dim * self.output_dim
        self.W2 = theta[idx:idx + w2_size].reshape(self.hidden_dim, self.output_dim).copy()
        idx += w2_size

        b2_size = self.output_dim
        self.b2 = theta[idx:idx + b2_size].copy()

    def get_flat_grads(self, grads: Dict[str, np.ndarray]) -> np.ndarray:
        """Flattens parameter gradient dictionary into a single 1D vector."""
        return np.concatenate([
            grads["W1"].ravel(),
            grads["b1"].ravel(),
            grads["W2"].ravel(),
            grads["b2"].ravel()
        ])

    def check_gradients_finite_difference(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        eps: float = 1e-6
    ) -> Dict[str, float]:
        """
        Performs rigorous symmetric two-sided finite difference gradient check:
            g_num[j] = (L(theta + eps * e_j) - L(theta - eps * e_j)) / (2 * eps)
        Returns:
            rel_error: ||g_ana - g_num||_2 / (||g_ana||_2 + ||g_num||_2)
            max_abs_diff: max_j |g_ana[j] - g_num[j]|
        """
        # Analytical gradients via backprop
        y_hat, cache = self.forward(X)
        grads = self.backward(y_true, cache)
        g_ana = self.get_flat_grads(grads)

        # Numerical gradients via symmetric perturbation
        orig_theta = self.get_flat_params()
        g_num = np.zeros_like(orig_theta)

        for j in range(len(orig_theta)):
            theta_plus = orig_theta.copy()
            theta_plus[j] += eps
            self.set_flat_params(theta_plus)
            y_plus, _ = self.forward(X)
            loss_plus = self.compute_loss(y_true, y_plus)

            theta_minus = orig_theta.copy()
            theta_minus[j] -= eps
            self.set_flat_params(theta_minus)
            y_minus, _ = self.forward(X)
            loss_minus = self.compute_loss(y_true, y_minus)

            g_num[j] = (loss_plus - loss_minus) / (2.0 * eps)

        # Restore original weights
        self.set_flat_params(orig_theta)

        # Compute error metrics
        numerator = np.linalg.norm(g_ana - g_num)
        denominator = np.linalg.norm(g_ana) + np.linalg.norm(g_num)
        rel_error = float(numerator / max(denominator, 1e-12))
        max_abs_diff = float(np.max(np.abs(g_ana - g_num)))

        return {
            "relative_error": rel_error,
            "max_abs_diff": max_abs_diff,
            "g_ana_norm": float(np.linalg.norm(g_ana)),
            "g_num_norm": float(np.linalg.norm(g_num))
        }
