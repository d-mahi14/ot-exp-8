"""
benchmark_functions.py
Analytical benchmark objective functions for continuous optimization.
Provides exact values, analytical gradients, and Hessians for convex,
non-convex, and saddle-point landscapes.
"""

from typing import Tuple, Optional
import numpy as np


class ConvexQuadraticBowl:
    """
    Anisotropic Convex Quadratic Bowl:
        f(x) = 0.5 * x^T A x - b^T x
    For A = diag([10.0, 1.0]) and b = [0, 0], condition number kappa = 10.
    The global minimum is at x* = [0, 0] with f(x*) = 0.
    """
    def __init__(self, A: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None):
        if A is None:
            # Default to ill-conditioned 2D ellipse (steep along x0, shallow along x1)
            self.A = np.array([[10.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        else:
            self.A = np.asarray(A, dtype=np.float64)
        
        if b is None:
            self.b = np.zeros(self.A.shape[0], dtype=np.float64)
        else:
            self.b = np.asarray(b, dtype=np.float64)
        
        # Verify symmetric positive definiteness
        eigvals = np.linalg.eigvalsh(self.A)
        if np.any(eigvals <= 0):
            raise ValueError("Matrix A must be strictly positive definite.")
        self.condition_number = float(np.max(eigvals) / np.min(eigvals))

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(0.5 * x.T @ self.A @ x - self.b.T @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (self.A @ x - self.b)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return self.A.copy()

    @property
    def optimal_point(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    @property
    def optimal_value(self) -> float:
        x_star = self.optimal_point
        return self.value(x_star)


class Rosenbrock:
    """
    Rosenbrock Function (Non-Convex Banana Valley):
        f(x, y) = (a - x)^2 + b * (y - x^2)^2
    Standard parameters: a = 1.0, b = 100.0.
    Global minimum is located at (a, a^2) = (1.0, 1.0) with f(x*) = 0.
    Features a parabolic, flat-bottomed valley that challenges gradient descent.
    """
    def __init__(self, a: float = 1.0, b: float = 100.0):
        self.a = float(a)
        self.b = float(b)

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        x0, x1 = x[0], x[1]
        return float((self.a - x0)**2 + self.b * (x1 - x0**2)**2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x0, x1 = x[0], x[1]
        g0 = -2.0 * (self.a - x0) - 4.0 * self.b * x0 * (x1 - x0**2)
        g1 = 2.0 * self.b * (x1 - x0**2)
        return np.array([g0, g1], dtype=np.float64)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x0, x1 = x[0], x[1]
        h00 = 2.0 - 4.0 * self.b * (x1 - 3.0 * x0**2)
        h01 = -4.0 * self.b * x0
        h10 = -4.0 * self.b * x0
        h11 = 2.0 * self.b
        return np.array([[h00, h01], [h10, h11]], dtype=np.float64)

    @property
    def optimal_point(self) -> np.ndarray:
        return np.array([self.a, self.a**2], dtype=np.float64)

    @property
    def optimal_value(self) -> float:
        return 0.0


class SaddlePointLandscape:
    """
    2D Hyperbolic Saddle Point:
        f(x, y) = x^2 - y^2
    Critical point at (0, 0) where nabla f(0, 0) = [0, 0].
    Hessian has eigenvalues +2 (convex along x) and -2 (concave along y).
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = float(alpha)
        self.beta = float(beta)

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(self.alpha * x[0]**2 - self.beta * x[1]**2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.array([2.0 * self.alpha * x[0], -2.0 * self.beta * x[1]], dtype=np.float64)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array([[2.0 * self.alpha, 0.0], [0.0, -2.0 * self.beta]], dtype=np.float64)


class NoisyQuadraticLoss:
    """
    Linear Regression Mean Squared Error with Simulated Data for SGD / Mini-Batch:
        L(w) = (1 / 2N) * ||X w - y||^2
    Allows computing both exact full-batch gradients and noisy sample/mini-batch gradients.
    """
    def __init__(self, num_samples: int = 500, num_features: int = 10, noise_std: float = 0.1, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.X = rng.randn(num_samples, num_features)
        # Ground truth weights
        self.w_true = rng.uniform(-2.0, 2.0, size=num_features)
        self.y = self.X @ self.w_true + rng.normal(0.0, noise_std, size=num_samples)
        self.num_samples = num_samples
        self.num_features = num_features

    def full_loss(self, w: np.ndarray) -> float:
        residuals = self.X @ w - self.y
        return float(0.5 * np.mean(residuals**2))

    def full_gradient(self, w: np.ndarray) -> np.ndarray:
        residuals = self.X @ w - self.y
        return (self.X.T @ residuals) / self.num_samples

    def mini_batch_gradient(self, w: np.ndarray, batch_size: int, rng: np.random.RandomState) -> np.ndarray:
        indices = rng.choice(self.num_samples, size=min(batch_size, self.num_samples), replace=False)
        X_batch = self.X[indices]
        y_batch = self.y[indices]
        residuals = X_batch @ w - y_batch
        return (X_batch.T @ residuals) / len(indices)
