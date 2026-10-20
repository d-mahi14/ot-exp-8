"""
first_order_optimizers.py
Pure-NumPy implementations of the 8 compulsory Chapter 6 first-order continuous optimizers:
1. Gradient Descent (Vanilla GD)
2. Stochastic Gradient Descent (SGD)
3. Mini-batch Gradient Descent
4. Momentum (Polyak Heavy-Ball)
5. Nesterov Accelerated Gradient (NAG)
6. AdaGrad
7. RMSProp
8. Adam (Adaptive Moment Estimation)

Designed with consistent APIs, state tracking, and clear mathematical documentation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class FirstOrderOptimizer(ABC):
    """Abstract base class for all first-order continuous optimizers."""
    def __init__(self, lr: float = 0.01):
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        self.lr = float(lr)
        self.iteration = 0

    @abstractmethod
    def reset(self, param_dim: int):
        """Resets internal state (velocities, accumulators, iteration count)."""
        pass

    @abstractmethod
    def compute_step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Computes the parameter update delta: Delta_theta = theta_{t+1} - theta_t.
        Returns:
            delta (np.ndarray): The change vector such that theta_{t+1} = theta_t + delta.
        """
        pass

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Applies one optimization step and updates internal iteration count."""
        delta = self.compute_step(theta, grad)
        self.iteration += 1
        return theta + delta


class GradientDescent(FirstOrderOptimizer):
    """
    Standard Batch Gradient Descent:
        theta_{t+1} = theta_t - eta * grad(f(theta_t))
    """
    def __init__(self, lr: float = 0.01):
        super().__init__(lr=lr)

    def reset(self, param_dim: int):
        self.iteration = 0

    def compute_step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return -self.lr * grad


class SGD(FirstOrderOptimizer):
    """
    Stochastic Gradient Descent (single-sample noisy gradient):
        theta_{t+1} = theta_t - eta * g_i(theta_t)
    """
    def __init__(self, lr: float = 0.01):
        super().__init__(lr=lr)

    def reset(self, param_dim: int):
        self.iteration = 0

    def compute_step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return -self.lr * grad


class MiniBatchGD(FirstOrderOptimizer):
    """
    Mini-Batch Gradient Descent:
        theta_{t+1} = theta_t - (eta / |B|) * sum_{i in B} grad(f_i(theta_t))
    """
    def __init__(self, lr: float = 0.01, batch_size: int = 32):
        super().__init__(lr=lr)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = batch_size

    def reset(self, param_dim: int):
        self.iteration = 0

    def compute_step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return -self.lr * grad


class Momentum(FirstOrderOptimizer):
    """
    Momentum Optimizer (Polyak Heavy-Ball):
        v_{t+1} = gamma * v_t + eta * grad_t
        theta_{t+1} = theta_t - v_{t+1}
    Accelerates across persistent dimensions and dampens high-frequency oscillations.
    """
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        super().__init__(lr=lr)
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Momentum factor gamma must be in [0, 1), got {momentum}")
        self.gamma = float(momentum)
        self.v = None

    def reset(self, param_dim: int):
        self.iteration = 0
        self.v = np.zeros(param_dim, dtype=np.float64)

    def compute_step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.reset(theta.shape[0])
        self.v = self.gamma * self.v + self.lr * grad
        return -self.v


class NAG(FirstOrderOptimizer):
    """
    Nesterov Accelerated Gradient (NAG):
        Evaluates the gradient at the 'look-ahead' position:
        theta_lookahead = theta_t - gamma * v_t
        v_{t+1} = gamma * v_t + eta * grad(theta_lookahead)
        theta_{t+1} = theta_t - v_{t+1}
    Provides predictive damping before overshooting steep ravines.
    """
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        super().__init__(lr=lr)
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Momentum factor gamma must be in [0, 1), got {momentum}")
        self.gamma = float(momentum)
        self.v = None

    def reset(self, param_dim: int):
        self.iteration = 0
        self.v = np.zeros(param_dim, dtype=np.float64)

    def get_lookahead_point(self, theta: np.ndarray) -> np.ndarray:
        """Returns the predictive look-ahead parameter vector."""
        if self.v is None:
            self.reset(theta.shape[0])
        return theta - self.gamma * self.v

    def compute_step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Note: grad should be evaluated at the lookahead point."""
        if self.v is None:
            self.reset(theta.shape[0])
        self.v = self.gamma * self.v + self.lr * grad
        return -self.v


class AdaGrad(FirstOrderOptimizer):
    """
    AdaGrad (Adaptive Gradient Algorithm):
        G_{t+1} = G_t + grad_t^2 (element-wise accumulator)
        theta_{t+1} = theta_t - (eta / (sqrt(G_{t+1}) + eps)) * grad_t
    Individually adapts learning rates for sparse or frequently updated features.
    """
    def __init__(self, lr: float = 0.1, eps: float = 1e-8):
        super().__init__(lr=lr)
        self.eps = float(eps)
        self.G = None

    def reset(self, param_dim: int):
        self.iteration = 0
        self.G = np.zeros(param_dim, dtype=np.float64)

    def compute_step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.G is None:
            self.reset(theta.shape[0])
        self.G += grad**2
        adjusted_lr = self.lr / (np.sqrt(self.G) + self.eps)
        return -adjusted_lr * grad


class RMSProp(FirstOrderOptimizer):
    """
    RMSProp (Root Mean Square Propagation):
        v_{t+1} = beta * v_t + (1 - beta) * grad_t^2
        theta_{t+1} = theta_t - (eta / (sqrt(v_{t+1}) + eps)) * grad_t
    Resolves AdaGrad's vanishing learning rate via exponential moving average.
    """
    def __init__(self, lr: float = 0.01, beta: float = 0.9, eps: float = 1e-8):
        super().__init__(lr=lr)
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"Decay factor beta must be in [0, 1), got {beta}")
        self.beta = float(beta)
        self.eps = float(eps)
        self.v = None

    def reset(self, param_dim: int):
        self.iteration = 0
        self.v = np.zeros(param_dim, dtype=np.float64)

    def compute_step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.reset(theta.shape[0])
        self.v = self.beta * self.v + (1.0 - self.beta) * (grad**2)
        adjusted_lr = self.lr / (np.sqrt(self.v) + self.eps)
        return -adjusted_lr * grad


class Adam(FirstOrderOptimizer):
    """
    Adam (Adaptive Moment Estimation):
        m_{t+1} = beta1 * m_t + (1 - beta1) * grad_t             (1st moment: mean)
        v_{t+1} = beta2 * v_t + (1 - beta2) * grad_t^2           (2nd moment: uncentered variance)
        m_hat = m_{t+1} / (1 - beta1^(t+1))                      (Bias correction)
        v_hat = v_{t+1} / (1 - beta2^(t+1))                      (Bias correction)
        theta_{t+1} = theta_t - (eta / (sqrt(v_hat) + eps)) * m_hat
    Combines momentum with adaptive per-parameter scaling and zero-bias correction.
    """
    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(lr=lr)
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m = None
        self.v = None

    def reset(self, param_dim: int):
        self.iteration = 0
        self.m = np.zeros(param_dim, dtype=np.float64)
        self.v = np.zeros(param_dim, dtype=np.float64)

    def compute_step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None or self.v is None:
            self.reset(theta.shape[0])
        
        t = self.iteration + 1  # 1-indexed for bias correction
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad**2)

        # Bias corrections
        m_hat = self.m / (1.0 - (self.beta1**t))
        v_hat = self.v / (1.0 - (self.beta2**t))

        delta = -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return delta
