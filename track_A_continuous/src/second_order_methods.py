"""
second_order_methods.py
Pure-NumPy implementations of Second-Order and Curvature-Based Optimization Methods:
1. Newton's Method (with Levenberg-Marquardt damping and spectral safeguarding)
2. Gauss-Newton Method (for non-linear least squares problems)
3. Empirical Fisher Information Matrix (Natural Gradient Descent)
4. Diagonal Hessian Curvature Approximation
5. Low-Rank Hessian Approximation (Rank-r spectral decomposition)
6. Hessian-Vector Products (HVP / Pearlmutter directional derivative)
"""

from typing import Callable, Tuple, Optional
import numpy as np


class NewtonOptimizer:
    """
    Newton's Method with Levenberg-Marquardt Damping:
        Delta_theta = - (H(theta) + mu * I)^(-1) * grad(theta)
        theta_{t+1} = theta_t + alpha * Delta_theta
    
    If the Hessian has non-positive eigenvalues (e.g. near saddle points),
    adaptive spectral damping increases mu to ensure strict positive-definiteness.
    """
    def __init__(self, damping: float = 1e-4, lr: float = 1.0, auto_safeguard: bool = True):
        self.damping = float(damping)
        self.lr = float(lr)
        self.auto_safeguard = auto_safeguard
        self.iteration = 0

    def compute_step(self, theta: np.ndarray, grad: np.ndarray, hess: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes the Newton step Delta_theta.
        Returns:
            delta (np.ndarray): The search direction.
            effective_damping (float): The actual damping mu used.
        """
        d = theta.shape[0]
        H = hess.copy()
        mu = self.damping

        if self.auto_safeguard:
            # Check smallest eigenvalue
            eigvals = np.linalg.eigvalsh(H)
            min_eig = float(np.min(eigvals))
            if min_eig <= 1e-5:
                # Add enough damping to make smallest eigenvalue at least 1e-3
                mu = max(mu, 1e-3 - min_eig)

        H_damped = H + mu * np.eye(d, dtype=np.float64)
        
        try:
            delta = -np.linalg.solve(H_damped, grad)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            delta = -np.linalg.pinv(H_damped) @ grad

        return delta, mu

    def step(self, theta: np.ndarray, grad: np.ndarray, hess: np.ndarray) -> np.ndarray:
        delta, _ = self.compute_step(theta, grad, hess)
        self.iteration += 1
        return theta + self.lr * delta


class GaussNewtonOptimizer:
    """
    Gauss-Newton Method for Non-Linear Least Squares:
        Objective: f(theta) = 0.5 * sum_{i=1}^m r_i(theta)^2 = 0.5 * ||R(theta)||^2
        Jacobian J_R in R^(m x d), Residuals R in R^m
        Gradient: g = J_R^T * R(theta)
        Gauss-Newton Curvature: H_GN = J_R^T * J_R
        Update: Delta_theta = - (J_R^T * J_R + mu * I)^(-1) * (J_R^T * R(theta))
    """
    def __init__(self, damping: float = 1e-4, lr: float = 1.0):
        self.damping = float(damping)
        self.lr = float(lr)
        self.iteration = 0

    def compute_step(self, residuals: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
        m, d = jacobian.shape
        grad = jacobian.T @ residuals
        H_gn = jacobian.T @ jacobian + self.damping * np.eye(d, dtype=np.float64)
        delta = -np.linalg.solve(H_gn, grad)
        return delta

    def step(self, theta: np.ndarray, residuals: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
        delta = self.compute_step(residuals, jacobian)
        self.iteration += 1
        return theta + self.lr * delta


class EmpiricalFisherInformation:
    """
    Empirical Fisher Information Matrix (FIM) & Natural Gradient Descent:
        Given per-sample loss gradients g_i = nabla_theta L_i(theta),
        F_emp = (1 / N) * sum_{i=1}^N g_i * g_i^T
        Natural Gradient Update: Delta_theta = - eta * (F_emp + lambda * I)^(-1) * grad_total
    Invariance to parameter coordinate transformations.
    """
    def __init__(self, lr: float = 0.1, reg: float = 1e-3):
        self.lr = float(lr)
        self.reg = float(reg)
        self.iteration = 0

    def compute_fisher_matrix(self, sample_gradients: np.ndarray) -> np.ndarray:
        """
        sample_gradients: shape (N, d) where N is number of samples, d is param dim.
        """
        N, d = sample_gradients.shape
        F = (sample_gradients.T @ sample_gradients) / N
        return F

    def compute_step(self, total_grad: np.ndarray, sample_gradients: np.ndarray) -> np.ndarray:
        d = total_grad.shape[0]
        F = self.compute_fisher_matrix(sample_gradients)
        F_reg = F + self.reg * np.eye(d, dtype=np.float64)
        nat_grad = np.linalg.solve(F_reg, total_grad)
        return -self.lr * nat_grad


class DiagonalCurvatureApproximation:
    """
    Diagonal Hessian Approximation:
        Ignores cross-parameter couplings H_ij (i != j):
        D_ii = max(|H_ii|, eps)
        Delta_theta_i = - (1 / D_ii) * grad_i
    Reduces O(d^3) matrix inversion cost to O(d) element-wise division.
    """
    def __init__(self, lr: float = 1.0, eps: float = 1e-4):
        self.lr = float(lr)
        self.eps = float(eps)

    def compute_step(self, grad: np.ndarray, hessian_diag: np.ndarray) -> np.ndarray:
        d_clamped = np.maximum(np.abs(hessian_diag), self.eps)
        return -self.lr * (grad / d_clamped)


class LowRankCurvatureApproximation:
    """
    Low-Rank Curvature Approximation (Truncated Spectral Hessian):
        Approximates H in R^(d x d) via its top-r principal eigenspace:
            H_approx = sum_{k=1}^r lambda_k u_k u_k^T + sigma^2 * (I - sum u_k u_k^T)
        where sigma^2 is the average background curvature of discarded modes:
            sigma^2 = (1 / (d - r)) * sum_{k=r+1}^d lambda_k  (or max(sigma^2, damping))
        Inverting this structured model gives:
            (H_approx)^(-1) = U_r Lambda_r^(-1) U_r^T + (1 / sigma^2) * (I - U_r U_r^T)
    Prevents explosive divergence in orthogonal directions while capturing dominant curvature.
    """
    def __init__(self, rank: int = 1, damping: float = 1e-3, lr: float = 1.0):
        self.rank = rank
        self.damping = float(damping)
        self.lr = float(lr)

    def compute_step(self, grad: np.ndarray, hessian: np.ndarray) -> np.ndarray:
        d = hessian.shape[0]
        r = min(max(1, self.rank), d)
        eigvals, eigvecs = np.linalg.eigh(hessian)

        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        sorted_vals = eigvals[idx]
        sorted_vecs = eigvecs[:, idx]

        top_vals = sorted_vals[:r]
        top_vecs = sorted_vecs[:, :r]

        # Estimate background curvature from trailing discarded eigenvalues
        if d > r:
            trailing_vals = sorted_vals[r:]
            sigma_sq = float(np.mean(np.maximum(trailing_vals, self.damping)))
        else:
            sigma_sq = self.damping

        # Curvature along principal eigenspace (damped spectral inverse)
        inv_top_vals = 1.0 / (np.maximum(top_vals, 0.0) + self.damping)
        step_proj = top_vecs @ (inv_top_vals * (top_vecs.T @ grad))

        # Curvature along orthogonal subspace (scaled by background curvature sigma^2)
        grad_proj = top_vecs @ (top_vecs.T @ grad)
        grad_orth = grad - grad_proj
        step_orth = (1.0 / (max(sigma_sq, 1e-4) + self.damping)) * grad_orth

        delta = -self.lr * (step_proj + step_orth)
        return delta


def hessian_vector_product_finite_diff(
    grad_func: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    v: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Computes H * v via symmetric two-sided finite difference of gradients
    without ever forming or storing the full d x d Hessian matrix:
        H v = (nabla f(theta + eps * v) - nabla f(theta - eps * v)) / (2 * eps)
    Memory: O(d) instead of O(d^2).
    """
    theta = np.asarray(theta, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return np.zeros_like(theta)

    g_plus = grad_func(theta + eps * v)
    g_minus = grad_func(theta - eps * v)
    return (g_plus - g_minus) / (2.0 * eps)
