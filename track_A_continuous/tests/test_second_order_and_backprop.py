"""
test_second_order_and_backprop.py
Independent unit tests for Track A Stage 2:
1. Newton's Method (1-step exact convergence on quadratic, spectral safeguarding)
2. Gauss-Newton (least-squares residual minimization)
3. Hessian-Vector Products (finite-difference vs exact analytical product)
4. Empirical Fisher Information & Diagonal/Low-rank curvature steps
5. Standalone MLP Backpropagation Engine (analytical vs numerical finite-difference gradient check)
"""

import pytest
import numpy as np

from track_A_continuous.src.benchmark_functions import ConvexQuadraticBowl, Rosenbrock, SaddlePointLandscape
from track_A_continuous.src.second_order_methods import (
    NewtonOptimizer, GaussNewtonOptimizer, EmpiricalFisherInformation,
    DiagonalCurvatureApproximation, LowRankCurvatureApproximation,
    hessian_vector_product_finite_diff
)
from track_A_continuous.src.mlp_backpropagation import TwoLayerMLPBackprop


class TestNewtonAndGaussNewton:
    """Verifies second-order step arithmetic and convergence guarantees."""

    def test_newton_exact_quadratic_convergence(self):
        """Newton's method must converge to the exact minimum in exactly 1 step on any quadratic bowl."""
        A = np.array([[6.0, 2.0], [2.0, 4.0]], dtype=np.float64)
        b = np.array([1.0, -3.0], dtype=np.float64)
        bowl = ConvexQuadraticBowl(A=A, b=b)
        x_star = bowl.optimal_point

        opt = NewtonOptimizer(damping=0.0, lr=1.0, auto_safeguard=False)
        x0 = np.array([10.0, -15.0])
        grad = bowl.gradient(x0)
        hess = bowl.hessian(x0)

        x1 = opt.step(x0, grad, hess)
        np.testing.assert_allclose(x1, x_star, atol=1e-12)
        assert bowl.value(x1) < 1e-20

    def test_newton_saddle_point_safeguarding(self):
        """On a hyperbolic saddle point f(x,y)=x^2 - y^2, auto-safeguard adds damping to ensure positive-definiteness."""
        saddle = SaddlePointLandscape(alpha=1.0, beta=1.0)
        x0 = np.array([0.5, 0.5])
        grad = saddle.gradient(x0)
        hess = saddle.hessian(x0)  # diag(2, -2), not positive definite

        opt = NewtonOptimizer(damping=1e-4, auto_safeguard=True)
        delta, effective_damping = opt.compute_step(x0, grad, hess)

        # Effective damping must be > 2.0 to make min_eig (2 - 2) > 0
        assert effective_damping > 2.0
        # Search direction along y (concave direction) should move away from the saddle
        assert delta is not None

    def test_gauss_newton_least_squares(self):
        """Gauss-Newton must minimize nonlinear residual system."""
        # Non-linear 2D residuals: r1(theta) = theta[0]^2 + theta[1] - 3; r2(theta) = theta[0] + theta[1]^2 - 3
        def residuals(theta):
            return np.array([theta[0]**2 + theta[1] - 3.0, theta[0] + theta[1]**2 - 3.0])

        def jacobian(theta):
            return np.array([
                [2.0 * theta[0], 1.0],
                [1.0, 2.0 * theta[1]]
            ])

        gn = GaussNewtonOptimizer(damping=1e-3, lr=1.0)
        theta = np.array([2.0, 2.0])  # Near true root (1, 1) or (-something)
        initial_res_norm = np.linalg.norm(residuals(theta))

        for _ in range(10):
            r = residuals(theta)
            J = jacobian(theta)
            theta = gn.step(theta, r, J)

        final_res_norm = np.linalg.norm(residuals(theta))
        assert final_res_norm < initial_res_norm
        assert final_res_norm < 1e-4


class TestCurvatureApproximations:
    """Verifies HVP accuracy, diagonal scaling, and low-rank decompositions."""

    def test_hvp_matches_exact_hessian(self):
        """Finite-difference HVP must match analytical H @ v with high relative accuracy."""
        bowl = ConvexQuadraticBowl()
        x = np.array([2.5, -1.8])
        v = np.array([0.7, -1.2])

        hvp_fd = hessian_vector_product_finite_diff(bowl.gradient, x, v, eps=1e-5)
        hvp_exact = bowl.hessian(x) @ v

        rel_error = np.linalg.norm(hvp_fd - hvp_exact) / np.linalg.norm(hvp_exact)
        assert rel_error < 1e-6, f"HVP relative error {rel_error} exceeded threshold"

    def test_diagonal_curvature_step(self):
        diag_opt = DiagonalCurvatureApproximation(lr=1.0, eps=1e-4)
        grad = np.array([4.0, -2.0])
        hess_diag = np.array([10.0, 2.0])
        step = diag_opt.compute_step(grad, hess_diag)
        expected = - np.array([4.0 / 10.0, -2.0 / 2.0])
        np.testing.assert_allclose(step, expected)

    def test_low_rank_approximation_reconstruction(self):
        low_rank = LowRankCurvatureApproximation(rank=2, damping=1e-3)
        H = np.array([[10.0, 0.0], [0.0, 1.0]])
        grad = np.array([1.0, 1.0])
        step = low_rank.compute_step(grad, H)
        assert step.shape == (2,)
        # Since rank=2 equals matrix dimension, low-rank step should match damped full solve
        expected = - np.linalg.solve(H + 1e-3 * np.eye(2), grad)
        np.testing.assert_allclose(step, expected, rtol=1e-4)


class TestMLPBackpropagationEngine:
    """Rigorous verification of standalone forward/backward engine and numerical gradient checks."""

    def test_forward_backward_shapes(self):
        mlp = TwoLayerMLPBackprop(input_dim=4, hidden_dim=8, output_dim=2, reg=0.01, seed=42)
        X = np.random.randn(10, 4)
        y = np.random.randn(10, 2)

        y_hat, cache = mlp.forward(X)
        assert y_hat.shape == (10, 2)
        loss = mlp.compute_loss(y, y_hat)
        assert loss > 0.0

        grads = mlp.backward(y, cache)
        assert grads["W1"].shape == (4, 8)
        assert grads["b1"].shape == (8,)
        assert grads["W2"].shape == (8, 2)
        assert grads["b2"].shape == (2,)

    def test_finite_difference_gradient_check(self):
        """
        Calculates symmetric two-sided finite difference gradients for all weights and biases.
        Asserts that measured relative error is strictly below 1e-7.
        """
        mlp = TwoLayerMLPBackprop(input_dim=3, hidden_dim=5, output_dim=1, reg=1e-3, seed=123)
        rng = np.random.RandomState(99)
        X = rng.randn(8, 3)
        y = rng.randn(8, 1)

        check_res = mlp.check_gradients_finite_difference(X, y, eps=1e-6)
        rel_error = check_res["relative_error"]
        max_abs_diff = check_res["max_abs_diff"]

        # Measured relative error must be strictly within precision
        assert rel_error < 1e-7, f"Gradient check failed with relative error: {rel_error}"
        assert max_abs_diff < 1e-6, f"Max absolute difference {max_abs_diff} too large"

    def test_mlp_training_reduces_loss(self):
        """Verifies that decoupled backprop engine integrated with an optimizer reduces MSE loss."""
        mlp = TwoLayerMLPBackprop(input_dim=2, hidden_dim=6, output_dim=1, reg=1e-4, seed=42)
        rng = np.random.RandomState(42)
        X = rng.uniform(-1.0, 1.0, size=(50, 2))
        y = np.sin(X[:, 0:1]) + 0.5 * X[:, 1:2]

        y_init, _ = mlp.forward(X)
        loss_init = mlp.compute_loss(y, y_init)

        # 100 gradient descent steps using analytical backprop
        lr = 0.05
        for _ in range(100):
            y_pred, cache = mlp.forward(X)
            grads = mlp.backward(y, cache)
            mlp.W1 -= lr * grads["W1"]
            mlp.b1 -= lr * grads["b1"]
            mlp.W2 -= lr * grads["W2"]
            mlp.b2 -= lr * grads["b2"]

        y_final, _ = mlp.forward(X)
        loss_final = mlp.compute_loss(y, y_final)
        assert loss_final < loss_init
        assert loss_final < 0.1 * loss_init
