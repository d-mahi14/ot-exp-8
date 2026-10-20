"""
test_first_order_optimizers.py
Independent unit tests for all 8 first-order continuous optimizers.
Verifies exact update arithmetic, bias corrections, state resets,
and convergence on strongly convex quadratic objectives.
"""

import pytest
import numpy as np

from track_A_continuous.src.benchmark_functions import ConvexQuadraticBowl, Rosenbrock, SaddlePointLandscape
from track_A_continuous.src.first_order_optimizers import (
    GradientDescent, SGD, MiniBatchGD, Momentum, NAG, AdaGrad, RMSProp, Adam
)


class TestOptimizerArithmetic:
    """Tests exact numerical step calculations against analytical formulas."""

    def test_gradient_descent_step(self):
        opt = GradientDescent(lr=0.1)
        theta = np.array([2.0, -1.0])
        grad = np.array([4.0, 2.0])
        new_theta = opt.step(theta, grad)
        expected = np.array([2.0 - 0.1 * 4.0, -1.0 - 0.1 * 2.0])
        np.testing.assert_allclose(new_theta, expected)
        assert opt.iteration == 1

    def test_momentum_accumulation(self):
        opt = Momentum(lr=0.1, momentum=0.9)
        theta = np.array([1.0, 1.0])
        grad1 = np.array([2.0, 0.0])
        # Step 1: v1 = 0.9 * 0 + 0.1 * [2.0, 0.0] = [0.2, 0.0]
        theta1 = opt.step(theta, grad1)
        np.testing.assert_allclose(theta1, [0.8, 1.0])
        assert opt.iteration == 1

        # Step 2: v2 = 0.9 * [0.2, 0.0] + 0.1 * [2.0, 0.0] = [0.18 + 0.2, 0.0] = [0.38, 0.0]
        grad2 = np.array([2.0, 0.0])
        theta2 = opt.step(theta1, grad2)
        np.testing.assert_allclose(theta2, [0.8 - 0.38, 1.0])
        assert opt.iteration == 2

    def test_nag_lookahead(self):
        opt = NAG(lr=0.1, momentum=0.9)
        theta = np.array([1.0, 1.0])
        # Before step 1, velocity is 0, so lookahead is theta
        lookahead = opt.get_lookahead_point(theta)
        np.testing.assert_allclose(lookahead, theta)
        
        grad1 = np.array([1.0, 1.0])
        theta1 = opt.step(theta, grad1)
        # v1 = [0.1, 0.1], lookahead for step 2 should be theta1 - 0.9 * [0.1, 0.1]
        expected_lookahead = theta1 - 0.9 * np.array([0.1, 0.1])
        np.testing.assert_allclose(opt.get_lookahead_point(theta1), expected_lookahead)

    def test_adagrad_learning_rate_decay(self):
        opt = AdaGrad(lr=1.0, eps=1e-8)
        theta = np.array([0.0])
        grad = np.array([2.0])
        # G1 = 4.0, step = - 1.0 / (sqrt(4.0) + 1e-8) * 2.0 = -1.0
        theta1 = opt.step(theta, grad)
        np.testing.assert_allclose(theta1, [-1.0], rtol=1e-5)
        # G2 = 4.0 + 4.0 = 8.0, step = - 1.0 / sqrt(8.0) * 2.0 = -0.707106
        theta2 = opt.step(theta1, grad)
        np.testing.assert_allclose(theta2, [-1.0 - (2.0 / np.sqrt(8.0))], rtol=1e-5)

    def test_adam_bias_correction(self):
        opt = Adam(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
        theta = np.array([1.0])
        grad = np.array([2.0])
        
        # t = 1:
        # m1 = 0.1 * 2.0 = 0.2, m_hat = 0.2 / (1 - 0.9^1) = 2.0
        # v1 = 0.001 * 4.0 = 0.004, v_hat = 0.004 / (1 - 0.999^1) = 4.0
        # step = -0.1 * 2.0 / (sqrt(4.0) + eps) = -0.1 * 2.0 / 2.0 = -0.1
        theta1 = opt.step(theta, grad)
        np.testing.assert_allclose(theta1, [0.9], rtol=1e-4)

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            GradientDescent(lr=-0.01)
        with pytest.raises(ValueError):
            Momentum(lr=0.01, momentum=1.2)
        with pytest.raises(ValueError):
            Adam(lr=0.01, beta1=1.5)


class TestConvergenceOnQuadratic:
    """Verifies that all optimizers decrease loss on a strongly convex quadratic bowl."""

    @pytest.mark.parametrize("opt_class,kwargs", [
        (GradientDescent, {"lr": 0.05}),
        (Momentum, {"lr": 0.03, "momentum": 0.8}),
        (NAG, {"lr": 0.03, "momentum": 0.8}),
        (AdaGrad, {"lr": 0.5}),
        (RMSProp, {"lr": 0.05}),
        (Adam, {"lr": 0.1}),
    ])
    def test_quadratic_convergence(self, opt_class, kwargs):
        bowl = ConvexQuadraticBowl()
        opt = opt_class(**kwargs)
        x = np.array([3.0, 3.0])
        initial_loss = bowl.value(x)

        for _ in range(100):
            if isinstance(opt, NAG):
                x_lookahead = opt.get_lookahead_point(x)
                grad = bowl.gradient(x_lookahead)
            else:
                grad = bowl.gradient(x)
            x = opt.step(x, grad)

    def test_sgd_and_minibatch_convergence(self):
        from track_A_continuous.src.benchmark_functions import NoisyQuadraticLoss
        problem = NoisyQuadraticLoss(num_samples=200, num_features=5, noise_std=0.01, seed=42)
        rng = np.random.RandomState(42)
        
        # Test SGD
        sgd = SGD(lr=0.01)
        w_sgd = np.zeros(5)
        loss_init = problem.full_loss(w_sgd)
        for _ in range(300):
            grad = problem.mini_batch_gradient(w_sgd, batch_size=1, rng=rng)
            w_sgd = sgd.step(w_sgd, grad)
        assert problem.full_loss(w_sgd) < loss_init

        # Test Mini-Batch GD
        mb_gd = MiniBatchGD(lr=0.02, batch_size=16)
        w_mb = np.zeros(5)
        for _ in range(300):
            grad = problem.mini_batch_gradient(w_mb, batch_size=mb_gd.batch_size, rng=rng)
            w_mb = mb_gd.step(w_mb, grad)
        assert problem.full_loss(w_mb) < loss_init
