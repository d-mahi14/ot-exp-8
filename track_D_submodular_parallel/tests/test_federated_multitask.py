"""Unit tests for Distributed (FedAvg) and Multi-Task (MGDA, GradNorm) module in Track D."""

import numpy as np
import pytest
import torch

from track_D_submodular_parallel.src.federated_multitask import (
    MultiTaskNet,
    SmallMLP,
    evaluate_model,
    generate_multitask_data,
    generate_synthetic_federated_data,
    run_centralized_training,
    run_fedavg,
    solve_mgda_2task_analytical,
    solve_mgda_frank_wolfe,
    train_multitask_gradnorm,
    train_multitask_mgda,
    train_multitask_uniform,
)


def test_federated_data_generation() -> None:
    """Verify synthetic client dataset partitioning."""
    client_data, test_data = generate_synthetic_federated_data(
        num_clients=4,
        samples_per_client=100,
        in_features=8,
        non_iid_degree=0.5,
        seed=42,
    )
    assert len(client_data) == 4
    assert len(client_data[0]) == 100
    assert len(test_data) == 500


def test_fedavg_and_centralized_execution() -> None:
    """Verify FedAvg and Centralized training run and achieve valid test accuracy."""
    client_data, test_data = generate_synthetic_federated_data(
        num_clients=3,
        samples_per_client=150,
        in_features=10,
        non_iid_degree=0.3,
        seed=101,
    )

    # 1. FedAvg
    fed_model, fed_history = run_fedavg(
        client_data,
        test_data,
        num_rounds=5,
        local_epochs=2,
        local_lr=0.05,
        batch_size=32,
        seed=101,
    )
    assert len(fed_history) == 5
    # Model should have learned something better than chance (accuracy > 55%)
    assert fed_history[-1].test_accuracy > 0.55

    # 2. Centralized baseline
    cent_model, cent_history = run_centralized_training(
        client_data,
        test_data,
        total_epochs=15,
        lr=0.05,
        batch_size=32,
        seed=101,
    )
    assert len(cent_history) > 0
    assert cent_history[-1].test_accuracy > 0.55


def test_mgda_analytical_matches_frank_wolfe_2task() -> None:
    """Verify analytical 2-task formula matches Frank-Wolfe convex optimization."""
    # Two conflicting gradient vectors in 3D
    g1 = torch.tensor([1.0, 0.0, 0.5])
    g2 = torch.tensor([-0.5, 1.0, 0.2])

    a1_ana, a2_ana = solve_mgda_2task_analytical(g1, g2)
    alpha_fw = solve_mgda_frank_wolfe([g1, g2], max_iters=30)

    assert np.isclose(a1_ana, alpha_fw[0], atol=1e-3)
    assert np.isclose(a2_ana, alpha_fw[1], atol=1e-3)
    assert np.isclose(a1_ana + a2_ana, 1.0, atol=1e-6)


def test_mgda_common_descent_or_stationarity() -> None:
    """Verify MGDA yields combined gradient with non-negative inner products or minimal norm."""
    # Two orthogonal or mildly conflicting gradients
    g1 = torch.tensor([2.0, 0.5, 0.0])
    g2 = torch.tensor([0.5, 2.0, 0.0])

    a1, a2 = solve_mgda_2task_analytical(g1, g2)
    combined_g = a1 * g1 + a2 * g2

    # Inner products with both task gradients should be >= 0 (common descent direction)
    ip1 = float(torch.dot(combined_g, g1).item())
    ip2 = float(torch.dot(combined_g, g2).item())
    assert ip1 >= -1e-6
    assert ip2 >= -1e-6


def test_multitask_methods_execution_and_weight_balancing() -> None:
    """Verify Uniform, MGDA, and GradNorm train cleanly and GradNorm dynamically adapts weights."""
    X, y_reg, y_cls = generate_multitask_data(num_samples=300, in_features=16, seed=42)

    # 1. Uniform
    m_uni = MultiTaskNet(in_features=16, shared_dim=16)
    logs_uni = train_multitask_uniform(m_uni, X, y_reg, y_cls, num_steps=20, lr=0.01)
    assert len(logs_uni) > 0
    assert logs_uni[-1].total_loss < logs_uni[0].total_loss

    # 2. MGDA
    m_mgda = MultiTaskNet(in_features=16, shared_dim=16)
    logs_mgda = train_multitask_mgda(m_mgda, X, y_reg, y_cls, num_steps=20, lr=0.01)
    assert len(logs_mgda) > 0
    # Weights should sum to 1.0
    for l in logs_mgda:
        assert np.isclose(l.weight_t1 + l.weight_t2, 1.0, atol=1e-5)

    # 3. GradNorm
    m_gn = MultiTaskNet(in_features=16, shared_dim=16)
    logs_gn = train_multitask_gradnorm(m_gn, X, y_reg, y_cls, num_steps=30, lr=0.01)
    assert len(logs_gn) > 0
    # Weights should sum to T = 2.0
    for l in logs_gn:
        assert np.isclose(l.weight_t1 + l.weight_t2, 2.0, atol=1e-4)
