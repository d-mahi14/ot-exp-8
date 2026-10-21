"""Distributed and Multi-Task Optimization Module for Track D.

Implements:
1. Federated Learning (FedAvg):
   - Federated client local training with local epochs E
   - Non-IID / IID synthetic client data partitioning
   - Server aggregation: w_{t+1} = sum_k (n_k / N) * w_k
   - Comparison with Centralized Training baseline
2. Multi-Task Learning (MTL):
   - Shared representation neural network with task-specific heads
   - Uniform Task Weighting baseline (equal weights)
   - Multiple Gradient Descent Algorithm (MGDA - Sener & Koltun 2018):
     Frank-Wolfe / analytical simplex solver for Pareto-stationarity
   - GradNorm (Chen et al. 2018):
     Dynamic task loss weight adaptation balancing gradient norms
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset


# ============================================================================
# Section 1: Federated Learning (FedAvg Simulation)
# ============================================================================

class SmallMLP(nn.Module):
    """Small MLP model for federated classification/regression."""

    def __init__(self, in_features: int = 10, hidden_dim: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_synthetic_federated_data(
    num_clients: int = 5,
    samples_per_client: int = 200,
    in_features: int = 10,
    non_iid_degree: float = 0.8,
    seed: int = 42,
) -> Tuple[List[TensorDataset], TensorDataset]:
    """Generate synthetic binary classification data partitioned across clients.

    non_iid_degree in [0, 1]:
      0.0 -> perfectly IID (identical class ratios ~ 50/50 per client)
      1.0 -> extreme non-IID (clients have skewed class ratios)
    Returns:
        (client_datasets, global_test_dataset)
    """
    rng = np.random.default_rng(seed)
    client_datasets: List[TensorDataset] = []

    # True latent hyperplane
    w_true = rng.standard_normal(in_features)
    w_true /= np.linalg.norm(w_true)

    # Test set
    n_test = 500
    x_test = rng.standard_normal((n_test, in_features)).astype(np.float32)
    logits_test = x_test @ w_true
    y_test = (logits_test > 0.0).astype(np.int64)
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    for k in range(num_clients):
        # Client feature bias
        client_shift = rng.standard_normal(in_features) * 0.3 * non_iid_degree
        x_client = (rng.standard_normal((samples_per_client, in_features)) + client_shift).astype(np.float32)

        # Class skew
        skew = (k / max(1, num_clients - 1) - 0.5) * 2.0 * non_iid_degree
        logits = x_client @ w_true + skew
        y_client = (logits > 0.0).astype(np.int64)

        client_datasets.append(TensorDataset(torch.from_numpy(x_client), torch.from_numpy(y_client)))

    return client_datasets, test_dataset


def evaluate_model(model: nn.Module, dataset: Dataset, batch_size: int = 64) -> Tuple[float, float]:
    """Evaluate accuracy and cross-entropy loss on a dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item())
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == y).sum().item())
            total += len(y)

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


@dataclass
class FederatedRoundLog:
    round_idx: int
    train_loss: float
    test_loss: float
    test_accuracy: float


def run_fedavg(
    client_datasets: Sequence[TensorDataset],
    test_dataset: TensorDataset,
    num_rounds: int = 15,
    local_epochs: int = 3,
    local_lr: float = 0.05,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[nn.Module, List[FederatedRoundLog]]:
    """Simulate FedAvg across clients for a set number of global rounds."""
    torch.manual_seed(seed)
    global_model = SmallMLP()
    client_sample_counts = [len(d) for d in client_datasets]
    total_samples = sum(client_sample_counts)

    history: List[FederatedRoundLog] = []
    criterion = nn.CrossEntropyLoss()

    for r in range(num_rounds):
        local_weights = []

        # Local updates on each client
        for k, dataset in enumerate(client_datasets):
            local_model = copy.deepcopy(global_model)
            local_model.train()
            optimizer = optim.SGD(local_model.parameters(), lr=local_lr, momentum=0.9)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for _ in range(local_epochs):
                for x, y in loader:
                    optimizer.zero_grad()
                    out = local_model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()

            local_weights.append(copy.deepcopy(local_model.state_dict()))

        # Server Aggregation: w_{t+1} = sum_k (n_k / N) * w_k
        global_dict = copy.deepcopy(global_weights := global_model.state_dict())
        for key in global_dict.keys():
            global_dict[key] = sum(
                (client_sample_counts[k] / total_samples) * local_weights[k][key]
                for k in range(len(client_datasets))
            )
        global_model.load_state_dict(global_dict)

        # Global evaluation
        test_loss, test_acc = evaluate_model(global_model, test_dataset)
        history.append(
            FederatedRoundLog(
                round_idx=r + 1,
                train_loss=0.0,
                test_loss=test_loss,
                test_accuracy=test_acc,
            )
        )

    return global_model, history


def run_centralized_training(
    client_datasets: Sequence[TensorDataset],
    test_dataset: TensorDataset,
    total_epochs: int = 45,
    lr: float = 0.05,
    batch_size: int = 32,
    seed: int = 42,
) -> Tuple[nn.Module, List[FederatedRoundLog]]:
    """Centralized training baseline: pool all client data and train standard SGD."""
    torch.manual_seed(seed)
    model = SmallMLP()
    all_x = torch.cat([d.tensors[0] for d in client_datasets], dim=0)
    all_y = torch.cat([d.tensors[1] for d in client_datasets], dim=0)
    pooled_dataset = TensorDataset(all_x, all_y)

    loader = DataLoader(pooled_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    history: List[FederatedRoundLog] = []

    for epoch in range(total_epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # Evaluate periodically
        if (epoch + 1) % 3 == 0:
            test_loss, test_acc = evaluate_model(model, test_dataset)
            history.append(
                FederatedRoundLog(
                    round_idx=(epoch + 1) // 3,
                    train_loss=0.0,
                    test_loss=test_loss,
                    test_accuracy=test_acc,
                )
            )

    return model, history


# ============================================================================
# Section 2: Multi-Task Learning (MGDA & GradNorm)
# ============================================================================

class MultiTaskNet(nn.Module):
    """Shared feature representation with distinct heads for two tasks.

    Task 1: Continuous regression (MSE Loss)
    Task 2: Binary classification (BCEWithLogits Loss)
    """

    def __init__(self, in_features: int = 16, shared_dim: int = 32) -> None:
        super().__init__()
        # Shared trunk
        self.shared_layer = nn.Linear(in_features, shared_dim)
        self.relu = nn.ReLU()
        # Task 1 Head (Regression)
        self.head_reg = nn.Linear(shared_dim, 1)
        # Task 2 Head (Classification)
        self.head_cls = nn.Linear(shared_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.relu(self.shared_layer(x))
        out1 = self.head_reg(feat)
        out2 = self.head_cls(feat)
        return out1, out2

    def get_shared_parameters(self) -> List[nn.Parameter]:
        return list(self.shared_layer.parameters())


def solve_mgda_2task_analytical(g1: torch.Tensor, g2: torch.Tensor) -> Tuple[float, float]:
    """Analytical solution for MGDA min-norm convex combination between 2 task gradient vectors:

    min_{alpha in [0, 1]} || alpha * g1 + (1 - alpha) * g2 ||_2^2
    Analytical stationary point:
        alpha = ((g2 - g1)^T * g2) / ||g1 - g2||_2^2
    clipped to [0, 1].
    """
    diff = g2 - g1
    diff_norm_sq = float(torch.dot(diff, diff).item())
    if diff_norm_sq < 1e-12:
        return 0.5, 0.5

    dot_val = float(torch.dot(diff, g2).item())
    alpha = dot_val / diff_norm_sq

    # Project to simplex [0, 1]
    alpha = max(0.0, min(1.0, alpha))
    return alpha, 1.0 - alpha


def solve_mgda_frank_wolfe(
    grads: List[torch.Tensor],
    max_iters: int = 20,
    tol: float = 1e-5,
) -> np.ndarray:
    """Frank-Wolfe algorithm for finding min-norm point in convex hull of gradient vectors:

    min_{alpha in Delta_T} || sum_t alpha_t g_t ||_2^2
    """
    T = len(grads)
    if T == 2:
        a1, a2 = solve_mgda_2task_analytical(grads[0], grads[1])
        return np.array([a1, a2], dtype=np.float32)

    # Initialize with uniform weights
    alpha = np.ones(T, dtype=np.float32) / T

    for _ in range(max_iters):
        # Current combined gradient
        current_g = sum(alpha[i] * grads[i] for i in range(T))
        # Linear subproblem: find corner vertex argmin_t <g_t, current_g>
        inner_prods = [float(torch.dot(g, current_g).item()) for g in grads]
        i_star = int(np.argmin(inner_prods))

        # Vertex direction d = e_{i_star} - alpha
        # Line search for step gamma in [0, 1]
        v_diff = grads[i_star] - current_g
        v_norm_sq = float(torch.dot(v_diff, v_diff).item())
        if v_norm_sq < 1e-12:
            break

        gamma = -float(torch.dot(current_g, v_diff).item()) / v_norm_sq
        gamma = max(0.0, min(1.0, gamma))

        if gamma < tol:
            break

        alpha = (1.0 - gamma) * alpha
        alpha[i_star] += gamma

    return alpha


def generate_multitask_data(
    num_samples: int = 1000,
    in_features: int = 16,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic 2-task dataset: (X, y_regression, y_classification)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((num_samples, in_features)).astype(np.float32)

    w1 = rng.standard_normal(in_features).astype(np.float32)
    y_reg = (X @ w1 + 0.1 * rng.standard_normal(num_samples)).astype(np.float32)[:, None]

    w2 = rng.standard_normal(in_features).astype(np.float32)
    y_cls = ((X @ w2) > 0.0).astype(np.float32)[:, None]

    return torch.from_numpy(X), torch.from_numpy(y_reg), torch.from_numpy(y_cls)


@dataclass
class MultiTaskTrainingLog:
    step: int
    task1_loss: float
    task2_loss: float
    total_loss: float
    weight_t1: float
    weight_t2: float
    grad_norm_t1: float
    grad_norm_t2: float
    cosine_sim: float


def train_multitask_uniform(
    model: MultiTaskNet,
    X: torch.Tensor,
    y_reg: torch.Tensor,
    y_cls: torch.Tensor,
    num_steps: int = 100,
    lr: float = 0.01,
) -> List[MultiTaskTrainingLog]:
    """Train with uniform (equal) task weighting: L_total = L1 + L2."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn_reg = nn.MSELoss()
    loss_fn_cls = nn.BCEWithLogitsLoss()
    logs: List[MultiTaskTrainingLog] = []

    for step in range(num_steps):
        optimizer.zero_grad()
        out1, out2 = model(X)
        l1 = loss_fn_reg(out1, y_reg)
        l2 = loss_fn_cls(out2, y_cls)

        total_loss = l1 + l2
        total_loss.backward()

        # Compute gradient norm of shared trunk
        shared_params = model.get_shared_parameters()
        grad_norm = float(torch.norm(torch.cat([p.grad.flatten() for p in shared_params if p.grad is not None])).item())

        optimizer.step()

        if (step + 1) % 10 == 0 or step == 0:
            logs.append(
                MultiTaskTrainingLog(
                    step=step + 1,
                    task1_loss=float(l1.item()),
                    task2_loss=float(l2.item()),
                    total_loss=float(total_loss.item()),
                    weight_t1=0.5,
                    weight_t2=0.5,
                    grad_norm_t1=grad_norm,
                    grad_norm_t2=grad_norm,
                    cosine_sim=0.0,
                )
            )

    return logs


def train_multitask_mgda(
    model: MultiTaskNet,
    X: torch.Tensor,
    y_reg: torch.Tensor,
    y_cls: torch.Tensor,
    num_steps: int = 100,
    lr: float = 0.01,
) -> List[MultiTaskTrainingLog]:
    """Train with MGDA (Multiple Gradient Descent Algorithm) on shared trunk parameters."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn_reg = nn.MSELoss()
    loss_fn_cls = nn.BCEWithLogitsLoss()
    logs: List[MultiTaskTrainingLog] = []

    for step in range(num_steps):
        # 1. Forward pass
        out1, out2 = model(X)
        l1 = loss_fn_reg(out1, y_reg)
        l2 = loss_fn_cls(out2, y_cls)

        # 2. Compute gradients for task 1
        optimizer.zero_grad()
        l1.backward(retain_graph=True)
        g1_shared = torch.cat([p.grad.flatten().clone() for p in model.get_shared_parameters()])

        # 3. Compute gradients for task 2
        optimizer.zero_grad()
        l2.backward(retain_graph=True)
        g2_shared = torch.cat([p.grad.flatten().clone() for p in model.get_shared_parameters()])

        # 4. Solve for Pareto-optimal combination weights alpha
        a1, a2 = solve_mgda_2task_analytical(g1_shared, g2_shared)

        # 5. Combined backward pass
        optimizer.zero_grad()
        combined_loss = a1 * l1 + a2 * l2
        combined_loss.backward()
        optimizer.step()

        # Cosine similarity between task gradients
        norm_prod = (torch.norm(g1_shared) * torch.norm(g2_shared)).item()
        cos_sim = float((torch.dot(g1_shared, g2_shared) / max(1e-9, norm_prod)).item())

        if (step + 1) % 10 == 0 or step == 0:
            logs.append(
                MultiTaskTrainingLog(
                    step=step + 1,
                    task1_loss=float(l1.item()),
                    task2_loss=float(l2.item()),
                    total_loss=float(combined_loss.item()),
                    weight_t1=float(a1),
                    weight_t2=float(a2),
                    grad_norm_t1=float(torch.norm(g1_shared).item()),
                    grad_norm_t2=float(torch.norm(g2_shared).item()),
                    cosine_sim=cos_sim,
                )
            )

    return logs


def train_multitask_gradnorm(
    model: MultiTaskNet,
    X: torch.Tensor,
    y_reg: torch.Tensor,
    y_cls: torch.Tensor,
    num_steps: int = 100,
    lr: float = 0.01,
    alpha_gradnorm: float = 0.5,
) -> List[MultiTaskTrainingLog]:
    """Train with GradNorm (Chen et al. 2018) dynamic gradient balancing."""
    optimizer_model = optim.Adam(model.parameters(), lr=lr)

    # Task weights w_t initialized to 1.0 (as nn.Parameter for dynamic updating)
    task_weights = torch.tensor([1.0, 1.0], requires_grad=True)
    optimizer_weights = optim.Adam([task_weights], lr=0.02)

    loss_fn_reg = nn.MSELoss()
    loss_fn_cls = nn.BCEWithLogitsLoss()
    logs: List[MultiTaskTrainingLog] = []

    # Initial losses L(0)
    with torch.no_grad():
        init_out1, init_out2 = model(X)
        l1_init = float(loss_fn_reg(init_out1, y_reg).item())
        l2_init = float(loss_fn_cls(init_out2, y_cls).item())

    for step in range(num_steps):
        # Forward
        out1, out2 = model(X)
        l1 = loss_fn_reg(out1, y_reg)
        l2 = loss_fn_cls(out2, y_cls)

        # Weighted loss
        # Note: weights are detached for model update, or used directly
        w1, w2 = task_weights[0], task_weights[1]
        weighted_loss = w1 * l1 + w2 * l2

        optimizer_model.zero_grad()
        weighted_loss.backward(retain_graph=True)

        # Compute gradient norms at shared layer W
        # Grad norm G_W^{(t)} = w_t * || nabla_W L_t ||
        shared_W = model.shared_layer.weight
        g1_norm = torch.norm(torch.autograd.grad(l1, shared_W, retain_graph=True)[0]) * w1
        g2_norm = torch.norm(torch.autograd.grad(l2, shared_W, retain_graph=True)[0]) * w2

        # Mean gradient norm
        g_bar = (g1_norm + g2_norm) / 2.0

        # Relative inverse training rate r_t(t) = (L_t(t) / L_t(0)) / mean(L / L_0)
        l1_ratio = float(l1.item()) / max(1e-8, l1_init)
        l2_ratio = float(l2.item()) / max(1e-8, l2_init)
        mean_ratio = (l1_ratio + l2_ratio) / 2.0

        r1 = l1_ratio / max(1e-8, mean_ratio)
        r2 = l2_ratio / max(1e-8, mean_ratio)

        # Target norms
        target_g1 = (g_bar.detach()) * (r1 ** alpha_gradnorm)
        target_g2 = (g_bar.detach()) * (r2 ** alpha_gradnorm)

        # GradNorm loss: L_grad = |G1 - target1| + |G2 - target2|
        l_grad = torch.abs(g1_norm - target_g1) + torch.abs(g2_norm - target_g2)

        # Update model parameters
        optimizer_model.step()

        # Update weights using GradNorm loss
        optimizer_weights.zero_grad()
        l_grad.backward()
        optimizer_weights.step()

        # Renormalize weights so sum(w) = T = 2
        with torch.no_grad():
            task_weights.data = torch.clamp(task_weights.data, min=0.01)
            task_weights.data = task_weights.data / task_weights.data.sum() * 2.0

        if (step + 1) % 10 == 0 or step == 0:
            logs.append(
                MultiTaskTrainingLog(
                    step=step + 1,
                    task1_loss=float(l1.item()),
                    task2_loss=float(l2.item()),
                    total_loss=float(weighted_loss.item()),
                    weight_t1=float(task_weights[0].item()),
                    weight_t2=float(task_weights[1].item()),
                    grad_norm_t1=float(g1_norm.item()),
                    grad_norm_t2=float(g2_norm.item()),
                    cosine_sim=0.0,
                )
            )

    return logs
