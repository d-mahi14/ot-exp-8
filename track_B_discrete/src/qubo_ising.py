"""
qubo_ising.py
QUBO (Quadratic Unconstrained Binary Optimization) and Ising Model Formulations:
1. Exact algebraic binary-to-spin mapping: s_i = 1 - 2*x_i <==> x_i = (1 - s_i)/2
2. Graph Maximum Cut (Max-Cut) formulated as QUBO and equivalent Ising Hamiltonian
3. Feature Selection formulated as a constrained QUBO with penalty term P*(sum(x_i) - K)^2
4. Exact Brute-Force ground state solver (ground truth for N <= 20)
5. Classical Simulated Annealing solver with O(N) single-bit-flip energy delta evaluation

Establishes the classical optimization baseline directly consumed by Track C (QAOA/VQE).
"""

import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class QUBOProblem:
    """
    Quadratic Unconstrained Binary Optimization (QUBO) Problem:
        Objective: Minimize E(x) = x^T Q x + C
        where x in {0, 1}^n, Q in R^(n x n), and C in R is a constant energy offset.
    """
    def __init__(self, Q: np.ndarray, constant: float = 0.0):
        self.Q = np.asarray(Q, dtype=np.float64)
        if self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError(f"Q must be a square 2D matrix, got shape {self.Q.shape}")
        self.n = self.Q.shape[0]
        self.constant = float(constant)

    def evaluate_energy(self, x: np.ndarray) -> float:
        """Computes total energy E(x) = x^T Q x + C."""
        x = np.asarray(x, dtype=np.float64)
        if len(x) != self.n:
            raise ValueError(f"State x length {len(x)} does not match problem dimension {self.n}")
        return float(x.T @ self.Q @ x + self.constant)

    def to_symmetric(self) -> 'QUBOProblem':
        """Converts Q to symmetric form: Q_sym = 0.5 * (Q + Q^T)."""
        Q_sym = 0.5 * (self.Q + self.Q.T)
        return QUBOProblem(Q_sym, self.constant)

    def to_upper_triangular(self) -> 'QUBOProblem':
        """Converts Q to an upper-triangular matrix with identical energy for all x."""
        Q_tri = np.zeros_like(self.Q)
        for i in range(self.n):
            Q_tri[i, i] = self.Q[i, i]
            for j in range(i + 1, self.n):
                Q_tri[i, j] = self.Q[i, j] + self.Q[j, i]
        return QUBOProblem(Q_tri, self.constant)

    def to_ising(self) -> 'IsingModel':
        """
        Converts QUBO to an equivalent classical Ising Spin Model using the exact substitution:
            x_i = (1 - s_i) / 2,   where s_i in {+1, -1}
        
        Derivation:
            x_i * x_j = (1 - s_i - s_j + s_i * s_j) / 4
            x_i^2 = x_i = (1 - s_i) / 2
            
            E(x) = sum_i Q_ii * (1 - s_i)/2 + sum_{i < j} (Q_ij + Q_ji) * (1 - s_i - s_j + s_i*s_j)/4 + C
                 = sum_{i < j} J_ij * s_i * s_j + sum_i h_i * s_i + C_ising
        """
        Q_tri = self.to_upper_triangular().Q
        n = self.n
        J = np.zeros((n, n), dtype=np.float64)
        h = np.zeros(n, dtype=np.float64)
        c_ising = self.constant

        # Diagonal contributions
        for i in range(n):
            q_ii = Q_tri[i, i]
            c_ising += q_ii / 2.0
            h[i] -= q_ii / 2.0

        # Off-diagonal contributions
        for i in range(n):
            for j in range(i + 1, n):
                q_ij = Q_tri[i, j]
                J[i, j] = q_ij / 4.0
                h[i] -= q_ij / 4.0
                h[j] -= q_ij / 4.0
                c_ising += q_ij / 4.0

        return IsingModel(J=J, h=h, constant=c_ising)


class IsingModel:
    """
    Classical Ising Spin Model:
        H(s) = sum_{i < j} J_ij * s_i * s_j + sum_{i=1}^n h_i * s_i + C_ising
        where s in {+1, -1}^n, J is an upper-triangular coupling matrix,
        h is the external magnetic field vector, and C_ising is a constant energy offset.
    """
    def __init__(self, J: np.ndarray, h: np.ndarray, constant: float = 0.0):
        self.J = np.asarray(J, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)
        self.n = len(self.h)
        self.constant = float(constant)

    def evaluate_energy(self, s: np.ndarray) -> float:
        """Computes Ising Hamiltonian energy H(s)."""
        s = np.asarray(s, dtype=np.float64)
        if len(s) != self.n:
            raise ValueError(f"Spin state length {len(s)} does not match dimension {self.n}")
        # sum_{i < j} J_ij * s_i * s_j
        coupling_energy = float(np.sum(np.triu(self.J, k=1) * np.outer(s, s)))
        field_energy = float(np.dot(self.h, s))
        return coupling_energy + field_energy + self.constant

    def to_qubo(self) -> QUBOProblem:
        """
        Converts Ising model back to QUBO using the inverse substitution:
            s_i = 1 - 2*x_i
        """
        n = self.n
        Q = np.zeros((n, n), dtype=np.float64)
        c_qubo = self.constant + np.sum(self.h) + np.sum(np.triu(self.J, k=1))

        # Linear field contribution: h_i * (1 - 2*x_i)
        for i in range(n):
            Q[i, i] -= 2.0 * self.h[i]

        # Quadratic coupling contribution: J_ij * (1 - 2*x_i) * (1 - 2*x_j) = J_ij * (1 - 2*x_i - 2*x_j + 4*x_i*x_j)
        for i in range(n):
            for j in range(i + 1, n):
                j_ij = self.J[i, j]
                Q[i, i] -= 2.0 * j_ij
                Q[j, j] -= 2.0 * j_ij
                Q[i, j] += 4.0 * j_ij

        return QUBOProblem(Q, c_qubo).to_upper_triangular()


# -------------------------------------------------------------
# Problem Formulations
# -------------------------------------------------------------

def maxcut_to_qubo(adjacency_matrix: np.ndarray) -> QUBOProblem:
    """
    Formulates Graph Maximum Cut (Max-Cut) as a QUBO problem.

    Problem:
        Maximize Cut(x) = sum_{(i,j) in E} W_ij * (x_i + x_j - 2*x_i*x_j)
        where x_i in {0, 1} indicates set membership (partition S vs S_bar).

    QUBO Minimization:
        Minimize E(x) = - Cut(x) = sum_{(i,j) in E} W_ij * (2*x_i*x_j - x_i - x_j)
        
    Matrix Construction:
        Q_ii = - sum_{j != i} W_ij  (negative weighted vertex degree)
        Q_ij = W_ij                 (for i < j, with factor 2 absorbed in upper triangle)
    """
    W = np.asarray(adjacency_matrix, dtype=np.float64)
    n = W.shape[0]
    Q = np.zeros((n, n), dtype=np.float64)

    degrees = np.sum(W, axis=1)
    for i in range(n):
        Q[i, i] = -degrees[i]
        for j in range(i + 1, n):
            if W[i, j] != 0:
                Q[i, j] = 2.0 * W[i, j]

    return QUBOProblem(Q, constant=0.0)


def maxcut_to_ising(adjacency_matrix: np.ndarray) -> IsingModel:
    """
    Direct derivation of the classical Ising Spin Hamiltonian for Max-Cut:
        Cut(s) = 0.5 * sum_{(i,j) in E} W_ij * (1 - s_i * s_j)
        Minimizing - Cut(s) gives:
        H(s) = sum_{i < j} (0.5 * W_ij) * s_i * s_j - 0.5 * sum_{i < j} W_ij
        where:
            J_ij = 0.5 * W_ij
            h_i  = 0.0          (zero external magnetic field)
            C_0  = - 0.5 * sum_{i < j} W_ij
    """
    W = np.asarray(adjacency_matrix, dtype=np.float64)
    n = W.shape[0]
    J = np.triu(0.5 * W, k=1)
    h = np.zeros(n, dtype=np.float64)
    c_0 = -0.5 * float(np.sum(np.triu(W, k=1)))
    return IsingModel(J=J, h=h, constant=c_0)


def feature_selection_to_qubo(
    relevance: np.ndarray,
    redundancy: np.ndarray,
    target_k: int,
    penalty_p: float,
    alpha: float = 1.0
) -> QUBOProblem:
    """
    Formulates Minimum Redundancy Maximum Relevance (mRMR) Feature Selection as a QUBO:
    
    Objective:
        Maximize relevance while minimizing pairwise collinearity:
            min_x  - sum_{i=1}^D r_i * x_i + alpha * sum_{i < j} c_ij * x_i * x_j
            subject to sum_{i=1}^D x_i = K
            
    Penalty Expansion:
        P * (sum x_i - K)^2 = P * [ sum x_i^2 + 2*sum_{i < j} x_i*x_j - 2K*sum x_i + K^2 ]
        Since x_i^2 = x_i:
        P * (sum x_i - K)^2 = P*(1 - 2K)*sum x_i + 2P*sum_{i < j} x_i*x_j + P*K^2
        
    Combined Matrix Q:
        Q_ii = - r_i + P * (1 - 2K)
        Q_ij = alpha * c_ij + 2 * P   (for i < j)
        Constant C = P * K^2
    """
    r = np.asarray(relevance, dtype=np.float64)
    C_mat = np.asarray(redundancy, dtype=np.float64)
    D = len(r)
    K = target_k
    P = float(penalty_p)

    Q = np.zeros((D, D), dtype=np.float64)

    # Diagonal
    for i in range(D):
        Q[i, i] = -r[i] + P * (1.0 - 2.0 * K)

    # Off-diagonal
    for i in range(D):
        for j in range(i + 1, D):
            Q[i, j] = alpha * C_mat[i, j] + 2.0 * P

    constant = P * (K**2)
    return QUBOProblem(Q, constant=constant)


# -------------------------------------------------------------
# Solvers: Brute-Force & Simulated Annealing
# -------------------------------------------------------------

def solve_qubo_brute_force(qubo: QUBOProblem) -> Dict[str, Any]:
    """
    Exact Brute-Force Solver: evaluates all 2^n binary bitstrings.
    Provides verifiable mathematical ground truth for small n <= 20.
    """
    t0 = time.perf_counter()
    n = qubo.n
    if n > 22:
        raise ValueError(f"Brute force intractable for n={n} (> 2^22 states)")

    Q = qubo.to_symmetric().Q
    c = qubo.constant

    best_energy = float("inf")
    best_state = None
    all_ground_states = []

    total_states = 1 << n
    for i in range(total_states):
        # Generate bitstring
        x = np.array([(i >> bit) & 1 for bit in range(n)], dtype=np.float64)
        energy = float(x.T @ Q @ x + c)

        if energy < best_energy - 1e-9:
            best_energy = energy
            best_state = x
            all_ground_states = [x]
        elif abs(energy - best_energy) <= 1e-9:
            all_ground_states.append(x)

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "best_state": best_state,
        "best_energy": best_energy,
        "runtime_ms": runtime_ms,
        "num_evaluated": total_states,
        "all_ground_states": all_ground_states,
        "is_exact": True
    }


def solve_qubo_simulated_annealing(
    qubo: QUBOProblem,
    initial_temp: float = 10.0,
    final_temp: float = 0.01,
    cooling_rate: float = 0.99,
    steps_per_temp: int = 50,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Classical Simulated Annealing for QUBO with O(n) single-bit-flip delta evaluation:
    
    Delta Energy Derivation:
        Let x' be x with bit k flipped: x'_k = 1 - x_k.
        Delta_E = E(x') - E(x) = (1 - 2*x_k) * [ Q_kk + sum_{j != k} (Q_kj + Q_jk) * x_j ]
        Evaluated in O(n) instead of O(n^2).
    """
    t0 = time.perf_counter()
    rng = np.random.RandomState(seed)
    n = qubo.n
    
    # Work with symmetric matrix for clean delta computation
    Q_sym = qubo.to_symmetric().Q
    c = qubo.constant

    # Random initial state
    x = rng.randint(0, 2, size=n).astype(np.float64)
    current_energy = float(x.T @ Q_sym @ x + c)

    best_state = x.copy()
    best_energy = current_energy

    T = float(initial_temp)
    total_steps = 0
    accepted_steps = 0

    while T > final_temp:
        for _ in range(steps_per_temp):
            total_steps += 1
            # Pick random bit to flip
            k = rng.randint(0, n)
            
            # Efficient O(n) delta energy computation
            # Delta_x_k = (1 - 2*x_k)
            delta_xk = 1.0 - 2.0 * x[k]
            # row_sum = 2 * sum_{j != k} Q_sym[k, j] * x_j
            row_sum = 2.0 * (np.dot(Q_sym[k], x) - Q_sym[k, k] * x[k])
            delta_E = delta_xk * (Q_sym[k, k] + row_sum)

            # Metropolis acceptance criterion
            if delta_E < 0 or rng.uniform(0.0, 1.0) < np.exp(-delta_E / T):
                x[k] = 1.0 - x[k]
                current_energy += delta_E
                accepted_steps += 1

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = x.copy()

        T *= cooling_rate

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "best_state": best_state,
        "best_energy": best_energy,
        "runtime_ms": runtime_ms,
        "total_steps": total_steps,
        "accepted_steps": accepted_steps,
        "acceptance_ratio": accepted_steps / max(1, total_steps)
    }
