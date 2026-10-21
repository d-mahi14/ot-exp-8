"""
qaoa_maxcut.py
Track C — Batch 1: Quantum Optimization Foundations and QAOA for Graph Max-Cut.

Theoretical Foundations:
1. Classical Max-Cut Objective:
   Given graph G = (V, E) with edge weights w_{ij}, maximize:
       C(x) = sum_{(i, j) in E} w_{ij} * (x_i + x_j - 2*x_i*x_j)
   where x_i in {0, 1}.

2. Ising / Pauli Hamiltonian Formulation:
   Via the spin substitution s_i = 1 - 2*x_i (or x_i = (1 - s_i)/2) and Pauli Z operator (Z |0> = |0>, Z |1> = -|1>):
       C_op = sum_{(i, j) in E} (w_{ij} / 2) * (I - Z_i Z_j)
   Maximizing C(x) is equivalent to minimizing the Cost Hamiltonian:
       H_C = sum_{(i, j) in E} (w_{ij} / 2) * Z_i Z_j

3. QAOA Ansatz (Alternating Operator Layers):
   Starting from the uniform superposition state |psi_0> = H^(tensor n) |0>^n:
       |psi(gamma, beta)> = prod_{l=1}^p [ U(B, beta_l) * U(C, gamma_l) ] |psi_0>
   where:
       - Problem unitary: U(C, gamma) = exp(-i * gamma * H_C)
         Implemented on each edge (i, j) by an R_ZZ(-gamma * w_{ij}) gate.
       - Mixer unitary: U(B, beta) = exp(-i * beta * H_M) with H_M = sum_i X_i
         Implemented on each qubit i by an R_X(2 * beta) gate.

4. Role of Parameter p (Circuit Depth):
   - At p -> inf, QAOA approaches continuous quantum adiabatic evolution,
     guaranteeing convergence to the true global ground state.
   - For finite p (e.g. p = 1, 2, 3), higher p expands ansatz expressiveness
     and improves the theoretical approximation ratio (e.g. >= 0.6924 for 3-regular at p=1),
     at the cost of increased circuit depth, CNOT count, and gate error exposure on NISQ hardware.

5. Bitstring Little-Endian Ordering Convention:
   Qiskit stores classical bitstrings in little-endian order: string 'b_{n-1}...b_1 b_0',
   where character index 0 corresponds to qubit (n - 1) and index (n - 1) corresponds to qubit 0.
   All decoding logic explicitly reverses or maps indices correctly.
"""

import time
from typing import Dict, List, Tuple, Any, Optional, Union, Sequence
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

# Reusable classical baselines from Track B
from track_B_discrete.src.qubo_ising import (
    QUBOProblem,
    IsingModel,
    maxcut_to_qubo,
    maxcut_to_ising,
    solve_qubo_brute_force,
    solve_qubo_simulated_annealing,
)


def parse_qiskit_bitstring(bitstring_str: str) -> np.ndarray:
    """
    Parses a Qiskit measurement bitstring string into a 0/1 numpy array where
    index i corresponds to qubit i.
    
    Qiskit format: 'q_{n-1} ... q_1 q_0'.
    Reversing the string maps character index i to qubit index i.
    """
    clean_str = bitstring_str.replace(" ", "")
    return np.array([int(b) for b in reversed(clean_str)], dtype=np.int32)


class CostHamiltonian:
    """
    Represents the Cost Hamiltonian for Graph Max-Cut.
    
    Attributes:
        n_qubits (int): Number of vertices / qubits.
        edges (List[Tuple[int, int, float]]): Weighted edges (u, v, w).
        adjacency_matrix (np.ndarray): Symmetric adjacency matrix.
        J (np.ndarray): Ising coupling matrix J_ij = 0.5 * w_ij.
        constant_offset (float): Energy constant offset -0.5 * sum w_ij.
    """
    def __init__(self, adjacency_matrix: np.ndarray):
        W = np.asarray(adjacency_matrix, dtype=np.float64)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got {W.shape}")
        self.adjacency_matrix = 0.5 * (W + W.T)  # Ensure exact symmetry
        self.n_qubits = self.adjacency_matrix.shape[0]

        # Extract edges (u < v with non-zero weight)
        self.edges: List[Tuple[int, int, float]] = []
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                w = float(self.adjacency_matrix[i, j])
                if abs(w) > 1e-12:
                    self.edges.append((i, j, w))

        self.num_edges = len(self.edges)
        self.total_weight = sum(w for _, _, w in self.edges)

        # Ising parameters matching Track B maxcut_to_ising
        self.J = np.triu(0.5 * self.adjacency_matrix, k=1)
        self.h = np.zeros(self.n_qubits, dtype=np.float64)
        self.constant_offset = -0.5 * self.total_weight

    @classmethod
    def from_edges(cls, num_nodes: int, edges: Sequence[Tuple[int, int, float]]) -> 'CostHamiltonian':
        """Constructs Hamiltonian directly from an edge list."""
        W = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for u, v, w in edges:
            W[u, v] += float(w)
            W[v, u] += float(w)
        return cls(W)

    def evaluate_cut(self, state: Union[str, np.ndarray, Sequence[int]]) -> float:
        """
        Calculates the classical cut value for a binary state vector x in {0, 1}^n:
            Cut(x) = sum_{(u, v) in E} w_{uv} * (x_u != x_v)
        """
        if isinstance(state, str):
            x = parse_qiskit_bitstring(state)
        else:
            x = np.asarray(state, dtype=np.int32)
        if len(x) != self.n_qubits:
            raise ValueError(f"State dimension {len(x)} does not match {self.n_qubits}")

        cut = 0.0
        for u, v, w in self.edges:
            if x[u] != x[v]:
                cut += w
        return float(cut)

    def evaluate_energy(self, state: Union[str, np.ndarray, Sequence[int]]) -> float:
        """
        Calculates energy E(x) = -Cut(x). Minimizing energy maximizes the cut.
        Matches Track B QUBO energy.
        """
        return -self.evaluate_cut(state)

    def to_qubo(self) -> QUBOProblem:
        """Constructs the equivalent Track B QUBO problem for cross-validation."""
        return maxcut_to_qubo(self.adjacency_matrix)

    def to_ising(self) -> IsingModel:
        """Constructs the equivalent Track B Ising model."""
        return maxcut_to_ising(self.adjacency_matrix)


def build_qaoa_circuit(
    hamiltonian: CostHamiltonian,
    p: int,
    gammas: Sequence[float],
    betas: Sequence[float],
    measure: bool = False
) -> QuantumCircuit:
    """
    Constructs the parameterized QAOA quantum circuit for Graph Max-Cut.
    
    Circuit Structure:
    1. Initial Layer: Hadamard H on all n qubits (|psi_0> = |+>^n).
    2. Alternating Layers (l = 0 .. p-1):
       a. Cost Unitary U(C, gamma_l):
          For each edge (u, v) with weight w_{uv}:
              R_ZZ(-gamma_l * w_{uv}) on qubits u and v.
       b. Mixer Unitary U(B, beta_l):
          For each qubit u:
              R_X(2 * beta_l) on qubit u.
    3. Optional Measurement: Measurement of all n qubits into n classical bits.
    
    Args:
        hamiltonian: CostHamiltonian containing graph edges and qubit count.
        p: Number of alternating QAOA layers (circuit depth parameter).
        gammas: Array or list of p cost layer angles [gamma_1, ..., gamma_p].
        betas: Array or list of p mixer layer angles [beta_1, ..., beta_p].
        measure: If True, adds classical register and measurement gates.
        
    Returns:
        QuantumCircuit: Constructed Qiskit circuit.
    """
    if len(gammas) != p or len(betas) != p:
        raise ValueError(f"Expected {p} gammas and {p} betas, got {len(gammas)} and {len(betas)}")

    n = hamiltonian.n_qubits
    qc = QuantumCircuit(n, n) if measure else QuantumCircuit(n)

    # Step 1: Initial state |+>^n
    qc.h(range(n))

    # Step 2: Alternating Cost and Mixer layers
    for layer in range(p):
        gamma_l = float(gammas[layer])
        beta_l = float(betas[layer])

        # Cost layer: exp(-i * gamma * H_C)
        # Using R_ZZ(-gamma * w_{uv})
        for u, v, w in hamiltonian.edges:
            qc.rzz(-gamma_l * w, u, v)

        # Mixer layer: exp(-i * beta * sum X_i)
        # Using R_X(2 * beta)
        for u in range(n):
            qc.rx(2.0 * beta_l, u)

    # Step 3: Measurement
    if measure:
        qc.measure(range(n), range(n))

    return qc


class QAOAMaxCutSolver:
    """
    Quantum Approximate Optimization Algorithm (QAOA) Solver for Graph Max-Cut.
    
    Executes on Qiskit Aer simulator with classical variational parameter optimization.
    """
    def __init__(
        self,
        hamiltonian: CostHamiltonian,
        p: int = 1,
        simulator: Optional[AerSimulator] = None,
        seed: int = 42
    ):
        self.hamiltonian = hamiltonian
        self.p = int(p)
        if self.p < 1:
            raise ValueError(f"QAOA depth p must be >= 1, got {self.p}")
        self.seed = int(seed)
        self.simulator = simulator if simulator is not None else AerSimulator(seed_simulator=self.seed)

        self.optimal_angles: Optional[np.ndarray] = None
        self.optimal_energy: Optional[float] = None
        self.optimization_result: Optional[Dict[str, Any]] = None

    def get_circuit_metrics(self) -> Dict[str, Any]:
        """Calculates circuit depth, 2-qubit gate count, and total gate count."""
        dummy_angles = [0.5] * self.p
        qc = build_qaoa_circuit(self.hamiltonian, self.p, dummy_angles, dummy_angles, measure=False)
        tqc = transpile(qc, self.simulator)
        ops = dict(tqc.count_ops())
        num_rzz = ops.get("rzz", 0)
        num_cx = ops.get("cx", 0)
        return {
            "num_qubits": self.hamiltonian.n_qubits,
            "p": self.p,
            "depth": int(tqc.depth()),
            "num_2q_gates": int(num_rzz + num_cx),
            "total_gates": int(sum(ops.values())),
            "gate_breakdown": ops
        }

    def compute_expectation(
        self,
        angles: Sequence[float],
        method: str = "statevector",
        shots: int = 1024
    ) -> float:
        """
        Computes the expected energy <E> = -<Cut> for a given parameter vector:
            angles = [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
            
        Minimizing <E> is equivalent to maximizing the expected cut <Cut>.
        """
        angles_arr = np.asarray(angles, dtype=np.float64)
        gammas = angles_arr[:self.p]
        betas = angles_arr[self.p:]

        if method == "statevector":
            qc = build_qaoa_circuit(self.hamiltonian, self.p, gammas, betas, measure=False)
            sv = Statevector.from_instruction(qc)
            probs = sv.probabilities_dict()

            expected_cut = 0.0
            for bitstr, prob in probs.items():
                cut_val = self.hamiltonian.evaluate_cut(bitstr)
                expected_cut += float(prob) * cut_val

            return -expected_cut  # Return negative cut (energy to minimize)

        elif method == "shots":
            qc = build_qaoa_circuit(self.hamiltonian, self.p, gammas, betas, measure=True)
            tqc = transpile(qc, self.simulator)
            job = self.simulator.run(tqc, shots=shots)
            counts = job.result().get_counts()

            expected_cut = 0.0
            for bitstr, count in counts.items():
                prob = count / shots
                cut_val = self.hamiltonian.evaluate_cut(bitstr)
                expected_cut += prob * cut_val

            return -expected_cut

        else:
            raise ValueError(f"Unknown expectation method: {method}")

    def optimize(
        self,
        initial_angles: Optional[Sequence[float]] = None,
        method: str = "COBYLA",
        maxiter: int = 100,
        expectation_method: str = "statevector",
        shots: int = 1024
    ) -> Dict[str, Any]:
        """
        Optimizes the 2p classical variational parameters (gammas, betas)
        using a classical optimizer from scipy.optimize.minimize.
        """
        t0 = time.perf_counter()

        if initial_angles is None:
            # Standard heuristic initialization: gamma in [0, pi], beta in [0, pi/2]
            rng = np.random.RandomState(self.seed)
            init_gammas = rng.uniform(0.1, np.pi / 2.0, size=self.p)
            init_betas = rng.uniform(0.1, np.pi / 4.0, size=self.p)
            x0 = np.concatenate([init_gammas, init_betas])
        else:
            x0 = np.asarray(initial_angles, dtype=np.float64)
            if len(x0) != 2 * self.p:
                raise ValueError(f"Expected 2*p = {2*self.p} angles, got {len(x0)}")

        history: List[float] = []

        def objective(x):
            val = self.compute_expectation(x, method=expectation_method, shots=shots)
            history.append(float(val))
            return val

        res = minimize(
            objective,
            x0,
            method=method,
            options={"maxiter": maxiter}
        )

        opt_time_ms = (time.perf_counter() - t0) * 1000.0

        self.optimal_angles = np.asarray(res.x, dtype=np.float64)
        self.optimal_energy = float(res.fun)
        expected_cut = -self.optimal_energy

        self.optimization_result = {
            "success": bool(res.success),
            "optimal_angles": self.optimal_angles,
            "optimal_gammas": self.optimal_angles[:self.p],
            "optimal_betas": self.optimal_angles[self.p:],
            "expected_cut": expected_cut,
            "minimized_energy": self.optimal_energy,
            "num_evaluations": int(res.nfev) if hasattr(res, "nfev") else len(history),
            "optimization_time_ms": opt_time_ms,
            "history": history,
            "optimizer": method,
            "status_message": str(res.message)
        }
        return self.optimization_result

    def sample_solutions(
        self,
        shots: int = 2048,
        angles: Optional[Sequence[float]] = None
    ) -> Dict[str, Any]:
        """
        Samples bitstrings from the QAOA circuit using Qiskit Aer simulator
        at the optimal (or provided) angles.
        """
        t0 = time.perf_counter()

        if angles is None:
            if self.optimal_angles is None:
                raise ValueError("Must run optimize() first or provide angles explicitly")
            eval_angles = self.optimal_angles
        else:
            eval_angles = np.asarray(angles, dtype=np.float64)

        gammas = eval_angles[:self.p]
        betas = eval_angles[self.p:]

        qc = build_qaoa_circuit(self.hamiltonian, self.p, gammas, betas, measure=True)
        tqc = transpile(qc, self.simulator)
        job = self.simulator.run(tqc, shots=shots)
        counts = job.result().get_counts()

        runtime_ms = (time.perf_counter() - t0) * 1000.0

        # Evaluate cut for all sampled bitstrings
        best_cut = -float("inf")
        best_bitstring = ""
        total_cut = 0.0
        cut_counts: Dict[float, int] = {}
        all_samples: List[Dict[str, Any]] = []

        for bitstr, count in counts.items():
            cut_val = self.hamiltonian.evaluate_cut(bitstr)
            total_cut += cut_val * count
            cut_counts[cut_val] = cut_counts.get(cut_val, 0) + count

            if cut_val > best_cut:
                best_cut = cut_val
                best_bitstring = bitstr

            all_samples.append({
                "bitstring": bitstr,
                "parsed_state": parse_qiskit_bitstring(bitstr),
                "count": count,
                "probability": count / shots,
                "cut": cut_val
            })

        # Sort samples by probability descending
        all_samples.sort(key=lambda s: s["count"], reverse=True)

        expected_cut = total_cut / shots
        most_frequent_bitstring = all_samples[0]["bitstring"]

        return {
            "best_cut": best_cut,
            "best_bitstring": best_bitstring,
            "best_state": parse_qiskit_bitstring(best_bitstring),
            "expected_cut": expected_cut,
            "most_frequent_bitstring": most_frequent_bitstring,
            "counts": counts,
            "cut_distribution": cut_counts,
            "top_samples": all_samples[:10],
            "shots": shots,
            "sample_runtime_ms": runtime_ms
        }


def explain_qaoa_depth_and_p() -> str:
    """
    Returns an academic, mathematically rigorous explanation of circuit depth
    and parameter p in QAOA for viva defense.
    """
    return (
        "### Mathematical & Physical Role of QAOA Depth Parameter p\n\n"
        "1. **Trotterized Adiabatic Analogy**:\n"
        "   QAOA is a variational discretisation of the Quantum Adiabatic Algorithm (QAA).\n"
        "   In QAA, the system starts in the ground state of mixer H_M and slowly interpolates\n"
        "   to cost H_C according to H(t) = (1 - t/T) H_M + (t/T) H_C over long evolution time T.\n"
        "   QAOA discretises this continuous evolution into p alternating unitary slices:\n"
        "       |psi> = prod_{l=1}^p [ exp(-i * beta_l * H_M) * exp(-i * gamma_l * H_C) ] |psi_0>.\n\n"
        "2. **Asymptotic Convergence (p -> inf)**:\n"
        "   By the Adiabatic Theorem and Trotter-Suzuki decomposition, as p -> inf,\n"
        "   QAOA parameters can be chosen to match adiabatic scheduling arbitrarily closely,\n"
        "   guaranteeing 100% convergence to the true global ground state (Max-Cut optimum).\n\n"
        "3. **Finite Depth Approximation Bounds**:\n"
        "   - At p = 1 on 3-regular graphs, Farhi et al. (2014) proved QAOA achieves at least\n"
        "     a 0.6924 approximation ratio.\n"
        "   - Increasing p strictly expands the expressive capacity of the variational ansatz subspace:\n"
        "     V_1 subset V_2 subset ... subset V_p subset H.\n"
        "     Consequently, optimal expected cut <C>_{p+1} >= <C>_p is monotonically non-decreasing.\n\n"
        "4. **NISQ Hardware Constraints & Trade-Offs**:\n"
        "   - Each QAOA layer introduces |E| two-qubit entangling gates (R_ZZ / CNOT pairs).\n"
        "   - Circuit depth scales as O(p * degree(G)), increasing total decoherence exposure (T1/T2)\n"
        "     and two-qubit gate errors (~10^-3 per gate).\n"
        "   - On noisy quantum processors, an empirical threshold exists where increasing p beyond\n"
        "     a critical depth p* degrades solution quality due to cumulative gate fidelity decay.\n"
    )
