"""
vqe_ising.py
Track C — Batch 2: Variational Quantum Eigensolver (VQE) for Ising Hamiltonians.

Theoretical Foundations:
1. Rayleigh-Ritz Variational Principle:
   For any parameterized quantum state |psi(theta)>, the expectation value of
   Hamiltonian H satisfies:
       <H>_theta = <psi(theta)| H |psi(theta)> / <psi(theta)|psi(theta)> >= E_0
   where E_0 is the true ground-state energy. Equality holds iff |psi(theta)>
   is an exact ground state.

2. Ising Hamiltonian as Commuting Pauli Operators:
   For classical spin models (e.g. Graph Max-Cut from Track B):
       H = sum_{i < j} J_{ij} Z_i Z_j + sum_i h_i Z_i + C_0 I
   Since all terms consist solely of Pauli Z and I operators, all terms mutually commute:
       [Z_i Z_j, Z_k Z_l] = 0,   [Z_i Z_j, Z_k] = 0
   Consequently, unlike general electronic Hamiltonians in quantum chemistry,
   the entire Ising Hamiltonian can be measured simultaneously in a single
   computational basis (Z-basis) measurement setup without basis rotations!

3. Hardware-Efficient Ansatz (RealAmplitudes / Ry-CX):
   Because the Ising Hamiltonian is diagonal in the computational basis, all its
   eigenvectors have purely real amplitudes. Restricting the ansatz to R_y rotations:
       |psi(theta)> = prod_{l=1}^L [ U_ent * prod_{i=1}^n R_y(theta_{l, i}) ] prod_{i=1}^n R_y(theta_{0, i}) |0>^n
   limits exploration strictly to the real subspace R^(2^n) subset C^(2^n),
   greatly reducing variational parameter overhead and phase redundancy.

4. Non-Convex Landscape & Local Minima:
   The variational energy landscape <H>(theta) is non-convex. Local optimizers like
   COBYLA can encounter local minima or plateaus depending on initial parameter seeds,
   motivating multi-start trials and depth comparisons.
"""

import time
from typing import Dict, List, Tuple, Any, Optional, Union, Sequence
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

from track_B_discrete.src.qubo_ising import IsingModel, QUBOProblem, maxcut_to_ising
from track_C_quantum.src.qaoa_maxcut import parse_qiskit_bitstring


class IsingPauliHamiltonian:
    """
    Constructs and manages an Ising Hamiltonian represented as a Qiskit SparsePauliOp.
    
    Qiskit Pauli String Little-Endian Convention:
    In Qiskit, string 'P_{n-1} ... P_1 P_0' maps character index (n - 1 - i) to qubit i.
    """
    def __init__(
        self,
        n_qubits: int,
        J: np.ndarray,
        h: Optional[np.ndarray] = None,
        constant: float = 0.0
    ):
        self.n_qubits = int(n_qubits)
        self.J = np.asarray(J, dtype=np.float64)
        if self.J.shape != (self.n_qubits, self.n_qubits):
            raise ValueError(f"J shape {self.J.shape} does not match n_qubits {self.n_qubits}")
        
        self.h = np.zeros(self.n_qubits, dtype=np.float64) if h is None else np.asarray(h, dtype=np.float64)
        if len(self.h) != self.n_qubits:
            raise ValueError(f"h length {len(self.h)} does not match n_qubits {self.n_qubits}")
            
        self.constant = float(constant)

        # Build SparsePauliOp
        self.pauli_terms: List[Tuple[str, float]] = []
        n = self.n_qubits

        # Coupling terms J_ij * Z_i Z_j
        for i in range(n):
            for j in range(i + 1, n):
                coeff = float(self.J[i, j])
                if abs(coeff) > 1e-12:
                    p = ['I'] * n
                    p[n - 1 - i] = 'Z'
                    p[n - 1 - j] = 'Z'
                    self.pauli_terms.append((''.join(p), coeff))

        # Linear field terms h_i * Z_i
        for i in range(n):
            coeff = float(self.h[i])
            if abs(coeff) > 1e-12:
                p = ['I'] * n
                p[n - 1 - i] = 'Z'
                self.pauli_terms.append((''.join(p), coeff))

        # Identity constant term C_0 * I
        if abs(self.constant) > 1e-12 or len(self.pauli_terms) == 0:
            self.pauli_terms.append(('I' * n, self.constant))

        self.sparse_pauli_op = SparsePauliOp.from_list(self.pauli_terms)

    @classmethod
    def from_ising_model(cls, ising: IsingModel) -> 'IsingPauliHamiltonian':
        """Constructs directly from Track B IsingModel."""
        return cls(n_qubits=ising.n, J=ising.J, h=ising.h, constant=ising.constant)

    @classmethod
    def from_maxcut_adjacency(cls, adjacency_matrix: np.ndarray) -> 'IsingPauliHamiltonian':
        """Constructs Max-Cut Ising Hamiltonian from graph adjacency matrix."""
        ising = maxcut_to_ising(adjacency_matrix)
        return cls.from_ising_model(ising)

    def exact_ground_state(self) -> Tuple[float, np.ndarray]:
        """
        Computes exact ground-state energy and ground-state bitstring via exact
        matrix diagonalization of the diagonal Ising Hamiltonian.
        """
        diag_matrix = self.sparse_pauli_op.to_matrix()
        diag_vals = np.real(np.diag(diag_matrix))
        min_idx = int(np.argmin(diag_vals))
        ground_energy = float(diag_vals[min_idx])

        # Convert index to binary bitstring
        ground_bits = np.array([(min_idx >> b) & 1 for b in range(self.n_qubits)], dtype=np.int32)
        return ground_energy, ground_bits

    def evaluate_bitstring_energy(self, state: Union[str, np.ndarray, Sequence[int]]) -> float:
        """Evaluates classical Ising energy H(s) for a given bitstring x in {0, 1}^n."""
        if isinstance(state, str):
            x = parse_qiskit_bitstring(state)
        else:
            x = np.asarray(state, dtype=np.int32)
        s = 1.0 - 2.0 * x  # Spin substitution: 0 -> +1, 1 -> -1

        coupling_energy = float(np.sum(np.triu(self.J, k=1) * np.outer(s, s)))
        field_energy = float(np.dot(self.h, s))
        return coupling_energy + field_energy + self.constant


class HardwareEfficientAnsatz:
    """
    Hardware-Efficient Parameterized Quantum Circuit (RealAmplitudes ansatz).
    Consists of alternating single-qubit R_y rotations and entangling CX gates.
    """
    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str = "linear"
    ):
        self.num_qubits = int(num_qubits)
        self.reps = int(reps)
        if self.reps < 1:
            raise ValueError(f"reps must be >= 1, got {self.reps}")
        self.entanglement = entanglement.lower()
        if self.entanglement not in ("linear", "circular", "full"):
            raise ValueError(f"Unsupported entanglement '{self.entanglement}', choose 'linear', 'circular', or 'full'")

        # Total parameters = n * (reps + 1)
        self.num_parameters = self.num_qubits * (self.reps + 1)
        self.param_vector = ParameterVector("theta", self.num_parameters)

    def _get_entangling_pairs(self) -> List[Tuple[int, int]]:
        """Generates entangling pairs based on strategy."""
        n = self.num_qubits
        if n < 2:
            return []
        if self.entanglement == "linear":
            return [(i, i + 1) for i in range(n - 1)]
        elif self.entanglement == "circular":
            pairs = [(i, i + 1) for i in range(n - 1)]
            pairs.append((n - 1, 0))
            return pairs
        elif self.entanglement == "full":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        return []

    def build_circuit(
        self,
        parameters: Optional[Sequence[float]] = None,
        measure: bool = False
    ) -> QuantumCircuit:
        """
        Builds the concrete or parameterized Qiskit QuantumCircuit.
        """
        n = self.num_qubits
        qc = QuantumCircuit(n, n) if measure else QuantumCircuit(n)

        param_idx = 0
        # Initial rotation layer
        for i in range(n):
            qc.ry(self.param_vector[param_idx], i)
            param_idx += 1

        entangling_pairs = self._get_entangling_pairs()

        # Repetition layers
        for _ in range(self.reps):
            for u, v in entangling_pairs:
                qc.cx(u, v)
            for i in range(n):
                qc.ry(self.param_vector[param_idx], i)
                param_idx += 1

        if measure:
            qc.measure(range(n), range(n))

        if parameters is not None:
            param_vals = np.asarray(parameters, dtype=np.float64)
            if len(param_vals) != self.num_parameters:
                raise ValueError(f"Expected {self.num_parameters} parameters, got {len(param_vals)}")
            # Assign parameters using dictionary
            param_dict = {self.param_vector[i]: param_vals[i] for i in range(self.num_parameters)}
            qc = qc.assign_parameters(param_dict)

        return qc


class VQESolver:
    """
    Variational Quantum Eigensolver (VQE) for Ising Hamiltonians.
    """
    def __init__(
        self,
        hamiltonian: IsingPauliHamiltonian,
        ansatz: Optional[HardwareEfficientAnsatz] = None,
        reps: int = 1,
        entanglement: str = "linear",
        simulator: Optional[AerSimulator] = None,
        seed: int = 42
    ):
        self.hamiltonian = hamiltonian
        self.seed = int(seed)
        self.simulator = simulator if simulator is not None else AerSimulator(seed_simulator=self.seed)

        if ansatz is None:
            self.ansatz = HardwareEfficientAnsatz(
                num_qubits=self.hamiltonian.n_qubits,
                reps=reps,
                entanglement=entanglement
            )
        else:
            self.ansatz = ansatz

        self.optimal_parameters: Optional[np.ndarray] = None
        self.optimal_energy: Optional[float] = None
        self.optimization_result: Optional[Dict[str, Any]] = None

    def get_circuit_metrics(self) -> Dict[str, Any]:
        """Returns circuit depth and gate counts."""
        dummy_params = [0.1] * self.ansatz.num_parameters
        qc = self.ansatz.build_circuit(dummy_params, measure=False)
        tqc = transpile(qc, self.simulator)
        ops = dict(tqc.count_ops())
        return {
            "num_qubits": self.hamiltonian.n_qubits,
            "reps": self.ansatz.reps,
            "num_parameters": self.ansatz.num_parameters,
            "depth": int(tqc.depth()),
            "num_cx_gates": int(ops.get("cx", 0)),
            "total_gates": int(sum(ops.values())),
            "gate_breakdown": ops
        }

    def compute_energy(
        self,
        parameters: Sequence[float],
        method: str = "statevector",
        shots: int = 2048
    ) -> float:
        """
        Computes the expectation value <psi(theta)| H |psi(theta)>.
        """
        if method == "statevector":
            qc = self.ansatz.build_circuit(parameters, measure=False)
            sv = Statevector.from_instruction(qc)
            exp_val = sv.expectation_value(self.hamiltonian.sparse_pauli_op).real
            return float(exp_val)

        elif method == "shots":
            qc = self.ansatz.build_circuit(parameters, measure=True)
            tqc = transpile(qc, self.simulator)
            job = self.simulator.run(tqc, shots=shots)
            counts = job.result().get_counts()

            exp_val = 0.0
            for bitstr, count in counts.items():
                p = count / shots
                e = self.hamiltonian.evaluate_bitstring_energy(bitstr)
                exp_val += p * e
            return float(exp_val)

        else:
            raise ValueError(f"Unknown expectation method: {method}")

    def optimize(
        self,
        initial_parameters: Optional[Sequence[float]] = None,
        method: str = "COBYLA",
        maxiter: int = 150,
        expectation_method: str = "statevector",
        shots: int = 2048,
        tol: float = 1e-5
    ) -> Dict[str, Any]:
        """
        Executes classical parameter optimization to find the ground-state energy.
        """
        t0 = time.perf_counter()

        if initial_parameters is None:
            rng = np.random.RandomState(self.seed)
            x0 = rng.uniform(-np.pi, np.pi, size=self.ansatz.num_parameters)
        else:
            x0 = np.asarray(initial_parameters, dtype=np.float64)
            if len(x0) != self.ansatz.num_parameters:
                raise ValueError(f"Expected {self.ansatz.num_parameters} parameters, got {len(x0)}")

        history: List[float] = []

        def cost_func(x):
            val = self.compute_energy(x, method=expectation_method, shots=shots)
            history.append(float(val))
            return val

        res = minimize(
            cost_func,
            x0,
            method=method,
            tol=tol,
            options={"maxiter": maxiter}
        )

        opt_time_ms = (time.perf_counter() - t0) * 1000.0

        self.optimal_parameters = np.asarray(res.x, dtype=np.float64)
        self.optimal_energy = float(res.fun)

        exact_e0, _ = self.hamiltonian.exact_ground_state()
        abs_error = abs(self.optimal_energy - exact_e0)

        self.optimization_result = {
            "success": bool(res.success),
            "optimal_parameters": self.optimal_parameters,
            "optimal_energy": self.optimal_energy,
            "exact_ground_energy": exact_e0,
            "absolute_energy_error": abs_error,
            "relative_error": abs_error / max(1e-9, abs(exact_e0)),
            "num_evaluations": int(res.nfev) if hasattr(res, "nfev") else len(history),
            "optimization_time_ms": opt_time_ms,
            "history": history,
            "optimizer": method,
            "status_message": str(res.message)
        }
        return self.optimization_result

    def sample_ground_state(
        self,
        shots: int = 2048,
        parameters: Optional[Sequence[float]] = None
    ) -> Dict[str, Any]:
        """
        Measures the quantum state at optimal parameters on AerSimulator to sample bitstrings.
        """
        t0 = time.perf_counter()

        if parameters is None:
            if self.optimal_parameters is None:
                raise ValueError("Must run optimize() first or provide parameters")
            eval_params = self.optimal_parameters
        else:
            eval_params = np.asarray(parameters, dtype=np.float64)

        qc = self.ansatz.build_circuit(eval_params, measure=True)
        tqc = transpile(qc, self.simulator)
        job = self.simulator.run(tqc, shots=shots)
        counts = job.result().get_counts()

        runtime_ms = (time.perf_counter() - t0) * 1000.0

        exact_e0, _ = self.hamiltonian.exact_ground_state()
        best_sampled_energy = float("inf")
        best_bitstring = ""
        total_energy = 0.0
        ground_state_hits = 0

        for bitstr, count in counts.items():
            energy = self.hamiltonian.evaluate_bitstring_energy(bitstr)
            total_energy += energy * count
            if energy < best_sampled_energy:
                best_sampled_energy = energy
                best_bitstring = bitstr
            if abs(energy - exact_e0) < 1e-5:
                ground_state_hits += count

        expected_sampled_energy = total_energy / shots
        ground_state_overlap = ground_state_hits / shots

        return {
            "counts": counts,
            "best_bitstring": best_bitstring,
            "best_state": parse_qiskit_bitstring(best_bitstring),
            "best_sampled_energy": best_sampled_energy,
            "expected_sampled_energy": expected_sampled_energy,
            "ground_state_overlap": ground_state_overlap,
            "shots": shots,
            "sample_runtime_ms": runtime_ms
        }

    def run_repeated_trials(
        self,
        n_trials: int = 5,
        maxiter: int = 150,
        seed_base: int = 42
    ) -> Dict[str, Any]:
        """
        Executes repeated independent VQE trials with different random seeds
        to report empirical variability, local minima, and failed/sub-optimal runs.
        """
        trials = []
        exact_e0, _ = self.hamiltonian.exact_ground_state()

        for trial_idx in range(n_trials):
            trial_seed = seed_base + trial_idx * 13
            rng = np.random.RandomState(trial_seed)
            x0 = rng.uniform(-np.pi, np.pi, size=self.ansatz.num_parameters)

            t0 = time.perf_counter()
            def cost(x):
                return self.compute_energy(x, method="statevector")

            res = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})
            runtime_ms = (time.perf_counter() - t0) * 1000.0

            energy = float(res.fun)
            abs_err = abs(energy - exact_e0)
            is_optimal = abs_err < 0.05  # Within 0.05 energy threshold

            trials.append({
                "trial": trial_idx + 1,
                "seed": trial_seed,
                "energy": energy,
                "absolute_error": abs_err,
                "is_optimal": is_optimal,
                "evaluations": int(res.nfev) if hasattr(res, "nfev") else maxiter,
                "runtime_ms": runtime_ms
            })

        energies = [t["energy"] for t in trials]
        success_count = sum(1 for t in trials if t["is_optimal"])

        return {
            "n_trials": n_trials,
            "trials": trials,
            "exact_ground_energy": exact_e0,
            "best_energy": float(np.min(energies)),
            "mean_energy": float(np.mean(energies)),
            "std_energy": float(np.std(energies)),
            "success_rate": success_count / n_trials,
            "num_optimal_trials": success_count
        }


def explain_vqe_foundations() -> str:
    """
    Returns an academic, mathematically rigorous explanation of VQE on Ising models for viva defense.
    """
    return (
        "### Mathematical & Algorithmic Foundations of VQE for Ising Models\n\n"
        "1. **Rayleigh-Ritz Variational Principle**:\n"
        "   The ground state energy E_0 = min_{|psi>} <psi| H |psi> / <psi|psi>.\n"
        "   By restricting |psi(theta)> to a parameterized quantum circuit (PQC), VQE turns\n"
        "   an exponentially large eigenvalue problem into a continuous parameter optimization task.\n\n"
        "2. **RealAmplitudes Subspace Restriction**:\n"
        "   Because Ising Hamiltonians are diagonal in the computational basis (Z), all ground states\n"
        "   have real amplitudes. By employing single-qubit R_y rotations and CNOT gates (Hardware-Efficient),\n"
        "   the ansatz restricts state preparation to the real subspace R^(2^n) subset C^(2^n).\n"
        "   This eliminates redundant complex phase dimensions and accelerates convergence.\n\n"
        "3. **Zero-Basis-Rotation Commutativity**:\n"
        "   All terms in an Ising Hamiltonian commute ([Z_i Z_j, Z_k] = 0). Therefore, unlike chemistry\n"
        "   Hamiltonians (which require measuring in separate X, Y, and Z bases), energy estimation\n"
        "   for Ising models requires only a SINGLE computational basis measurement register!\n\n"
        "4. **Classical Optimizer & Local Minima**:\n"
        "   COBYLA (Constrained Optimization BY Linear Approximation) constructs linear simplex approximations\n"
        "   of the cost landscape. While gradient-free and noise-tolerant, it can be sensitive to initialization\n"
        "   seeds on shallow non-convex landscapes, necessitating multi-start trials.\n\n"
        "5. **Algorithmic Role (VQE vs Classical Solvers)**:\n"
        "   On classical NP-hard problems (e.g. Max-Cut on small graphs), classical Simulated Annealing\n"
        "   and Branch-and-Bound are far faster than quantum simulators. VQE serves as a foundational\n"
        "   paradigm demonstration of hybrid quantum-classical optimization and state preparation.\n"
    )
