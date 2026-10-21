"""
test_qaoa_maxcut.py
Independent unit tests for Track C Batch 1: Quantum Optimization & QAOA for Max-Cut.

Covers:
1. Hamiltonian construction, couplings J_ij, fields h_i, and Track B QUBO/Ising equivalence.
2. Bitstring parsing under Qiskit little-endian conventions.
3. Quantum circuit construction, layer scaling with depth p, and gate validation.
4. Error handling for parameter length and depth mismatches.
5. Circuit execution on Qiskit AerSimulator and sampling statistics.
6. Exact Farhi (2014) analytic ground truth verification on 2-qubit system.
7. Optimization and approximation ratio improvement from p=1 to p=2 on C_4.
8. Cross-verification against Track B brute-force ground truth on K_4 and random graphs.
"""

import numpy as np
import pytest

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from track_B_discrete.src.qubo_ising import (
    maxcut_to_qubo,
    maxcut_to_ising,
    solve_qubo_brute_force,
    solve_qubo_simulated_annealing,
)
from track_C_quantum.src.qaoa_maxcut import (
    CostHamiltonian,
    build_qaoa_circuit,
    QAOAMaxCutSolver,
    parse_qiskit_bitstring,
    explain_qaoa_depth_and_p,
)


class TestCostHamiltonian:
    """Tests for CostHamiltonian construction and classical objective evaluations."""

    def test_construction_and_track_b_equivalence(self):
        # 3-node triangle graph K_3
        W = np.array([
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 3.0],
            [1.0, 3.0, 0.0]
        ])
        ham = CostHamiltonian(W)
        assert ham.n_qubits == 3
        assert ham.num_edges == 3
        assert ham.total_weight == 6.0
        assert ham.constant_offset == -3.0

        # Check Track B Ising equivalence
        track_b_ising = maxcut_to_ising(W)
        np.testing.assert_allclose(ham.J, track_b_ising.J)
        np.testing.assert_allclose(ham.h, track_b_ising.h)
        assert abs(ham.constant_offset - track_b_ising.constant) < 1e-9

        # Check Track B QUBO equivalence on all 2^3 states
        qubo = ham.to_qubo()
        for i in range(8):
            x = np.array([(i >> b) & 1 for b in range(3)], dtype=np.int32)
            cut_val = ham.evaluate_cut(x)
            energy_val = ham.evaluate_energy(x)
            qubo_energy = qubo.evaluate_energy(x)
            assert energy_val == -cut_val
            assert abs(energy_val - qubo_energy) < 1e-9

    def test_construction_from_edges(self):
        edges = [(0, 1, 4.0), (1, 2, 5.0), (2, 0, 1.0)]
        ham = CostHamiltonian.from_edges(3, edges)
        assert ham.num_edges == 3
        assert ham.total_weight == 10.0
        assert ham.evaluate_cut([1, 0, 0]) == 5.0  # cuts (0,1) weight 4 and (2,0) weight 1

    def test_known_graphs_ground_truth(self):
        # 1. 2-node graph with 1 edge of weight 1
        W2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        ham2 = CostHamiltonian(W2)
        assert ham2.evaluate_cut([0, 0]) == 0.0
        assert ham2.evaluate_cut([1, 1]) == 0.0
        assert ham2.evaluate_cut([0, 1]) == 1.0
        assert ham2.evaluate_cut([1, 0]) == 1.0

        # 2. 4-node cycle C_4
        W4 = np.array([
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0]
        ])
        ham4 = CostHamiltonian(W4)
        assert ham4.evaluate_cut([0, 1, 0, 1]) == 4.0
        assert ham4.evaluate_cut([1, 0, 1, 0]) == 4.0
        assert ham4.evaluate_cut([0, 0, 1, 1]) == 2.0


class TestBitstringParsing:
    """Tests for bitstring conversion and Qiskit little-endian handling."""

    def test_parse_qiskit_bitstring(self):
        # String '100' in Qiskit: char 0 is qubit 2 ('1'), char 1 is qubit 1 ('0'), char 2 is qubit 0 ('0')
        # Reversing maps qubit i to element i: [0, 0, 1]
        parsed = parse_qiskit_bitstring("100")
        assert list(parsed) == [0, 0, 1]

        parsed_4 = parse_qiskit_bitstring("1010")
        assert list(parsed_4) == [0, 1, 0, 1]


class TestCircuitConstruction:
    """Tests for QAOA QuantumCircuit generation and parameter handling."""

    def test_circuit_structure_and_depth(self):
        W = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ])
        ham = CostHamiltonian(W)

        # p = 1 without measurement
        qc_p1 = build_qaoa_circuit(ham, p=1, gammas=[0.5], betas=[0.3], measure=False)
        assert qc_p1.num_qubits == 3
        assert qc_p1.num_clbits == 0

        # p = 2 with measurement
        qc_p2 = build_qaoa_circuit(ham, p=2, gammas=[0.5, 0.6], betas=[0.3, 0.4], measure=True)
        assert qc_p2.num_qubits == 3
        assert qc_p2.num_clbits == 3

    def test_circuit_parameter_validation(self):
        W = np.eye(2)
        ham = CostHamiltonian(W)
        with pytest.raises(ValueError, match="Expected 2 gammas"):
            build_qaoa_circuit(ham, p=2, gammas=[0.5], betas=[0.3, 0.4])

        with pytest.raises(ValueError, match="p must be >= 1"):
            QAOAMaxCutSolver(ham, p=0)


class TestQAOASolver:
    """Tests for QAOA simulation, optimization, and sampling."""

    def test_farhi_2qubit_analytic_exact_solution(self):
        """
        Farhi et al. (2014) analytic solution:
        For a single edge with weight 1, at gamma = pi/2 and beta = pi/8:
        Expected cut is mathematically 1.0, and optimal states |01> and |10>
        have combined probability 1.0 (exact ground state preparation).
        """
        W = np.array([[0.0, 1.0], [1.0, 0.0]])
        ham = CostHamiltonian(W)
        solver = QAOAMaxCutSolver(ham, p=1)

        gamma_exact = np.pi / 2.0
        beta_exact = np.pi / 8.0
        energy = solver.compute_expectation([gamma_exact, beta_exact], method="statevector")
        expected_cut = -energy

        assert abs(expected_cut - 1.0) < 1e-6

        # Sample with AerSimulator
        res = solver.sample_solutions(shots=1000, angles=[gamma_exact, beta_exact])
        assert res["best_cut"] == 1.0
        # Optimal probability (|01> + |10>) should be 100%
        counts = res["counts"]
        opt_counts = counts.get("01", 0) + counts.get("10", 0)
        assert opt_counts == 1000

    def test_c4_optimization_p_depth_monotonicity(self):
        """
        On C_4 (4-cycle), tests that QAOA achieves higher expected cut at p=2 than p=1.
        """
        W4 = np.array([
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0]
        ])
        ham = CostHamiltonian(W4)

        # p = 1
        solver_p1 = QAOAMaxCutSolver(ham, p=1, seed=42)
        opt_p1 = solver_p1.optimize(maxiter=80)
        cut_p1 = opt_p1["expected_cut"]
        sample_p1 = solver_p1.sample_solutions(shots=1000)
        assert sample_p1["best_cut"] == 4.0
        assert cut_p1 >= 2.8  # Theoretical max expectation at p=1 is ~3.0

        # p = 2
        solver_p2 = QAOAMaxCutSolver(ham, p=2, seed=42)
        opt_p2 = solver_p2.optimize(maxiter=100)
        cut_p2 = opt_p2["expected_cut"]
        sample_p2 = solver_p2.sample_solutions(shots=1000)
        assert sample_p2["best_cut"] == 4.0
        # p=2 should strictly improve expected cut on C_4 towards 4.0
        assert cut_p2 >= cut_p1

    def test_qaoa_vs_track_b_brute_force_and_sa(self):
        """
        Validates QAOA against Track B Brute-Force and Simulated Annealing
        on a 4-node complete graph K_4.
        """
        rng = np.random.RandomState(42)
        W_raw = rng.randint(1, 6, size=(4, 4)).astype(np.float64)
        W = np.triu(W_raw, k=1) + np.triu(W_raw, k=1).T

        ham = CostHamiltonian(W)
        qubo = ham.to_qubo()

        # Track B Brute Force Ground Truth
        bf_res = solve_qubo_brute_force(qubo)
        exact_max_cut = -bf_res["best_energy"]

        # Track B Simulated Annealing
        sa_res = solve_qubo_simulated_annealing(qubo, seed=42)
        sa_cut = -sa_res["best_energy"]
        assert sa_cut == exact_max_cut

        # QAOA at p=2
        solver = QAOAMaxCutSolver(ham, p=2, seed=42)
        solver.optimize(maxiter=100)
        qaoa_res = solver.sample_solutions(shots=2048)

        # QAOA should discover the exact ground state cut among its samples
        assert qaoa_res["best_cut"] == exact_max_cut
        approx_ratio = qaoa_res["expected_cut"] / exact_max_cut
        assert approx_ratio >= 0.70  # Sound quantum approximation ratio

    def test_explain_text(self):
        explanation = explain_qaoa_depth_and_p()
        assert "Trotterized Adiabatic Analogy" in explanation
        assert "Asymptotic Convergence" in explanation
        assert "NISQ Hardware Constraints" in explanation
