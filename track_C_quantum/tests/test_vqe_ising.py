"""
test_vqe_ising.py
Independent unit tests for Track C Batch 2: Variational Quantum Eigensolver (VQE).

Covers:
1. IsingPauliHamiltonian construction, Pauli string little-endian mapping, and Track B IsingModel equivalence.
2. HardwareEfficientAnsatz parameter count scaling, entanglement modes (linear, circular, full), and parameter binding.
3. Energy evaluation consistency: Statevector expectation value vs computational basis probabilities.
4. AerSimulator shot-based energy evaluation within statistical variance bounds.
5. VQE optimization on known analytical ground-truth Hamiltonians:
   - 2-qubit single coupling (E_0 = -1.0)
   - 3-qubit Triangle K_3 (E_0 = -2.0)
   - 4-qubit Cycle C_4 (E_0 = -4.0)
6. Ground-state sampling and modal state overlap verification.
7. Multi-start trials reporting variability, success rate, and trial outcomes.
8. Rigorous explanation text for student viva defense.
"""

import numpy as np
import pytest

from qiskit_aer import AerSimulator
from track_B_discrete.src.qubo_ising import IsingModel, maxcut_to_ising
from track_C_quantum.src.vqe_ising import (
    IsingPauliHamiltonian,
    HardwareEfficientAnsatz,
    VQESolver,
    explain_vqe_foundations,
)


class TestIsingPauliHamiltonian:
    """Tests for Ising Hamiltonian conversion to Qiskit SparsePauliOp."""

    def test_construction_and_pauli_mapping(self):
        # 3-qubit line: J_01 = 1.0, J_12 = 2.0, h_0 = -0.5, C = 3.0
        J = np.zeros((3, 3))
        J[0, 1] = 1.0
        J[1, 2] = 2.0
        h = np.array([-0.5, 0.0, 0.0])
        ham = IsingPauliHamiltonian(n_qubits=3, J=J, h=h, constant=3.0)

        assert ham.n_qubits == 3
        assert len(ham.pauli_terms) == 4  # 2 couplings, 1 field, 1 constant

        # Check exact ground state via diagonal extraction
        ground_energy, ground_state = ham.exact_ground_state()
        assert len(ground_state) == 3

        # Cross-verify with Track B IsingModel evaluation on all 8 states
        ising = IsingModel(J=J, h=h, constant=3.0)
        energies = []
        for i in range(8):
            bits = np.array([(i >> b) & 1 for b in range(3)], dtype=np.int32)
            s = 1.0 - 2.0 * bits
            e_track_b = ising.evaluate_energy(s)
            e_pauli = ham.evaluate_bitstring_energy(bits)
            assert abs(e_track_b - e_pauli) < 1e-9
            energies.append(e_pauli)

        assert abs(ground_energy - min(energies)) < 1e-9

    def test_from_maxcut_adjacency(self):
        W = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ])
        ham = IsingPauliHamiltonian.from_maxcut_adjacency(W)
        ground_energy, ground_state = ham.exact_ground_state()
        # Max-Cut on K_3 is 2.0, so minimum energy is -2.0
        assert abs(ground_energy - (-2.0)) < 1e-9

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError, match="J shape"):
            IsingPauliHamiltonian(n_qubits=2, J=np.zeros((3, 3)))
        with pytest.raises(ValueError, match="h length"):
            IsingPauliHamiltonian(n_qubits=2, J=np.zeros((2, 2)), h=np.zeros(3))


class TestHardwareEfficientAnsatz:
    """Tests for ansatz parameterization and entanglement modes."""

    def test_parameter_counts_and_entanglement(self):
        # 4 qubits, 1 repetition: 4 * (1 + 1) = 8 parameters
        ansatz_lin = HardwareEfficientAnsatz(num_qubits=4, reps=1, entanglement="linear")
        assert ansatz_lin.num_parameters == 8
        qc_lin = ansatz_lin.build_circuit()
        ops_lin = qc_lin.count_ops()
        assert ops_lin.get("cx", 0) == 3  # (0,1), (1,2), (2,3)
        assert ops_lin.get("ry", 0) == 8

        # Circular entanglement: 4 CX gates
        ansatz_circ = HardwareEfficientAnsatz(num_qubits=4, reps=1, entanglement="circular")
        qc_circ = ansatz_circ.build_circuit()
        ops_circ = qc_circ.count_ops()
        assert ops_circ.get("cx", 0) == 4

        # Full entanglement: 4 * 3 / 2 = 6 CX gates
        ansatz_full = HardwareEfficientAnsatz(num_qubits=4, reps=1, entanglement="full")
        qc_full = ansatz_full.build_circuit()
        ops_full = qc_full.count_ops()
        assert ops_full.get("cx", 0) == 6

    def test_parameter_binding_validation(self):
        ansatz = HardwareEfficientAnsatz(num_qubits=3, reps=1)
        with pytest.raises(ValueError, match="Expected 6 parameters"):
            ansatz.build_circuit(parameters=[0.1, 0.2])


class TestVQESolver:
    """Tests for VQE execution, optimization, and ground-state discovery."""

    def test_vqe_2qubit_exact_ground_state(self):
        # 2 qubits with J_01 = 1.0 (antiferromagnetic), ground state has opposite spins: E_0 = -1.0
        J = np.array([[0.0, 1.0], [0.0, 0.0]])
        ham = IsingPauliHamiltonian(n_qubits=2, J=J, constant=0.0)
        solver = VQESolver(ham, reps=1, seed=42)

        # Optimize
        opt = solver.optimize(method="COBYLA", maxiter=100)
        assert opt["success"]
        assert abs(opt["optimal_energy"] - (-1.0)) < 1e-5
        assert opt["absolute_energy_error"] < 1e-5

        # Sample solutions
        samples = solver.sample_ground_state(shots=1000)
        assert samples["best_sampled_energy"] == -1.0
        assert samples["ground_state_overlap"] > 0.90

    def test_vqe_triangle_k3_ground_state(self):
        # K_3 Max-Cut Ising Hamiltonian (E_0 = -2.0)
        W = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        ham = IsingPauliHamiltonian.from_maxcut_adjacency(W)
        solver = VQESolver(ham, reps=1, seed=42)

        opt = solver.optimize(method="COBYLA", maxiter=120)
        assert abs(opt["optimal_energy"] - (-2.0)) < 1e-4

        samples = solver.sample_ground_state(shots=1500)
        assert samples["best_sampled_energy"] == -2.0
        assert samples["ground_state_overlap"] > 0.85

    def test_vqe_cycle_c4_ground_state(self):
        # C_4 Max-Cut Ising Hamiltonian (E_0 = -4.0)
        W = np.array([
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0]
        ])
        ham = IsingPauliHamiltonian.from_maxcut_adjacency(W)
        solver = VQESolver(ham, reps=1, seed=42)

        opt = solver.optimize(method="COBYLA", maxiter=150)
        assert abs(opt["optimal_energy"] - (-4.0)) < 1e-4

        samples = solver.sample_ground_state(shots=2048)
        assert samples["best_sampled_energy"] == -4.0
        assert samples["ground_state_overlap"] > 0.80

    def test_shot_based_energy_evaluation(self):
        J = np.array([[0.0, 1.0], [0.0, 0.0]])
        ham = IsingPauliHamiltonian(n_qubits=2, J=J)
        solver = VQESolver(ham, reps=1, seed=42)

        params = [0.5, 0.5, 0.5, 0.5]
        e_sv = solver.compute_energy(params, method="statevector")
        e_shots = solver.compute_energy(params, method="shots", shots=4096)

        # Shot-based estimation should match statevector within sampling standard error
        assert abs(e_sv - e_shots) < 0.1

    def test_repeated_trials_variability(self):
        W = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        ham = IsingPauliHamiltonian.from_maxcut_adjacency(W)
        solver = VQESolver(ham, reps=1, seed=123)

        results = solver.run_repeated_trials(n_trials=3, maxiter=80)
        assert results["n_trials"] == 3
        assert len(results["trials"]) == 3
        assert results["best_energy"] <= -1.90
        assert "mean_energy" in results
        assert "std_energy" in results
        assert results["success_rate"] > 0.0

    def test_vqe_explanation_text(self):
        text = explain_vqe_foundations()
        assert "Rayleigh-Ritz Variational Principle" in text
        assert "RealAmplitudes Subspace Restriction" in text
        assert "Zero-Basis-Rotation Commutativity" in text
        assert "COBYLA" in text
