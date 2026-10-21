"""
test_quantum_gradients.py
Independent unit tests for quantum gradient methods:
- Parameter-Shift Rule
- Finite-Difference baseline
- Simultaneous Perturbation Stochastic Approximation (SPSA)
- Quantum Fisher Information Matrix (QFIM) and Quantum Natural Gradient (QNG)
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from track_C_quantum.src.quantum_gradients import (
    evaluate_circuit_energy,
    parameter_shift_gradient,
    finite_difference_gradient,
    spsa_gradient,
    quantum_fisher_information_matrix,
    quantum_natural_gradient_step,
    explain_quantum_gradients,
)


class TestQuantumGradientsAnalytical:
    """Verifies gradient methods against exact analytical calculus."""

    def test_single_qubit_ry_z_observable(self):
        # Circuit: |psi(theta)> = Ry(theta)|0>
        # Observable: H = Z
        # <Z>(theta) = cos(theta)
        # Analytical gradient: d<Z>/dtheta = -sin(theta)
        # Analytical QFIM: F = 1.0
        pv = ParameterVector("theta", 1)
        qc = QuantumCircuit(1)
        qc.ry(pv[0], 0)
        H = SparsePauliOp.from_list([("Z", 1.0)])

        for test_theta in [0.25, 0.785, 1.57, 2.35]:
            analytical_grad = -np.sin(test_theta)

            # 1. Parameter-Shift
            ps_grad = parameter_shift_gradient(qc, H, [test_theta])[0]
            assert abs(ps_grad - analytical_grad) < 1e-12

            # 2. Finite-Difference
            fd_grad = finite_difference_gradient(qc, H, [test_theta], eps=1e-5)[0]
            assert abs(fd_grad - analytical_grad) < 1e-6

            # 3. QFIM
            qfim = quantum_fisher_information_matrix(qc, [test_theta])[0, 0]
            assert abs(qfim - 1.0) < 1e-6

    def test_two_qubit_entangled_gradient_consistency(self):
        # 2-qubit circuit with entanglement and 3 parameters
        pv = ParameterVector("theta", 3)
        qc = QuantumCircuit(2)
        qc.ry(pv[0], 0)
        qc.ry(pv[1], 1)
        qc.cx(0, 1)
        qc.ry(pv[2], 0)

        H = SparsePauliOp.from_list([("ZZ", 1.0), ("ZI", -0.5), ("IZ", 0.3)])
        params = [0.6, -1.2, 0.9]

        ps_grad = parameter_shift_gradient(qc, H, params)
        fd_grad = finite_difference_gradient(qc, H, params, eps=1e-5)

        # Parameter-shift and finite difference should agree across all coordinates
        np.testing.assert_allclose(ps_grad, fd_grad, atol=1e-5)


class TestSPSAUnbiasedness:
    """Tests SPSA gradient estimator properties."""

    def test_spsa_mean_converges_to_exact_gradient(self):
        # Average of SPSA evaluations should converge to the exact parameter-shift gradient
        pv = ParameterVector("theta", 2)
        qc = QuantumCircuit(2)
        qc.ry(pv[0], 0)
        qc.ry(pv[1], 1)
        qc.cx(0, 1)

        H = SparsePauliOp.from_list([("ZZ", 1.0)])
        params = [0.8, -0.5]

        exact_grad = parameter_shift_gradient(qc, H, params)

        # Average 200 SPSA estimates
        rng = np.random.RandomState(42)
        spsa_estimates = []
        for _ in range(200):
            g, _ = spsa_gradient(qc, H, params, c_k=0.05, rng=rng)
            spsa_estimates.append(g)

        spsa_mean = np.mean(spsa_estimates, axis=0)
        # Verify SPSA matches within statistical Monte Carlo error
        np.testing.assert_allclose(spsa_mean, exact_grad, atol=0.08)


class TestQuantumNaturalGradient:
    """Tests Quantum Natural Gradient step computation."""

    def test_qng_step_reduces_energy(self):
        pv = ParameterVector("theta", 2)
        qc = QuantumCircuit(2)
        qc.ry(pv[0], 0)
        qc.ry(pv[1], 1)
        qc.cx(0, 1)

        H = SparsePauliOp.from_list([("ZZ", 1.0)])
        params = np.array([0.9, 0.7])

        e_initial = evaluate_circuit_energy(qc, H, params)
        new_params, grad_nat, F = quantum_natural_gradient_step(qc, H, params, lr=0.15, damping=1e-3)
        e_after = evaluate_circuit_energy(qc, H, new_params)

        # Natural gradient step should successfully decrease energy
        assert e_after < e_initial
        # QFIM must be symmetric and positive semi-definite
        np.testing.assert_allclose(F, F.T, atol=1e-8)
        eigvals = np.linalg.eigvalsh(F)
        assert np.all(eigvals >= -1e-6)

    def test_explanation_text(self):
        text = explain_quantum_gradients()
        assert "Parameter-Shift Rule" in text
        assert "SPSA for Scaling" in text
        assert "Quantum Natural Gradient" in text
