"""
Unit tests for Track I: Quantum Algorithms and Quantum Machine Learning.
Verifies Deutsch-Jozsa, Bernstein-Vazirani, QFT, QPE, Shor-15, Grover search,
NISQ noise evaluation, Quantum Kernel Estimation (QSVM), and benchmark artifacts.
"""

import os
import math
import numpy as np
# pyrefly: ignore[missing-import]
import pytest
# pyrefly: ignore[missing-import]
from qiskit.quantum_info import Operator
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from track_I_quantum_algorithms.src.quantum_algorithms import (
    run_deutsch_jozsa,
    run_bernstein_vazirani,
    qft_circuit,
    inverse_qft_circuit,
    run_qpe,
    run_shor_15_period_finding,
    run_grover_search,
    optimal_grover_iterations,
)
from track_I_quantum_algorithms.src.quantum_machine_learning import (
    QuantumKernelEstimator,
    evaluate_qml_vs_classical,
)
from track_I_quantum_algorithms.src.quantum_noise_evaluation import (
    build_hardware_noise_model,
    evaluate_grover_noise_impact,
)


def test_deutsch_jozsa():
    """Verifies Deutsch-Jozsa determines constant vs balanced in 1 query."""
    assert run_deutsch_jozsa(3, "constant_0") == "constant"
    assert run_deutsch_jozsa(3, "constant_1") == "constant"
    assert run_deutsch_jozsa(3, "balanced") == "balanced"


def test_bernstein_vazirani():
    """Verifies Bernstein-Vazirani reconstructs arbitrary hidden bitstrings in 1 query."""
    for hidden_s in ["101", "0110", "11101"]:
        recovered = run_bernstein_vazirani(hidden_s)
        assert recovered == hidden_s, f"Expected {hidden_s}, got {recovered}"


def test_qft_unitary_and_inversion():
    """Verifies QFT circuit matches analytical DFT matrix and QFT^dagger * QFT = I."""
    for n in [2, 3]:
        N = 2 ** n
        qc_qft = qft_circuit(n)
        U_qft = Operator(qc_qft).data

        # Analytical DFT matrix
        omega = np.exp(2j * np.pi / N)
        dft_mat = np.array([[omega ** (j * k) / np.sqrt(N) for k in range(N)] for j in range(N)])

        # Max difference from analytical DFT
        diff = np.max(np.abs(U_qft - dft_mat))
        assert diff < 1e-12, f"QFT matrix differs from analytical DFT by {diff}"

        # Unitary invertibility: QFT^dagger * QFT == I
        qc_iqft = inverse_qft_circuit(n)
        U_iqft = Operator(qc_iqft).data
        prod = np.dot(U_iqft, U_qft)
        eye = np.eye(N)
        assert np.max(np.abs(prod - eye)) < 1e-12, "QFT inverse does not invert QFT"


def test_quantum_phase_estimation():
    """Verifies QPE estimates fractional phase with zero error for dyadic phases."""
    test_cases = [(3, 0.125), (3, 0.250), (3, 0.375), (3, 0.500), (3, 0.750)]
    for n_eval, phase in test_cases:
        est_phase, error, bitstr = run_qpe(n_eval_qubits=n_eval, phase=phase)
        assert error < 1e-6, f"Phase {phase} estimated as {est_phase} with error {error}"


def test_shor_15_factorization():
    """Verifies Shor's period finding extracts period r=4 and finds factors {3, 5} for N=15."""
    res = run_shor_15_period_finding(a=7)
    assert res["success"] is True, f"Shor factoring failed: {res}"
    assert res["extracted_period"] == 4, f"Expected period 4, got {res['extracted_period']}"
    assert res["factors"] == [3, 5], f"Expected factors [3, 5], got {res['factors']}"


def test_optimal_grover_iterations():
    """Verifies theoretical optimal Grover iteration count formula."""
    assert optimal_grover_iterations(2) == 1  # N=4 -> 1 iteration gives 100%
    assert optimal_grover_iterations(3) == 2  # N=8 -> 2 iterations gives ~95%
    assert optimal_grover_iterations(4) == 3  # N=16 -> 3 iterations gives ~96%
    assert optimal_grover_iterations(5) == 4  # N=32 -> 4 iterations gives ~99%


def test_grover_search_ideal_simulation():
    """Verifies Grover's search achieves >90% success probability on ideal simulator."""
    for n in [2, 3]:
        target = "1" * n
        res = run_grover_search(n, target, shots=2048)
        assert res["top_bitstring"] == target, f"Expected top bitstring {target}, got {res['top_bitstring']}"
        assert res["success_probability"] >= 0.90, f"Expected P(succ) >= 0.90, got {res['success_probability']}"


def test_grover_hardware_noise_evaluation():
    """Verifies documented NISQ noise model degrades success probability and increases with CX depth."""
    noise_results = evaluate_grover_noise_impact(qubit_range=[2, 3], shots=2048)
    assert len(noise_results) == 2
    for r in noise_results:
        assert r["p_success_noisy"] > 0.0
        assert r["p_success_noisy"] < r["p_success_ideal"]
        assert r["relative_degradation_pct"] > 0.0
        assert r["two_qubit_cx_count"] > 0


def test_quantum_kernel_properties():
    """Verifies quantum fidelity kernel is symmetric, bounded [0, 1], with unit diagonal."""
    qk = QuantumKernelEstimator(n_features=2)
    x1 = np.array([0.5, 1.2])
    x2 = np.array([1.8, 2.3])

    # Unit diagonal (self-overlap)
    assert math.isclose(qk.compute_kernel_entry(x1, x1), 1.0, rel_tol=1e-5)

    # Symmetry
    k12 = qk.compute_kernel_entry(x1, x2)
    k21 = qk.compute_kernel_entry(x2, x1)
    assert math.isclose(k12, k21, rel_tol=1e-5)
    assert 0.0 <= k12 <= 1.0

    # Gram matrix symmetry
    X = np.array([[0.2, 0.4], [0.8, 1.5], [1.2, 2.0]])
    K = qk.compute_kernel_matrix(X, X)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), np.ones(3))


def test_qml_vs_classical_svm():
    """Verifies QSVM achieves competitive test accuracy (>=80%) on structured classification."""
    blobs_data = make_blobs(
        n_samples=50,
        centers=[[-1.2, -1.2], [1.2, 1.2]],
        cluster_std=0.8,
        random_state=42
    )
    X_data, y_data = blobs_data[0], blobs_data[1]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_data, y_data, test_size=0.30, random_state=42, stratify=y_data
    )
    res = evaluate_qml_vs_classical(X_tr, y_tr, X_te, y_te, C=5.0, feature_range=(0.0, float(np.pi)))

    qsvm_acc = res["qsvm"]["test_accuracy"]
    rbf_acc = res["classical_rbf"]["test_accuracy"]

    assert qsvm_acc >= 0.80, f"QSVM test accuracy {qsvm_acc} below 0.80 threshold"
    assert rbf_acc >= 0.80, f"RBF test accuracy {rbf_acc} below 0.80 threshold"
    assert res["qsvm"]["n_support_vectors"] > 0
    assert res["classical_rbf"]["n_support_vectors"] > 0


def test_benchmark_artifacts_exist():
    """Verifies benchmark CSV, Markdown report, and PNG visualization exist on disk."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(base_dir, "results")

    csv_path = os.path.join(results_dir, "track_i_metrics.csv")
    md_path = os.path.join(results_dir, "track_i_metrics.md")
    png_path = os.path.join(results_dir, "track_i_benchmarks.png")

    assert os.path.isfile(csv_path), f"Missing metrics CSV at {csv_path}"
    assert os.path.isfile(md_path), f"Missing Markdown summary at {md_path}"
    assert os.path.isfile(png_path), f"Missing plot PNG at {png_path}"

    assert os.path.getsize(csv_path) > 100
    assert os.path.getsize(md_path) > 500
    assert os.path.getsize(png_path) > 1000
