"""
Quantum Machine Learning (QML):
Fidelity Quantum Kernel Estimation and Quantum Support Vector Classifier (QSVM).
Compares Quantum Kernel Estimation against Classical RBF and Linear SVM baselines.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


def zz_feature_map_circuit(x: np.ndarray) -> QuantumCircuit:
    """
    Constructs an explicit 2-qubit ZZ-feature map circuit for data vector x in R^2:
    U_Phi(x) = exp( i (pi - x0)(pi - x1) Z0 Z1 ) * exp(i x0 Z0) * exp(i x1 Z1) * H^(x2).
    """
    n_qubits = len(x)
    qc = QuantumCircuit(n_qubits, name=f"ZZFeatureMap({x.round(2)})")

    # 1. First layer: Hadamards
    for i in range(n_qubits):
        qc.h(i)

    # 2. Single-qubit phase rotations
    for i in range(n_qubits):
        qc.p(2.0 * x[i], i)

    # 3. Two-qubit ZZ entangling interaction: CNOT -> Phase((pi-x0)(pi-x1)) -> CNOT
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            angle = 2.0 * (np.pi - x[i]) * (np.pi - x[j])
            qc.cx(i, j)
            qc.p(angle, j)
            qc.cx(i, j)

    return qc


class QuantumKernelEstimator:
    r"""
    Computes fidelity quantum kernel matrices K(x_i, x_j) = |<Phi(x_i)|Phi(x_j)>|^2.
    Evaluated using statevector overlap: |<0| U_Phi^\dagger(x_j) U_Phi(x_i) |0>|^2.
    """

    def __init__(self, n_features: int = 2):
        self.n_features = n_features
        self.evaluation_counts = 0

    def compute_kernel_entry(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes overlap |<Phi(x1)|Phi(x2)>|^2 for a pair of classical data points."""
        qc1 = zz_feature_map_circuit(x1)
        qc2 = zz_feature_map_circuit(x2)

        # Full fidelity circuit: U(x1) followed by U^\dagger(x2)
        fidelity_circ = qc1.compose(qc2.inverse())
        sv = Statevector.from_instruction(fidelity_circ)

        # Overlap with ground state |0...0> is sv.data[0]
        overlap = abs(sv.data[0]) ** 2
        self.evaluation_counts += 1
        return float(np.clip(overlap, 0.0, 1.0))

    def compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Computes the full Gram/kernel matrix between dataset X1 and X2."""
        n1 = len(X1)
        n2 = len(X2)
        is_symmetric = (X1 is X2) or (n1 == n2 and np.array_equal(X1, X2))

        K = np.zeros((n1, n2))

        for i in range(n1):
            start_j = i if is_symmetric else 0
            for j in range(start_j, n2):
                if is_symmetric and i == j:
                    K[i, j] = 1.0
                else:
                    val = self.compute_kernel_entry(X1[i], X2[j])
                    K[i, j] = val
                    if is_symmetric:
                        K[j, i] = val

        return K


def evaluate_qml_vs_classical(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 1.0,
    feature_range: Tuple[float, float] = (0.0, float(np.pi))
) -> Dict[str, Any]:
    """
    Trains and benchmarks Quantum Kernel SVM (QSVM) vs Classical RBF SVM and Linear SVM
    on identical training and testing splits.
    """
    # Preprocessing: Scale features to feature_range (default [0, pi]) for periodic quantum embedding
    scaler = MinMaxScaler(feature_range=feature_range)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------
    # 1. Quantum Kernel Estimation & QSVM
    # -------------------------------------------------------------
    q_kernel = QuantumKernelEstimator(n_features=X_train.shape[1])
    t0_q_train = time.perf_counter()
    K_train = q_kernel.compute_kernel_matrix(X_train_scaled, X_train_scaled)
    qsvm = SVC(kernel="precomputed", C=C)
    qsvm.fit(K_train, y_train)
    t1_q_train = time.perf_counter()

    t0_q_eval = time.perf_counter()
    K_test = q_kernel.compute_kernel_matrix(X_test_scaled, X_train_scaled)
    y_pred_qsvm = qsvm.predict(K_test)
    t1_q_eval = time.perf_counter()

    acc_qsvm = accuracy_score(y_test, y_pred_qsvm)
    f1_qsvm = f1_score(y_test, y_pred_qsvm, average="weighted")
    n_sv_qsvm = len(qsvm.support_)

    # -------------------------------------------------------------
    # 2. Classical RBF Support Vector Machine
    # -------------------------------------------------------------
    rbf_svm = SVC(kernel="rbf", C=C, gamma="scale")
    t0_rbf_train = time.perf_counter()
    rbf_svm.fit(X_train_scaled, y_train)
    t1_rbf_train = time.perf_counter()

    t0_rbf_eval = time.perf_counter()
    y_pred_rbf = rbf_svm.predict(X_test_scaled)
    t1_rbf_eval = time.perf_counter()

    acc_rbf = accuracy_score(y_test, y_pred_rbf)
    f1_rbf = f1_score(y_test, y_pred_rbf, average="weighted")
    n_sv_rbf = len(rbf_svm.support_)

    # -------------------------------------------------------------
    # 3. Classical Linear Support Vector Machine
    # -------------------------------------------------------------
    lin_svm = SVC(kernel="linear", C=C)
    t0_lin_train = time.perf_counter()
    lin_svm.fit(X_train_scaled, y_train)
    t1_lin_train = time.perf_counter()

    t0_lin_eval = time.perf_counter()
    y_pred_lin = lin_svm.predict(X_test_scaled)
    t1_lin_eval = time.perf_counter()

    acc_lin = accuracy_score(y_test, y_pred_lin)
    f1_lin = f1_score(y_test, y_pred_lin, average="weighted")
    n_sv_lin = len(lin_svm.support_)

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "qsvm": {
            "test_accuracy": float(acc_qsvm),
            "test_f1": float(f1_qsvm),
            "n_support_vectors": int(n_sv_qsvm),
            "train_time_sec": float(t1_q_train - t0_q_train),
            "eval_time_sec": float(t1_q_eval - t0_q_eval),
            "kernel_evaluations": q_kernel.evaluation_counts,
            "predictions": y_pred_qsvm.tolist()
        },
        "classical_rbf": {
            "test_accuracy": float(acc_rbf),
            "test_f1": float(f1_rbf),
            "n_support_vectors": int(n_sv_rbf),
            "train_time_sec": float(t1_rbf_train - t0_rbf_train),
            "eval_time_sec": float(t1_rbf_eval - t0_rbf_eval),
            "predictions": y_pred_rbf.tolist()
        },
        "classical_linear": {
            "test_accuracy": float(acc_lin),
            "test_f1": float(f1_lin),
            "n_support_vectors": int(n_sv_lin),
            "train_time_sec": float(t1_lin_train - t0_lin_train),
            "eval_time_sec": float(t1_lin_eval - t0_lin_eval),
            "predictions": y_pred_lin.tolist()
        }
    }
