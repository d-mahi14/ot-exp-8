"""
quantum_gradients.py
Track C: Quantum Gradient Estimation and Quantum Natural Gradient (QNG).

Theoretical Foundations:
1. Exact Parameter-Shift Rule:
   For parameterized quantum circuits with unitary generators of the form
   U(theta) = exp(-i * theta * G / 2) where G has two distinct eigenvalues +/- r:
       d<H>/d theta = r * [ <H>(theta + pi / (4r)) - <H>(theta - pi / (4r)) ]
   For standard Pauli generators (X, Y, Z with r = 1/2):
       d<H>/d theta_j = [ <H>(theta + pi/2 * e_j) - <H>(theta - pi/2 * e_j) ] / 2
   This yields the EXACT analytical gradient directly on physical quantum hardware
   without finite-difference truncation error and without wave-function access!

2. Simultaneous Perturbation Stochastic Approximation (SPSA):
   While parameter-shift requires 2 * D circuit evaluations for a D-dimensional parameter vector,
   SPSA estimates the entire gradient vector using ONLY 2 circuit evaluations:
       g_j(theta) = [ <H>(theta + c_k * Delta) - <H>(theta - c_k * Delta) ] / (2 * c_k * Delta_j)
   where Delta in {-1, +1}^D is a Rademacher random vector. In expectation:
       E[g_j(theta)] = d<H>/d theta_j
   making SPSA highly scalable for noisy, multi-parameter quantum circuits.

3. Quantum Natural Gradient (QNG) & Fubini-Study Metric:
   Standard gradient descent follows the steepest descent direction in the Euclidean parameter space R^D.
   However, equal Euclidean steps can correspond to wildly unequal distances in Hilbert space!
   QNG optimizes along the Riemannian manifold of quantum states using the Fubini-Study metric
   (Quantum Fisher Information Matrix, QFIM F):
       theta_{t+1} = theta_t - eta * (F(theta_t) + lambda * I)^(-1) * grad <H>
   where:
       F_{jk} = 4 * Re[ <d_j psi | d_k psi> - <d_j psi | psi> <psi | d_k psi> ]
   and lambda * I is a Levenberg-Marquardt regularization damping term ensuring stable inversion.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator


def evaluate_circuit_energy(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    parameters: Sequence[float],
    simulator: Optional[AerSimulator] = None,
    shots: Optional[int] = None
) -> float:
    """
    Evaluates expectation value <psi(theta)| H |psi(theta)>.
    If shots is None: uses exact Statevector simulation.
    If shots is an int: uses AerSimulator with finite-shot sampling.
    """
    param_dict = dict(zip(circuit.parameters, parameters))
    bound_circ = circuit.assign_parameters(param_dict)

    if shots is None:
        sv = Statevector.from_instruction(bound_circ)
        return float(sv.expectation_value(observable).real)
    else:
        sim = simulator if simulator is not None else AerSimulator()
        # Measure in Z basis
        n = bound_circ.num_qubits
        meas_circ = bound_circ.copy()
        meas_circ.measure_all()
        job = sim.run(meas_circ, shots=shots)
        counts = job.result().get_counts()

        # Compute empirical expectation value of the diagonal observable
        diag_matrix = observable.to_matrix()
        diag_vals = np.real(np.diag(diag_matrix))
        total_energy = 0.0
        for bitstr, count in counts.items():
            clean_b = bitstr.replace(" ", "")
            idx = int(clean_b, 2)
            total_energy += count * diag_vals[idx]
        return float(total_energy / shots)


def parameter_shift_gradient(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    parameters: Sequence[float],
    shift: float = np.pi / 2.0,
    simulator: Optional[AerSimulator] = None,
    shots: Optional[int] = None
) -> np.ndarray:
    """
    Computes exact gradient vector using the Parameter-Shift Rule:
        grad_j = [ E(theta + shift * e_j) - E(theta - shift * e_j) ] / (2 * sin(shift))
    Requires 2 * D circuit evaluations.
    """
    params = np.asarray(parameters, dtype=np.float64)
    d = len(params)
    grad = np.zeros(d, dtype=np.float64)
    denom = 2.0 * np.sin(shift)

    for j in range(d):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[j] += shift
        p_minus[j] -= shift

        e_plus = evaluate_circuit_energy(circuit, observable, p_plus, simulator=simulator, shots=shots)
        e_minus = evaluate_circuit_energy(circuit, observable, p_minus, simulator=simulator, shots=shots)

        grad[j] = (e_plus - e_minus) / denom

    return grad


def finite_difference_gradient(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    parameters: Sequence[float],
    eps: float = 1e-5
) -> np.ndarray:
    """
    Central finite-difference gradient:
        grad_j = [ E(theta + eps * e_j) - E(theta - eps * e_j) ] / (2 * eps)
    Serves as an independent numerical verification baseline for parameter shift.
    """
    params = np.asarray(parameters, dtype=np.float64)
    d = len(params)
    grad = np.zeros(d, dtype=np.float64)

    for j in range(d):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[j] += eps
        p_minus[j] -= eps

        e_plus = evaluate_circuit_energy(circuit, observable, p_plus)
        e_minus = evaluate_circuit_energy(circuit, observable, p_minus)

        grad[j] = (e_plus - e_minus) / (2.0 * eps)

    return grad


def spsa_gradient(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    parameters: Sequence[float],
    c_k: float = 0.1,
    rng: Optional[np.random.RandomState] = None,
    simulator: Optional[AerSimulator] = None,
    shots: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simultaneous Perturbation Stochastic Approximation (SPSA) gradient:
    Evaluates entire D-dimensional gradient in ONLY 2 circuit evaluations
    using a Rademacher random perturbation vector Delta in {-1, +1}^D.
    
    Returns:
        (grad_estimate, delta_vector)
    """
    params = np.asarray(parameters, dtype=np.float64)
    d = len(params)
    if rng is None:
        rng = np.random.RandomState()

    # Rademacher distribution: +/- 1 with equal probability
    delta = rng.choice([-1.0, 1.0], size=d)

    p_plus = params + c_k * delta
    p_minus = params - c_k * delta

    e_plus = evaluate_circuit_energy(circuit, observable, p_plus, simulator=simulator, shots=shots)
    e_minus = evaluate_circuit_energy(circuit, observable, p_minus, simulator=simulator, shots=shots)

    diff = (e_plus - e_minus) / (2.0 * c_k)
    grad = diff / delta  # delta_j is +/-1, so 1 / delta_j = delta_j

    return grad, delta


def quantum_fisher_information_matrix(
    circuit: QuantumCircuit,
    parameters: Sequence[float],
    eps: float = 1e-5
) -> np.ndarray:
    """
    Computes the exact Quantum Fisher Information Matrix (QFIM) / Fubini-Study metric tensor:
        F_{jk} = 4 * Re[ <d_j psi | d_k psi> - <d_j psi | psi> <psi | d_k psi> ]
    
    Uses central finite differences of the statevector for general parameterizations.
    """
    params = np.asarray(parameters, dtype=np.float64)
    d = len(params)

    # Base statevector |psi(theta)>
    param_dict = dict(zip(circuit.parameters, params))
    psi = Statevector.from_instruction(circuit.assign_parameters(param_dict)).data

    # Numerical partial derivatives d_j |psi>
    d_psi = np.zeros((d, len(psi)), dtype=np.complex128)
    for j in range(d):
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[j] += eps
        p_minus[j] -= eps

        dict_plus = dict(zip(circuit.parameters, p_plus))
        dict_minus = dict(zip(circuit.parameters, p_minus))

        psi_plus = Statevector.from_instruction(circuit.assign_parameters(dict_plus)).data
        psi_minus = Statevector.from_instruction(circuit.assign_parameters(dict_minus)).data

        d_psi[j] = (psi_plus - psi_minus) / (2.0 * eps)

    # Overlaps <psi | d_j psi>
    overlaps = np.array([np.vdot(psi, d_psi[j]) for j in range(d)])

    # Metric tensor F_{jk}
    F = np.zeros((d, d), dtype=np.float64)
    for j in range(d):
        for k in range(j, d):
            inner_jk = np.vdot(d_psi[j], d_psi[k])
            metric_val = 4.0 * np.real(inner_jk - np.conj(overlaps[j]) * overlaps[k])
            F[j, k] = metric_val
            F[k, j] = metric_val

    return F


def quantum_natural_gradient_step(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    parameters: Sequence[float],
    lr: float = 0.1,
    damping: float = 1e-3,
    qfim: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes one step of Quantum Natural Gradient (QNG):
        grad_nat = (F + damping * I)^(-1) * grad_euclidean
        theta_{t+1} = theta_t - lr * grad_nat
    
    Returns:
        (new_parameters, grad_nat, F)
    """
    params = np.asarray(parameters, dtype=np.float64)
    grad_euc = parameter_shift_gradient(circuit, observable, params)

    if qfim is None:
        F = quantum_fisher_information_matrix(circuit, params)
    else:
        F = np.asarray(qfim, dtype=np.float64)

    # Damped inversion (Levenberg-Marquardt regularization)
    d = len(params)
    F_reg = F + damping * np.eye(d)
    grad_nat = np.linalg.solve(F_reg, grad_euc)

    new_params = params - lr * grad_nat
    return new_params, grad_nat, F


def explain_quantum_gradients() -> str:
    """
    Academic explanation of quantum gradients for student viva defense.
    """
    return (
        "### Quantum Gradient Methods & Riemannian Optimization\n\n"
        "1. **Exact Parameter-Shift Rule**:\n"
        "   Unlike classical automatic differentiation (which traverses computational graphs in software),\n"
        "   quantum gradients cannot be backpropagated through non-clonable quantum wavefunctions.\n"
        "   The parameter-shift rule leverages the trigonometric structure of unitary rotations exp(-i*theta*G/2)\n"
        "   to compute EXACT derivatives via macro-shifts (+/- pi/2), completely immune to finite-difference\n"
        "   precision errors and executable directly on physical quantum hardware.\n\n"
        "2. **SPSA for Scaling Under Noise**:\n"
        "   While parameter-shift requires 2*D circuit evaluations per gradient step, SPSA perturbs all\n"
        "   coordinates simultaneously using a random Rademacher vector, reducing execution complexity to 2 evaluations\n"
        "   regardless of parameter count D. It acts as the quantum counterpart of classical stochastic gradients.\n\n"
        "3. **Quantum Natural Gradient (QNG)**:\n"
        "   Euclidean gradients assume parameters live in flat Cartesian space. Quantum Natural Gradient replaces\n"
        "   Euclidean geometry with the Fubini-Study metric (Quantum Fisher Information Matrix F),\n"
        "   ensuring constant-velocity transport along the Riemannian manifold of quantum states.\n"
        "   Damping with lambda*I prevents singularity on ill-conditioned or over-parameterized circuits.\n"
    )
