"""
quantum_noise_experiments.py
Track C: Finite-Shot Measurement Statistics, Quantum Noise Models, and Barren Plateaus.

Theoretical Foundations:
1. Finite-Shot Measurement Variance:
   On physical quantum computers and shot-based simulators, quantum expectation values
   are estimated from M discrete projective measurements:
       <O>_hat = 1/M * sum_{m=1}^M o_m
   By the Central Limit Theorem, the empirical variance scales inversely with shot count:
       Var[<O>_hat] = ( <O^2> - <O>^2 ) / M  ~  O(1 / M)
       Std[<O>_hat] ~ O(1 / sqrt(M))
   Comparing M in {100, 1000, 10000} empirically verifies the sqrt(10) ~ 3.16x reduction
   in standard error per 10x increase in quantum shot budget.

2. Quantum Hardware Noise Models:
   Real NISQ quantum processors suffer from physical decoherence and gate infidelities:
   - Depolarizing Noise: Models isotropic loss of quantum information.
     Single-qubit: E(rho) = (1 - p1)*rho + p1/3 * (X*rho*X + Y*rho*Y + Z*rho*Z)
     Two-qubit: Entangling gates (CX/RZZ) have ~10x higher error rates (p2 ~ 10*p1).
   - Readout (Measurement) Noise: Asymmetric or symmetric bit-flip errors during projective readout:
     P(measure 1 | state 0) = p_meas,  P(measure 0 | state 1) = p_meas.

3. Barren Plateau Phenomenon (McClean et al., 2018):
   For random parameterized quantum circuits with Haar-distributed random parameter vectors,
   the variance of the gradient of local/global observables vanishes exponentially with qubit count N:
       Var_theta[ d<H>/d theta_1 ] ~ O( 1 / 2^N )
   Consequently, for N >= 10, gradients become exponentially flat across the parameter landscape,
   rendering standard gradient-based optimizers incapable of distinguishing downhill directions
   from statistical shot noise without specialized initialization (identity blocks, local cost functions).
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from track_C_quantum.src.quantum_gradients import parameter_shift_gradient


def build_depolarizing_noise_model(
    p1: float = 0.01,
    p2: float = 0.05
) -> NoiseModel:
    """
    Constructs a Qiskit Aer NoiseModel with 1-qubit and 2-qubit depolarizing gate errors.
    """
    nm = NoiseModel()
    err_1q = depolarizing_error(p1, 1)
    err_2q = depolarizing_error(p2, 2)

    nm.add_all_qubit_quantum_error(err_1q, ["rx", "ry", "rz", "h"])
    nm.add_all_qubit_quantum_error(err_2q, ["cx", "rzz"])
    return nm


def build_readout_noise_model(p_meas: float = 0.03) -> NoiseModel:
    """
    Constructs a Qiskit Aer NoiseModel with symmetric bit-flip readout error.
    """
    nm = NoiseModel()
    ro_err = ReadoutError([[1.0 - p_meas, p_meas], [p_meas, 1.0 - p_meas]])
    nm.add_all_qubit_readout_error(ro_err)
    return nm


def build_combined_nisq_noise_model(
    p1: float = 0.005,
    p2: float = 0.02,
    p_meas: float = 0.03
) -> NoiseModel:
    """
    Constructs a realistic NISQ superconducting transmon noise model combining
    single-qubit, two-qubit, and readout errors.
    """
    nm = NoiseModel()
    err_1q = depolarizing_error(p1, 1)
    err_2q = depolarizing_error(p2, 2)
    ro_err = ReadoutError([[1.0 - p_meas, p_meas], [p_meas, 1.0 - p_meas]])

    nm.add_all_qubit_quantum_error(err_1q, ["rx", "ry", "rz", "h"])
    nm.add_all_qubit_quantum_error(err_2q, ["cx", "rzz"])
    nm.add_all_qubit_readout_error(ro_err)
    return nm


def evaluate_finite_shot_variance(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    shots_list: Sequence[int] = (100, 1000, 10000),
    n_trials: int = 30,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Empirically measures expectation value variance across different shot budgets M.
    Verifies that standard error scales as O(1 / sqrt(M)).
    """
    # Exact ground truth from statevector
    meas_circ = circuit.copy()
    if meas_circ.num_clbits == 0:
        meas_circ.measure_all()

    ideal_energy = float(Statevector.from_instruction(circuit).expectation_value(observable).real)
    diag_vals = np.real(np.diag(observable.to_matrix()))

    shot_stats = []
    for shots in shots_list:
        trial_estimates = []
        for t in range(n_trials):
            sim = AerSimulator(seed_simulator=seed + t * 101 + shots)
            job = sim.run(meas_circ, shots=shots)
            counts = job.result().get_counts()

            sample_sum = 0.0
            for bstr, count in counts.items():
                clean_b = bstr.replace(" ", "")
                idx = int(clean_b, 2)
                sample_sum += count * diag_vals[idx]
            trial_estimates.append(sample_sum / shots)

        mean_est = float(np.mean(trial_estimates))
        var_est = float(np.var(trial_estimates, ddof=1))
        std_est = float(np.std(trial_estimates, ddof=1))
        scale_constant = std_est * np.sqrt(shots)  # Should remain ~ constant across shots

        shot_stats.append({
            "shots": int(shots),
            "n_trials": int(n_trials),
            "ideal_expectation": ideal_energy,
            "mean_estimate": mean_est,
            "absolute_bias": abs(mean_est - ideal_energy),
            "variance": var_est,
            "std_error": std_est,
            "scaling_metric": scale_constant
        })

    return {
        "ideal_expectation": ideal_energy,
        "results": shot_stats
    }


def evaluate_noise_impact(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    shots: int = 2048,
    p1: float = 0.01,
    p2: float = 0.04,
    p_meas: float = 0.03,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates circuit expectation value under 4 distinct hardware noise environments:
    1. Ideal (Noiseless)
    2. Readout noise only
    3. Depolarizing gate noise only
    4. Combined NISQ noise
    """
    meas_circ = circuit.copy()
    if meas_circ.num_clbits == 0:
        meas_circ.measure_all()

    diag_vals = np.real(np.diag(observable.to_matrix()))
    ideal_energy = float(Statevector.from_instruction(circuit).expectation_value(observable).real)

    noise_environments = [
        ("Ideal (Noiseless)", None),
        ("Readout Noise Only", build_readout_noise_model(p_meas)),
        ("Depolarizing Noise Only", build_depolarizing_noise_model(p1, p2)),
        ("Combined NISQ Model", build_combined_nisq_noise_model(p1, p2, p_meas)),
    ]

    records = []
    for env_name, nm in noise_environments:
        sim = AerSimulator(noise_model=nm, seed_simulator=seed)
        tqc = transpile(meas_circ, sim)
        job = sim.run(tqc, shots=shots)
        counts = job.result().get_counts()

        total = 0.0
        for bstr, count in counts.items():
            clean_b = bstr.replace(" ", "")
            idx = int(clean_b, 2)
            total += count * diag_vals[idx]
        measured_energy = total / shots
        shift = abs(measured_energy - ideal_energy)

        records.append({
            "environment": env_name,
            "measured_energy": float(measured_energy),
            "ideal_energy": float(ideal_energy),
            "energy_shift": float(shift),
            "relative_error": float(shift / max(1e-9, abs(ideal_energy)))
        })

    return {
        "ideal_energy": ideal_energy,
        "environments": records
    }


def run_barren_plateaus_study(
    qubit_range: Sequence[int] = range(2, 11),
    n_samples: int = 40,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Empirical verification of McClean et al. (2018) barren plateau phenomenon.
    Evaluates the variance of partial derivatives Var[ d<H>/d theta_1 ] across qubit counts
    N in [2..10] for deep random hardware-efficient ansatzes.
    """
    results = []
    rng = np.random.RandomState(seed)

    for n in qubit_range:
        t0 = time.perf_counter()
        # Observable: global parity Z^(tensor n)
        H = SparsePauliOp.from_list([("Z" * n, 1.0)])

        # Construct random ansatz with depth L = n layers
        num_layers = n
        num_params = n * (num_layers + 1)
        pv = ParameterVector("theta", num_params)

        qc = QuantumCircuit(n)
        param_idx = 0
        for i in range(n):
            qc.ry(pv[param_idx], i)
            param_idx += 1

        for _ in range(num_layers):
            for i in range(n - 1):
                qc.cx(i, i + 1)
            for i in range(n):
                qc.ry(pv[param_idx], i)
                param_idx += 1

        # Sample gradients for parameter theta_0
        gradients = []
        for _ in range(n_samples):
            rand_angles = rng.uniform(-np.pi, np.pi, size=num_params)
            
            # Exact parameter shift on coordinate 0
            p_plus = rand_angles.copy()
            p_minus = rand_angles.copy()
            p_plus[0] += np.pi / 2.0
            p_minus[0] -= np.pi / 2.0

            dict_plus = dict(zip(qc.parameters, p_plus))
            dict_minus = dict(zip(qc.parameters, p_minus))

            sv_plus = Statevector.from_instruction(qc.assign_parameters(dict_plus))
            sv_minus = Statevector.from_instruction(qc.assign_parameters(dict_minus))

            e_plus = sv_plus.expectation_value(H).real
            e_minus = sv_minus.expectation_value(H).real

            grad = 0.5 * (e_plus - e_minus)
            gradients.append(float(grad))

        mean_grad = float(np.mean(gradients))
        var_grad = float(np.var(gradients))
        runtime_ms = (time.perf_counter() - t0) * 1000.0

        results.append({
            "num_qubits": int(n),
            "num_parameters": int(num_params),
            "circuit_depth": qc.depth(),
            "n_samples": int(n_samples),
            "gradient_mean": mean_grad,
            "gradient_variance": var_grad,
            "log10_variance": float(np.log10(max(1e-15, var_grad))),
            "runtime_ms": runtime_ms
        })

    # Linear fit of log10(Var) vs N
    n_vals = np.array([r["num_qubits"] for r in results])
    log_vars = np.array([r["log10_variance"] for r in results])
    slope, intercept = np.polyfit(n_vals, log_vars, 1)

    return {
        "results": results,
        "slope": float(slope),
        "intercept": float(intercept),
        "decay_factor_per_qubit": float(10 ** slope)
    }


def explain_noise_and_barren_plateaus() -> str:
    """Academic explanation for student viva defense."""
    return (
        "### Quantum Noise Models and Barren Plateau Scaling\n\n"
        "1. **Finite-Shot Measurement Scaling**:\n"
        "   Quantum mechanics dictates that observable expectation values are reconstructed\n"
        "   from discrete projective measurements. The standard deviation scales strictly as O(1/sqrt(M)).\n"
        "   Increasing shots from 100 to 10,000 reduces statistical variance by a factor of ~100x\n"
        "   (standard error drops ~10x), representing a critical computational cost/accuracy trade-off.\n\n"
        "2. **Hardware Noise (Depolarizing & Readout)**:\n"
        "   - Readout Error: Distorts observed bitstring probabilities, shifting diagonal expectation values\n"
        "     without altering the underlying quantum state before measurement.\n"
        "   - Depolarizing Noise: Contracts the density matrix towards the maximally mixed state I/2^n,\n"
        "     attenuating quantum expectation values towards zero and degrading ground-state fidelity.\n\n"
        "3. **Barren Plateau Phenomenon**:\n"
        "   In deep, unstructured parameterized circuits, the Haar measure over the unitary group U(2^n)\n"
        "   induces concentration of measure. As qubit count N grows from 2 to 10, the gradient variance\n"
        "   vanishes exponentially: Var[d<H>/d theta] ~ O(1 / 2^N). For large N, gradient magnitudes\n"
        "   fall below physical shot noise, causing gradient descent to stall unless specialized ansatzes\n"
        "   (e.g., shallow circuits, local cost functions, or pre-trained initialization) are deployed.\n"
    )
