"""
Quantum Noise Evaluation:
Constructs documented NISQ hardware noise models (depolarizing and readout error)
and evaluates quantum algorithms (e.g., Grover's search) under ideal vs noisy simulation.
Clearly distinguishes ideal simulator, noisy simulator, and real hardware constraints.
"""

from typing import Dict, List, Any, Optional
# pyrefly: ignore[missing-import]
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
# pyrefly: ignore[missing-import]
from qiskit_aer import AerSimulator
from track_I_quantum_algorithms.src.quantum_algorithms import build_grover_circuit


def build_hardware_noise_model(
    p1_error: float = 0.001,
    p2_error: float = 0.012,
    p_meas01: float = 0.020,
    p_meas10: float = 0.015
) -> NoiseModel:
    """
    Constructs a documented realistic hardware noise model based on typical
    superconducting transmon quantum processor parameters (e.g., IBM Falcon/Eagle tier):
      - Single-qubit depolarizing error rate: p1 = 0.1%
      - Two-qubit gate (CX/CZ) depolarizing error rate: p2 = 1.2%
      - Readout assignment error: P(0|1) = 2.0%, P(1|0) = 1.5%
    """
    noise_model = NoiseModel()

    # 1. Single-qubit gate errors (X, H, P, etc.)
    err_1 = depolarizing_error(p1_error, 1)
    noise_model.add_all_qubit_quantum_error(err_1, ['h', 'x', 'p', 'z', 'sx', 'rz'])

    # 2. Two-qubit gate errors (CX, CZ)
    err_2 = depolarizing_error(p2_error, 2)
    noise_model.add_all_qubit_quantum_error(err_2, ['cx', 'cz'])

    # 3. Readout assignment errors
    ro_probabilities = [[1.0 - p_meas10, p_meas10], [p_meas01, 1.0 - p_meas01]]
    readout_err = ReadoutError(ro_probabilities)
    noise_model.add_all_qubit_readout_error(readout_err)

    return noise_model


def evaluate_grover_noise_impact(
    qubit_range: List[int] = [2, 3, 4, 5],
    shots: int = 4096,
    noise_model: Optional[NoiseModel] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates Grover's search across varying qubit numbers comparing:
      - Ideal statevector / shots simulator
      - Documented NISQ hardware noise model
    Measures the decay in success probability as circuit depth and multi-qubit gate count increase.
    """
    if noise_model is None:
        noise_model = build_hardware_noise_model()

    ideal_backend = AerSimulator()
    noisy_backend = AerSimulator(noise_model=noise_model)

    results = []

    for n in qubit_range:
        # Target state: all ones "|11...1>"
        target_state = "1" * n
        qc = build_grover_circuit(n, target_state)

        # Decompose circuit to count fundamental 1-qubit and 2-qubit gates
        decomp_qc = qc.decompose()
        # Decompose once more to resolve MCX into CX/H gates
        for _ in range(2):
            decomp_qc = decomp_qc.decompose()

        depth = decomp_qc.depth()
        gate_counts = dict(decomp_qc.count_ops())
        cx_count = gate_counts.get("cx", 0) + gate_counts.get("cz", 0)

        # 1. Ideal simulation
        job_ideal = ideal_backend.run(qc, shots=shots)
        counts_ideal = job_ideal.result().get_counts()
        p_success_ideal = counts_ideal.get(target_state, 0) / shots

        # 2. Noisy simulation
        job_noisy = noisy_backend.run(qc, shots=shots)
        counts_noisy = job_noisy.result().get_counts()
        p_success_noisy = counts_noisy.get(target_state, 0) / shots

        rel_degradation = (
            (p_success_ideal - p_success_noisy) / p_success_ideal
            if p_success_ideal > 0 else 0.0
        )

        results.append({
            "n_qubits": n,
            "search_space_N": 2 ** n,
            "target_state": target_state,
            "circuit_depth": depth,
            "two_qubit_cx_count": cx_count,
            "shots": shots,
            "p_success_ideal": float(p_success_ideal),
            "p_success_noisy": float(p_success_noisy),
            "relative_degradation_pct": float(rel_degradation * 100.0),
            "hardware_category": "Noisy Simulator (IBM Falcon-like depolarizing + readout model)",
            "real_hardware_status": "Simulated (Live quantum hardware requires IBM Quantum cloud credentials)"
        })

    return results
