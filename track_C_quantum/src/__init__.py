"""
Track C: Quantum Optimization (QAOA, VQE, PQCs, Quantum Gradients, Noise & Barren Plateaus)
"""

from track_C_quantum.src.qaoa_maxcut import (
    CostHamiltonian,
    build_qaoa_circuit,
    QAOAMaxCutSolver,
    parse_qiskit_bitstring,
    explain_qaoa_depth_and_p,
)
from track_C_quantum.src.vqe_ising import (
    IsingPauliHamiltonian,
    HardwareEfficientAnsatz,
    VQESolver,
    explain_vqe_foundations,
)
from track_C_quantum.src.quantum_gradients import (
    parameter_shift_gradient,
    finite_difference_gradient,
    spsa_gradient,
    quantum_fisher_information_matrix,
    quantum_natural_gradient_step,
    explain_quantum_gradients,
)
from track_C_quantum.src.pqc_qnn import (
    generate_concentric_circles_dataset,
    ZZFeatureMap,
    QuantumClassifierPQC,
    ClassicalMLPClassifier,
    explain_pqc_and_qnn,
)
from track_C_quantum.src.quantum_noise_experiments import (
    build_depolarizing_noise_model,
    build_readout_noise_model,
    build_combined_nisq_noise_model,
    evaluate_finite_shot_variance,
    evaluate_noise_impact,
    run_barren_plateaus_study,
    explain_noise_and_barren_plateaus,
)

__all__ = [
    # QAOA
    "CostHamiltonian",
    "build_qaoa_circuit",
    "QAOAMaxCutSolver",
    "parse_qiskit_bitstring",
    "explain_qaoa_depth_and_p",
    # VQE
    "IsingPauliHamiltonian",
    "HardwareEfficientAnsatz",
    "VQESolver",
    "explain_vqe_foundations",
    # Quantum Gradients
    "parameter_shift_gradient",
    "finite_difference_gradient",
    "spsa_gradient",
    "quantum_fisher_information_matrix",
    "quantum_natural_gradient_step",
    "explain_quantum_gradients",
    # PQC & QNN
    "generate_concentric_circles_dataset",
    "ZZFeatureMap",
    "QuantumClassifierPQC",
    "ClassicalMLPClassifier",
    "explain_pqc_and_qnn",
    # Noise & Barren Plateaus
    "build_depolarizing_noise_model",
    "build_readout_noise_model",
    "build_combined_nisq_noise_model",
    "evaluate_finite_shot_variance",
    "evaluate_noise_impact",
    "run_barren_plateaus_study",
    "explain_noise_and_barren_plateaus",
]
