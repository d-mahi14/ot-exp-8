"""
Track C: Quantum Optimization (QAOA, VQE, PQCs)
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

__all__ = [
    "CostHamiltonian",
    "build_qaoa_circuit",
    "QAOAMaxCutSolver",
    "parse_qiskit_bitstring",
    "explain_qaoa_depth_and_p",
    "IsingPauliHamiltonian",
    "HardwareEfficientAnsatz",
    "VQESolver",
    "explain_vqe_foundations",
]
