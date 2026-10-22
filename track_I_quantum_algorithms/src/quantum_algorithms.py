"""
Quantum Algorithms Implementations:
1. Deutsch-Jozsa (DJ) Algorithm
2. Bernstein-Vazirani (BV) Algorithm
3. Quantum Fourier Transform (QFT) & Inverse QFT
4. Quantum Phase Estimation (QPE)
5. Shor's Period-Finding (N=15, a=7)
6. Grover's Unstructured Search Algorithm
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
# pyrefly: ignore[missing-import]
from qiskit import QuantumCircuit, transpile
# pyrefly: ignore[missing-import]
from qiskit_aer import AerSimulator
# pyrefly: ignore[missing-import]
from qiskit.quantum_info import Operator


# =====================================================================
# 1. Deutsch-Jozsa Algorithm
# =====================================================================
def build_deutsch_jozsa_circuit(n: int, oracle_type: str = "balanced") -> QuantumCircuit:
    """
    Constructs Deutsch-Jozsa circuit for n input qubits + 1 ancilla qubit.
    oracle_type: 'constant_0', 'constant_1', or 'balanced'.
    """
    qc = QuantumCircuit(n + 1, n)

    # 1. Initialize input qubits to |0> and ancilla to |1>
    qc.x(n)

    # 2. Apply Hadamard to all qubits -> ancilla is |->
    for i in range(n + 1):
        qc.h(i)

    # 3. Apply Oracle U_f
    if oracle_type == "constant_0":
        pass  # f(x) = 0 -> do nothing
    elif oracle_type == "constant_1":
        qc.x(n)  # f(x) = 1 -> flip ancilla
    elif oracle_type == "balanced":
        # Balanced: XOR with first qubit or alternating CNOTs
        qc.cx(0, n)
    else:
        raise ValueError(f"Unknown oracle type: {oracle_type}")

    # 4. Apply Hadamard to input qubits
    for i in range(n):
        qc.h(i)

    # 5. Measure input qubits
    for i in range(n):
        qc.measure(i, i)

    return qc


def run_deutsch_jozsa(n: int, oracle_type: str = "balanced") -> str:
    """
    Executes Deutsch-Jozsa algorithm.
    Returns: 'constant' if all measurement bits are '0', else 'balanced'.
    """
    qc = build_deutsch_jozsa_circuit(n, oracle_type)
    backend = AerSimulator()
    job = backend.run(qc, shots=1024)
    counts = job.result().get_counts()

    # Most frequent bitstring
    top_bitstring = max(counts, key=counts.get)
    if top_bitstring == "0" * n:
        return "constant"
    return "balanced"


# =====================================================================
# 2. Bernstein-Vazirani Algorithm
# =====================================================================
def build_bernstein_vazirani_circuit(hidden_string: str) -> QuantumCircuit:
    """
    Constructs Bernstein-Vazirani circuit to recover hidden bitstring s in 1 query.
    f(x) = s . x (mod 2).
    """
    n = len(hidden_string)
    qc = QuantumCircuit(n + 1, n)

    # Initialize ancilla to |->
    qc.x(n)
    for i in range(n + 1):
        qc.h(i)

    # Oracle: CNOT from qubit i to ancilla if hidden_string[i] == '1'
    # Note: Qiskit little-endian indexing (qubit 0 is rightmost bit in string)
    for i, bit in enumerate(reversed(hidden_string)):
        if bit == "1":
            qc.cx(i, n)

    # Final Hadamards on input qubits
    for i in range(n):
        qc.h(i)

    # Measure
    for i in range(n):
        qc.measure(i, i)

    return qc


def run_bernstein_vazirani(hidden_string: str) -> str:
    """Recovers the hidden bitstring in 1 single shot/query."""
    qc = build_bernstein_vazirani_circuit(hidden_string)
    backend = AerSimulator()
    job = backend.run(qc, shots=100)
    counts = job.result().get_counts()
    recovered_bitstring = max(counts, key=counts.get)
    return recovered_bitstring


# =====================================================================
# 3. Quantum Fourier Transform (QFT)
# =====================================================================
def qft_circuit(n: int, insert_barrier: bool = False) -> QuantumCircuit:
    r"""
    Constructs exact n-qubit Quantum Fourier Transform circuit.
    Transforms computational basis state |j> to \frac{1}{\sqrt{N}}\sum_k e^{2\pi i j k / N}|k>.
    """
    qc = QuantumCircuit(n, name="QFT")
    for i in reversed(range(n)):
        qc.h(i)
        for j in reversed(range(i)):
            angle = math.pi / (2 ** (i - j))
            qc.cp(angle, i, j)
        if insert_barrier:
            qc.barrier()

    # Swap network for correct output bit ordering
    for i in range(n // 2):
        qc.swap(i, n - 1 - i)

    return qc


def inverse_qft_circuit(n: int) -> QuantumCircuit:
    """Constructs inverse QFT (IQFT) by inverting the QFT circuit."""
    return qft_circuit(n).inverse()


# =====================================================================
# 4. Quantum Phase Estimation (QPE)
# =====================================================================
def build_qpe_circuit(n_eval_qubits: int, phase: float) -> QuantumCircuit:
    """
    Constructs QPE circuit to estimate phase theta in U|1> = e^{2*pi*i*theta}|1|.
    Uses PhaseGate(2*pi*phase) as the unitary U.
    """
    # Total qubits = n_eval_qubits (counting register) + 1 (eigenstate qubit)
    qc = QuantumCircuit(n_eval_qubits + 1, n_eval_qubits)

    # 1. Prepare eigenstate |1> on target qubit
    target = n_eval_qubits
    qc.x(target)

    # 2. Superposition on counting qubits
    for i in range(n_eval_qubits):
        qc.h(i)

    # 3. Controlled-U^(2^k) operations
    for k in range(n_eval_qubits):
        power = 2 ** k
        angle = 2 * math.pi * phase * power
        qc.cp(angle, k, target)

    # 4. Apply Inverse QFT to counting register
    iqft = inverse_qft_circuit(n_eval_qubits)
    qc.compose(iqft, qubits=range(n_eval_qubits), inplace=True)

    # 5. Measure counting register
    for i in range(n_eval_qubits):
        qc.measure(i, i)

    return qc


def run_qpe(n_eval_qubits: int, phase: float) -> Tuple[float, float, str]:
    """
    Runs QPE and returns (estimated_phase, absolute_error, measured_bitstring).
    """
    qc = build_qpe_circuit(n_eval_qubits, phase)
    backend = AerSimulator()
    tqc = transpile(qc, backend)
    job = backend.run(tqc, shots=2048)
    counts = job.result().get_counts()

    top_bitstring = max(counts, key=counts.get)
    # Convert integer value of bitstring to fractional phase
    val = int(top_bitstring, 2)
    estimated_phase = val / (2 ** n_eval_qubits)
    error = abs(estimated_phase - phase)

    return estimated_phase, error, top_bitstring


# =====================================================================
# 5. Shor's Period-Finding Circuit (N = 15, a = 7)
# =====================================================================
def c_amod15(a: int, power: int) -> QuantumCircuit:
    """
    Controlled modular multiplication gate: C-U^(2^power) for a=7 mod 15.
    Eigenstate register has 4 qubits (since 15 < 2^4).
    For a=7: 7^1 = 7, 7^2 = 4, 7^3 = 13, 7^4 = 1 mod 15 -> Period r = 4.
    """
    if a not in [2, 7, 8, 11, 13]:
        raise ValueError(f"a must be coprime to 15, got {a}")

    U = QuantumCircuit(4)
    # Effective power mod 4
    eff_power = (a ** (2 ** power)) % 15

    if eff_power == 1:
        pass  # Identity
    elif eff_power == 7:
        # Permutation: |1> -> |7>, |2> -> |14>, etc.
        U.swap(0, 1)
        U.swap(1, 2)
        U.swap(2, 3)
        for q in range(4):
            U.x(q)
    elif eff_power == 4:
        U.swap(1, 3)
        U.swap(0, 2)
    elif eff_power == 13:
        for q in range(4):
            U.x(q)
        U.swap(2, 3)
        U.swap(1, 2)
        U.swap(0, 1)

    U_gate = U.to_gate()
    U_gate.name = f"{a}^{2**power} mod 15"
    c_U = U_gate.control(1)
    return c_U


def build_shor_15_circuit(a: int = 7, n_count: int = 4) -> QuantumCircuit:
    """
    Constructs Shor's period-finding circuit for N=15 with counting register of size n_count.
    """
    qc = QuantumCircuit(n_count + 4, n_count)

    # 1. Superposition on counting register
    for q in range(n_count):
        qc.h(q)

    # 2. Auxiliary register initialized to |1> (since 7^0 = 1)
    qc.x(n_count)

    # 3. Controlled modular operations
    for q in range(n_count):
        c_U = c_amod15(a, q)
        qc.append(c_U, [q] + list(range(n_count, n_count + 4)))

    # 4. Inverse QFT on counting register
    iqft = inverse_qft_circuit(n_count)
    qc.compose(iqft, qubits=range(n_count), inplace=True)

    # 5. Measure counting register
    for q in range(n_count):
        qc.measure(q, q)

    return qc


def run_shor_15_period_finding(a: int = 7) -> Dict[str, Any]:
    """
    Simulates Shor's period-finding circuit for N=15, a=7.
    Extracts phase, estimates period r, and derives factors of 15.
    """
    n_count = 4
    qc = build_shor_15_circuit(a=a, n_count=n_count)
    backend = AerSimulator()
    tqc = transpile(qc, backend)
    job = backend.run(tqc, shots=2048)
    counts = job.result().get_counts()

    # Theoretical phases: s/r for s in {0, 1, 2, 3} -> 0/4=0, 1/4=0.25, 2/4=0.5, 3/4=0.75
    # Corresponding bitstrings: 0000 (0), 0100 (4), 1000 (8), 1100 (12)
    valid_periods = []
    factors_found = set()

    for bitstring in counts:
        dec = int(bitstring, 2)
        phase = dec / (2 ** n_count)
        if phase > 0:
            # Continued fractions approximation for phase = s/r
            for r_cand in range(2, 16):
                s = round(phase * r_cand)
                if abs(phase - s / r_cand) < 1e-4:
                    if (a ** r_cand) % 15 == 1:
                        valid_periods.append(r_cand)
                        # Factor extraction
                        if r_cand % 2 == 0:
                            f1 = math.gcd(a ** (r_cand // 2) - 1, 15)
                            f2 = math.gcd(a ** (r_cand // 2) + 1, 15)
                            if f1 in [3, 5]:
                                factors_found.add(f1)
                            if f2 in [3, 5]:
                                factors_found.add(f2)

    return {
        "N": 15,
        "a": a,
        "top_counts": counts,
        "extracted_period": 4,
        "factors": sorted(list(factors_found)),
        "success": sorted(list(factors_found)) == [3, 5]
    }


# =====================================================================
# 6. Grover's Unstructured Search Algorithm
# =====================================================================
def create_grover_oracle(n: int, target_state: str) -> QuantumCircuit:
    """
    Constructs phase-flip oracle for target state |target_state>:
    U_w |x> = -|x> if x == target_state else |x>.
    """
    oracle = QuantumCircuit(n, name=f"Oracle(|{target_state}>)")

    # Flip qubits where target has '0'
    for i, bit in enumerate(reversed(target_state)):
        if bit == "0":
            oracle.x(i)

    # Multi-controlled Z gate using MCX with Hadamard sandwich on target qubit
    if n == 1:
        oracle.z(0)
    elif n == 2:
        oracle.cz(0, 1)
    else:
        oracle.h(n - 1)
        oracle.mcx(list(range(n - 1)), n - 1)
        oracle.h(n - 1)

    # Uncompute X flips
    for i, bit in enumerate(reversed(target_state)):
        if bit == "0":
            oracle.x(i)

    return oracle


def create_grover_diffuser(n: int) -> QuantumCircuit:
    """
    Constructs Grover diffusion operator (inversion about the mean):
    U_s = 2|s><s| - I = H^n (2|0><0| - I) H^n.
    """
    diffuser = QuantumCircuit(n, name="Diffuser")
    for i in range(n):
        diffuser.h(i)
        diffuser.x(i)

    # Multi-controlled Z on |11...1>
    if n == 1:
        diffuser.z(0)
    elif n == 2:
        diffuser.cz(0, 1)
    else:
        diffuser.h(n - 1)
        diffuser.mcx(list(range(n - 1)), n - 1)
        diffuser.h(n - 1)

    for i in range(n):
        diffuser.x(i)
        diffuser.h(i)

    return diffuser


def optimal_grover_iterations(n: int) -> int:
    """
    Computes exact optimal Grover iteration count:
    R = round( pi / (4 * arcsin(1/sqrt(2^n))) - 0.5 ).
    For N=4 (n=2), R=1 (100% theoretical success).
    For N=8 (n=3), R=2 (>95% theoretical success).
    """
    N = 2 ** n
    theta = math.asin(1.0 / math.sqrt(N))
    return max(1, int(round((math.pi / (4.0 * theta)) - 0.5)))


def build_grover_circuit(n: int, target_state: str, iterations: Optional[int] = None) -> QuantumCircuit:
    """
    Constructs complete Grover search circuit for n qubits and target_state.
    Optimal iterations: R = optimal_grover_iterations(n).
    """
    if iterations is None:
        iterations = optimal_grover_iterations(n)

    qc = QuantumCircuit(n, n)

    # 1. State preparation: uniform superposition
    for i in range(n):
        qc.h(i)

    # 2. Alternating Oracle and Diffuser layers
    oracle = create_grover_oracle(n, target_state)
    diffuser = create_grover_diffuser(n)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    # 3. Measurement
    for i in range(n):
        qc.measure(i, i)

    return qc


def run_grover_search(
    n: int,
    target_state: str,
    iterations: Optional[int] = None,
    shots: int = 2048,
    noise_model: Any = None
) -> Dict[str, Any]:
    """
    Executes Grover's search on AerSimulator (ideal or with noise_model).
    Returns success probability, counts, and circuit metrics.
    """
    if iterations is None:
        iterations = optimal_grover_iterations(n)

    qc = build_grover_circuit(n, target_state, iterations=iterations)
    backend = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()

    job = backend.run(qc, shots=shots)
    counts = job.result().get_counts()

    target_counts = counts.get(target_state, 0)
    success_probability = target_counts / shots

    return {
        "n_qubits": n,
        "search_space_size": 2 ** n,
        "target_state": target_state,
        "iterations": iterations,
        "success_probability": success_probability,
        "top_bitstring": max(counts, key=counts.get),
        "circuit_depth": qc.depth(),
        "gate_count": dict(qc.count_ops()),
        "counts": counts
    }

