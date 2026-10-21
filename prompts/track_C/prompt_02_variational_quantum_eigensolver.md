# Prompt Record 10: Track C Batch 2 — Variational Quantum Eigensolver (VQE)

- **Prompt ID**: `P-TRC-02`
- **Date & Time**: 2026-10-21 19:55:04 IST
- **Track**: Track C (Quantum Optimization)
- **AI Tool**: Antigravity (Powered by Gemini 3.8 Flash)
- **Commit Reference**: Pending commit

---

### Purpose
Implement Batch 2 of Track C (Quantum Optimization):
1. Construct the Ising Hamiltonian as mutually commuting Pauli operators (`SparsePauliOp`) in Qiskit.
2. Build a Hardware-Efficient Parameterized Quantum Circuit (`RealAmplitudes` / $R_y$-$CX$) restricted to the real computational subspace.
3. Implement energy estimation via exact statevector and finite-shot $Z$-basis measurement sampling on `AerSimulator`.
4. Optimize variational parameters using classical `COBYLA` optimizer.
5. Benchmark VQE against exact Brute-Force ground state energy and classical Simulated Annealing across 8 graph topologies.
6. Conduct multi-trial evaluations (5 independent seeds per configuration) to explicitly measure empirical variability, success rates, and local minima traps.
7. Add independent unit tests covering Hamiltonian construction, ansatz parameterization, energy evaluation, VQE convergence, and ground-state sampling.
8. Save results as CSV and Markdown, and generate comparison plots.

---

### Original Prompt
```text
Continue my Optimization Techniques Lab Assignment 8 in C:\dev\ot-8.

Implement ONLY Track C — Batch 2: Variational Quantum Eigensolver (VQE).

### Tasks

1. Inspect CHAPTER_6_CHECKLIST.md, the existing QAOA implementation, Track B's QUBO/Ising code, tests, benchmark scripts, and prompt logs. Reuse suitable code.

2. Implement VQE using Qiskit and AerSimulator to find the ground-state energy of small Ising Hamiltonians.
   - Build a parameterized quantum circuit (ansatz).
   - Construct the Hamiltonian using Pauli operators.
   - Estimate energy from the circuit state or measurements.
   - Optimize parameters using a classical optimizer such as COBYLA.
   - Clearly document the ansatz, objective function, optimizer settings, and stopping criteria.
   - Verify that circuit execution is genuinely performed by the quantum simulator; do not substitute a classical solver for VQE.

3. Use small Max-Cut/Ising instances that can be solved exactly by brute force. Compare VQE against the exact ground-state energy and report absolute energy error, solution quality, runtime, circuit depth, and number of qubits.

4. Set random seeds where supported. Run repeated trials when practical and report actual outcomes, including failed or non-optimal runs.

5. Add independent tests for Hamiltonian construction, energy evaluation, circuit execution, parameter validation, and comparison with exact solutions on small instances.

6. Save actual benchmark results as CSV and Markdown, with clear plots. Include all relevant simulator and optimizer settings.

7. Update CHAPTER_6_CHECKLIST.md, PROMPT_LOG.md, and a verbatim prompt record under prompts/track_C/. Regenerate AI_Prompt_Log.pdf.

8. Record only genuine AI mistakes and fixes in AI_FAILURE_LOG.md. Do not invent results or claim quantum hardware experiments.

9. Run the full test suite with py -3.12 -m pytest -v. Report the actual number of tests passed/failed and investigate any failures.

10. Commit the work locally with a commit message beginning [AI:Antigravity].

### Constraints

- Do not start PQCs/QNNs, parameter-shift gradients, SPSA, Quantum Natural Gradient, noise experiments, or barren-plateau analysis. These belong to later batches.
- Do not create a GitHub repository, add a remote, push, publish, or upload anything.
- Do not switch branches or rewrite Git history.
- Do not claim VQE is better than classical optimization based only on small-instance results.
- Keep the implementation reproducible, modular, and explainable for a viva.
- Stop after Batch 2. Summarize changed files, dependencies, actual test results, benchmark results, commit hash, and remaining issues.
```

---

### AI Response Summary
- Created `track_C_quantum/src/vqe_ising.py`:
  - `IsingPauliHamiltonian`: Maps Ising couplings $J_{ij}$, fields $h_i$, and constant offset $C_0$ to Qiskit `SparsePauliOp` using standard little-endian indexing. Computes exact ground-state energy and diagonal spectrum.
  - `HardwareEfficientAnsatz`: Parameterized RealAmplitudes ansatz ($R_y$-$CX$) with linear, circular, and full entanglement modes, leveraging real subspace restriction for Ising ground states.
  - `VQESolver`: Evaluates energy expectation $\langle\psi(\boldsymbol{\theta})|H|\psi(\boldsymbol{\theta})\rangle$ on statevector and `AerSimulator`, optimizes via `COBYLA`, measures ground state overlap with shots, and supports multi-trial evaluations.
  - `explain_vqe_foundations`: Rigorous viva text covering the Rayleigh-Ritz variational principle, zero-basis-rotation commutativity of Ising models, and classical vs quantum performance realities.
- Created `track_C_quantum/tests/test_vqe_ising.py`:
  - 11 unit tests covering Hamiltonian construction, Track B equivalence, ansatz parameterization, statevector vs shot evaluation, analytic ground truth on 2-qubit, $K_3$, and $C_4$ instances, multi-trial variance, and explanation text.
- Created `track_C_quantum/experiments/run_vqe_benchmarks.py`:
  - Evaluated 8 graph topologies across ansatz depths $L \in \{1, 2\}$ and 5 independent seeds (80 total VQE runs).
  - Recorded best energy, mean energy, std, success rate, and explicit local-minima non-optimal runs.
  - Generated `vqe_benchmark_metrics.csv`, `vqe_benchmark_metrics.md`, and `vqe_benchmarks.png`.
- Updated `CHAPTER_6_CHECKLIST.md`, `PROMPT_LOG.md`, and recompiled `AI_Prompt_Log.pdf`.

---

### Verification & Testing
- Track C VQE Tests: `py -3.12 -m pytest track_C_quantum/tests/test_vqe_ising.py -v` (11/11 passed in 2.59s).
- Full Test Suite: `py -3.12 -m pytest -v` (68/68 passed across Tracks A, B, and C).
- Benchmark Execution: 80 VQE trial optimizations + AerSimulator sampling runs completed cleanly.
