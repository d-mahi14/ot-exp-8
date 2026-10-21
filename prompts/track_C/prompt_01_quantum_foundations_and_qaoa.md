# Prompt Record 09: Track C Batch 1 — Quantum Optimization Foundations and QAOA

- **Prompt ID**: `P-TRC-01`
- **Date & Time**: 2026-10-21 19:35:28 IST
- **Track**: Track C (Quantum Optimization)
- **AI Tool**: Antigravity (Powered by Gemini 3.8 Flash)
- **Commit Reference**: Pending commit

---

### Purpose
Implement Batch 1 of Track C (Quantum Optimization):
1. Construct the cost Hamiltonian and parameterized quantum circuit for Graph Max-Cut.
2. Implement alternating cost and mixer layers ($U(C, \gamma)$ and $U(B, \beta)$).
3. Implement classical variational parameter optimization (COBYLA) for $(\boldsymbol{\gamma}, \boldsymbol{\beta})$.
4. Evaluate measured bitstrings and calculate Max-Cut objective values and expectation.
5. Provide a rigorous mathematical explanation of circuit depth and parameter $p$.
6. Check quantum simulator packages in the environment and verify Qiskit Aer (`AerSimulator`).
7. Compare QAOA against exact Brute-Force and classical Simulated Annealing on identical graph instances.
8. Benchmark multiple QAOA depths ($p \in \{1, 2, 3\}$) across reproducible graph topologies with fixed random seeds.
9. Add independent unit tests covering Hamiltonian construction, circuit execution, objective evaluation, parameter validation, Farhi analytic solution, and cross-verification against Track B.
10. Generate publication-quality comparison plots and save results as CSV and Markdown.

---

### Original Prompt
```text
Continue my Optimization Techniques Lab Assignment 8 in C:\dev\ot-8.

Implement ONLY Track C — Batch 1: Quantum Optimization Foundations and QAOA.

### Tasks

1. First inspect CHAPTER_6_CHECKLIST.md, existing Track B QUBO/Ising code, tests, results, and prompt logs. Reuse existing implementations where appropriate.

2. Implement QAOA (Quantum Approximate Optimization Algorithm) for the existing Graph Max-Cut problem:
   - Construct the cost Hamiltonian and parameterized quantum circuit.
   - Implement the alternating cost and mixer layers.
   - Optimize circuit parameters using a classical optimizer.
   - Evaluate the resulting bitstrings and calculate the Max-Cut objective.
   - Explain circuit depth and the role of the QAOA parameter p.

3. Use an available quantum simulator, such as Qiskit Aer, if compatible with the existing environment. First inspect installed packages. If dependencies are missing, explain the options before making substantial environment changes. Do not pretend a classical calculation is a quantum-circuit simulation.

4. Compare QAOA against:
   - Exact brute force on small graphs.
   - The existing classical Simulated Annealing solver.
   - Use identical graph instances and report objective values, approximation ratios, and runtime.

5. Run experiments on several small reproducible graphs and test multiple QAOA depths where computationally practical. Set and record random seeds. Explain that stochastic results may vary.

6. Add independent unit tests for Hamiltonian construction, circuit execution, objective evaluation, parameter handling, and correctness on small graphs. Test against known answers where possible.

7. Save actual results as CSV and Markdown, and generate clear comparison plots. Include the number of qubits, QAOA depth, shots, optimizer settings, and simulator details in the results.

8. Update CHAPTER_6_CHECKLIST.md, PROMPT_LOG.md, and a verbatim prompt record under prompts/track_C/. Regenerate AI_Prompt_Log.pdf.

9. Record only genuine AI mistakes and their actual corrections in AI_FAILURE_LOG.md. Do not fabricate failures, results, or quantum hardware experiments.

10. Run all Track A, B, and new Track C tests. Report the exact commands, actual test counts, benchmark findings, limitations, and unresolved issues.

11. Commit the completed work locally with a commit message beginning [AI:Antigravity].

### Strict constraints

- Do not start SPSA, parameter-shift gradient implementation, Quantum Natural Gradient, noise experiments, or barren-plateau analysis yet. Those belong to later batches.
- Do not create a GitHub repository, add a Git remote, push, publish, or upload anything.
- Do not switch branches or rewrite Git history.
- Do not mark a requirement complete unless it has actually been implemented and tested.
- Keep the code modular, documented, reproducible, and easy to explain in a viva.
- Stop after Batch 1. Summarize files changed, dependencies, test results, measured comparisons, commit hash, and remaining work.
```

---

### AI Response Summary
- Checked Python environment: verified Python 3.12 via Windows `py -3.12`. Inspected dependencies specified in `requirements.txt` and completed dry-run compatibility check for `qiskit` (v2.6.0) and `qiskit-aer` (v0.18.0).
- Implemented `track_C_quantum/src/qaoa_maxcut.py`:
  - `CostHamiltonian`: Derives cost couplings $J_{ij} = 0.5 W_{ij}$, fields $h_i = 0$, and constant offset $-0.5 \sum W_{ij}$ with exact isomorphism to Track B `QUBOProblem` and `IsingModel`.
  - `build_qaoa_circuit`: Parameterized Qiskit `QuantumCircuit` with initial Hadamard state $|+\rangle^{\otimes n}$, alternating cost $R_{ZZ}(-\gamma_l w_{uv})$ layers and mixer $R_X(2 \beta_l)$ layers, with optional measurement registers.
  - `QAOAMaxCutSolver`: Complete solver wrapping classical variational optimization (COBYLA) and quantum circuit sampling on Qiskit Aer (`AerSimulator`).
  - `parse_qiskit_bitstring`: Handles Qiskit little-endian qubit mapping cleanly.
  - `explain_qaoa_depth_and_p`: Rigorous theoretical text detailing Trotterized adiabatic state preparation, $p \to \infty$ adiabatic theorem convergence, and NISQ depth/decoherence trade-offs.
- Created `track_C_quantum/tests/test_qaoa_maxcut.py`:
  - 10 unit tests covering Hamiltonian construction, Track B equivalence, bitstring parsing, circuit depth scaling, parameter validation, Farhi analytic solution (100% optimal probability at $\gamma=\pi/2, \beta=\pi/8$), depth monotonicity on $C_4$, and ground truth verification against Brute-Force and Simulated Annealing.
- Implemented `track_C_quantum/experiments/run_qaoa_benchmarks.py`:
  - Benchmarked QAOA ($p \in \{1, 2, 3\}$) against exact Brute-Force and Simulated Annealing across 9 graph topologies.
  - Generated metrics CSV, markdown report, and 4-panel comparison plots.
- Updated `CHAPTER_6_CHECKLIST.md`, `PROMPT_LOG.md`, and recompiled `AI_Prompt_Log.pdf`.

---

### Verification & Testing
- Track C Unit Tests: `py -3.12 -m pytest track_C_quantum/tests/test_qaoa_maxcut.py -v` (10/10 passed in 3.13s).
- Full Test Suite: `py -3.12 -m pytest` (57/57 passed across Tracks A, B, and C).
- Benchmark Execution: Clean execution on `AerSimulator` across 22 experimental configurations with 2048 shots per run.
