# Master AI Prompt Log Index
**Course**: Optimization Techniques (Lab Assignment 8 — The 14-Day Vibe-Coding Challenge)  
**Student Project**: Optimize Anything — 10-Track Optimization Portfolio  
**Repository**: `ot-8`  
**Generated PDF Document**: [`AI_Prompt_Log.pdf`](file:///c:/dev/ot-8/AI_Prompt_Log.pdf)

---

## 1. Prompt Logging Policy & Rules
1. **Verbatim Preservation**: Every single user prompt is recorded exactly as entered.
2. **AI Disclosure**: Every use of AI (model, tool, date, purpose) is disclosed per Section 8.
3. **Commit Traceability**: AI-assisted commits use the prefix `[AI:<tool-name>]`, while human manual corrections use `[Manual:Fix]`.
4. **Independent Verification**: No experiment or claim is accepted without reproducible tests and verified metric outputs.
5. **No Secrets**: No API keys, passwords, or credentials may ever be saved in logs.

---

## 2. Master Prompt Registry

| Prompt ID | Date & Time (IST) | Track | Purpose | Markdown Log Link | Key Files Changed / Created | Verification Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `P-GEN-01` | 2026-10-20 20:12 | General | Initial Project Planning, Chapter 6 Syllabus Mapping, and Schedule Formulation | [`prompt_01_planning_and_architecture.md`](file:///c:/dev/ot-8/prompts/general/prompt_01_planning_and_architecture.md) | None (read-only system inspection) | Passed (Plan approved) |
| `P-GEN-02` | 2026-10-20 20:16 | General | Architectural Confirmation, xv6 Toolchain Diagnostics, Prompt Log Infrastructure & PDF Pipeline Setup | [`prompt_02_logging_and_approval.md`](file:///c:/dev/ot-8/prompts/general/prompt_02_logging_and_approval.md) | `PROMPT_LOG.md`, `AI_TOOLS_REGISTER.md`, `AI_FAILURE_LOG.md`, `scripts/generate_prompt_pdf.py`, `AI_Prompt_Log.pdf` | Passed (PDF generated & verified) |
| `P-TRA-01` | 2026-10-20 20:25 | Track A | Stage 1 First-Order Continuous Optimizers: GD, SGD, Mini-batch, Momentum, NAG, AdaGrad, RMSProp, Adam | [`prompt_01_continuous_optimization.md`](file:///c:/dev/ot-8/prompts/track_A/prompt_01_continuous_optimization.md) | `track_A_continuous/src/`, `track_A_continuous/tests/`, `track_A_continuous/experiments/`, `track_A_continuous/results/` | Passed (13/13 unit tests passed, metrics & plots generated) |
| `P-TRA-02` | 2026-10-20 20:36 | Track A | Empirical Audit, Benchmark Findings Correction, Hyperparameter Sensitivity Rationale & Failure Log 02 | [`prompt_02_review_and_correction.md`](file:///c:/dev/ot-8/prompts/track_A/prompt_02_review_and_correction.md) | `first_order_metrics.md`, `run_first_order_benchmarks.py`, `AI_FAILURE_LOG.md`, `AI_Prompt_Log.pdf` | Passed (Discrepancy resolved, Failure 02 recorded) |
| `P-TRA-03` | 2026-10-20 20:41 | Track A | Stage 2 Second-Order Methods (Newton, Gauss-Newton), Curvature Approximations (Fisher, Diag/Low-rank, HVP) & MLP Backprop Engine | [`prompt_03_second_order_and_backprop.md`](file:///c:/dev/ot-8/prompts/track_A/prompt_03_second_order_and_backprop.md) | `second_order_methods.py`, `mlp_backpropagation.py`, tests, benchmarks, plots, metrics | Passed (22/22 tests passed, gradient error 8.94e-11, plots & metrics generated) |
| `P-TRB-01` | 2026-10-20 20:52 | Track B | Batch 1 Discrete Optimization: 0-1 Knapsack Problem Formulation, Greedy Heuristic & Exact Dynamic Programming | [`prompt_01_discrete_foundations.md`](file:///c:/dev/ot-8/prompts/track_B/prompt_01_discrete_foundations.md) | `knapsack_solvers.py`, `test_knapsack.py`, `run_knapsack_benchmarks.py`, metrics & plots | Passed (8/8 Track B tests passed, 30/30 total tests, metrics & plots generated) |
| `P-TRB-02` | 2026-10-20 20:58 | Track B | Batch 2 Discrete Optimization: Branch-and-Bound with Fractional Relaxation & Integer Linear Programming (ILP) | [`prompt_02_branch_and_bound_ilp.md`](file:///c:/dev/ot-8/prompts/track_B/prompt_02_branch_and_bound_ilp.md) | `knapsack_solvers.py`, `test_knapsack.py`, `run_knapsack_benchmarks.py`, metrics & plots | Passed (12/12 Track B tests passed, 34/34 total tests, 4 solvers benchmarked) |
| `P-TRB-03` | 2026-10-21 00:19 | Track B | Batch 3 Discrete Optimization: QUBO & Ising Formulations, Max-Cut, Penalty Derivations, Simulated Annealing & Dual-AI Setup | [`prompt_03_qubo_ising_annealing.md`](file:///c:/dev/ot-8/prompts/track_B/prompt_03_qubo_ising_annealing.md) | `qubo_ising.py`, `test_qubo_ising.py`, `run_qubo_benchmarks.py`, `tool_comparison_antigravity_vs_claude.md`, metrics & plots | Passed (13/13 QUBO tests, 47/47 total tests passed, SA 100% exact, plots & metrics generated) |
| `P-TRC-01` | 2026-10-21 19:35 | Track C | Batch 1 Quantum Optimization Foundations: QAOA for Max-Cut, Qiskit Aer simulation, parameter optimization, depth p scaling | [`prompt_01_quantum_foundations_and_qaoa.md`](file:///c:/dev/ot-8/prompts/track_C/prompt_01_quantum_foundations_and_qaoa.md) | `qaoa_maxcut.py`, `test_qaoa_maxcut.py`, `run_qaoa_benchmarks.py`, metrics CSV/MD & comparison plots | Passed (10/10 Track C tests, 57/57 total tests passed, 22 QAOA experiments, plots & metrics generated) |
| `P-TRC-02` | 2026-10-21 19:55 | Track C | Batch 2 Variational Quantum Eigensolver (VQE): Ising Hamiltonians, RealAmplitudes ansatz, COBYLA, multi-trial variance | [`prompt_02_variational_quantum_eigensolver.md`](file:///c:/dev/ot-8/prompts/track_C/prompt_02_variational_quantum_eigensolver.md) | `vqe_ising.py`, `test_vqe_ising.py`, `run_vqe_benchmarks.py`, metrics CSV/MD & comparison plots | Passed (11/11 VQE tests, 68/68 total tests passed, 80 multi-trial experiments, plots & metrics generated) |

---

## 3. Directory Structure of Prompt Logs
```text
prompts/
├── general/
│   ├── prompt_01_planning_and_architecture.md
│   └── prompt_02_logging_and_approval.md
├── track_A/        # Continuous Optimization
├── track_B/        # Discrete & QUBO Optimization
├── track_C/        # Quantum Optimization (QAOA, VQE)
├── track_D/        # Submodular, Parallel & Distributed
├── track_E/        # Main Case Study (EV Fleet Smart Charging)
├── track_F/        # Mini-Compiler Optimization
├── track_G/        # xv6 Operating System
├── track_H/        # DBMS Optimization
├── track_I/        # Quantum Algorithms & QML Advantage
└── track_J/        # Software Engineering Optimization
```

---

## 4. Re-Generating the Master PDF
To re-compile all Markdown logs into the official submission PDF `AI_Prompt_Log.pdf`, run:
```powershell
py -3.12 scripts/generate_prompt_pdf.py
```
