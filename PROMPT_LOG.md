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
| `P-TRC-03` | 2026-10-21 20:08 | Track C | Batch 3 Quantum Completion: PQCs, QNNs vs Classical MLP, Parameter-Shift, SPSA, QNG, Noise Models, Barren Plateaus | [`prompt_03_qnn_gradients_and_noise.md`](file:///c:/dev/ot-8/prompts/track_C/prompt_03_qnn_gradients_and_noise.md) | `quantum_gradients.py`, `pqc_qnn.py`, `quantum_noise_experiments.py`, tests, benchmarks, metrics CSV/MD & plots | Passed (40/40 Track C tests, 87/87 total tests passed, 5 advanced benchmarks, plots & metrics generated) |
| `P-TRD-01` | 2026-10-21 23:15 | Track D | Submodular, Parallel, Distributed (FedAvg) and Multi-Task Optimization (MGDA, GradNorm) | [`prompt_01_submodular_parallel_distributed.md`](file:///c:/dev/ot-8/prompts/track_D/prompt_01_submodular_parallel_distributed.md) | `submodular_maximization.py`, `parallel_optimization.py`, `federated_multitask.py`, tests, benchmarks, metrics CSV/MD & plots | Passed (18/18 Track D tests, 105/105 total tests passed, metrics & plots generated) |
| `P-TRE-01` | 2026-10-21 23:36 | Track E | Main Real-Process Case Study: EV Fleet Smart Charging Scheduling (SLSQP, HiGHS, Greedy) | [`prompt_01_real_process_case_study.md`](file:///c:/dev/ot-8/prompts/track_E/prompt_01_real_process_case_study.md) | `ev_charging_problem.py`, `ev_optimizers.py`, tests, benchmarks, metrics CSV/MD & plots | Passed (7/7 Track E tests, 112/112 total tests passed, -67% cost reduction, metrics & plots generated) |
| `P-TRF-01` | 2026-10-21 23:50 | Track F | Mini-Compiler Optimization: TAC IR Passes (ConstFold, CSE, LICM, DCE) & ILP Register Allocation | [`prompt_01_compiler_optimization.md`](file:///c:/dev/ot-8/prompts/track_F/prompt_01_compiler_optimization.md) | `ir.py`, `interpreter.py`, `optimizations.py`, `compiler.py`, tests, benchmarks, metrics CSV/MD & plots | Passed (8/8 Track F tests, 120/120 total tests passed, output verified, metrics & plots generated) |
| `P-TRG-01` | 2026-10-22 00:05 | Track G | xv6 Teaching OS Optimization: Stride Scheduler, Dynamic Wakeup Pass Compensation & Buffer Cache Hit/Miss Profiling | [`prompt_01_xv6_optimization.md`](file:///c:/dev/ot-8/prompts/track_G/prompt_01_xv6_optimization.md) | `kernel/proc.c`, `kernel/bio.c`, `user/schedtest.c`, `user/bcachetest.c`, `xv6_stride_and_bcache.patch`, tests, metrics CSV/MD & plot | Passed (5/5 Track G tests, 125/125 total tests, forktest OK, QEMU verified, metrics & plot generated) |
| `P-TRH-01` | 2026-10-22 00:24 | Track H | DBMS Optimization: TPC-H Benchmark, 0-1 Knapsack ILP vs Greedy Index Selection & System-R Join DP | [`prompt_01_dbms_optimization.md`](file:///c:/dev/ot-8/prompts/track_H/prompt_01_dbms_optimization.md) | `tpch_generator.py`, `index_selection_ilp.py`, `join_order_dp.py`, `run_dbms_benchmarks.py`, tests, metrics CSV/MD & plot | Passed (6/6 Track H tests, 131/131 total tests, 15.9x speedup, hash verified, metrics & plot generated) |
| `P-TRI-01` | 2026-10-22 19:40 | Track I | Quantum Algorithms & QML: DJ, BV, QFT, QPE, Shor-15, Grover Search, NISQ Noise Modeling & QSVM vs Classical SVM | [`prompt_01_quantum_algorithms.md`](file:///c:/dev/ot-8/prompts/track_I/prompt_01_quantum_algorithms.md) | `quantum_algorithms.py`, `quantum_machine_learning.py`, `quantum_noise_evaluation.py`, `run_track_i_benchmarks.py`, tests, metrics CSV/MD & plot | Passed (11/11 Track I tests, 142/142 total tests, Grover 100%, Shor 15=3x5, QSVM 85.7%, noise evaluated) |


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
