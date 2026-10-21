# Prompt Record 12: Track D — Submodular, Parallel, Distributed and Multi-Task Optimization

- **Prompt ID**: `P-TRD-01`
- **Date & Time**: 2026-10-21 23:15:57 IST
- **Track**: Track D (Submodular, Parallel & Distributed Optimization)
- **AI Tool**: Antigravity (Powered by Gemini 3.8 Flash)
- **Commit Reference**: Pending commit

---

### Purpose
Complete Track D: Submodular, Parallel, Distributed and Multi-Task Optimization:
1. **Submodular Optimization**:
   - Implement monotone submodular sensor placement / spatial coverage problem formulation.
   - Define expected coverage utility function $f(S) = \sum_{u} w_u [1 - \prod_{s \in S} (1 - p(s, u))]$ with cost budget constraint.
   - Mathematically and empirically verify diminishing returns (submodularity) $f(A \cup \{e\}) - f(A) \ge f(B \cup \{e\}) - f(B)$ for $A \subseteq B$.
   - Implement exact combinatorial brute-force solver for ground-truth optimum on small instances ($N \le 20$).
   - Implement Standard Greedy algorithm (Nemhauser, Wolsey, Fisher 1978) with empirical check of the $(1 - 1/e) \approx 0.6321$ theoretical approximation guarantee.
   - Implement Accelerated Lazy Greedy (Minoux 1978) utilizing priority-queue marginal bound pruning.
   - Compare against Random and Top-K Isolated baselines.
2. **Parallel Program Optimization**:
   - Implement compute-intensive deterministic workload with GIL-releasing BLAS operations.
   - Implement Sequential baseline, Static scheduling (thread pool), Dynamic work queue scheduling, and Work-stealing queue simulation.
   - Measure wall-clock runtime $T(p)$, speedup $S(p) = T(1) / T(p)$, efficiency $E(p) = S(p) / p$, and parallel overhead $T_{\text{overhead}}(p) = p \cdot T(p) - T(1)$ across worker pools $p \in \{1, 2, 4, 8\}$.
   - Fit Amdahl's Law $S(p) = \frac{1}{(1-s) + s/p}$ and Gustafson's Law $S(p) = p - \alpha(p - 1)$ to empirical speedup measurements.
   - Model and analyze Data Parallelism (Ring All-Reduce communication overhead) vs. Pipeline Parallelism (1F1B schedule bubble fraction).
3. **Distributed and Multi-Task Optimization**:
   - Implement Federated Learning simulation (FedAvg, McMahan et al. 2017) across 5 clients with IID ($E=1$) and Non-IID ($E=5$) data partitioning.
   - Compare FedAvg convergence and client drift against Centralized SGD baseline.
   - Implement Multi-Task Learning (MTL) with shared representation neural trunk and task-specific regression/classification heads.
   - Compare Uniform Task Weighting vs. Multiple Gradient Descent Algorithm (MGDA, Sener & Koltun 2018) for Pareto-stationarity vs. GradNorm (Chen et al. 2018) for dynamic gradient norm balancing.
4. **Verification & Artifacts**:
   - 18 unit tests in `track_D_submodular_parallel/tests/`.
   - Comprehensive reproducible benchmark script `run_track_d_benchmarks.py`.
   - Outputs saved to CSV, native Markdown tables, and 4-panel comparison figure `track_d_benchmarks.png`.

---

### Original Prompt
```text
Continue Assignment 8 in `C:\dev\ot-8`. Complete Track D: Submodular, Parallel, Distributed and Multi-Task Optimization.

First inspect the repository, assignment checklist, existing Tracks A–C, prompt logs, failure log, and Git status. Preserve completed work and follow existing project conventions.

### Required work

1. **Submodular optimization:** implement a suitable greedy algorithm for a submodular maximization problem, such as budgeted sensor placement or maximum coverage. Define the objective, constraints, baseline, and approximation guarantee with its assumptions. Compare against an exact solution on small instances.

2. **Parallel optimization:** implement and benchmark parallel execution for a suitable workload. Include load balancing, scheduling, data/model/pipeline parallelism where appropriate. Measure speedup, efficiency, overhead, and scalability against a sequential baseline.

3. **Distributed and multi-task optimization:** implement FedAvg for a small federated-learning simulation and compare with centralized training. Implement MGDA and GradNorm for a small multi-task learning problem. Explain objective weighting, convergence behavior, and limitations.

### Verification and artifacts
- Write unit tests and deterministic, reproducible benchmark scripts.
- Save measured results to CSV, readable Markdown summaries, and plots.
- Verify constraints, objective values, and baselines independently.
- Explain assumptions, trade-offs, and cases where each method may fail.
- Do not claim theoretical guarantees unless their assumptions hold.
- Save the exact prompt under `prompts/track_D/`.
- Update `PROMPT_LOG.md`, `CHAPTER_6_CHECKLIST.md`, and `AI_Prompt_Log.pdf` according to existing conventions.
- Add failure-log entries only for genuine failures.
- Run `py -3.12 -m pytest -v` and report the actual results.
- Commit local changes with a message starting `[AI:Antigravity]`.

Do not create a GitHub repository, configure a remote, push, or publish anything. Do not switch branches or rewrite history without asking. Do not modify completed Tracks A–C except for necessary, documented compatibility fixes.

Stop after Track D and summarize files, tests, measurements, commit hash, and remaining work.
```

---

### Execution & Verification Summary
1. **Source Code**:
   - `track_D_submodular_parallel/src/submodular_maximization.py`: SensorPlacementProblem, diminishing returns verification, brute-force exact, standard greedy, lazy greedy, baselines.
   - `track_D_submodular_parallel/src/parallel_optimization.py`: compute kernels, static/dynamic/work-stealing schedulers, Amdahl/Gustafson curve fitting, DDP/Pipeline simulation models.
   - `track_D_submodular_parallel/src/federated_multitask.py`: FedAvg simulator, centralized SGD baseline, analytical & Frank-Wolfe MGDA, GradNorm trainer.
   - `track_D_submodular_parallel/src/__init__.py`: module exports.
2. **Unit Tests**:
   - `track_D_submodular_parallel/tests/test_submodular.py`: 7 tests.
   - `track_D_submodular_parallel/tests/test_parallel.py`: 6 tests.
   - `track_D_submodular_parallel/tests/test_federated_multitask.py`: 5 tests.
   - Total: 18/18 Track D tests passed.
3. **Benchmarks & Artifacts**:
   - `run_track_d_benchmarks.py` executed cleanly.
   - `track_D_submodular_parallel/results/submodular_metrics.csv` & `.md`.
   - `track_D_submodular_parallel/results/parallel_scaling_metrics.csv` & `.md`.
   - `track_D_submodular_parallel/results/federated_multitask_metrics.csv` & `.md`.
   - `track_D_submodular_parallel/results/track_d_benchmarks.png`.
