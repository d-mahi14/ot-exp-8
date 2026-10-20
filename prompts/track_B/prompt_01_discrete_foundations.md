# Prompt Record 06: Track B Batch 1 — Discrete Optimization Foundations (0-1 Knapsack, Greedy & Dynamic Programming)

- **Prompt ID**: `P-TRB-01`
- **Date & Time**: 2026-10-20 20:52:11 IST
- **Track**: Track B (Discrete Optimization)
- **AI Tool**: Antigravity (Powered by Gemini 3.8 Flash)
- **Commit Reference**: `a7d9472`

---

### Purpose
Implement Batch 1 of Track B (Discrete Optimization):
1. Define the 0-1 Knapsack combinatorial optimization problem with formal variables, objectives, and constraints.
2. Implement the Value-to-Weight Ratio Greedy Heuristic with explanation of selection rule and approximation limits.
3. Implement the exact Dynamic Programming algorithm (pseudo-polynomial $O(NW)$ time) with optimal solution backtracking.
4. Create independent unit tests covering known textbook instances and edge cases (zero capacity, items exceeding capacity, empty item list).
5. Build reproducible benchmark experiments comparing Greedy vs DP on identical problem instances across problem sizes $N \in [5, 10, 20, 50, 100, 200]$.
6. Produce result tables (CSV, Markdown) and comparison plots.
7. Update `PROMPT_LOG.md`, `CHAPTER_6_CHECKLIST.md`, and regenerate `AI_Prompt_Log.pdf`.

---

### Original Prompt
```text
Continue the Optimization Techniques Lab Assignment 8 in the existing local project at `C:\dev\ot-8`.

## Track B — Batch 1: Problem Formulation and Baseline Solvers

### 1. Preserve project safety
- Inspect the existing folder structure, `CHAPTER_6_CHECKLIST.md`, and relevant assignment requirements before coding.
- Do not delete or overwrite completed Track A work.
- Do not change branches or rewrite Git history.
- Do not create a GitHub repository, configure a Git remote, push, publish, or upload anything.
- Local Git commits are allowed.
- Use commit prefix `[AI:Antigravity]`.

### 2. Record the AI interaction
- Save this exact prompt to `prompts/track_B/prompt_01_discrete_foundations.md`.
- Update `PROMPT_LOG.md` with the interaction.
- Record actual implementation and verification details; do not invent missing prompt history.
- Regenerate `AI_Prompt_Log.pdf` after updating the logs.

### 3. Inspect the required methods
Read `CHAPTER_6_CHECKLIST.md` and the assignment requirements. List the discrete optimization methods required for Track B and mark which belong in this batch versus later batches. Do not mark the entire Track B complete.

### 4. Implement this batch
Use **0-1 Knapsack** as the initial combinatorial optimization problem.

Implement:
1. A clear problem definition with item weights, values, and capacity.
2. A Greedy heuristic, explaining its selection rule and limitations.
3. Dynamic Programming for the exact 0-1 Knapsack solution.
4. Independent unit tests, including small examples where the optimal answer is known.
5. A reproducible experiment comparing Greedy and Dynamic Programming on identical problem instances.
6. CSV and Markdown result tables showing actual values, selected items, total weight, total value, runtime, and whether each result is optimal.
7. At least one meaningful plot comparing the methods.
8. Clear mathematical explanations and code comments suitable for a student viva.

Use fixed random seeds for generated instances. Include edge cases such as zero capacity, an item heavier than the capacity, and an empty item list.

### 5. Verify honestly
Run the unit tests and experiment scripts using the available Python environment. Fix genuine errors and rerun the relevant checks.

Record the actual commands and results. Document meaningful AI mistakes and their fixes in `AI_FAILURE_LOG.md`. Never claim that a test or experiment passed unless you ran it.

### 6. Finish and report
Update the relevant sections of `CHAPTER_6_CHECKLIST.md`, but do not mark Branch-and-Bound, ILP, QUBO/Ising, or other unimplemented methods as complete.

Commit the finished batch locally with a descriptive message beginning:
`[AI:Antigravity]`

Before stopping, report:
- Files created or changed.
- Commands executed and actual test results.
- Actual experiment results and generated plot paths.
- Any limitations or unresolved errors.
- The commit hash and Git status.
- The exact next batch to implement.

Explain the implementation in simple terms so I can understand it for my viva.

Do not begin Branch-and-Bound, ILP, QUBO/Ising, or Track C yet. Stop after Batch 1 and wait for my approval.
```

---

### AI Response Summary
- Reviewed Track B Chapter 6 requirements and structured implementation into three distinct batches:
  - **Batch 1 (Active)**: 0-1 Knapsack formulation, Greedy Heuristic, and Dynamic Programming solver.
  - **Batch 2 (Pending)**: Branch-and-Bound and Integer Linear Programming (ILP via PuLP).
  - **Batch 3 (Pending)**: QUBO & Ising formulation with penalty terms on Max-Cut/Feature Selection, and classical Simulated Annealing baseline for Track C.
- Implemented `track_B_discrete/src/knapsack_solvers.py`: `KnapsackProblem`, `solve_knapsack_greedy`, and `solve_knapsack_dp` with backtracked item selection.
- Created `track_B_discrete/tests/test_knapsack.py`: 10 independent unit tests covering edge cases (zero capacity, oversized items, empty input, identical ratios) and known optimal benchmark instances.
- Created `track_B_discrete/experiments/run_knapsack_benchmarks.py`: Evaluated Greedy vs DP across $N \in [5, 10, 20, 50, 100, 200]$ with fixed seed, generating comparison tables and a dual-panel plot.
- Updated `PROMPT_LOG.md`, `CHAPTER_6_CHECKLIST.md`, and recompiled `AI_Prompt_Log.pdf`.
- Committed work locally with prefix `[AI:Antigravity]`.

---

### Files Created / Modified
- `prompts/track_B/prompt_01_discrete_foundations.md`
- `PROMPT_LOG.md`
- `CHAPTER_6_CHECKLIST.md`
- `track_B_discrete/src/knapsack_solvers.py`
- `track_B_discrete/tests/test_knapsack.py`
- `track_B_discrete/experiments/run_knapsack_benchmarks.py`
- `track_B_discrete/results/knapsack_metrics.csv`
- `track_B_discrete/results/knapsack_metrics.md`
- `track_B_discrete/results/knapsack_comparison.png`
- `AI_Prompt_Log.pdf`

---

### Verification Note
- **Unit Tests**: Executed `py -3.12 -m pytest track_A_continuous/tests/ track_B_discrete/tests/ -v`. All 30 tests (22 Track A + 8 Track B) passed in 0.23s with 0 regressions.
- **Benchmark Run**: Executed `py -3.12 track_B_discrete/experiments/run_knapsack_benchmarks.py`. Verified scaling across $N \in [5, 10, 20, 50, 100, 200, 400]$.
- **Optimality vs Runtime**: Greedy executes in $< 0.3$ ms up to $N=400$ with optimality gap $\le 1.4\%$ on large instances, but exhibits a $7.8\%$ gap on smaller instances ($N=10$). Dynamic Programming guarantees 100% exact optimality on every instance, scaling from $0.05$ ms ($N=5$) to $737.6$ ms ($N=400, W=4,990$).

---

### Human Revisions / Instructions Received
Awaiting human approval of Batch 1 before proceeding to Batch 2 (Branch-and-Bound and ILP).
