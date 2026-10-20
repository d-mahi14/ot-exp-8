# Prompt Record 07: Track B Batch 2 — Branch-and-Bound & Integer Linear Programming (ILP)

- **Prompt ID**: `P-TRB-02`
- **Date & Time**: 2026-10-20 20:58:18 IST
- **Track**: Track B (Discrete Optimization)
- **AI Tool**: Antigravity (Powered by Gemini 3.8 Flash)
- **Commit Reference**: `a3b77c5`

---

### Purpose
Implement Batch 2 of Track B (Discrete Optimization):
1. Implement Branch-and-Bound for 0-1 Knapsack using best-first search, state-space tree traversal, fractional knapsack relaxation upper bounding, and bounding pruning.
2. Implement Integer Linear Programming (ILP) for 0-1 Knapsack using `scipy.optimize.milp` (backed by the HiGHS dual-simplex branch-and-cut solver), with documentation on PuLP 4.x compatibility.
3. Add independent unit tests validating Branch-and-Bound and ILP against known textbook instances, edge cases, and 100% agreement with Dynamic Programming on small/medium instances.
4. Run comprehensive benchmarks comparing Greedy, Dynamic Programming, Branch-and-Bound, and ILP on identical instances, analyzing runtime scaling, node counts, and capacity limitations.
5. Save results to CSV/Markdown and generate updated comparison plots.
6. Update `CHAPTER_6_CHECKLIST.md`, `PROMPT_LOG.md`, and regenerate `AI_Prompt_Log.pdf`.

---

### Original Prompt
```text
Continue my Optimization Techniques Lab Assignment 8 in C:\dev\ot-8.

Implement ONLY Track B — Batch 2:
1. Branch-and-Bound for the 0–1 Knapsack problem, using a valid fractional-knapsack upper bound and pruning.
2. Integer Linear Programming (ILP) for the same problem, preferably using PuLP with an available solver. If unavailable, investigate a reliable installed alternative and document the choice; do not silently fake ILP results.
3. Benchmark Greedy, Dynamic Programming, Branch-and-Bound, and ILP on identical instances. Record objective value, runtime, and any meaningful solver statistics available. Explain limitations for large capacities.
4. Add independent tests for correctness, edge cases, and agreement with exact solutions on small instances.
5. Save actual benchmark results as CSV/Markdown and generate a comparison plot.
6. Update CHAPTER_6_CHECKLIST.md, PROMPT_LOG.md, the verbatim prompt record under prompts/track_B/, and regenerate AI_Prompt_Log.pdf.
7. Record only genuine AI mistakes and their actual fixes in AI_FAILURE_LOG.md. Do not invent failures or experimental results.
8. Run the new tests and the combined Track A + Track B tests. Report exact commands and actual outcomes.
9. Commit completed local changes using a message beginning [AI:Antigravity].

IMPORTANT:
- First inspect the existing code, tests, results, and checklist.
- Reuse the existing KnapsackProblem and solver interfaces where appropriate.
- Keep the methods mathematically correct and make fair comparisons.
- Do not begin QUBO, Ising, or simulated annealing; those belong to Batch 3.
- Do not create a GitHub repository, add a remote, push, publish, or upload anything.
- Do not change branches or rewrite Git history.
- Stop after Batch 2, summarize files changed, tests, measured results, commit hash, and any unresolved issues. Do not start another batch.
```

---

### AI Response Summary
- Investigated ILP solvers: examined PuLP 4.1.0 and discovered constructor interface changes in its recent Rust core; verified `scipy.optimize.milp` powered by the C++ HiGHS branch-and-cut solver is fully available in the local environment and provides exact branch-and-bound node counts (`mip_node_count`), gap, and bounds. Supported both interfaces with primary implementation on `scipy.optimize.milp`.
- Extended `track_B_discrete/src/knapsack_solvers.py` with:
  - `solve_knapsack_branch_and_bound()`: Best-first priority queue search using fractional knapsack upper bound relaxation, tracking nodes explored and pruned.
  - `solve_knapsack_ilp()`: Exact 0-1 Integer Linear Program using HiGHS branch-and-cut.
- Extended `track_B_discrete/tests/test_knapsack.py` with 7 new unit tests verifying Branch-and-Bound and ILP correctness, edge cases, and exact agreement with Dynamic Programming on random instances.
- Updated `track_B_discrete/experiments/run_knapsack_benchmarks.py` to evaluate all 4 solvers (Greedy, DP, Branch-and-Bound, ILP) across instance scales, recording runtime, objective value, nodes explored, and memory limits.
- Generated updated metrics table and a 4-solver comparison plot `track_B_discrete/results/knapsack_comparison.png`.
- Updated `CHAPTER_6_CHECKLIST.md`, `PROMPT_LOG.md`, and recompiled `AI_Prompt_Log.pdf`.
- Committed locally with prefix `[AI:Antigravity]`.

---

### Files Created / Modified
- `prompts/track_B/prompt_02_branch_and_bound_ilp.md`
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
- **Unit Tests**: Executed `py -3.12 -m pytest track_A_continuous/tests/ track_B_discrete/tests/ -v`. All 34 tests (22 Track A + 12 Track B) passed in 0.96s with 0 regressions.
- **Benchmark Run**: Executed `py -3.12 track_B_discrete/experiments/run_knapsack_benchmarks.py`. All 4 solvers benchmarked across $N \in [5, 200]$ and Large-Capacity stress test ($N=25, W=263,700$).
- **100% Exact Agreement**: DP, Branch-and-Bound, and ILP (HiGHS) achieved 100% identical exact optimal values across every single instance evaluated.
- **Large Capacity Bottleneck**: In the stress test ($W=263,700$), DP required $3,814.45$ ms due to $O(NW)$ table allocation, while Branch-and-Bound solved it in **$26.76$ ms** ($142\times$ faster) and ILP solved it in **$206.29$ ms** ($18\times$ faster).

---

### Human Revisions / Instructions Received
Awaiting human approval of Batch 2 before moving to Batch 3 (QUBO / Ising formulation and Simulated Annealing baseline for Track C).
