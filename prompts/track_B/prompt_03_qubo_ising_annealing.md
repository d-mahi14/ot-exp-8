# Prompt Record 08: Track B Batch 3 — QUBO, Ising Formulation, and Simulated Annealing

- **Prompt ID**: `P-TRB-03`
- **Date & Time**: 2026-10-21 00:19:40 IST
- **Track**: Track B (Discrete Optimization)
- **AI Tool**: Antigravity (Powered by Gemini 3.8 Flash)
- **Commit Reference**: Pending commit

---

### Purpose
Implement Batch 3 of Track B (Discrete Optimization):
1. Formulate Graph Maximum Cut (Max-Cut) as a Quadratic Unconstrained Binary Optimization (QUBO) problem and derive its exact equivalent Ising spin Hamiltonian ($J_{ij}, h_i, C_0$).
2. Implement Feature Selection as a constrained QUBO problem with penalty terms $P(\sum x_i - K)^2$, documenting the mathematical derivation and the critical impact of penalty multiplier $P$.
3. Formulate and explain the binary-to-spin transformation $s_i = 1 - 2x_i$.
4. Implement a classical Simulated Annealing solver for QUBO with $O(N)$ single-bit-flip energy delta updates.
5. Implement an exact Brute-Force solver for small instances to serve as ground truth.
6. Benchmark Simulated Annealing against exact Brute-Force on Max-Cut and Feature Selection instances, reporting cut values, energies, feasibility, and runtimes.
7. Prepare a dual-AI comparison template and exact prompt for comparing Antigravity and Claude on the QUBO penalty formulation.
8. Add independent unit tests, update checklists and prompt logs, recompile `AI_Prompt_Log.pdf`, and commit locally.

---

### Original Prompt
```text
Continue my Optimization Techniques Lab Assignment 8 in C:\dev\ot-8.

Implement ONLY Track B — Batch 3: QUBO, Ising Formulation, and Simulated Annealing.

1. First inspect CHAPTER_6_CHECKLIST.md, existing Track B code, tests, and prompt logs.
2. Implement Graph Max-Cut as a QUBO problem and derive its equivalent Ising formulation.
3. Implement feature selection as a QUBO problem, clearly documenting the objective and penalty terms.
4. Derive and explain the binary-to-spin transformation s_i = 1 - 2x_i.
5. Implement and test constraint penalties of the form P(sum(x_i) - K)^2. Explain how P affects feasibility and solution quality.
6. Implement a classical Simulated Annealing solver for QUBO instances. Compare it against exact brute-force solutions on small instances and report objective quality, feasibility, and runtime.
7. Generate reproducible benchmark results as CSV/Markdown and create comparison plots. Record actual measurements only.
8. Add independent unit tests covering QUBO energy calculations, Ising equivalence, penalty correctness, edge cases, and simulated annealing against brute force.
9. Update CHAPTER_6_CHECKLIST.md, PROMPT_LOG.md, and a verbatim prompt record under prompts/track_B/. Regenerate AI_Prompt_Log.pdf.
10. Conduct the required dual-AI comparison between Antigravity and Claude for the QUBO penalty formulation. Do not fabricate Claude's output or claim Claude was used unless it actually was. If I must run Claude separately, prepare the exact prompt and a comparison template for me.
11. Record only genuine AI mistakes and fixes in AI_FAILURE_LOG.md.
12. Run all Track A and Track B tests and report the actual results.
13. Commit the completed work locally with a commit message beginning [AI:Antigravity].

Important constraints:
- Do not start Track C or any other track.
- Do not create a GitHub repository, add a remote, push, publish, or upload anything.
- Do not switch branches or rewrite Git history.
- Do not mark methods complete unless implemented and tested.
- Explain the mathematical formulation clearly enough for a student to defend it in a viva.
- Stop after Batch 3 and summarize changed files, test results, benchmark findings, commit hash, and unresolved issues.
```

---

### AI Response Summary
- Built `track_B_discrete/src/qubo_ising.py`:
  - `QUBOProblem`: Evaluation, conversion to upper-triangular, symmetric, and Ising formulations.
  - `IsingModel`: Direct spin evaluation $H(s) = \sum J_{ij} s_i s_j + \sum h_i s_i + C_0$.
  - `maxcut_to_qubo` & `maxcut_to_ising`: Exact derivations.
  - `feature_selection_to_qubo`: mRMR objective + quadratic penalty $P(\sum x_i - K)^2$.
  - `solve_qubo_brute_force`: Exact ground truth for small $N \le 20$.
  - `solve_qubo_simulated_annealing`: Geometric cooling with $O(N)$ single-bit-flip delta energy.
- Created `track_B_discrete/tests/test_qubo_ising.py`: 13 comprehensive unit tests verifying binary-to-spin equivalence, Max-Cut energy matching, penalty thresholding, brute-force exactness, and simulated annealing convergence.
- Built `track_B_discrete/experiments/run_qubo_benchmarks.py`: Evaluated Simulated Annealing vs Brute-Force on Max-Cut graphs (Ring, Complete, Random) and Feature Selection with varying penalty weights $P$.
- Created dual-AI comparison file `prompts/general/tool_comparison_antigravity_vs_claude.md` with verbatim benchmark prompt, Antigravity formulation, and standardized comparison template for Claude.
- Updated `CHAPTER_6_CHECKLIST.md`, `PROMPT_LOG.md`, and recompiled `AI_Prompt_Log.pdf`.
- Committed locally with prefix `[AI:Antigravity]`.

---

### Files Created / Modified
- `prompts/track_B/prompt_03_qubo_ising_annealing.md`
- `PROMPT_LOG.md`
- `CHAPTER_6_CHECKLIST.md`
- `prompts/general/tool_comparison_antigravity_vs_claude.md`
- `track_B_discrete/src/qubo_ising.py`
- `track_B_discrete/tests/test_qubo_ising.py`
- `track_B_discrete/experiments/run_qubo_benchmarks.py`
- `track_B_discrete/results/qubo_benchmark_metrics.csv`
- `track_B_discrete/results/qubo_benchmark_metrics.md`
- `track_B_discrete/results/qubo_penalty_and_annealing.png`
- `AI_Prompt_Log.pdf`

---

### Verification Note
- All 13 unit tests in `test_qubo_ising.py` passed cleanly.
- Full test suite (22 Track A + 12 Knapsack + 13 QUBO = 47 tests) passed in 15.92s with 0 regressions.
- Exact equivalence verified: For every tested binary state $x$, $E_{\text{QUBO}}(x) == E_{\text{Ising}}(1 - 2x)$ with numerical precision $< 10^{-11}$.
- Simulated Annealing found the exact global ground state on 100% of tested Max-Cut instances ($N \in [6, 14]$).

---

### Human Revisions / Instructions Received
Awaiting human evaluation and execution of Claude comparison before proceeding to Track C (Quantum Optimization).
