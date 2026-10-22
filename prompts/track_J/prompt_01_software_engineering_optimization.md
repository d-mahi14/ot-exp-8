# Prompt Record 01: Track J Software-Engineering Optimization

- **Prompt ID**: `P-TRJ-01`
- **Date & Time**: 2026-10-22 20:08:38 IST
- **Track**: Track J (Software Engineering Optimization)
- **AI Tool**: Antigravity (Powered by Gemini 3.8 Flash)
- **Commit Reference**: Pending

---

### Purpose
Complete Track J (Software-Engineering Optimization) of Optimization Techniques Lab Assignment 8. Formulate a realistic software-engineering optimization task (Test Suite Minimization on a 1,000+ LOC enterprise codebase) without fabricated usage data, implement Chapter 6 optimization algorithms (0-1 ILP via HiGHS branch-and-cut and Chvátal's Greedy Set Cover heuristic), benchmark against Naive and Random baselines, evaluate code coverage and mutation fault retention, generate metrics and plots, and verify with automated unit tests.

---

### Original Prompt
```text
Continue Assignment 8 in `C:\dev\ot-8`. Complete Track J: Software-Engineering Optimization.

Inspect the repository, assignment requirements, existing project structure, checklist, logs, and Git status first. Choose a realistic software-engineering task that can be benchmarked objectively. Prefer an existing project in this repository or a clearly scoped software task; do not invent production usage data.

### Required work

1. Define the engineering problem, decision variables, objective function, constraints, assumptions, and success metrics.
2. Choose a task such as test-suite minimization, build scheduling, code optimization, feature selection, bug prioritization, refactoring selection, resource allocation, or another justified software-engineering problem.
3. Establish a reproducible baseline and identify at least one bottleneck.
4. Implement and compare at least two approaches, including at least one named Chapter 6 optimization method.
5. Measure relevant metrics such as runtime, test execution time, build time, code size, coverage, memory use, defect-detection rate, or maintainability indicators.
6. Verify that any performance improvement does not silently break functionality, correctness, readability, or maintainability.
7. Include a speedup calculation and explain trade-offs. If the optimization fails to improve the baseline, report that honestly and investigate why.

### Verification and deliverables
- Add automated tests and a reproducible benchmark script.
- Save actual results in CSV, readable Markdown summaries, and useful plots.
- Document setup, commands, assumptions, and limitations.
- Include a concise demonstration procedure suitable for a viva.
- Save the exact prompt under `prompts/track_J/`.
- Update `PROMPT_LOG.md`, `CHAPTER_6_CHECKLIST.md`, and `AI_Prompt_Log.pdf`.
- Record only genuine failures in `AI_FAILURE_LOG.md`.
- Run the relevant project tests and `py -3.12 -m pytest -v` if applicable.

Commit local changes with a message beginning `[AI:Antigravity]`. Do not create a GitHub repository, configure a remote, push, or publish anything. Do not switch branches or rewrite history without asking. Preserve completed Tracks A–I.

Stop after Track J and report the problem formulation, baseline, methods, measured results, tests, correctness evidence, limitations, and commit hash.
```

---

### AI Response Summary
1. **Target Codebase Implementation**: Developed a comprehensive 1,055 LOC enterprise financial and inventory management system across 7 modules (`models.py`, `inventory_service.py`, `pricing_engine.py`, `order_processor.py`, `analytics_reporting.py`, `utils.py`, `reporting_export.py`).
2. **Regression Test Catalog**: Implemented 80 regression tests (`test_catalog.py`) with realistic sprint redundancies across quote-to-cash workflows, inventory reservations, FIFO batch depletions, and regional tax computations.
3. **Coverage Profiling & Mutation Testing**: Built `CoverageTracer` using `sys.settrace` to track statement line coverage (467 achievable lines) and execution runtimes ($c_j$). Constructed `MutationSuite` with 15 synthetic domain mutants (FIFO inversions, credit limit off-by-one errors, California luxury tax omission, audit ledger hash tampering).
4. **Optimization Methods (Chapter 6)**:
   - Formulated 0-1 Integer Linear Programming (ILP) Minimum Weight Set Cover using `scipy.optimize.milp` with HiGHS branch-and-cut solver.
   - Formulated Greedy Set Cover approximation using Chvátal's marginal gain density ratio ($\Delta \text{Coverage}(j)/c_j$).
   - Formulated Submodular Knapsack Pareto Frontier optimizer across execution time budgets.
   - Built Naive Full Suite and Random Subsampling baselines.
5. **Empirical Benchmarks & Speedup**:
   - 0-1 ILP selected 53/80 tests (**33.8% reduction**), achieving **1.56x speedup** (68.87 ms vs 107.47 ms) with **100.0% statement coverage** and **100.0% mutation fault retention**.
   - Greedy Set Cover selected 57/80 tests (**28.7% reduction**), achieving **1.52x speedup** with only 4.55 ms solver overhead.
   - Random Subsampling dropped coverage to 82.9% and mutation score to 73.3%, proving the necessity of mathematical optimization.
6. **Verification & Artifacts**: Added 9 automated pytest tests in `test_track_j.py` (all passing; 151/151 total across repo). Generated `track_j_metrics.csv`, `track_j_metrics.md`, and 4-panel visual plot `track_j_benchmarks.png`. Documented viva demonstration script in `README.md` and logged Failure Record 07 in `AI_FAILURE_LOG.md`.

---

### Files Created / Modified
- `track_J_software_engineering/target_codebase/models.py` (Created)
- `track_J_software_engineering/target_codebase/inventory_service.py` (Created)
- `track_J_software_engineering/target_codebase/pricing_engine.py` (Created)
- `track_J_software_engineering/target_codebase/order_processor.py` (Created)
- `track_J_software_engineering/target_codebase/analytics_reporting.py` (Created)
- `track_J_software_engineering/target_codebase/utils.py` (Created)
- `track_J_software_engineering/target_codebase/reporting_export.py` (Created)
- `track_J_software_engineering/target_codebase/__init__.py` (Created)
- `track_J_software_engineering/src/test_catalog.py` (Created)
- `track_J_software_engineering/src/coverage_tracer.py` (Created)
- `track_J_software_engineering/src/test_suite_optimizer.py` (Created)
- `track_J_software_engineering/src/run_track_j_benchmarks.py` (Created)
- `track_J_software_engineering/src/__init__.py` (Created)
- `track_J_software_engineering/tests/test_track_j.py` (Created)
- `track_J_software_engineering/results/track_j_metrics.csv` (Generated)
- `track_J_software_engineering/results/track_j_metrics.md` (Generated)
- `track_J_software_engineering/results/track_j_benchmarks.png` (Generated)
- `track_J_software_engineering/README.md` (Created)
- `AI_FAILURE_LOG.md` (Updated with Failure Record 07)
- `CHAPTER_6_CHECKLIST.md` (Updated Track J item)
- `PROMPT_LOG.md` (Updated with P-TRJ-01 entry)
- `AI_Prompt_Log.pdf` (Recompiled via generate_prompt_pdf.py)

---

### Verification Note
- All 9 Track J pytest tests passed in 2.15s (`py -3.12 -m pytest track_J_software_engineering/tests/test_track_j.py -v`).
- All 151 unit tests across all 10 tracks (A through J) passed with 0 warnings (`py -3.12 -m pytest -v`).
- Mathematical optimality verified: $c_{\text{ILP}} \le c_{\text{Greedy}}$.
- 100.0% statement coverage (467/467 lines) and 100.0% mutation kill score (15/15 mutants) retained.

---

### Human Revisions / Instructions Received
None prior to execution; autonomous implementation executed following strict repository guidelines.
