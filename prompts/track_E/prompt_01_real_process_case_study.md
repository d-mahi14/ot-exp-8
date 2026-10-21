# Prompt Record 13: Track E — Main Real-Process Case Study (EV Fleet Smart Charging Scheduling)

- **Prompt ID**: `P-TRE-01`
- **Date & Time**: 2026-10-21 23:36:05 IST
- **Track**: Track E (Main Real-Process Case Study)
- **AI Tool**: Antigravity (Powered by Gemini 3.8 Flash)
- **Commit Reference**: Pending commit

---

### Purpose
Complete Track E: Main Real-Process Case Study:
1. **Process Identification**:
   - Formulate commercial EV fleet charging scheduling at a central delivery depot.
   - Address operational bottlenecks: Time-of-Use (ToU) electricity tariffs (4.75x peak-to-off-peak ratio), depot transformer peak demand limits ($P_{\text{grid\_max}}$), and battery degradation.
2. **Mathematical Formulation**:
   - Continuous decision variables $p_{k, t} \ge 0$ (charging power kW per vehicle $k$ at hour $t$) and $P_{\text{peak}} \ge 0$.
   - Objective: $\min \sum c_t (\sum p_{k, t}) \Delta t + \lambda_{\text{peak}} P_{\text{peak}} + \lambda_{\text{deg}} \sum p_{k, t}^2 \Delta t$.
   - Linear constraints: Transformer capacity headroom, peak power envelope, vehicle charger power rating bounds, dwell availability windows, and target SoC energy fulfillment.
3. **Simulated Baseline**:
   - Uncontrolled "As-Soon-As-Possible" (ASAP) charging: vehicles plug in immediately on return (16:00–19:00) and charge at maximum rate until target SoC.
   - Demonstrates severe peak load coincidence during evening peak tariff hours and transformer capacity violations.
4. **Chapter 6 Optimization Methods**:
   - **Method 1: Continuous Convex Optimization (SLSQP)**: Smooth convex formulation with exact analytical gradients and constraint Jacobian.
   - **Method 2: Mixed-Integer / Linear Programming (HiGHS)**: Exact linear programming formulation with linearized peak demand envelope.
   - **Method 3: Priority-Based Greedy Valley-Filling Heuristic**: Ultra-fast $O(K T \log T)$ online heuristic sorting by ToU tariff and vehicle urgency.
5. **Sensitivity, Trade-offs & Failure Modes**:
   - Fleet scale sweep across $K \in \{5, 10, 15, 20\}$ vehicles.
   - Transformer capacity headroom stress test ($40\%$ to $120\%$).
   - Explicit identification of mathematical infeasibility boundaries (e.g. insufficient dwell time relative to minimum required charging duration).
6. **Deliverables & Verification**:
   - 7 unit tests in `track_E_case_study/tests/test_ev_charging.py`.
   - Comprehensive reproducible benchmarks in `run_ev_case_study_benchmarks.py`.
   - CSV results, native Markdown report, and 4-panel visual plot `ev_fleet_charging_profiles.png`.

---

### Original Prompt
```text
Continue Assignment 8 in `C:\dev\ot-8`. Complete Track E: Main Real-Process Case Study.

Inspect the assignment requirements, repository, completed Tracks A–D, existing documentation, and Git status before making changes. Choose a suitable real-world or realistically simulated process from the work already completed, where possible. Do not invent a real deployment or claim that simulated measurements came from a real organization.

### Required work

1. Identify the process and explain the problem it solves.
2. Describe the original workflow, bottleneck, assumptions, available data, and constraints.
3. Define decision variables, objective function, constraints, and measurable success criteria.
4. Measure or reproducibly simulate a baseline using fixed inputs and documented conditions.
5. Select at least two suitable optimization methods from Chapter 6, implement them, and justify why they fit the problem.
6. Compare the baseline and optimized alternatives using objective value, runtime, resource use, and other relevant metrics.
7. Investigate trade-offs, sensitivity, scalability, and conditions under which the methods may fail.
8. Verify correctness independently and include tests and reproducible experiments.

### Required deliverables
- Problem formulation and baseline documentation.
- Source code and tests.
- CSV results, Markdown analysis, and useful plots.
- A concise explanation of the measured improvement and its limitations.
- Exact prompt saved under `prompts/track_E/`.
- Updates to `PROMPT_LOG.md`, `CHAPTER_6_CHECKLIST.md`, and `AI_Prompt_Log.pdf`.
- Genuine AI failure records only.
- Run `py -3.12 -m pytest -v` and report actual results.

Keep the study focused and suitable for demonstrating during a viva. Reuse previous work where appropriate without duplicating entire tracks.

Commit local changes with a message beginning `[AI:Antigravity]`. Do not create a GitHub repository, add a remote, push, or publish anything. Do not switch branches or rewrite history without asking.

Stop after Track E. Report the problem, mathematical formulation, baseline, methods, measured results, tests, limitations, and commit hash.
```

---

### Execution & Verification Summary
1. **Source Code**:
   - `track_E_case_study/src/ev_charging_problem.py`: Dataclasses, constraints, feasibility evaluator, ASAP baseline.
   - `track_E_case_study/src/ev_optimizers.py`: SLSQP, HiGHS LP, and Priority Greedy Valley-Filling.
   - `track_E_case_study/src/__init__.py`: Package exports.
   - `track_E_case_study/src/export_dataset.py`: Deterministic fleet dataset exporter.
2. **Unit Tests**:
   - `track_E_case_study/tests/test_ev_charging.py`: 7 tests passing.
3. **Benchmarks & Artifacts**:
   - `track_E_case_study/results/ev_fleet_metrics.csv`
   - `track_E_case_study/results/ev_fleet_metrics.md`
   - `track_E_case_study/results/ev_fleet_charging_profiles.png`
   - `track_E_case_study/data/synthetic_fleet_dataset.csv` & `.json`.
