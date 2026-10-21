# Prompt Record 16: Track H — DBMS Optimization (TPC-H, 0-1 Knapsack Index Selection & System-R Join DP)

- **Prompt ID**: `P-TRH-01`
- **Date & Time**: 2026-10-22 00:24:19 IST
- **Track**: Track H (Database Management System Optimization)
- **AI Tool**: Antigravity (Powered by Gemini 3.8 Flash)
- **Commit Reference**: Pending commit

---

### Purpose
Complete Track H: Database Management System Optimization:
1. **Documented Dataset & Workload**:
   - Engine: SQLite Version 3.49.1 (Python 3.12 embedded C-engine).
   - Dataset: TPC-H decision-support relational schema with 7 tables (`region`, `nation`, `supplier`, `part`, `customer`, `orders`, `lineitem`), 50,004 lineitems, 20,000 orders, 4,000 customers (9.27 MB file size).
   - Standard Analytical Queries: Q1 (Pricing Summary), Q3 (Shipping Priority 3-way join), Q5 (Local Supplier Volume 6-way join), Q6 (Revenue Forecasting selective range scan), and Q10 (Returned Item Reporting 4-way join).
2. **Two Core Chapter 6 Optimization Formulations**:
   - **Index Selection (0-1 Knapsack ILP vs Greedy Heuristic)**:
     - Formulated secondary storage budget-constrained index configuration problem across 10 candidate single-column, composite, covering, and spurious indexes.
     - Implemented marginal gain Greedy selection ($\Delta \text{Benefit} / \text{cost}$).
     - Implemented exact 0-1 Knapsack Integer Linear Programming (ILP) with per-query baseline capping, solved using HiGHS branch-and-cut via `scipy.optimize.milp`.
     - Compared selections across budgets $B \in [2.0\,\text{MB}, 5.0\,\text{MB}, 10.0\,\text{MB}]$.
   - **Join-Order Selection (Selinger / System-R Dynamic Programming)**:
     - Formulated Left-Deep join tree bottom-up dynamic programming across subsets using intermediate cardinality estimation and join selectivity factors.
     - Evaluated TPC-H Q5 6-table join graph connectivity, avoiding Cartesian cross-products and evaluating all 63 non-empty relation subsets.
   - **Buffer-Pool Cache Tuning**:
     - Measured page caching effects via `PRAGMA cache_size` (2 MB vs 64 MB).
3. **Empirical Measurements & Query Plan Evolution**:
   - `EXPLAIN QUERY PLAN` verification: demonstrated transition from full `SCAN lineitem` (50,004 rows) to `SEARCH lineitem USING COVERING INDEX` and pre-materialized B-tree seeks (eliminating ephemeral automatic B-trees).
   - Massive speedups on selective queries (e.g. **15.9x speedup** on Q6).
   - Demonstrated index trade-offs: low-selectivity range filters (Q1) suffer a 0.35x slowdown under unclustered index scans due to random page pointer dereferencing vs sequential scans.
   - 100% output equivalence: SHA-256 result hash matches baseline on all queries.
4. **Testing & Deliverables**:
   - `track_H_dbms/src/`: `tpch_generator.py`, `dbms_workload.py`, `index_selection_ilp.py`, `join_order_dp.py`, `run_dbms_benchmarks.py`.
   - `track_H_dbms/results/`: `dbms_metrics.csv`, `dbms_metrics.md`, `dbms_benchmarks.png`.
   - `track_H_dbms/tests/test_dbms.py`: 6 pytest test cases passing.

---

### Original Prompt
```text
Continue Assignment 8 in `C:\dev\ot-8`. Complete Track H: Database Management System Optimization.

Inspect the assignment, repository, existing DBMS environment, checklist, prompt logs, and Git status before editing. Use an existing database such as PostgreSQL, MySQL, or SQLite, or a small database application if that is more practical. Document the exact database and version.

### Required work

1. Create or identify a documented dataset and queries representative of a realistic workload.
2. Establish a reproducible baseline using the original query plans, execution times, and relevant resource metrics.
3. Formulate at least one optimization problem involving query rewriting, join-order selection, index or materialized-view selection, partitioning, buffer-pool configuration, or another justified DBMS optimization.
4. Implement and compare at least two Chapter 6 methods where appropriate, such as greedy selection, dynamic programming, or integer linear programming.
5. Use at least one accepted benchmark, such as a documented TPC-H workload, or a clearly described synthetic dataset. If using a synthetic dataset, state its size, schema, generation procedure, and assumptions.
6. Compare baseline and optimized queries or configurations under identical conditions.
7. Report query plans, latency, throughput where relevant, and storage or memory overhead. Repeat runs where feasible and explain variability.

### Correctness and reporting
- Verify query outputs remain equivalent after optimization.
- Include tests, SQL scripts, benchmark commands, raw CSV results, Markdown analysis, and plots where useful.
- Explain trade-offs, limitations, and when an index or optimization may hurt performance.
- Do not claim an optimization improved performance unless the measurements support it.

Save the exact prompt under `prompts/track_H/`. Update `PROMPT_LOG.md`, `CHAPTER_6_CHECKLIST.md`, and `AI_Prompt_Log.pdf`. Add only genuine failures to `AI_FAILURE_LOG.md`.

Run relevant database tests and `py -3.12 -m pytest -v` if applicable. Report actual outcomes.

Commit local changes with a message starting `[AI:Antigravity]`. Do not create a GitHub repository, configure a remote, push, or publish anything. Do not change branches or rewrite history without asking.

Stop after Track H and report the database, dataset, formulation, optimization methods, measured results, correctness checks, tests, limitations, and commit hash.
```

---

### Execution & Verification Summary
- **DBMS**: SQLite Version 3.49.1.
- **Dataset**: TPC-H benchmark (9.27 MB, 50,004 lineitems, 20,000 orders).
- **Speedup**: Up to **15.91x speedup** on selective queries with covering indexes.
- **Storage Overhead**: 4.92 MB secondary storage index set for 5.0 MB budget.
- **Tests**: 131/131 passing across all tracks (6/6 for Track H).
