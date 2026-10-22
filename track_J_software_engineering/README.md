# Track J: Software-Engineering Optimization — Test Suite Minimization (TSM)

## 1. Problem Formulation & Theoretical Foundations

In continuous integration (CI/CD) pipelines, regression testing suites expand monotonically as new features and bug fixes are committed. Running the entire test suite on every commit causes build queue congestion, delayed developer feedback, and high compute infrastructure expenses.

### Mathematical Formulation: Minimum Weight Set Cover
Let $T = \{t_1, t_2, \dots, t_n\}$ be the full regression test suite of $n$ tests.
Each test $j \in \{1, \dots, n\}$ has an empirical execution cost $c_j > 0$ (runtime in milliseconds).
Let $R = \{r_1, r_2, \dots, r_m\}$ be the set of $m$ achievable code requirements (statement lines and synthetic mutation faults).
The binary requirement coverage matrix is $A \in \{0, 1\}^{m \times n}$, where:
$$A_{ij} = \begin{cases} 1 & \text{if test } j \text{ covers requirement } r_i \\ 0 & \text{otherwise} \end{cases}$$

- **Decision Variables**:
  $$x_j \in \{0, 1\} \quad \forall j \in \{1, \dots, n\}$$
  where $x_j = 1$ if test $j$ is retained in the minimized suite, and $0$ if discarded.

- **Objective Function (Cost Minimization)**:
  $$\min_{x} \sum_{j=1}^n c_j x_j$$

- **Constraints (100% Coverage & Fault Retention)**:
  $$\sum_{j=1}^n A_{ij} x_j \ge 1 \quad \forall i \in \{1, \dots, m\}$$
  $$x_j \in \{0, 1\} \quad \forall j \in \{1, \dots, n\}$$

- **Time-Budgeted Dual Formulation (Submodular Knapsack Pareto Frontier)**:
  Under a hard CI turnaround time budget $B$:
  $$\max_{S \subseteq T} \left| \bigcup_{j \in S} \text{Cov}(j) \right| \quad \text{subject to } \sum_{j \in S} c_j \le B$$

---

## 2. Target Codebase Architecture (1,055 LOC)

The target system is an enterprise-grade financial and inventory management engine located under `track_J_software_engineering/target_codebase/`:
- **`models.py`** (245 lines): Data entities (`Item`, `Customer`, `Order`, `OrderLine`, `Invoice`, `Warehouse`, `StockBatch`), domain enums, gross margin/markup calculations, credit limit checks.
- **`inventory_service.py`** (188 lines): Multi-warehouse allocation, FIFO batch depletion, perishable expiration skipping, Wilson Economic Order Quantity (EOQ), analytical safety stock ($SS = Z \cdot \sigma_L$), Reorder Point (ROP) triggers.
- **`pricing_engine.py`** (168 lines): Tiered volume discounts, promotional coupon validation with expiry and discount caps, multi-tier regional tax policies (state, county, and luxury surcharges), multi-currency exchange conversion.
- **`order_processor.py`** (218 lines): Lifecycle state machine (`DRAFT` &rarr; `VALIDATED` &rarr; `PAYMENT_AUTHORIZED` &rarr; `ALLOCATED` &rarr; `SHIPPED`), anti-fraud risk scoring heuristics, credit verification, reservation rollback.
- **`analytics_reporting.py`** (139 lines): ABC Pareto 80/20 inventory classification, inventory turnover ratio, dead stock identification, financial revenue aggregation.
- **`utils.py`** (130 lines): In-memory TTL cache with LRU eviction, tamper-evident SHA-256 cryptographic audit chaining, financial currency formatters.
- **`reporting_export.py`** (52 lines): Tabular CSV serialization and audit summary reporting.

The regression test suite consists of **80 distinct test cases** (`track_J_software_engineering/src/test_catalog.py`).

---

## 3. Implemented Chapter 6 Optimization Methods

1. **Exact 0-1 Integer Linear Programming (HiGHS Branch-and-Cut)**:
   - Implemented via `scipy.optimize.milp` with binary integrality constraints.
   - Finds the global optimum $x^*$, proving the lowest possible execution cost that satisfies all coverage and mutant constraints.
2. **Greedy Set Cover Approximation (Chvátal's Algorithm)**:
   - Iteratively selects test $j^* = \arg\max_j \frac{\Delta \text{Coverage}(j)}{c_j}$.
   - Provable $(1 - 1/e)$ approximation ratio with near-zero solver overhead ($<5$ ms).
3. **Submodular Knapsack Pareto Frontier**:
   - Sweeps time budgets from 15% to 100% of optimal runtime to generate the complete cost-vs-coverage Pareto trade-off curve.
4. **Baselines**:
   - **Exhaustive Full Suite Baseline**: Executes all 80 tests.
   - **Random Subsampling Baseline**: Randomly samples $k$ tests, demonstrating steep coverage degradation when mathematical optimization is omitted.

---

## 4. Empirical Benchmark Results

Measured on Python 3.12 (Windows 11):

| Method | Selected Tests | Suite Size Reduction | Suite Runtime | Speedup | Statement Coverage | Mutation Score | Solver Overhead |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Full Suite** | 80 / 80 | 0.0% | 107.5 ms | **1.00x** | 100.0% (467/467) | 100.0% (15/15) | 0.00 ms |
| **Greedy Set Cover (Chvátal)** | 57 / 80 | **28.7%** | 70.9 ms | **1.52x** | **100.0%** (467/467) | **100.0%** (15/15) | 4.55 ms |
| **0-1 ILP (HiGHS)** | 53 / 80 | **33.8%** | 68.9 ms | **1.56x** | **100.0%** (467/467) | **100.0%** (15/15) | 24.32 ms |
| **Random Subsampling (k=53)** | 53 / 80 | 33.8% | 80.2 ms | 1.34x | **82.9%** (387/467) | **73.3%** (11/15) | 0.00 ms |

### Key Observations
1. **Mathematical Optimality**: 0-1 ILP achieved the provable minimum execution time (68.87 ms), selecting 53 tests and discarding 27 redundant tests while retaining 100.0% statement coverage and killing 100.0% of mutants.
2. **Speedup**: 0-1 ILP delivered a **1.56x speedup** on regression testing with zero defect-detection loss.
3. **Greedy Efficiency**: Chvátal's Greedy heuristic selected 57 tests in only **4.55 ms** solver time, achieving **1.52x speedup** (within 2.9% of the exact ILP global optimum).
4. **Random Failure**: Uniform random selection dropped coverage by **17.1%** and missed **26.7%** of injected domain faults, demonstrating why mathematical optimization is mandatory.

---

## 5. Reproduction & Verification Commands

```powershell
# 1. Run the comprehensive Track J benchmarks and generate CSV, MD, PNG artifacts:
py -3.12 track_J_software_engineering/src/run_track_j_benchmarks.py

# 2. Run Track J automated verification tests:
py -3.12 -m pytest track_J_software_engineering/tests/test_track_j.py -v

# 3. Run all 151 unit tests across all Tracks A through J:
py -3.12 -m pytest -v
```

---

## 6. Viva Demonstration Script

1. **Step 1: Codebase Verification**:
   Show that `track_J_software_engineering/target_codebase/` contains 1,055 LOC across 7 production-grade modules.
2. **Step 2: Redundancy Demonstration**:
   Run `run_track_j_benchmarks.py` live. Explain to the examiner that 80 tests cover 467 statement lines, but many integration tests repeatedly exercise common initialization, pricing, and validation routines.
3. **Step 3: 0-1 ILP vs Greedy**:
   Show how 0-1 ILP eliminates 27 redundant tests (33.8% reduction) with a 1.56x speedup. Point out that Greedy is near-instantaneous (4.5 ms) and achieves 1.52x speedup.
4. **Step 4: Zero Defect Escape (Mutation Score)**:
   Point to the 15 synthetic domain mutants (off-by-one errors, FIFO inversions, luxury tax errors, crypto audit tampering). Show that both the 0-1 ILP and Greedy minimized suites killed 15/15 mutants (100.0%).
5. **Step 5: Visual Plots**:
   Open `track_J_software_engineering/results/track_j_benchmarks.png` to walk through the 4 panels: Suite Size, Runtime Speedup, Coverage vs Mutation Score, and the Pareto Time-Budget Curve.
