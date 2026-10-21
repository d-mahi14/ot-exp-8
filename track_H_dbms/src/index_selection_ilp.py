"""
Index Selection Optimization using 0-1 Knapsack Integer Linear Programming (ILP)
and Greedy Heuristic with Diminishing Returns.
Formulates the secondary storage budget-constrained index configuration problem.
"""

import sqlite3
import os
from typing import Dict, List, Set, Tuple, Any
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# 10 Diverse Candidate Indexes spanning join keys, selective filters, covering columns, and spurious indexes
CANDIDATE_INDEXES = {
    "idx_lineitem_shipdate": {
        "table": "lineitem",
        "sql": "CREATE INDEX idx_lineitem_shipdate ON lineitem(l_shipdate);",
        "target_queries": ["Q1_Pricing_Summary", "Q3_Shipping_Priority", "Q6_Forecast_Revenue_Change"],
        "description": "Index on l_shipdate for range scans and join filters."
    },
    "idx_orders_custdate": {
        "table": "orders",
        "sql": "CREATE INDEX idx_orders_custdate ON orders(o_custkey, o_orderdate);",
        "target_queries": ["Q3_Shipping_Priority", "Q5_Local_Supplier_Volume", "Q10_Returned_Item_Reporting"],
        "description": "Composite index on orders(o_custkey, o_orderdate) for customer joins."
    },
    "idx_customer_mktsegment": {
        "table": "customer",
        "sql": "CREATE INDEX idx_customer_mktsegment ON customer(c_mktsegment, c_custkey);",
        "target_queries": ["Q3_Shipping_Priority"],
        "description": "Filter index on customer market segment."
    },
    "idx_lineitem_orderkey": {
        "table": "lineitem",
        "sql": "CREATE INDEX idx_lineitem_orderkey ON lineitem(l_orderkey);",
        "target_queries": ["Q3_Shipping_Priority", "Q5_Local_Supplier_Volume", "Q10_Returned_Item_Reporting"],
        "description": "Primary join index on lineitem(l_orderkey)."
    },
    "idx_orders_orderdate": {
        "table": "orders",
        "sql": "CREATE INDEX idx_orders_orderdate ON orders(o_orderdate);",
        "target_queries": ["Q3_Shipping_Priority", "Q5_Local_Supplier_Volume", "Q10_Returned_Item_Reporting"],
        "description": "Date range index on orders(o_orderdate)."
    },
    "idx_lineitem_supp_order": {
        "table": "lineitem",
        "sql": "CREATE INDEX idx_lineitem_supp_order ON lineitem(l_suppkey, l_orderkey);",
        "target_queries": ["Q5_Local_Supplier_Volume"],
        "description": "Composite index for supplier-order join correlation."
    },
    "idx_customer_nation": {
        "table": "customer",
        "sql": "CREATE INDEX idx_customer_nation ON customer(c_nationkey, c_custkey);",
        "target_queries": ["Q5_Local_Supplier_Volume", "Q10_Returned_Item_Reporting"],
        "description": "Composite index on customer nation foreign key."
    },
    "idx_supplier_nation": {
        "table": "supplier",
        "sql": "CREATE INDEX idx_supplier_nation ON supplier(s_nationkey, s_suppkey);",
        "target_queries": ["Q5_Local_Supplier_Volume"],
        "description": "Composite index on supplier nation foreign key."
    },
    "idx_lineitem_q6_covering": {
        "table": "lineitem",
        "sql": "CREATE INDEX idx_lineitem_q6_covering ON lineitem(l_shipdate, l_discount, l_quantity, l_extendedprice);",
        "target_queries": ["Q6_Forecast_Revenue_Change"],
        "description": "Covering index containing all projection and predicate attributes for Q6."
    },
    "idx_orders_status_spurious": {
        "table": "orders",
        "sql": "CREATE INDEX idx_orders_status_spurious ON orders(o_orderstatus);",
        "target_queries": [],
        "description": "Low-selectivity index on orderstatus (trade-off/negative example)."
    }
}


def drop_all_indexes(conn: sqlite3.Connection):
    """Drops all non-system indexes from the database."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_autoindex%';")
    indexes = [row[0] for row in cur.fetchall()]
    for idx in indexes:
        cur.execute(f"DROP INDEX IF EXISTS {idx};")
    conn.commit()


def measure_index_storage_costs(conn: sqlite3.Connection) -> Dict[str, float]:
    """Measures the exact physical storage footprint of each candidate index in MB."""
    cur = conn.cursor()
    costs_mb = {}

    for idx_name, info in CANDIDATE_INDEXES.items():
        drop_all_indexes(conn)
        cur.execute("PRAGMA page_size;")
        page_size = cur.fetchone()[0]
        cur.execute("PRAGMA page_count;")
        pages_before = cur.fetchone()[0]
        cur.execute("PRAGMA freelist_count;")
        free_before = cur.fetchone()[0]

        cur.execute(info["sql"])
        conn.commit()

        cur.execute("PRAGMA page_count;")
        pages_after = cur.fetchone()[0]
        cur.execute("PRAGMA freelist_count;")
        free_after = cur.fetchone()[0]

        used_pages = (pages_after - pages_before) + (free_before - free_after)
        used_bytes = max(page_size, used_pages * page_size)
        delta_mb = used_bytes / (1024.0 * 1024.0)
        costs_mb[idx_name] = round(delta_mb, 3)

    drop_all_indexes(conn)
    return costs_mb


def compute_benefit_matrix(
    conn: sqlite3.Connection,
    queries: Dict[str, Any]
) -> Tuple[Dict[str, float], np.ndarray, List[str], List[str]]:
    """
    Computes baseline query execution times without indexes and the individual
    latency reduction (benefit) provided by each candidate index.
    Returns: (baseline_latencies, benefit_matrix, index_names, query_names)
    """
    from dbms_workload import measure_query_latency

    drop_all_indexes(conn)
    q_names = list(queries.keys())
    idx_names = list(CANDIDATE_INDEXES.keys())

    # 1. Baseline latencies with no indexes
    baseline_lat = {}
    for q in q_names:
        stats = measure_query_latency(conn, queries[q]["sql"], n_warmup=1, n_runs=3)
        baseline_lat[q] = stats["mean_ms"]

    # 2. Individual benefit of each candidate index
    benefit_matrix = np.zeros((len(idx_names), len(q_names)))

    cur = conn.cursor()
    for i, idx_name in enumerate(idx_names):
        drop_all_indexes(conn)
        cur.execute(CANDIDATE_INDEXES[idx_name]["sql"])
        conn.commit()

        for j, q in enumerate(q_names):
            stats = measure_query_latency(conn, queries[q]["sql"], n_warmup=1, n_runs=3)
            # Latency reduction (ms saved)
            lat_with_index = stats["mean_ms"]
            benefit = max(0.0, baseline_lat[q] - lat_with_index)
            benefit_matrix[i, j] = benefit

    drop_all_indexes(conn)
    return baseline_lat, benefit_matrix, idx_names, q_names


def compute_joint_objective(
    selected_indices: List[int],
    benefit_matrix: np.ndarray,
    query_weights: np.ndarray,
    baseline_latencies: np.ndarray = None
) -> float:
    """Computes the total weighted benefit of an index set, capped at baseline query latency."""
    if not selected_indices:
        return 0.0
    query_benefits = np.sum(benefit_matrix[selected_indices, :], axis=0)
    if baseline_latencies is not None:
        query_benefits = np.minimum(query_benefits, baseline_latencies)
    return float(np.dot(query_benefits, query_weights))


def solve_index_selection_greedy(
    budget_mb: float,
    costs_mb: Dict[str, float],
    benefit_matrix: np.ndarray,
    query_weights: np.ndarray,
    idx_names: List[str],
    baseline_latencies: np.ndarray = None
) -> Tuple[List[str], float, float]:
    """
    Greedy heuristic for Index Selection using Benefit-to-Cost Ratio.
    Iteratively picks the candidate index yielding the highest marginal gain per MB
    under per-query baseline capping.
    Returns: (selected_indexes, total_benefit, total_cost_mb)
    """
    costs = np.array([costs_mb[name] for name in idx_names])
    n_indexes = len(idx_names)
    selected_indices = []
    used_budget = 0.0
    current_obj = 0.0

    while True:
        best_candidate = None
        best_ratio = -1.0
        best_gain = 0.0

        for i in range(n_indexes):
            if i in selected_indices:
                continue
            c = costs[i]
            if used_budget + c > budget_mb:
                continue

            trial_indices = selected_indices + [i]
            trial_obj = compute_joint_objective(
                trial_indices, benefit_matrix, query_weights, baseline_latencies
            )
            gain = trial_obj - current_obj
            if gain <= 1e-4:
                continue

            eff_cost = max(c, 0.01)
            ratio = gain / eff_cost
            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate = i
                best_gain = gain

        if best_candidate is None:
            break

        selected_indices.append(best_candidate)
        used_budget += costs[best_candidate]
        current_obj += best_gain

    selected_names = [idx_names[i] for i in selected_indices]
    return selected_names, current_obj, round(used_budget, 3)


def solve_index_selection_ilp(
    budget_mb: float,
    costs_mb: Dict[str, float],
    benefit_matrix: np.ndarray,
    query_weights: np.ndarray,
    idx_names: List[str],
    baseline_latencies: np.ndarray
) -> Tuple[List[str], float, float]:
    """
    Exact 0-1 Knapsack Integer Linear Programming (ILP) Index Selection Solver.
    Formulation:
      Variables:
        x_i in {0, 1} for each candidate index i
        y_j in [0, Baseline_j] representing latency reduction for query j
      Maximize:
        sum_j (w_j * y_j)
      Subject to:
        sum_i (cost_i * x_i) <= Budget
        y_j <= sum_i (benefit_{i,j} * x_i)   for all j
        y_j <= Baseline_j                     for all j
    """
    n_idx = len(idx_names)
    n_q = len(query_weights)
    n_vars = n_idx + n_q

    # Objective: maximize sum_j (w_j * y_j) -> minimize -sum_j (w_j * y_j)
    c_obj = np.zeros(n_vars)
    for j in range(n_q):
        c_obj[n_idx + j] = -query_weights[j]

    # Variable bounds and integrality
    # x_i in {0, 1} (integer = 1)
    # y_j in [0, baseline_latencies[j]] (continuous = 0)
    integrality = np.zeros(n_vars)
    integrality[:n_idx] = 1  # binary

    lb = np.zeros(n_vars)
    ub = np.zeros(n_vars)
    ub[:n_idx] = 1.0
    for j in range(n_q):
        ub[n_idx + j] = baseline_latencies[j]

    bounds = Bounds(lb, ub)

    # Constraint 1: Storage Budget constraint
    # sum_i (cost_i * x_i) <= Budget
    A_budget = np.zeros((1, n_vars))
    for i in range(n_idx):
        A_budget[0, i] = costs_mb[idx_names[i]]
    lhs_budget = np.array([-np.inf])
    rhs_budget = np.array([budget_mb])

    # Constraint 2: Coupling constraints: y_j - sum_i (benefit_{i,j} * x_i) <= 0
    A_couple = np.zeros((n_q, n_vars))
    for j in range(n_q):
        A_couple[j, n_idx + j] = 1.0  # +y_j
        for i in range(n_idx):
            A_couple[j, i] = -benefit_matrix[i, j]  # -benefit_{i,j} * x_i
    lhs_couple = np.full(n_q, -np.inf)
    rhs_couple = np.zeros(n_q)

    # Combine constraints
    A_total = np.vstack([A_budget, A_couple])
    lhs_total = np.concatenate([lhs_budget, lhs_couple])
    rhs_total = np.concatenate([rhs_budget, rhs_couple])

    constraints = LinearConstraint(A_total, lhs_total, rhs_total)

    # Solve with HiGHS
    res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints)

    if not res.success:
        raise RuntimeError(f"ILP Index Selection solver failed: {res.status}")

    x_sol = res.x[:n_idx]
    selected_indexes = [idx_names[i] for i in range(n_idx) if x_sol[i] > 0.5]
    total_cost = sum(costs_mb[name] for name in selected_indexes)
    total_benefit = -res.fun

    return selected_indexes, total_benefit, round(total_cost, 3)
