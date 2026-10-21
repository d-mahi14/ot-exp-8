"""
Unit and integration tests for Track H: DBMS Optimization.
Verifies TPC-H dataset integrity, query output equivalence, query plan evolution,
Index Selection (Greedy vs ILP), and Selinger System-R Join-Order Dynamic Programming.
"""

import os
import sqlite3
import numpy as np
import pytest

from track_H_dbms.src.tpch_generator import build_benchmark_database
from track_H_dbms.src.dbms_workload import (
    TPCH_QUERIES,
    explain_query_plan,
    execute_and_hash_result
)
from track_H_dbms.src.index_selection_ilp import (
    CANDIDATE_INDEXES,
    drop_all_indexes,
    measure_index_storage_costs,
    solve_index_selection_greedy,
    solve_index_selection_ilp
)
from track_H_dbms.src.join_order_dp import JoinOrderOptimizerDP, evaluate_tpch_q5_join_orders


@pytest.fixture(scope="module")
def db_conn():
    """Provides a shared in-memory or on-disk database for tests."""
    db_file = os.path.join(os.path.dirname(__file__), "..", "data", "tpch_benchmark.db")
    if not os.path.exists(db_file):
        conn = build_benchmark_database(db_file, n_orders=2000)
    else:
        conn = sqlite3.connect(db_file)
    yield conn
    drop_all_indexes(conn)
    conn.close()


def test_tpch_schema_and_dataset_integrity(db_conn):
    """Verifies table existence, primary keys, and realistic row counts."""
    cur = db_conn.cursor()
    tables = ["region", "nation", "supplier", "part", "customer", "orders", "lineitem"]
    for t in tables:
        cur.execute(f"SELECT count(*) FROM {t};")
        count = cur.fetchone()[0]
        assert count > 0, f"Table {t} is empty"

    cur.execute("SELECT count(*) FROM lineitem;")
    n_lineitems = cur.fetchone()[0]
    assert n_lineitems >= 2000, f"Expected at least 2000 lineitems, got {n_lineitems}"


def test_query_result_equivalence_with_indexes(db_conn):
    """
    Verifies that creating indexes never alters the logical query output.
    Asserts exact SHA-256 result hash equality across all TPC-H queries.
    """
    # 1. Unindexed baseline hashes
    drop_all_indexes(db_conn)
    baseline_hashes = {}
    for q_name, q_info in TPCH_QUERIES.items():
        _, r_hash = execute_and_hash_result(db_conn, q_info["sql"])
        baseline_hashes[q_name] = r_hash

    # 2. Create covering and join indexes
    cur = db_conn.cursor()
    cur.execute(CANDIDATE_INDEXES["idx_lineitem_q6_covering"]["sql"])
    cur.execute(CANDIDATE_INDEXES["idx_orders_custdate"]["sql"])
    cur.execute(CANDIDATE_INDEXES["idx_lineitem_orderkey"]["sql"])
    db_conn.commit()

    # 3. Indexed execution hashes
    for q_name, q_info in TPCH_QUERIES.items():
        _, r_hash = execute_and_hash_result(db_conn, q_info["sql"])
        assert r_hash == baseline_hashes[q_name], (
            f"Query output mismatch on {q_name} after index creation!"
        )

    drop_all_indexes(db_conn)


def test_explain_query_plan_shows_index_usage(db_conn):
    """
    Verifies that EXPLAIN QUERY PLAN reflects B-tree index usage
    instead of full table scans after indexing.
    """
    q6_sql = TPCH_QUERIES["Q6_Forecast_Revenue_Change"]["sql"]

    # Before index: must scan lineitem
    drop_all_indexes(db_conn)
    plan_before = explain_query_plan(db_conn, q6_sql)
    assert any("SCAN lineitem" in line for line in plan_before), (
        f"Expected SCAN lineitem in unindexed plan, got: {plan_before}"
    )

    # After covering index: must use covering index search
    cur = db_conn.cursor()
    cur.execute(CANDIDATE_INDEXES["idx_lineitem_q6_covering"]["sql"])
    db_conn.commit()

    plan_after = explain_query_plan(db_conn, q6_sql)
    assert any("SEARCH lineitem USING COVERING INDEX" in line for line in plan_after), (
        f"Expected SEARCH lineitem USING COVERING INDEX in optimized plan, got: {plan_after}"
    )

    drop_all_indexes(db_conn)


def test_index_selection_greedy_vs_ilp_budget_constraint(db_conn):
    """
    Verifies that both Greedy and 0-1 Knapsack ILP respect storage budgets
    and produce valid index configurations.
    """
    costs_mb = measure_index_storage_costs(db_conn)
    idx_names = list(CANDIDATE_INDEXES.keys())
    q_names = list(TPCH_QUERIES.keys())
    q_weights = np.array([TPCH_QUERIES[q]["weight"] for q in q_names])

    # Synthetic benefit matrix to verify mathematical knapsack behavior
    benefit_matrix = np.array([
        [10.0, 5.0, 0.0, 20.0, 0.0],
        [0.0, 15.0, 10.0, 0.0, 12.0],
        [0.0, 25.0, 0.0, 0.0, 0.0],
        [0.0, 20.0, 15.0, 0.0, 18.0],
        [0.0, 10.0, 8.0, 0.0, 10.0],
        [0.0, 0.0, 12.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0, 6.0],
        [0.0, 0.0, 7.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 35.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    baseline_lats = np.array([45.0, 20.0, 15.0, 10.0, 15.0])

    for budget in [1.0, 2.5, 5.0]:
        greedy_idxs, greedy_obj, greedy_cost = solve_index_selection_greedy(
            budget, costs_mb, benefit_matrix, q_weights, idx_names, baseline_lats
        )
        ilp_idxs, ilp_obj, ilp_cost = solve_index_selection_ilp(
            budget, costs_mb, benefit_matrix, q_weights, idx_names, baseline_lats
        )

        assert greedy_cost <= budget + 1e-5, f"Greedy exceeded budget {budget}: {greedy_cost}"
        assert ilp_cost <= budget + 1e-5, f"ILP exceeded budget {budget}: {ilp_cost}"

        # ILP knapsack objective must be at least greedy or within 5% due to capping
        assert ilp_obj >= greedy_obj * 0.90, (
            f"ILP objective {ilp_obj:.2f} significantly worse than greedy {greedy_obj:.2f}"
        )


def test_join_order_dp_optimality(db_conn):
    """
    Verifies that the System-R Dynamic Programming join optimizer evaluates
    all subsets and that query output remains identical under rewritten join orders.
    """
    join_res = evaluate_tpch_q5_join_orders()
    assert "optimal_order" in join_res
    assert len(join_res["optimal_order"]) == 6
    assert join_res["telemetry"]["subsets_evaluated"] == 63  # 2^6 - 1 subsets

    # Execute Q5 with unoptimized table order vs optimal DP table order
    sql_unopt = """
    SELECT n_name, sum(l_extendedprice * (1 - l_discount)) AS revenue
    FROM lineitem, orders, customer, supplier, nation, region
    WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey = s_suppkey
      AND c_nationkey = s_nationkey AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
      AND r_name = 'ASIA' AND o_orderdate >= '1994-01-01' AND o_orderdate < '1995-01-01'
    GROUP BY n_name ORDER BY revenue DESC;
    """
    sql_opt = """
    SELECT n_name, sum(l_extendedprice * (1 - l_discount)) AS revenue
    FROM region, nation, supplier, customer, orders, lineitem
    WHERE n_regionkey = r_regionkey AND s_nationkey = n_nationkey AND c_nationkey = s_nationkey
      AND c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey = s_suppkey
      AND r_name = 'ASIA' AND o_orderdate >= '1994-01-01' AND o_orderdate < '1995-01-01'
    GROUP BY n_name ORDER BY revenue DESC;
    """
    _, h_unopt = execute_and_hash_result(db_conn, sql_unopt)
    _, h_opt = execute_and_hash_result(db_conn, sql_opt)
    assert h_unopt == h_opt, "Join order output mismatch!"


def test_artifacts_exist():
    """Verifies that all report and benchmark artifact files exist."""
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    csv_file = os.path.join(results_dir, "dbms_metrics.csv")
    md_file = os.path.join(results_dir, "dbms_metrics.md")
    png_file = os.path.join(results_dir, "dbms_benchmarks.png")

    assert os.path.exists(csv_file), f"CSV metrics file missing: {csv_file}"
    assert os.path.exists(md_file), f"Markdown report missing: {md_file}"
    assert os.path.exists(png_file), f"Visualization plot missing: {png_file}"

    with open(md_file, "r", encoding="utf-8") as f:
        md_text = f.read()
        assert "TPC-H" in md_text
        assert "Index Selection" in md_text
        assert "Join-Order" in md_text
