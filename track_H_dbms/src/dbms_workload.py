"""
TPC-H Benchmark Workload Definitions and Query Plan Telemetry.
Provides standardized analytical queries, plan inspection via EXPLAIN QUERY PLAN,
and high-precision latency measurement.
"""

import sqlite3
import time
import hashlib
from typing import Dict, List, Tuple, Any

# Standard TPC-H Queries adapted for SQLite
TPCH_QUERIES = {
    "Q1_Pricing_Summary": {
        "description": "TPC-H Q1: Pricing summary report (large aggregation on lineitem)",
        "weight": 2.0,
        "sql": """
        SELECT
            l_returnflag,
            l_linestatus,
            sum(l_quantity) AS sum_qty,
            sum(l_extendedprice) AS sum_base_price,
            sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
            avg(l_quantity) AS avg_qty,
            avg(l_extendedprice) AS avg_price,
            avg(l_discount) AS avg_disc,
            count(*) AS count_order
        FROM lineitem
        WHERE l_shipdate <= '1998-09-01'
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus;
        """
    },
    "Q3_Shipping_Priority": {
        "description": "TPC-H Q3: Shipping priority query (3-way join: customer, orders, lineitem)",
        "weight": 3.0,
        "sql": """
        SELECT
            l_orderkey,
            sum(l_extendedprice * (1 - l_discount)) AS revenue,
            o_orderdate,
            o_shippriority
        FROM customer, orders, lineitem
        WHERE c_mktsegment = 'BUILDING'
          AND c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate < '1995-03-15'
          AND l_shipdate > '1995-03-15'
        GROUP BY l_orderkey, o_orderdate, o_shippriority
        ORDER BY revenue DESC, o_orderdate
        LIMIT 10;
        """
    },
    "Q5_Local_Supplier_Volume": {
        "description": "TPC-H Q5: Local supplier volume query (6-way join across region, nation, customer, supplier, orders, lineitem)",
        "weight": 4.0,
        "sql": """
        SELECT
            n_name,
            sum(l_extendedprice * (1 - l_discount)) AS revenue
        FROM customer, orders, lineitem, supplier, nation, region
        WHERE c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND l_suppkey = s_suppkey
          AND c_nationkey = s_nationkey
          AND s_nationkey = n_nationkey
          AND n_regionkey = r_regionkey
          AND r_name = 'ASIA'
          AND o_orderdate >= '1994-01-01'
          AND o_orderdate < '1995-01-01'
        GROUP BY n_name
        ORDER BY revenue DESC;
        """
    },
    "Q6_Forecast_Revenue_Change": {
        "description": "TPC-H Q6: Forecasting revenue change (selective multi-predicate range scan on lineitem)",
        "weight": 1.5,
        "sql": """
        SELECT
            sum(l_extendedprice * l_discount) AS revenue
        FROM lineitem
        WHERE l_shipdate >= '1994-01-01'
          AND l_shipdate < '1995-01-01'
          AND l_discount BETWEEN 0.05 AND 0.07
          AND l_quantity < 24;
        """
    },
    "Q10_Returned_Item_Reporting": {
        "description": "TPC-H Q10: Returned item reporting (4-way join: customer, orders, lineitem, nation)",
        "weight": 2.5,
        "sql": """
        SELECT
            c_custkey,
            c_name,
            sum(l_extendedprice * (1 - l_discount)) AS revenue,
            c_acctbal,
            n_name,
            c_address,
            c_phone
        FROM customer, orders, lineitem, nation
        WHERE c_custkey = o_custkey
          AND l_orderkey = o_orderkey
          AND o_orderdate >= '1993-10-01'
          AND o_orderdate < '1994-01-01'
          AND l_returnflag = 'R'
          AND c_nationkey = n_nationkey
        GROUP BY c_custkey, c_name, c_acctbal, c_phone, n_name, c_address
        ORDER BY revenue DESC
        LIMIT 20;
        """
    }
}


def explain_query_plan(conn: sqlite3.Connection, sql: str) -> List[str]:
    """Retrieves SQLite's EXPLAIN QUERY PLAN node descriptions."""
    cur = conn.cursor()
    cur.execute(f"EXPLAIN QUERY PLAN {sql}")
    rows = cur.fetchall()
    # Format: (id, parent, notused, detail)
    plan_lines = [row[3] for row in rows]
    return plan_lines


def execute_and_hash_result(conn: sqlite3.Connection, sql: str) -> Tuple[List[Any], str]:
    """Executes query, returns top rows and deterministic SHA-256 result hash for equivalence verification."""
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    # Normalize rows to string for hashing
    hasher = hashlib.sha256()
    for row in rows:
        row_str = "|".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in row)
        hasher.update(row_str.encode("utf-8"))
    return rows, hasher.hexdigest()


def measure_query_latency(
    conn: sqlite3.Connection,
    sql: str,
    n_warmup: int = 1,
    n_runs: int = 5
) -> Dict[str, float]:
    """Measures query execution latency with warmup and statistical metrics."""
    cur = conn.cursor()

    # Warmup
    for _ in range(n_warmup):
        cur.execute(sql)
        _ = cur.fetchall()

    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        cur.execute(sql)
        _ = cur.fetchall()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms.sort()
    mean_lat = sum(latencies_ms) / len(latencies_ms)
    p50_lat = latencies_ms[len(latencies_ms) // 2]
    p95_lat = latencies_ms[int(len(latencies_ms) * 0.95)]
    std_lat = (sum((x - mean_lat) ** 2 for x in latencies_ms) / len(latencies_ms)) ** 0.5

    return {
        "mean_ms": mean_lat,
        "p50_ms": p50_lat,
        "p95_ms": p95_lat,
        "std_ms": std_lat,
        "min_ms": latencies_ms[0],
        "max_ms": latencies_ms[-1],
    }
