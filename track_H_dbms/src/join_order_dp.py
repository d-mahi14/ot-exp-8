"""
Join-Order Optimization using Selinger / System-R Dynamic Programming.
Finds optimal Left-Deep join trees for multi-relation queries to minimize
intermediate tuple materialization and disk/memory join processing costs.
"""

from typing import Dict, List, Set, Tuple, Any
import itertools

class JoinOrderOptimizerDP:
    """
    Selinger / System-R Dynamic Programming Join-Order Optimizer.
    Enumerates join trees bottom-up across subset sizes 1 to n,
    evaluating join graph connectivity and selectivity factors.
    """

    def __init__(
        self,
        table_cardinalities: Dict[str, int],
        join_predicates: List[Tuple[str, str, float]]  # (table1, table2, selectivity)
    ):
        self.tables = list(table_cardinalities.keys())
        self.cardinalities = table_cardinalities
        self.predicates = join_predicates

        # Adjacency map for join connectivity
        self.adj: Dict[str, Set[str]] = {t: set() for t in self.tables}
        self.selectivity_map: Dict[Tuple[str, str], float] = {}

        for t1, t2, sel in join_predicates:
            self.adj[t1].add(t2)
            self.adj[t2].add(t1)
            self.selectivity_map[(t1, t2)] = sel
            self.selectivity_map[(t2, t1)] = sel

    def _cross_selectivity(self, set1: frozenset, set2: frozenset) -> float:
        """Computes the product of selectivities for all edges connecting set1 and set2."""
        total_sel = 1.0
        has_edge = False
        for t1 in set1:
            for t2 in set2:
                if (t1, t2) in self.selectivity_map:
                    total_sel *= self.selectivity_map[(t1, t2)]
                    has_edge = True
        return total_sel if has_edge else 1.0  # Cartesian product if no edge

    def _has_join_edge(self, set1: frozenset, set2: frozenset) -> bool:
        """Returns True if there is at least one join predicate connecting set1 and set2."""
        for t1 in set1:
            for t2 in set2:
                if (t1, t2) in self.selectivity_map:
                    return True
        return False

    def optimize_left_deep(self) -> Tuple[List[str], float, Dict[str, Any]]:
        """
        Executes bottom-up dynamic programming over relation subsets.
        Restricts search to Left-Deep join trees (standard in production optimizers).
        Returns: (optimal_order_list, total_cost, dp_telemetry)
        """
        # opt[subset] = (cost, cardinality, join_order_list)
        opt: Dict[frozenset, Tuple[float, float, List[str]]] = {}

        # 1. Base cases: size = 1
        for t in self.tables:
            card = float(self.cardinalities[t])
            scan_cost = card  # table scan cost is proportional to row count
            opt[frozenset([t])] = (scan_cost, card, [t])

        # 2. Dynamic programming for size = 2 to n
        n = len(self.tables)
        evaluations_count = 0

        for size in range(2, n + 1):
            for subset in itertools.combinations(self.tables, size):
                subset_fs = frozenset(subset)
                best_cost = float("inf")
                best_card = float("inf")
                best_order = []

                # In left-deep trees, subset = left_subset + single_relation
                for t in subset:
                    left_subset = subset_fs - frozenset([t])
                    if left_subset not in opt:
                        continue

                    evaluations_count += 1
                    left_cost, left_card, left_order = opt[left_subset]
                    right_card = float(self.cardinalities[t])

                    # Calculate joint selectivity
                    connected = self._has_join_edge(left_subset, frozenset([t]))
                    sel = self._cross_selectivity(left_subset, frozenset([t]))

                    # If not connected, apply Cartesian penalty factor
                    penalty = 1.0 if connected else 10.0

                    # New intermediate cardinality
                    new_card = left_card * right_card * sel

                    # Nested-loop / Hash join cost model:
                    # Cost = Cost(Left) + Cost(Scan(Right)) + Intermediate_Materialization_Cost
                    join_cost = left_cost + right_card + (new_card * 0.5) * penalty

                    if join_cost < best_cost:
                        best_cost = join_cost
                        best_card = new_card
                        best_order = left_order + [t]

                opt[subset_fs] = (best_cost, best_card, best_order)

        final_subset = frozenset(self.tables)
        optimal_cost, final_card, optimal_order = opt[final_subset]

        telemetry = {
            "n_tables": n,
            "subsets_evaluated": len(opt),
            "candidate_pairs_evaluated": evaluations_count,
            "estimated_final_cardinality": final_card,
            "optimal_cost": optimal_cost
        }

        return optimal_order, optimal_cost, telemetry


def evaluate_tpch_q5_join_orders() -> Dict[str, Any]:
    """
    Sets up the 6-table join graph for TPC-H Q5:
    Tables: region (R), nation (N), customer (C), supplier (S), orders (O), lineitem (L).
    Compares the optimal DP order against unoptimized/worst-case orders.
    """
    # Realistic filtered cardinalities based on TPC-H Q5 predicates
    # Filter: r_name = 'ASIA' -> only 1 region row
    # o_orderdate in [1994-01-01, 1995-01-01] -> ~15% of orders
    cardinalities = {
        "region": 1,        # 1 matching region ('ASIA')
        "nation": 5,         # 5 nations in Asia
        "supplier": 200,     # ~200 suppliers in Asia
        "customer": 800,     # ~800 customers in Asia
        "orders": 3000,      # ~3000 orders in 1994
        "lineitem": 50000    # 50,000 total lineitems
    }

    # Foreign key join selectivities (1 / TargetTablePKCount)
    predicates = [
        ("nation", "region", 1.0 / 5.0),
        ("customer", "nation", 1.0 / 25.0),
        ("supplier", "nation", 1.0 / 25.0),
        ("orders", "customer", 1.0 / 4000.0),
        ("lineitem", "orders", 1.0 / 20000.0),
        ("lineitem", "supplier", 1.0 / 1000.0),
    ]

    optimizer = JoinOrderOptimizerDP(cardinalities, predicates)
    optimal_order, optimal_cost, telemetry = optimizer.optimize_left_deep()

    # Worst-case order (e.g. large tables first without early selective filtering)
    worst_order = ["lineitem", "orders", "customer", "supplier", "nation", "region"]

    return {
        "optimal_order": optimal_order,
        "optimal_cost": optimal_cost,
        "worst_order": worst_order,
        "telemetry": telemetry
    }
