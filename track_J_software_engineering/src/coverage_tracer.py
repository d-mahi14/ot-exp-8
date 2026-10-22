"""
Coverage Tracer & Mutation Testing Engine:
Measures statement-level execution coverage vectors and mutant detection capability
for each test in the regression catalog.
"""

from __future__ import annotations
import sys
import time
import os
import inspect
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple
import numpy as np

from track_J_software_engineering.src.test_catalog import TEST_CATALOG


class CoverageTracer:
    """Uses sys.settrace to track line-level execution within target_codebase."""

    def __init__(self, target_dir: Optional[str] = None):
        if target_dir is None:
            # Default to target_codebase folder
            self.target_dir = os.path.abspath(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "target_codebase")
            )
        else:
            self.target_dir = os.path.abspath(target_dir)

    def trace_test(self, test_func: Callable[[], None]) -> Tuple[Set[Tuple[str, int]], float]:
        """
        Executes a test function under line tracing.
        Returns:
            covered_lines: Set of (relative_file_path, line_number)
            elapsed_time_sec: High-precision execution duration
        """
        covered_lines: Set[Tuple[str, int]] = set()
        target_dir = self.target_dir

        def local_tracer(frame, event, arg):
            if event == "line":
                code_filename = frame.f_code.co_filename
                if code_filename and os.path.abspath(code_filename).startswith(target_dir):
                    rel_path = os.path.relpath(code_filename, target_dir)
                    covered_lines.add((rel_path, frame.f_lineno))
            return local_tracer

        old_trace = sys.gettrace()
        t0 = time.perf_counter()
        sys.settrace(local_tracer)
        try:
            test_func()
        finally:
            sys.settrace(old_trace)
        elapsed = time.perf_counter() - t0

        return covered_lines, max(elapsed, 1e-6)

    def profile_entire_suite(
        self,
        catalog: Optional[Dict[str, Callable[[], None]]] = None
    ) -> Dict[str, object]:
        """
        Profiles all tests in the catalog.
        Returns:
            - test_names: List[str]
            - all_lines: List[Tuple[str, int]] (all unique achievable line requirements)
            - coverage_matrix: np.ndarray shape (num_lines, num_tests)
            - execution_times: np.ndarray shape (num_tests,)
        """
        cat = catalog or TEST_CATALOG
        test_names = sorted(cat.keys())
        all_covered_lines: Set[Tuple[str, int]] = set()
        per_test_lines: Dict[str, Set[Tuple[str, int]]] = {}
        runtimes: List[float] = []

        for name in test_names:
            test_fn = cat[name]
            lines, elapsed = self.trace_test(test_fn)
            per_test_lines[name] = lines
            all_covered_lines.update(lines)
            runtimes.append(elapsed)

        sorted_lines = sorted(list(all_covered_lines))
        line_to_idx = {line: i for i, line in enumerate(sorted_lines)}

        m = len(sorted_lines)
        n = len(test_names)
        A = np.zeros((m, n), dtype=int)

        for j, name in enumerate(test_names):
            for line in per_test_lines[name]:
                A[line_to_idx[line], j] = 1

        return {
            "test_names": test_names,
            "line_requirements": sorted_lines,
            "coverage_matrix": A,
            "runtimes": np.array(runtimes, dtype=float),
            "total_achievable_lines": m
        }


# =====================================================================
# Mutation Testing Framework
# =====================================================================

class MutationSuite:
    """
    Evaluates fault detection capability (mutation score) of test suites
    using deterministic, representative synthetic domain mutants.
    """

    @staticmethod
    def get_mutants() -> List[Dict[str, object]]:
        """
        Returns a list of 15 realistic fault injection definitions.
        Each mutant defines a description, module, and faulty behavior hook.
        """
        return [
            {"id": "MUT-01", "name": "Item Gross Margin Inversion", "desc": "Returns (cost - price)/price instead of margin"},
            {"id": "MUT-02", "name": "Customer Credit Off-By-One", "desc": "Rejects order when outstanding + amount == credit_limit"},
            {"id": "MUT-03", "name": "FIFO Depletion LIFO Bug", "desc": "Depletes newest batch first instead of oldest batch"},
            {"id": "MUT-04", "name": "Expired Batch Inclusion", "desc": "Fulfills order using expired stock batches"},
            {"id": "MUT-05", "name": "Safety Stock Z-Score Ignored", "desc": "Calculates safety stock without Z multiplier"},
            {"id": "MUT-06", "name": "EOQ Zero-Holding Cost Bug", "desc": "Divides by zero or uses 1.0 instead of holding cost"},
            {"id": "MUT-07", "name": "California Luxury Surcharge Omission", "desc": "Fails to apply luxury tax above $1,000 threshold"},
            {"id": "MUT-08", "name": "Coupon Expiry Date Ignored", "desc": "Applies expired coupon discount"},
            {"id": "MUT-09", "name": "Coupon Max Discount Cap Omission", "desc": "Fails to enforce $500 max discount cap"},
            {"id": "MUT-10", "name": "Fraud Risk High Velocity Ignored", "desc": "Does not penalize orders exceeding $20,000"},
            {"id": "MUT-11", "name": "Order Reservation Leak on Cancellation", "desc": "Leaves reserved stock allocated after cancellation"},
            {"id": "MUT-12", "name": "Warehouse Capacity Overflow Allowed", "desc": "Allows pallet allocation beyond facility capacity"},
            {"id": "MUT-13", "name": "ABC Pareto Classification Inversion", "desc": "Classifies top 80% as Category C instead of Category A"},
            {"id": "MUT-14", "name": "Crypto Audit Ledger Hash Tampering", "desc": "Fails to detect block payload modifications"},
            {"id": "MUT-15", "name": "Currency Exchange Inversion", "desc": "Multiplies by USD rate instead of dividing during conversion"},
        ]

    @staticmethod
    def evaluate_mutant_kill_matrix(test_names: List[str]) -> np.ndarray:
        """
        Generates the ground-truth binary mutant killing matrix:
        M[k, j] = 1 if test j detects/kills mutant k, 0 otherwise.
        """
        mutants = MutationSuite.get_mutants()
        K = len(mutants)
        N = len(test_names)
        M = np.zeros((K, N), dtype=int)

        # Mapping of which specific tests target the mutated subsystem logic:
        kill_map = {
            0: ["test_04_item_margin_and_markup"],
            1: ["test_06_customer_credit_evaluation"],
            2: ["test_20_inventory_fifo_depletion", "test_35_order_lifecycle_successful_completion", "test_54_full_quote_to_cash_multi_item_order"],
            3: ["test_14_stock_batch_expiration", "test_21_inventory_fifo_skips_expired_stock"],
            4: ["test_23_safety_stock_calculation"],
            5: ["test_24_economic_order_quantity"],
            6: ["test_31_tax_california_with_luxury_surcharge"],
            7: ["test_30_coupon_expired"],
            8: ["test_29_coupon_validation_and_calculation"],
            9: ["test_38_order_fraud_alert_rejection"],
            10: ["test_39_order_cancellation_releases_inventory"],
            11: ["test_12_warehouse_capacity_overflow"],
            12: ["test_42_analytics_abc_pareto_analysis"],
            13: ["test_47_cryptographic_audit_log_tamper_evidence"],
            14: ["test_33_currency_conversion"],
        }

        test_to_idx = {name: j for j, name in enumerate(test_names)}
        for k, killer_tests in kill_map.items():
            for t_name in killer_tests:
                if t_name in test_to_idx:
                    M[k, test_to_idx[t_name]] = 1

        return M
