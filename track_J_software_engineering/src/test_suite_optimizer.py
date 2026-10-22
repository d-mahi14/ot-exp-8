"""
Test Suite Minimization & Prioritization Optimizers:
Implements Chapter 6 Optimization Methods:
1. Exact 0-1 Integer Linear Programming (ILP) via HiGHS branch-and-cut (scipy.optimize.milp)
2. Greedy Set Cover Approximation with cost-benefit ratio (Chvátal's algorithm)
3. Budget-Constrained Submodular Knapsack Coverage (Pareto Frontier)
4. Naive Full Suite & Random Subsampling Baselines
"""

from __future__ import annotations
import time
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


class TestSuiteOptimizationResult:
    __test__ = False

    def __init__(
        self,
        method_name: str,
        selected_indices: List[int],
        selected_tests: List[str],
        total_tests_count: int,
        total_suite_runtime: float,
        covered_requirements_count: int,
        total_achievable_requirements: int,
        coverage_percentage: float,
        mutants_killed_count: int,
        total_mutants_count: int,
        mutation_score_pct: float,
        solver_time_sec: float,
        status: str = "OPTIMAL"
    ):
        self.method_name = method_name
        self.selected_indices = selected_indices
        self.selected_tests = selected_tests
        self.selected_count = len(selected_indices)
        self.total_tests_count = total_tests_count
        self.total_suite_runtime = total_suite_runtime
        self.covered_requirements_count = covered_requirements_count
        self.total_achievable_requirements = total_achievable_requirements
        self.coverage_percentage = coverage_percentage
        self.mutants_killed_count = mutants_killed_count
        self.total_mutants_count = total_mutants_count
        self.mutation_score_pct = mutation_score_pct
        self.solver_time_sec = solver_time_sec
        self.status = status

    @property
    def reduction_percentage(self) -> float:
        if self.total_tests_count == 0:
            return 0.0
        return ((self.total_tests_count - self.selected_count) / self.total_tests_count) * 100.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "method": self.method_name,
            "selected_count": self.selected_count,
            "total_count": self.total_tests_count,
            "reduction_pct": round(self.reduction_percentage, 2),
            "suite_runtime_sec": round(self.total_suite_runtime, 5),
            "covered_reqs": self.covered_requirements_count,
            "total_reqs": self.total_achievable_requirements,
            "coverage_pct": round(self.coverage_percentage, 2),
            "mutants_killed": self.mutants_killed_count,
            "total_mutants": self.total_mutants_count,
            "mutation_score_pct": round(self.mutation_score_pct, 2),
            "solver_time_sec": round(self.solver_time_sec, 5),
            "status": self.status
        }


# =====================================================================
# 1. Exact 0-1 Integer Linear Programming (ILP) Solver
# =====================================================================

class ILPTestSuiteOptimizer:
    """
    Formulates Test Suite Minimization as a 0-1 Minimum Weight Set Cover:
        min  c^T x
        s.t. A x >= 1   (for each achievable requirement i)
             x_j in {0, 1}
    Solved using HiGHS branch-and-cut via scipy.optimize.milp.
    """

    @staticmethod
    def solve(
        A: np.ndarray,
        costs: np.ndarray,
        test_names: List[str],
        mutant_matrix: Optional[np.ndarray] = None,
        enforce_mutation_coverage: bool = True
    ) -> TestSuiteOptimizationResult:
        m, n = A.shape
        t0 = time.perf_counter()

        if mutant_matrix is not None and enforce_mutation_coverage:
            full_req_matrix = np.vstack([A, mutant_matrix])
        else:
            full_req_matrix = A

        # Filter out unreachable requirements if any
        row_sums = full_req_matrix.sum(axis=1)
        reachable_rows = np.where(row_sums >= 1)[0]
        A_eff = full_req_matrix[reachable_rows, :]
        m_eff = len(reachable_rows)

        # Objective vector c
        c = costs.copy()

        # Constraints: A_eff @ x >= 1, <= infinity
        lhs = np.ones(m_eff, dtype=float)
        rhs = np.full(m_eff, np.inf, dtype=float)
        constraints = LinearConstraint(A_eff, lhs, rhs)

        # Integrality: all 1 (pure binary 0-1 decision variables)
        integrality = np.ones(n, dtype=int)
        # Bounds: 0 <= x_j <= 1
        bounds = Bounds(0, 1)

        res = milp(
            c=c,
            integrality=integrality,
            constraints=constraints,
            bounds=bounds
        )
        t_solver = time.perf_counter() - t0

        if not res.success:
            raise RuntimeError(f"ILP optimization failed: {res.status_message}")

        x_opt = np.round(res.x).astype(int)
        selected_idx = np.where(x_opt == 1)[0].tolist()
        selected_tests = [test_names[i] for i in selected_idx]

        # Calculate metrics
        selected_cost = float(costs[selected_idx].sum())
        covered_mask = (A[:, selected_idx].sum(axis=1) > 0)
        covered_count = int(covered_mask.sum())
        cov_pct = (covered_count / m) * 100.0 if m > 0 else 100.0

        mut_killed = 0
        total_mut = 0
        mut_pct = 0.0
        if mutant_matrix is not None:
            total_mut = mutant_matrix.shape[0]
            mut_killed = int((mutant_matrix[:, selected_idx].sum(axis=1) > 0).sum())
            mut_pct = (mut_killed / total_mut) * 100.0 if total_mut > 0 else 100.0

        return TestSuiteOptimizationResult(
            method_name="0-1 ILP (HiGHS Branch-and-Cut)",
            selected_indices=selected_idx,
            selected_tests=selected_tests,
            total_tests_count=n,
            total_suite_runtime=selected_cost,
            covered_requirements_count=covered_count,
            total_achievable_requirements=m,
            coverage_percentage=cov_pct,
            mutants_killed_count=mut_killed,
            total_mutants_count=total_mut,
            mutation_score_pct=mut_pct,
            solver_time_sec=t_solver,
            status="OPTIMAL"
        )


# =====================================================================
# 2. Greedy Set Cover Approximation (Chvátal's Algorithm)
# =====================================================================

class GreedySetCoverOptimizer:
    """
    Chapter 6 Greedy Heuristic:
    Iteratively selects test j* maximizing marginal coverage gain per unit cost:
        j* = argmax_{j} [ Delta_Cov(j) / c_j ]
    Achieves provable (1 - 1/e) / O(ln m) approximation ratio for Set Cover.
    """

    @staticmethod
    def solve(
        A: np.ndarray,
        costs: np.ndarray,
        test_names: List[str],
        mutant_matrix: Optional[np.ndarray] = None,
        enforce_mutation_coverage: bool = True
    ) -> TestSuiteOptimizationResult:
        m, n = A.shape
        t0 = time.perf_counter()

        if mutant_matrix is not None and enforce_mutation_coverage:
            full_req_matrix = np.vstack([A, mutant_matrix])
        else:
            full_req_matrix = A

        reachable_rows = set(np.where(full_req_matrix.sum(axis=1) >= 1)[0])
        uncovered = set(reachable_rows)
        selected_idx: List[int] = []

        # Precompute requirement sets for each test
        test_covers = [set(np.where(full_req_matrix[:, j] == 1)[0]) for j in range(n)]

        while uncovered:
            best_j = -1
            best_ratio = -1.0
            best_gain = 0

            for j in range(n):
                if j in selected_idx:
                    continue
                newly_covered = len(test_covers[j].intersection(uncovered))
                if newly_covered == 0:
                    continue
                cost = max(costs[j], 1e-6)
                ratio = newly_covered / cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_j = j
                    best_gain = newly_covered

            if best_j == -1 or best_gain == 0:
                # No remaining test covers anything new
                break

            selected_idx.append(best_j)
            uncovered.difference_update(test_covers[best_j])

        t_solver = time.perf_counter() - t0

        selected_tests = [test_names[i] for i in selected_idx]
        selected_cost = float(costs[selected_idx].sum())
        covered_mask = (A[:, selected_idx].sum(axis=1) > 0)
        covered_count = int(covered_mask.sum())
        cov_pct = (covered_count / m) * 100.0 if m > 0 else 100.0

        mut_killed = 0
        total_mut = 0
        mut_pct = 0.0
        if mutant_matrix is not None:
            total_mut = mutant_matrix.shape[0]
            mut_killed = int((mutant_matrix[:, selected_idx].sum(axis=1) > 0).sum())
            mut_pct = (mut_killed / total_mut) * 100.0 if total_mut > 0 else 100.0

        return TestSuiteOptimizationResult(
            method_name="Greedy Set Cover (Chvátal)",
            selected_indices=selected_idx,
            selected_tests=selected_tests,
            total_tests_count=n,
            total_suite_runtime=selected_cost,
            covered_requirements_count=covered_count,
            total_achievable_requirements=m,
            coverage_percentage=cov_pct,
            mutants_killed_count=mut_killed,
            total_mutants_count=total_mut,
            mutation_score_pct=mut_pct,
            solver_time_sec=t_solver,
            status="HEURISTIC_SUCCESS"
        )


# =====================================================================
# 3. Budget-Constrained Submodular Knapsack Optimizer (Pareto Frontier)
# =====================================================================

class BudgetedCoverageOptimizer:
    """
    Selects a subset S of tests within a strict CI execution time budget B
    to maximize covered requirements (Submodular Knapsack).
    """

    @staticmethod
    def solve(
        A: np.ndarray,
        costs: np.ndarray,
        test_names: List[str],
        budget_sec: float,
        mutant_matrix: Optional[np.ndarray] = None
    ) -> TestSuiteOptimizationResult:
        m, n = A.shape
        t0 = time.perf_counter()

        test_covers = [set(np.where(A[:, j] == 1)[0]) for j in range(n)]
        selected_idx: List[int] = []
        covered_so_far: Set[int] = set()
        accumulated_cost = 0.0

        remaining_candidates = set(range(n))

        while remaining_candidates:
            best_j = -1
            best_density = -1.0

            for j in remaining_candidates:
                if accumulated_cost + costs[j] > budget_sec:
                    continue
                new_gain = len(test_covers[j].difference(covered_so_far))
                cost = max(costs[j], 1e-6)
                density = new_gain / cost
                if density > best_density:
                    best_density = density
                    best_j = j

            if best_j == -1:
                break

            selected_idx.append(best_j)
            remaining_candidates.remove(best_j)
            covered_so_far.update(test_covers[best_j])
            accumulated_cost += costs[best_j]

        t_solver = time.perf_counter() - t0

        selected_tests = [test_names[i] for i in selected_idx]
        cov_pct = (len(covered_so_far) / m) * 100.0 if m > 0 else 100.0

        mut_killed = 0
        total_mut = 0
        mut_pct = 0.0
        if mutant_matrix is not None:
            total_mut = mutant_matrix.shape[0]
            if selected_idx:
                mut_killed = int((mutant_matrix[:, selected_idx].sum(axis=1) > 0).sum())
            mut_pct = (mut_killed / total_mut) * 100.0 if total_mut > 0 else 100.0

        return TestSuiteOptimizationResult(
            method_name=f"Budgeted Greedy (Budget={budget_sec*1000:.1f}ms)",
            selected_indices=selected_idx,
            selected_tests=selected_tests,
            total_tests_count=n,
            total_suite_runtime=accumulated_cost,
            covered_requirements_count=len(covered_so_far),
            total_achievable_requirements=m,
            coverage_percentage=cov_pct,
            mutants_killed_count=mut_killed,
            total_mutants_count=total_mut,
            mutation_score_pct=mut_pct,
            solver_time_sec=t_solver,
            status="BUDGET_EXHAUSTED"
        )


# =====================================================================
# 4. Naive Full Suite & Random Baselines
# =====================================================================

class BaselineEvaluator:
    @staticmethod
    def evaluate_full_suite(
        A: np.ndarray,
        costs: np.ndarray,
        test_names: List[str],
        mutant_matrix: Optional[np.ndarray] = None
    ) -> TestSuiteOptimizationResult:
        m, n = A.shape
        selected_idx = list(range(n))
        total_cost = float(costs.sum())
        covered_count = int((A.sum(axis=1) > 0).sum())
        cov_pct = (covered_count / m) * 100.0 if m > 0 else 100.0

        mut_killed = 0
        total_mut = 0
        mut_pct = 0.0
        if mutant_matrix is not None:
            total_mut = mutant_matrix.shape[0]
            mut_killed = int((mutant_matrix.sum(axis=1) > 0).sum())
            mut_pct = (mut_killed / total_mut) * 100.0 if total_mut > 0 else 100.0

        return TestSuiteOptimizationResult(
            method_name="Exhaustive Full Suite (Naive Baseline)",
            selected_indices=selected_idx,
            selected_tests=test_names,
            total_tests_count=n,
            total_suite_runtime=total_cost,
            covered_requirements_count=covered_count,
            total_achievable_requirements=m,
            coverage_percentage=cov_pct,
            mutants_killed_count=mut_killed,
            total_mutants_count=total_mut,
            mutation_score_pct=mut_pct,
            solver_time_sec=0.0,
            status="BASELINE"
        )

    @staticmethod
    def evaluate_random_subsampling(
        A: np.ndarray,
        costs: np.ndarray,
        test_names: List[str],
        sample_size: int,
        seed: int = 42,
        mutant_matrix: Optional[np.ndarray] = None
    ) -> TestSuiteOptimizationResult:
        m, n = A.shape
        rng = np.random.default_rng(seed)
        selected_idx = sorted(rng.choice(n, size=min(sample_size, n), replace=False).tolist())
        selected_cost = float(costs[selected_idx].sum())
        covered_mask = (A[:, selected_idx].sum(axis=1) > 0)
        covered_count = int(covered_mask.sum())
        cov_pct = (covered_count / m) * 100.0 if m > 0 else 100.0

        mut_killed = 0
        total_mut = 0
        mut_pct = 0.0
        if mutant_matrix is not None:
            total_mut = mutant_matrix.shape[0]
            mut_killed = int((mutant_matrix[:, selected_idx].sum(axis=1) > 0).sum())
            mut_pct = (mut_killed / total_mut) * 100.0 if total_mut > 0 else 100.0

        return TestSuiteOptimizationResult(
            method_name=f"Random Subsampling (k={sample_size})",
            selected_indices=selected_idx,
            selected_tests=[test_names[i] for i in selected_idx],
            total_tests_count=n,
            total_suite_runtime=selected_cost,
            covered_requirements_count=covered_count,
            total_achievable_requirements=m,
            coverage_percentage=cov_pct,
            mutants_killed_count=mut_killed,
            total_mutants_count=total_mut,
            mutation_score_pct=mut_pct,
            solver_time_sec=0.0,
            status="STOCHASTIC_BASELINE"
        )
