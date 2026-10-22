"""
Automated Pytest Suite for Track J: Software-Engineering Optimization.
Verifies target codebase integrity, coverage tracing, 0-1 ILP mathematical optimality,
Greedy approximation ratio, mutation fault preservation, and artifact existence.
"""

import os
import pytest
import numpy as np

from track_J_software_engineering.src.test_catalog import TEST_CATALOG
from track_J_software_engineering.src.coverage_tracer import CoverageTracer, MutationSuite
from track_J_software_engineering.src.test_suite_optimizer import (
    ILPTestSuiteOptimizer,
    GreedySetCoverOptimizer,
    BudgetedCoverageOptimizer,
    BaselineEvaluator
)
from track_J_software_engineering.src.run_track_j_benchmarks import run_all_benchmarks


@pytest.fixture(scope="module")
def profile_fixture():
    tracer = CoverageTracer()
    data = tracer.profile_entire_suite(TEST_CATALOG)
    mut_matrix = MutationSuite.evaluate_mutant_kill_matrix(data["test_names"])
    return {
        "A": data["coverage_matrix"],
        "costs": data["runtimes"],
        "test_names": data["test_names"],
        "mut_matrix": mut_matrix,
        "m_lines": data["total_achievable_lines"]
    }


def test_target_codebase_all_catalog_tests_pass():
    """Verifies that all 55 regression test cases pass cleanly without error."""
    assert len(TEST_CATALOG) >= 50
    for name, test_fn in TEST_CATALOG.items():
        test_fn()  # Should execute and pass all assertions


def test_coverage_profiling_matrix_structure(profile_fixture):
    """Verifies the binary coverage matrix properties and dimension sanity."""
    A = profile_fixture["A"]
    costs = profile_fixture["costs"]
    m, n = A.shape
    assert m > 100, f"Expected >100 line requirements traced, got {m}"
    assert n == len(TEST_CATALOG)
    # Check that each test covers at least one line
    assert (A.sum(axis=0) > 0).all()
    # Check that each tracked line is covered by at least one test
    assert (A.sum(axis=1) >= 1).all()
    # Costs must be strictly positive
    assert (costs > 0).all()


def test_mutation_suite_kill_matrix(profile_fixture):
    """Verifies mutation testing fault matrix has 100% kill rate under full suite."""
    M = profile_fixture["mut_matrix"]
    k, n = M.shape
    assert k == 15, "Expected 15 synthetic mutants"
    # Verify that every mutant is killed by at least one test in the full catalog
    killed_per_mutant = M.sum(axis=1)
    assert (killed_per_mutant >= 1).all(), "Every mutant must be caught by the exhaustive test suite"


def test_ilp_solver_optimality_and_full_coverage(profile_fixture):
    """
    Verifies that 0-1 ILP (HiGHS) achieves 100% statement coverage
    and 100% mutation score while minimizing test count and runtime.
    """
    A = profile_fixture["A"]
    costs = profile_fixture["costs"]
    names = profile_fixture["test_names"]
    M = profile_fixture["mut_matrix"]

    res_ilp = ILPTestSuiteOptimizer.solve(A, costs, names, M)

    assert res_ilp.status == "OPTIMAL"
    assert res_ilp.coverage_percentage == 100.0, "ILP must retain 100% statement coverage"
    assert res_ilp.mutation_score_pct == 100.0, "ILP must retain 100% mutation score"
    assert res_ilp.selected_count < res_ilp.total_tests_count
    assert res_ilp.reduction_percentage >= 25.0, f"Expected >=25% reduction, got {res_ilp.reduction_percentage:.1f}%"


def test_greedy_solver_full_coverage_and_heuristic_quality(profile_fixture):
    """
    Verifies that Chvátal's Greedy heuristic achieves 100% statement coverage
    and 100% mutation score with low solver overhead.
    """
    A = profile_fixture["A"]
    costs = profile_fixture["costs"]
    names = profile_fixture["test_names"]
    M = profile_fixture["mut_matrix"]

    res_greedy = GreedySetCoverOptimizer.solve(A, costs, names, M)

    assert res_greedy.coverage_percentage == 100.0
    assert res_greedy.mutation_score_pct == 100.0
    assert res_greedy.reduction_percentage >= 20.0
    assert res_greedy.solver_time_sec < 0.10, "Greedy heuristic should be near instantaneous"


def test_ilp_cost_less_than_or_equal_to_greedy(profile_fixture):
    """
    Verifies the theoretical property that the exact 0-1 ILP minimum cost
    is less than or equal to the Greedy heuristic cost: c(ILP) <= c(Greedy).
    """
    A = profile_fixture["A"]
    costs = profile_fixture["costs"]
    names = profile_fixture["test_names"]

    res_ilp = ILPTestSuiteOptimizer.solve(A, costs, names)
    res_greedy = GreedySetCoverOptimizer.solve(A, costs, names)

    # Allow tiny floating-point margin (1e-6)
    assert res_ilp.total_suite_runtime <= res_greedy.total_suite_runtime + 1e-6, (
        f"ILP cost ({res_ilp.total_suite_runtime:.6f}) must be <= Greedy cost ({res_greedy.total_suite_runtime:.6f})"
    )


def test_budgeted_greedy_respects_time_budget(profile_fixture):
    """Verifies that the budgeted submodular knapsack optimizer strictly respects the runtime budget."""
    A = profile_fixture["A"]
    costs = profile_fixture["costs"]
    names = profile_fixture["test_names"]

    total_cost = costs.sum()
    budget = total_cost * 0.40  # 40% budget

    res_budget = BudgetedCoverageOptimizer.solve(A, costs, names, budget_sec=budget)

    assert res_budget.total_suite_runtime <= budget + 1e-7, (
        f"Selected runtime {res_budget.total_suite_runtime} exceeded budget {budget}"
    )
    assert res_budget.coverage_percentage > 50.0


def test_random_subsampling_demonstrates_inferiority(profile_fixture):
    """
    Verifies that random subsampling with the same test count as ILP
    fails to achieve 100% coverage or 100% mutation score.
    """
    A = profile_fixture["A"]
    costs = profile_fixture["costs"]
    names = profile_fixture["test_names"]
    M = profile_fixture["mut_matrix"]

    res_ilp = ILPTestSuiteOptimizer.solve(A, costs, names, M)
    res_rand = BaselineEvaluator.evaluate_random_subsampling(
        A, costs, names, sample_size=res_ilp.selected_count, seed=42, mutant_matrix=M
    )

    # Random selection with small k should not reach 100% on both coverage and mutants
    assert (res_rand.coverage_percentage < 100.0) or (res_rand.mutation_score_pct < 100.0), (
        "Random subsampling should be demonstrably inferior to mathematically guided selection"
    )


def test_benchmark_artifacts_generation_and_integrity():
    """Runs the benchmark runner and verifies CSV, MD, and PNG artifacts exist and have non-zero size."""
    run_all_benchmarks()

    base_results = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    csv_file = os.path.join(base_results, "track_j_metrics.csv")
    md_file = os.path.join(base_results, "track_j_metrics.md")
    png_file = os.path.join(base_results, "track_j_benchmarks.png")

    assert os.path.exists(csv_file) and os.path.getsize(csv_file) > 100
    assert os.path.exists(md_file) and os.path.getsize(md_file) > 200
    assert os.path.exists(png_file) and os.path.getsize(png_file) > 1000
