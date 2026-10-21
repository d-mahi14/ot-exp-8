"""Unit tests for Submodular Maximization module in Track D."""

import numpy as np
import pytest

from track_D_submodular_parallel.src.submodular_maximization import (
    SensorPlacementProblem,
    solve_submodular_brute_force,
    solve_submodular_greedy,
    solve_submodular_lazy_greedy,
    solve_submodular_random_baseline,
    solve_submodular_topk_isolated_baseline,
    verify_submodularity,
)


@pytest.fixture
def small_sensor_problem() -> SensorPlacementProblem:
    """Fixture providing a small 12-candidate, 25-target 2D coverage instance."""
    rng = np.random.default_rng(123)
    targets = rng.uniform(0.0, 1.0, size=(25, 2))
    candidates = rng.uniform(0.0, 1.0, size=(12, 2))
    target_weights = np.ones(25)
    sensor_costs = np.ones(12)  # Unit cost
    return SensorPlacementProblem(
        targets=targets,
        candidates=candidates,
        target_weights=target_weights,
        sensor_costs=sensor_costs,
        decay_sigma=0.3,
        r_max=0.6,
    )


def test_submodularity_verification(small_sensor_problem: SensorPlacementProblem) -> None:
    """Verify that the sensor coverage function satisfies the diminishing returns definition."""
    def obj_fn(s_indices):
        return small_sensor_problem.evaluate(s_indices)

    is_submod, min_slack = verify_submodularity(
        obj_fn,
        ground_set=list(range(small_sensor_problem.num_candidates)),
        num_samples=60,
        seed=42,
    )
    assert is_submod is True
    assert min_slack >= -1e-9


def test_empty_set_evaluation(small_sensor_problem: SensorPlacementProblem) -> None:
    """Evaluate empty set returns 0."""
    assert small_sensor_problem.evaluate([]) == 0.0


def test_monotonicity(small_sensor_problem: SensorPlacementProblem) -> None:
    """Adding an element must not decrease the objective value."""
    s1 = [0, 2]
    s2 = [0, 2, 5]
    val1 = small_sensor_problem.evaluate(s1)
    val2 = small_sensor_problem.evaluate(s2)
    assert val2 >= val1 - 1e-12


def test_greedy_vs_brute_force_approximation_ratio(small_sensor_problem: SensorPlacementProblem) -> None:
    """Verify Greedy achieves at least (1 - 1/e) approx 0.6321 of exact brute-force optimum."""
    k = 4
    sol_exact = solve_submodular_brute_force(small_sensor_problem, k=k)
    sol_greedy = solve_submodular_greedy(small_sensor_problem, k=k)

    assert sol_exact.objective_value > 0.0
    ratio = sol_greedy.objective_value / sol_exact.objective_value
    # Theoretical guarantee under monotone submodular objective:
    one_minus_one_over_e = 1.0 - 1.0 / np.e  # ~0.63212
    assert ratio >= one_minus_one_over_e - 1e-9, f"Greedy ratio {ratio} fell below (1 - 1/e)"
    # In practice greedy is often >= 90%
    assert ratio >= 0.80


def test_lazy_greedy_matches_standard_greedy_and_saves_evaluations(small_sensor_problem: SensorPlacementProblem) -> None:
    """Verify Minoux lazy greedy reaches the exact same objective while executing fewer evaluations."""
    k = 4
    sol_greedy = solve_submodular_greedy(small_sensor_problem, k=k)
    sol_lazy = solve_submodular_lazy_greedy(small_sensor_problem, k=k)

    # Values must match closely (or be identical)
    assert np.isclose(sol_lazy.objective_value, sol_greedy.objective_value, atol=1e-7)
    assert sol_lazy.selected_set == sol_greedy.selected_set
    # Lazy greedy evaluation count should be <= standard greedy
    assert sol_lazy.evaluations_count <= sol_greedy.evaluations_count


def test_budgeted_greedy_constraint() -> None:
    """Verify sensor selection strictly respects a non-uniform knapsack budget."""
    rng = np.random.default_rng(99)
    targets = rng.uniform(0.0, 1.0, size=(20, 2))
    candidates = rng.uniform(0.0, 1.0, size=(10, 2))
    target_weights = np.ones(20)
    # Non-uniform costs between 2.0 and 8.0
    costs = np.array([2.0, 3.0, 5.0, 7.0, 4.0, 2.5, 6.0, 8.0, 3.5, 4.5])
    budget = 10.0

    problem = SensorPlacementProblem(
        targets=targets,
        candidates=candidates,
        target_weights=target_weights,
        sensor_costs=costs,
    )

    sol_greedy = solve_submodular_greedy(problem, k=5, max_budget=budget)
    assert sol_greedy.total_cost <= budget + 1e-9

    sol_exact = solve_submodular_brute_force(problem, k=5, max_budget=budget)
    assert sol_exact.total_cost <= budget + 1e-9


def test_greedy_outperforms_baselines(small_sensor_problem: SensorPlacementProblem) -> None:
    """Verify standard greedy outperforms single random selection and top-k isolated baseline."""
    k = 3
    sol_greedy = solve_submodular_greedy(small_sensor_problem, k=k)
    sol_random = solve_submodular_random_baseline(small_sensor_problem, k=k, num_trials=1, seed=42)
    sol_topk = solve_submodular_topk_isolated_baseline(small_sensor_problem, k=k)

    # Greedy should strictly outperform a single random baseline sample
    assert sol_greedy.objective_value >= sol_random.objective_value
    # Top-k isolated picks top 3 individual sensors ignoring spatial overlap
    assert sol_greedy.objective_value >= sol_topk.objective_value
