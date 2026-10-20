"""
test_knapsack.py
Independent unit tests for 0-1 Knapsack combinatorial problem:
1. Exact textbook instances with verified analytical solutions
2. Adversarial counterexamples proving Greedy sub-optimality
3. Edge cases: zero capacity, items heavier than capacity, empty item list, single-item exact fit
4. Validation constraints and backtracking consistency
"""

import pytest
import numpy as np

from track_B_discrete.src.knapsack_solvers import (
    KnapsackProblem, KnapsackSolution, solve_knapsack_greedy, solve_knapsack_dp,
    solve_knapsack_branch_and_bound, solve_knapsack_ilp
)


class TestKnapsackBasics:
    """Verifies edge cases, parameter validation, and feasibility."""

    def test_empty_problem(self):
        problem = KnapsackProblem(values=[], weights=[], capacity=10)
        sol_greedy = solve_knapsack_greedy(problem)
        sol_dp = solve_knapsack_dp(problem)

        assert sol_greedy.total_value == 0.0
        assert sol_greedy.selected_items == []
        assert sol_dp.total_value == 0.0
        assert sol_dp.selected_items == []

    def test_zero_capacity(self):
        problem = KnapsackProblem(values=[10, 20, 30], weights=[5, 10, 15], capacity=0)
        sol_greedy = solve_knapsack_greedy(problem)
        sol_dp = solve_knapsack_dp(problem)

        assert sol_greedy.total_value == 0.0
        assert sol_greedy.total_weight == 0
        assert sol_dp.total_value == 0.0
        assert sol_dp.total_weight == 0

    def test_items_heavier_than_capacity(self):
        problem = KnapsackProblem(values=[100, 200], weights=[50, 60], capacity=30)
        sol_greedy = solve_knapsack_greedy(problem)
        sol_dp = solve_knapsack_dp(problem)

        assert sol_greedy.total_value == 0.0
        assert sol_greedy.selected_items == []
        assert sol_dp.total_value == 0.0
        assert sol_dp.selected_items == []

    def test_all_items_fit(self):
        problem = KnapsackProblem(values=[10, 20, 30], weights=[2, 3, 4], capacity=20)
        sol_greedy = solve_knapsack_greedy(problem)
        sol_dp = solve_knapsack_dp(problem)

        assert sol_greedy.total_value == 60.0
        assert sol_greedy.total_weight == 9
        assert sol_greedy.selected_items == [0, 1, 2]
        assert sol_dp.total_value == 60.0
        assert sol_dp.total_weight == 9
        assert sol_dp.selected_items == [0, 1, 2]

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            KnapsackProblem(values=[10], weights=[5], capacity=-1)
        with pytest.raises(ValueError):
            KnapsackProblem(values=[10, 20], weights=[5], capacity=10)
        with pytest.raises(ValueError):
            KnapsackProblem(values=[-5], weights=[10], capacity=10)
        with pytest.raises(ValueError):
            KnapsackProblem(values=[5], weights=[-10], capacity=10)


class TestKnapsackAlgorithms:
    """Verifies textbook accuracy and algorithm comparison."""

    def test_classic_textbook_instance(self):
        """
        Classic 4-item instance:
            Values:  [3, 4, 5, 6]
            Weights: [2, 3, 4, 5]
            Capacity: 5
        Optimal solution: items 0 and 1 (weights 2+3=5, value 3+4=7).
        """
        problem = KnapsackProblem(values=[3, 4, 5, 6], weights=[2, 3, 4, 5], capacity=5)
        sol_dp = solve_knapsack_dp(problem)

        assert sol_dp.total_value == 7.0
        assert sol_dp.total_weight == 5
        assert sol_dp.selected_items == [0, 1]

        # Verify evaluation helper
        feasible, w, v = problem.evaluate_solution(sol_dp.selected_items)
        assert feasible is True
        assert w == 5
        assert v == 7.0

    def test_adversarial_counterexample_greedy_vs_dp(self):
        """
        Demonstrates that pure Greedy fails to achieve the optimal solution:
            Capacity W = 100
            Item 0: w=1, v=2    (density = 2.0)
            Item 1: w=100, v=150 (density = 1.5)
        Pure greedy takes item 0 (value 2), leaving 99 capacity unused.
        DP takes item 1 (value 150).
        """
        problem = KnapsackProblem(values=[2.0, 150.0], weights=[1, 100], capacity=100)
        
        # Pure greedy without modified check
        sol_pure_greedy = solve_knapsack_greedy(problem, use_modified_greedy=False)
        sol_dp = solve_knapsack_dp(problem)

        assert sol_pure_greedy.selected_items == [0]
        assert sol_pure_greedy.total_value == 2.0
        assert sol_dp.selected_items == [1]
        assert sol_dp.total_value == 150.0

        # Modified greedy selects item 1
        sol_mod_greedy = solve_knapsack_greedy(problem, use_modified_greedy=True)
        assert sol_mod_greedy.selected_items == [1]
        assert sol_mod_greedy.total_value == 150.0

    def test_backtracking_exactness(self):
        """Random instance: verify DP backtracked item sum equals reported totals."""
        rng = np.random.RandomState(42)
        values = list(rng.randint(10, 100, size=15))
        weights = list(rng.randint(5, 30, size=15))
        capacity = 80
        problem = KnapsackProblem(values, weights, capacity)

        sol_dp = solve_knapsack_dp(problem)
        feasible, total_wt, total_val = problem.evaluate_solution(sol_dp.selected_items)

        assert feasible is True
        assert total_wt == sol_dp.total_weight
        assert total_val == sol_dp.total_value
        assert total_wt <= capacity


class TestBranchAndBoundAndILP:
    """Independent unit tests for Branch-and-Bound and ILP (HiGHS) solvers."""

    def test_bb_and_ilp_classic_instance(self):
        problem = KnapsackProblem(values=[3, 4, 5, 6], weights=[2, 3, 4, 5], capacity=5)
        sol_bb = solve_knapsack_branch_and_bound(problem)
        sol_ilp = solve_knapsack_ilp(problem)

        assert sol_bb.total_value == 7.0
        assert sol_bb.total_weight == 5
        assert sol_bb.selected_items == [0, 1]
        assert sol_bb.nodes_explored >= 1

        assert sol_ilp.total_value == 7.0
        assert sol_ilp.total_weight == 5
        assert sol_ilp.selected_items == [0, 1]

    def test_bb_and_ilp_adversarial_counterexample(self):
        problem = KnapsackProblem(values=[2.0, 150.0], weights=[1, 100], capacity=100)
        sol_bb = solve_knapsack_branch_and_bound(problem)
        sol_ilp = solve_knapsack_ilp(problem)

        assert sol_bb.total_value == 150.0
        assert sol_bb.selected_items == [1]

        assert sol_ilp.total_value == 150.0
        assert sol_ilp.selected_items == [1]

    def test_bb_and_ilp_edge_cases(self):
        # Zero capacity
        p_zero = KnapsackProblem(values=[10, 20], weights=[5, 10], capacity=0)
        assert solve_knapsack_branch_and_bound(p_zero).total_value == 0.0
        assert solve_knapsack_ilp(p_zero).total_value == 0.0

        # Empty problem
        p_empty = KnapsackProblem(values=[], weights=[], capacity=10)
        assert solve_knapsack_branch_and_bound(p_empty).total_value == 0.0
        assert solve_knapsack_ilp(p_empty).total_value == 0.0

        # Oversized items
        p_over = KnapsackProblem(values=[100], weights=[50], capacity=20)
        assert solve_knapsack_branch_and_bound(p_over).total_value == 0.0
        assert solve_knapsack_ilp(p_over).total_value == 0.0

    def test_all_four_solvers_agreement_on_random_instances(self):
        """
        Verify that DP, Branch-and-Bound, and ILP reach 100% identical exact optimal values,
        and that Greedy produces a feasible lower bound.
        """
        rng = np.random.RandomState(101)
        for trial in range(5):
            n = 12
            weights = list(rng.randint(3, 25, size=n))
            values = [w + rng.randint(1, 10) for w in weights]
            capacity = int(0.5 * sum(weights))
            problem = KnapsackProblem(values, weights, capacity)

            sol_greedy = solve_knapsack_greedy(problem)
            sol_dp = solve_knapsack_dp(problem)
            sol_bb = solve_knapsack_branch_and_bound(problem)
            sol_ilp = solve_knapsack_ilp(problem)

            # Feasibility
            for sol in (sol_greedy, sol_dp, sol_bb, sol_ilp):
                feasible, w, v = problem.evaluate_solution(sol.selected_items)
                assert feasible is True
                assert w == sol.total_weight
                assert np.isclose(v, sol.total_value)

            # Exact solvers must match 100%
            np.testing.assert_allclose(sol_dp.total_value, sol_bb.total_value, rtol=1e-5)
            np.testing.assert_allclose(sol_dp.total_value, sol_ilp.total_value, rtol=1e-5)
            # Greedy lower bound
            assert sol_greedy.total_value <= sol_dp.total_value + 1e-6
