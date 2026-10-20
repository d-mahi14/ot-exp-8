"""
knapsack_solvers.py
Formulation and Solvers for the 0-1 Knapsack Combinatorial Optimization Problem.

Problem Formulation:
    Maximize:   sum_{i=1}^N v_i * x_i
    Subject to: sum_{i=1}^N w_i * x_i <= W
                x_i in {0, 1} for all i in {1, ..., N}

Algorithms:
1. Greedy Density Heuristic (selection by value-to-weight ratio v_i / w_i)
2. Exact Dynamic Programming (pseudo-polynomial time O(N * W) with backtracking)
"""

import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class KnapsackSolution:
    """Stores the evaluation metrics of a knapsack solver run."""
    selected_items: List[int]
    total_value: float
    total_weight: int
    runtime_ms: float
    is_optimal: Optional[bool] = None
    method_name: str = ""


class KnapsackProblem:
    """
    0-1 Knapsack Problem Instance:
        weights: integer array of item weights (w_i >= 0)
        values: array of item values (v_i >= 0)
        capacity: knapsack capacity (W >= 0)
    """
    def __init__(self, values: List[float], weights: List[int], capacity: int):
        if capacity < 0:
            raise ValueError(f"Capacity must be non-negative, got {capacity}")
        if len(values) != len(weights):
            raise ValueError(f"Mismatched values ({len(values)}) and weights ({len(weights)}) length")

        self.values = [float(v) for v in values]
        self.weights = [int(w) for w in weights]
        self.capacity = int(capacity)
        self.n_items = len(values)

        # Validate non-negativity
        for i, (v, w) in enumerate(zip(self.values, self.weights)):
            if v < 0:
                raise ValueError(f"Item {i} has negative value: {v}")
            if w < 0:
                raise ValueError(f"Item {i} has negative weight: {w}")

    def evaluate_solution(self, selected_indices: List[int]) -> Tuple[bool, int, float]:
        """
        Evaluates a candidate solution set.
        Returns:
            is_feasible (bool): True if total_weight <= capacity.
            total_weight (int): Sum of weights of selected items.
            total_value (float): Sum of values of selected items.
        """
        tot_w = 0
        tot_v = 0.0
        for idx in selected_indices:
            if idx < 0 or idx >= self.n_items:
                raise IndexError(f"Item index {idx} out of range [0, {self.n_items-1}]")
            tot_w += self.weights[idx]
            tot_v += self.values[idx]

        is_feasible = (tot_w <= self.capacity)
        return is_feasible, tot_w, tot_v


def solve_knapsack_greedy(problem: KnapsackProblem, use_modified_greedy: bool = True) -> KnapsackSolution:
    """
    Greedy Heuristic for 0-1 Knapsack:
    
    Selection Rule:
        Computes the value density (efficiency ratio) r_i = v_i / w_i.
        Sorts items in descending order of density.
        Greedily adds items into the knapsack in order of highest density as long
        as the item's weight does not exceed remaining capacity.
        
    Theoretical Limitations:
        - Unlike the continuous (fractional) knapsack problem where greedy is optimal,
          0-1 knapsack can leave empty capacity that cannot be filled by smaller high-value items.
        - Adversarial counterexample: Capacity W = 100. Item 1: w=1, v=2 (ratio 2.0).
          Item 2: w=100, v=150 (ratio 1.5). Greedy selects Item 1 (value 2), leaving
          99 capacity unused, missing optimal Item 2 (value 150), yielding arbitrarily bad ratio!
        - Modified Greedy: Taking max(Greedy_Selection, Single_Best_Item) guarantees a 2-approximation
          (value >= 0.5 * OPT).
    """
    t0 = time.perf_counter()
    
    if problem.n_items == 0 or problem.capacity == 0:
        return KnapsackSolution(
            selected_items=[], total_value=0.0, total_weight=0,
            runtime_ms=(time.perf_counter() - t0) * 1000.0, method_name="Greedy"
        )

    # Compute density ratios
    items = []
    for i in range(problem.n_items):
        w = problem.weights[i]
        v = problem.values[i]
        ratio = (v / w) if w > 0 else float("inf")
        items.append((ratio, v, w, i))

    # Sort descending by ratio; break ties with higher value
    items.sort(key=lambda x: (x[0], x[1]), reverse=True)

    rem_cap = problem.capacity
    greedy_selection = []
    greedy_val = 0.0
    greedy_wt = 0

    for ratio, v, w, orig_idx in items:
        if w <= rem_cap:
            greedy_selection.append(orig_idx)
            greedy_val += v
            greedy_wt += w
            rem_cap -= w

    # Standard Modified Greedy: check if single best feasible item beats greedy combination
    best_single_idx = None
    best_single_val = -1.0
    for i in range(problem.n_items):
        if problem.weights[i] <= problem.capacity and problem.values[i] > best_single_val:
            best_single_val = problem.values[i]
            best_single_idx = i

    if use_modified_greedy and best_single_idx is not None and best_single_val > greedy_val:
        final_selection = [best_single_idx]
        final_val = best_single_val
        final_wt = problem.weights[best_single_idx]
    else:
        final_selection = sorted(greedy_selection)
        final_val = greedy_val
        final_wt = greedy_wt

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    return KnapsackSolution(
        selected_items=final_selection,
        total_value=final_val,
        total_weight=final_wt,
        runtime_ms=runtime_ms,
        method_name="Greedy"
    )


def solve_knapsack_dp(problem: KnapsackProblem) -> KnapsackSolution:
    """
    Exact Dynamic Programming Solver for 0-1 Knapsack:
    
    Recurrence Relation:
        Let V[i, w] denote the maximum value achievable considering a subset
        of the first i items (1 <= i <= N) with maximum capacity w (0 <= w <= W).
        
        V[i, w] = V[i-1, w]                                          if w_i > w
        V[i, w] = max(V[i-1, w], V[i-1, w - w_i] + v_i)             if w_i <= w
        
        Base Cases:
        V[0, w] = 0 for all w in {0, ..., W}
        V[i, 0] = 0 for all i in {0, ..., N}
        
    Complexity:
        Time:  O(N * W)  (Pseudo-polynomial time)
        Space: O(N * W)  (Allows exact solution reconstruction via backtracking)
    """
    t0 = time.perf_counter()
    N = problem.n_items
    W = problem.capacity

    if N == 0 or W == 0:
        return KnapsackSolution(
            selected_items=[], total_value=0.0, total_weight=0,
            runtime_ms=(time.perf_counter() - t0) * 1000.0, is_optimal=True, method_name="Dynamic Programming"
        )

    # 2D DP Table: (N + 1) rows x (W + 1) columns
    dp = np.zeros((N + 1, W + 1), dtype=np.float64)

    weights = problem.weights
    values = problem.values

    # Fill DP table
    for i in range(1, N + 1):
        wt_i = weights[i - 1]
        val_i = values[i - 1]
        for w in range(W + 1):
            if wt_i <= w:
                val_with = dp[i - 1, w - wt_i] + val_i
                val_without = dp[i - 1, w]
                dp[i, w] = val_with if val_with > val_without else val_without
            else:
                dp[i, w] = dp[i - 1, w]

    opt_value = float(dp[N, W])

    # Backtracking to reconstruct selected items
    selected_items = []
    curr_w = W
    for i in range(N, 0, -1):
        # If value differs from row above, item i-1 was included
        if dp[i, curr_w] != dp[i - 1, curr_w]:
            item_idx = i - 1
            selected_items.append(item_idx)
            curr_w -= weights[item_idx]

    selected_items.sort()
    total_weight = sum(weights[idx] for idx in selected_items)
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return KnapsackSolution(
        selected_items=selected_items,
        total_value=opt_value,
        total_weight=total_weight,
        runtime_ms=runtime_ms,
        is_optimal=True,
        method_name="Dynamic Programming"
    )
