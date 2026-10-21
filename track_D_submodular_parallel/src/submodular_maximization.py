"""Submodular Maximization Module for Track D.

Implements:
1. Monotone Submodular Coverage / Sensor Placement problem formulations.
2. Diminishing returns (submodularity) formal verification.
3. Solvers:
   - Exact Brute-Force (Combinatorial search for ground-truth on small instances)
   - Standard Greedy (Nemhauser et al. 1978, (1 - 1/e) guarantee)
   - Lazy Greedy (Minoux 1978, accelerated priority-queue pruning)
   - Baseline heuristics (Random selection, Top-k isolated marginals)
4. Cost-budgeted greedy variant (density-greedy).
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class SensorPlacementProblem:
    """Monotone submodular sensor placement / spatial coverage problem.

    Given a 2D domain with target locations U and candidate sensor locations V.
    Each sensor s in V covers target u with probability:
        p(s, u) = exp(- ||s - u||^2 / (2 * sigma^2)) if ||s - u|| <= r_max else 0.

    The expected coverage utility function over subset S subseteq V is:
        f(S) = sum_{u in U} w_u * [ 1 - prod_{s in S} (1 - p(s, u)) ]

    Properties:
    - Normalized: f(emptyset) = 0
    - Strictly Monotone: S subseteq T => f(S) <= f(T)
    - Submodular: f(A cup {e}) - f(A) >= f(B cup {e}) - f(B) for all A subseteq B and e notin B.
    """

    targets: np.ndarray  # shape (M, 2)
    candidates: np.ndarray  # shape (N, 2)
    target_weights: np.ndarray  # shape (M,)
    sensor_costs: np.ndarray  # shape (N,)
    decay_sigma: float = 0.25
    r_max: float = 0.5

    def __post_init__(self) -> None:
        self.num_targets = len(self.targets)
        self.num_candidates = len(self.candidates)
        assert len(self.target_weights) == self.num_targets
        assert len(self.sensor_costs) == self.num_candidates

        # Precompute coverage probabilities matrix P[s, u] of shape (N, M)
        dists = np.linalg.norm(
            self.candidates[:, None, :] - self.targets[None, :, :], axis=2
        )  # (N, M)
        prob = np.exp(- (dists ** 2) / (2.0 * (self.decay_sigma ** 2)))
        prob[dists > self.r_max] = 0.0
        self.prob_matrix = np.clip(prob, 0.0, 1.0)  # (N, M)

    def evaluate(self, selected_indices: Iterable[int]) -> float:
        """Compute submodular objective value f(S)."""
        idx_list = list(selected_indices)
        if not idx_list:
            return 0.0
        # For targets, prob of NOT being covered is prod_{s in S} (1 - P[s, u])
        not_covered_prob = np.prod(1.0 - self.prob_matrix[idx_list, :], axis=0)  # (M,)
        covered_prob = 1.0 - not_covered_prob  # (M,)
        return float(np.dot(self.target_weights, covered_prob))

    def marginal_gain(self, current_set: Iterable[int], candidate: int) -> float:
        """Compute f(S cup {candidate}) - f(S)."""
        idx_list = list(current_set)
        if candidate in idx_list:
            return 0.0
        val_before = self.evaluate(idx_list)
        val_after = self.evaluate(idx_list + [candidate])
        return max(0.0, val_after - val_before)


def verify_submodularity(
    objective_fn: Callable[[Set[int]], float],
    ground_set: Sequence[int],
    num_samples: int = 50,
    seed: int = 42,
) -> Tuple[bool, float]:
    """Empirically verify diminishing returns property on random subset pairs A subseteq B.

    Diminishing returns condition:
        f(A cup {e}) - f(A) >= f(B cup {e}) - f(B) - tol
    for all A subseteq B and e notin B.

    Returns:
        (is_submodular, min_slack): min_slack is min of [gain(A, e) - gain(B, e)].
        If min_slack >= -1e-9, submodularity holds empirically.
    """
    rng = np.random.default_rng(seed)
    n = len(ground_set)
    ground_set_list = list(ground_set)
    min_slack = float("inf")

    for _ in range(num_samples):
        # Sample subset size for B
        if n <= 2:
            break
        k_b = rng.integers(1, n)
        b_indices = set(rng.choice(ground_set_list, size=k_b, replace=False))
        # Sample subset A subseteq B
        k_a = rng.integers(0, k_b)
        a_indices = set(rng.choice(list(b_indices), size=k_a, replace=False))

        # Sample element e not in B
        remaining = [x for x in ground_set_list if x not in b_indices]
        if not remaining:
            continue
        e = rng.choice(remaining)

        gain_a = objective_fn(a_indices | {e}) - objective_fn(a_indices)
        gain_b = objective_fn(b_indices | {e}) - objective_fn(b_indices)

        slack = gain_a - gain_b
        if slack < min_slack:
            min_slack = slack

        if slack < -1e-9:
            return False, float(slack)

    return True, float(min_slack if min_slack != float("inf") else 0.0)


@dataclass
class SubmodularSolution:
    selected_set: List[int]
    objective_value: float
    total_cost: float
    evaluations_count: int
    execution_time_sec: float
    method: str


def solve_submodular_brute_force(
    problem: SensorPlacementProblem,
    k: int,
    max_budget: Optional[float] = None,
) -> SubmodularSolution:
    """Exact combinatorial brute force solver over all subsets of size <= k (or within budget).

    Used as ground-truth oracle for small problem instances (|V| <= 20).
    """
    import time
    t0 = time.perf_counter()
    n = problem.num_candidates
    best_set: List[int] = []
    best_val = -1.0
    best_cost = 0.0
    eval_count = 0

    all_candidates = list(range(n))

    # Examine subsets of cardinality up to k
    for r in range(1, k + 1):
        for subset in itertools.combinations(all_candidates, r):
            cost = float(np.sum(problem.sensor_costs[list(subset)]))
            if max_budget is not None and cost > max_budget:
                continue
            val = problem.evaluate(subset)
            eval_count += 1
            if val > best_val:
                best_val = val
                best_set = list(subset)
                best_cost = cost

    t1 = time.perf_counter()
    return SubmodularSolution(
        selected_set=sorted(best_set),
        objective_value=best_val if best_val >= 0 else 0.0,
        total_cost=best_cost,
        evaluations_count=eval_count,
        execution_time_sec=t1 - t0,
        method="BruteForceExact",
    )


def solve_submodular_greedy(
    problem: SensorPlacementProblem,
    k: int,
    max_budget: Optional[float] = None,
) -> SubmodularSolution:
    r"""Standard Greedy Algorithm (Nemhauser, Wolsey, Fisher 1978).

    At each step:
        s* = argmax_{s in V \setminus S} [ f(S cup {s}) - f(S) ]
    For unit costs, achieves (1 - 1/e) approx ~= 0.6321 of the optimal value.
    """
    import time
    t0 = time.perf_counter()

    selected: List[int] = []
    remaining = set(range(problem.num_candidates))
    current_cost = 0.0
    eval_count = 0

    for _ in range(k):
        best_candidate = -1
        best_gain = -1.0
        best_candidate_cost = 0.0

        for c in remaining:
            c_cost = float(problem.sensor_costs[c])
            if max_budget is not None and (current_cost + c_cost) > max_budget:
                continue

            gain = problem.marginal_gain(selected, c)
            eval_count += 1

            if gain > best_gain:
                best_gain = gain
                best_candidate = c
                best_candidate_cost = c_cost

        if best_candidate == -1 or best_gain <= 1e-12:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        current_cost += best_candidate_cost

    final_val = problem.evaluate(selected)
    t1 = time.perf_counter()

    return SubmodularSolution(
        selected_set=sorted(selected),
        objective_value=final_val,
        total_cost=current_cost,
        evaluations_count=eval_count,
        execution_time_sec=t1 - t0,
        method="StandardGreedy",
    )


def solve_submodular_lazy_greedy(
    problem: SensorPlacementProblem,
    k: int,
    max_budget: Optional[float] = None,
) -> SubmodularSolution:
    """Accelerated Lazy Greedy Algorithm (Minoux 1978).

    Maintains a max-priority queue of marginal gains.
    Exploits submodularity: since marginal gains monotonically decrease over iterations,
    if the top element's re-evaluated marginal gain is >= the next upper bound in the queue,
    it is guaranteed to be optimal for this round without evaluating any other candidates.
    """
    import time
    t0 = time.perf_counter()

    # Priority queue stores (-upper_bound, candidate_id, iteration_when_evaluated)
    # Python heapq is a min-heap, so we negate upper_bound
    pq: List[Tuple[float, int, int]] = []
    eval_count = 0

    # Initialize queue with initial marginal gain f({s}) - f(emptyset) = f({s})
    for s in range(problem.num_candidates):
        gain = problem.marginal_gain([], s)
        eval_count += 1
        heapq.heappush(pq, (-gain, s, 0))

    selected: List[int] = []
    current_cost = 0.0
    current_iter = 0

    while pq and len(selected) < k:
        neg_bound, cand, last_iter = heapq.heappop(pq)
        cand_cost = float(problem.sensor_costs[cand])

        if max_budget is not None and (current_cost + cand_cost) > max_budget:
            continue

        # If this bound was computed in current_iter, it is exact and maximal!
        if last_iter == current_iter:
            if -neg_bound <= 1e-12:
                break  # No further positive gain
            selected.append(cand)
            current_cost += cand_cost
            current_iter += 1
        else:
            # Recompute marginal gain for candidate with current selected set
            new_gain = problem.marginal_gain(selected, cand)
            eval_count += 1
            # Push back with updated iteration stamp
            heapq.heappush(pq, (-new_gain, cand, current_iter))

    final_val = problem.evaluate(selected)
    t1 = time.perf_counter()

    return SubmodularSolution(
        selected_set=sorted(selected),
        objective_value=final_val,
        total_cost=current_cost,
        evaluations_count=eval_count,
        execution_time_sec=t1 - t0,
        method="LazyGreedy",
    )


def solve_submodular_random_baseline(
    problem: SensorPlacementProblem,
    k: int,
    num_trials: int = 50,
    seed: int = 42,
    max_budget: Optional[float] = None,
) -> SubmodularSolution:
    """Random baseline heuristic: samples random subsets of size k (respecting budget) and returns best."""
    import time
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    best_val = -1.0
    best_set: List[int] = []
    best_cost = 0.0
    eval_count = 0

    all_cands = np.arange(problem.num_candidates)

    for _ in range(num_trials):
        perm = rng.permutation(all_cands)
        chosen = []
        cost = 0.0
        for c in perm:
            c_cost = float(problem.sensor_costs[c])
            if max_budget is not None and (cost + c_cost) > max_budget:
                continue
            chosen.append(int(c))
            cost += c_cost
            if len(chosen) >= k:
                break

        val = problem.evaluate(chosen)
        eval_count += 1
        if val > best_val:
            best_val = val
            best_set = chosen
            best_cost = cost

    t1 = time.perf_counter()
    return SubmodularSolution(
        selected_set=sorted(best_set),
        objective_value=best_val if best_val >= 0 else 0.0,
        total_cost=best_cost,
        evaluations_count=eval_count,
        execution_time_sec=t1 - t0,
        method="RandomBaseline",
    )


def solve_submodular_topk_isolated_baseline(
    problem: SensorPlacementProblem,
    k: int,
    max_budget: Optional[float] = None,
) -> SubmodularSolution:
    """Top-k isolated baseline: selects elements with highest individual coverage f({s}), ignoring overlap."""
    import time
    t0 = time.perf_counter()

    eval_count = 0
    scores = []
    for s in range(problem.num_candidates):
        score = problem.evaluate([s])
        eval_count += 1
        scores.append((score, s, float(problem.sensor_costs[s])))

    # Sort descending by score
    scores.sort(key=lambda x: x[0], reverse=True)

    selected = []
    total_cost = 0.0
    for score, s, c_cost in scores:
        if max_budget is not None and (total_cost + c_cost) > max_budget:
            continue
        selected.append(s)
        total_cost += c_cost
        if len(selected) >= k:
            break

    final_val = problem.evaluate(selected)
    t1 = time.perf_counter()

    return SubmodularSolution(
        selected_set=sorted(selected),
        objective_value=final_val,
        total_cost=total_cost,
        evaluations_count=eval_count,
        execution_time_sec=t1 - t0,
        method="TopKIsolated",
    )
