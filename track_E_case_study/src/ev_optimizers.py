"""Optimization Solvers for EV Fleet Smart Charging (Track E).

Implements:
1. Method 1: Continuous Convex Optimization (SLSQP / Quadratic Programming)
   - Minimizes Time-of-Use energy cost, peak power demand, and battery degradation
   - Guarantees global optimality via convex formulation
2. Method 2: Mixed-Integer / Linear Programming (HiGHS Branch-and-Cut)
   - Linearized peak demand with exact simplex / interior-point / branch-and-cut
3. Method 3: Priority-Based Greedy Valley-Filling Heuristic
   - Fast O(K T log T) online valley filling for real-time edge embedded dispatch
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linprog, minimize

from track_E_case_study.src.ev_charging_problem import (
    FleetDepotScenario,
    ScheduleEvaluationResult,
    evaluate_schedule,
)


def solve_ev_continuous_convex_slsqp(
    scenario: FleetDepotScenario,
    max_iter: int = 500,
    tol: float = 1e-5,
) -> ScheduleEvaluationResult:
    """Method 1: Continuous Convex Optimization via SLSQP.

    Variables:
        x = [p_{0, 0}, ..., p_{K-1, T-1}, P_peak] of dimension (K*T + 1)
    Objective:
        min sum_t c_t (sum_k p_{k, t}) dt + lambda_peak * P_peak + lambda_deg * sum_{k, t} p_{k, t}^2 dt
    Subject to:
        sum_k p_{k, t} <= P_peak           (for all t)
        sum_k p_{k, t} <= P_grid_max       (for all t)
        sum_t p_{k, t} * eta * dt >= E_k   (for each vehicle k)
        0 <= p_{k, t} <= P_max * A_{k, t}  (bounds)
        0 <= P_peak <= P_grid_max          (bound)
    """
    t0 = time.perf_counter()
    K = scenario.num_vehicles
    T = scenario.num_time_steps
    dt = scenario.time_step_hours
    eta = scenario.charger_efficiency
    c_tou = scenario.tou_tariff_per_kwh
    l_peak = scenario.peak_demand_charge_per_kw
    l_deg = scenario.degradation_coeff
    p_grid_max = scenario.depot_transformer_limit_kw
    avail = scenario.availability_matrix()

    n_vars = K * T + 1  # Last variable is P_peak

    # Objective function
    def objective(x: np.ndarray) -> float:
        P = x[:K * T].reshape((K, T))
        P_peak = x[-1]
        agg_p = np.sum(P, axis=0)
        tou_cost = np.sum(agg_p * c_tou * dt)
        peak_cost = l_peak * P_peak
        deg_cost = l_deg * np.sum(P ** 2) * dt
        return float(tou_cost + peak_cost + deg_cost)

    # Objective gradient
    def objective_grad(x: np.ndarray) -> np.ndarray:
        P = x[:K * T].reshape((K, T))
        grad = np.zeros(n_vars, dtype=np.float64)
        grad_P = np.zeros((K, T), dtype=np.float64)
        for t in range(T):
            grad_P[:, t] = c_tou[t] * dt + 2.0 * l_deg * P[:, t] * dt
        grad[:K * T] = grad_P.flatten()
        grad[-1] = l_peak
        return grad

    # Variable bounds
    bounds = []
    for k in range(K):
        for t in range(T):
            upper = scenario.vehicles[k].max_charge_power_kw if avail[k, t] else 0.0
            bounds.append((0.0, upper))
    bounds.append((0.0, p_grid_max))  # P_peak bound

    # Constraints list
    constraints = []

    # 1. P_peak bounding constraints: P_peak - sum_k p_{k, t} >= 0
    for t in range(T):
        def c_peak(x: np.ndarray, t_idx=t) -> float:
            P = x[:K * T].reshape((K, T))
            P_peak = x[-1]
            return float(P_peak - np.sum(P[:, t_idx]))
        constraints.append({"type": "ineq", "fun": c_peak})

    # 2. Depot transformer capacity constraints: P_grid_max - sum_k p_{k, t} >= 0
    for t in range(T):
        def c_grid(x: np.ndarray, t_idx=t) -> float:
            P = x[:K * T].reshape((K, T))
            return float(p_grid_max - np.sum(P[:, t_idx]))
        constraints.append({"type": "ineq", "fun": c_grid})

    # 3. Energy delivery requirement per vehicle: sum_t p_{k, t} * eta * dt - E_k >= 0
    for k in range(K):
        e_req = scenario.vehicles[k].energy_required_kwh
        def c_energy(x: np.ndarray, k_idx=k, req=e_req) -> float:
            P = x[:K * T].reshape((K, T))
            return float(np.sum(P[k_idx, :]) * eta * dt - req)
        constraints.append({"type": "ineq", "fun": c_energy})

    # Initial guess: distribute required energy evenly across available slots
    x0 = np.zeros(n_vars, dtype=np.float64)
    P0 = np.zeros((K, T), dtype=np.float64)
    for k in range(K):
        avail_count = np.sum(avail[k, :])
        if avail_count > 0:
            power_per_slot = scenario.vehicles[k].energy_required_kwh / (avail_count * eta * dt)
            power_per_slot = min(power_per_slot, scenario.vehicles[k].max_charge_power_kw)
            P0[k, avail[k, :]] = power_per_slot
    x0[:K * T] = P0.flatten()
    x0[-1] = float(np.max(np.sum(P0, axis=0)))

    res = minimize(
        objective,
        x0,
        jac=objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": tol, "disp": False},
    )

    p_sol = res.x[:K * T].reshape((K, T))
    # Clip numerical noise
    p_sol = np.maximum(0.0, p_sol) * avail

    t1 = time.perf_counter()
    return evaluate_schedule(scenario, p_sol, method_name="Continuous_Convex_SLSQP", exec_time_sec=t1 - t0)


def solve_ev_linear_programming_highs(
    scenario: FleetDepotScenario,
) -> ScheduleEvaluationResult:
    """Method 2: Mixed-Integer / Linear Programming via HiGHS.

    Formulates the exact linear cost model:
        min c^T x
    where x = [p_{0, 0}, ..., p_{K-1, T-1}, P_peak]
    Linearizes peak demand charges exactly and achieves provably optimal solutions.
    """
    t0 = time.perf_counter()
    K = scenario.num_vehicles
    T = scenario.num_time_steps
    dt = scenario.time_step_hours
    eta = scenario.charger_efficiency
    c_tou = scenario.tou_tariff_per_kwh
    l_peak = scenario.peak_demand_charge_per_kw
    p_grid_max = scenario.depot_transformer_limit_kw
    avail = scenario.availability_matrix()

    n_vars = K * T + 1

    # Objective vector c:
    # c_p = c_tou[t] * dt
    # c_peak = l_peak
    c = np.zeros(n_vars, dtype=np.float64)
    for k in range(K):
        for t in range(T):
            idx = k * T + t
            c[idx] = c_tou[t] * dt
    c[-1] = l_peak

    # Inequality constraints A_ub * x <= b_ub
    # 1. P_peak definition: sum_k p_{k, t} - P_peak <= 0  (T rows)
    # 2. Grid limit: sum_k p_{k, t} <= P_grid_max          (T rows)
    # 3. Energy fulfillment: - sum_t p_{k, t} * eta * dt <= - E_k (K rows)
    num_constraints = 2 * T + K
    A_ub = np.zeros((num_constraints, n_vars), dtype=np.float64)
    b_ub = np.zeros(num_constraints, dtype=np.float64)

    row = 0
    # 1. sum_k p_{k, t} - P_peak <= 0
    for t in range(T):
        for k in range(K):
            A_ub[row, k * T + t] = 1.0
        A_ub[row, -1] = -1.0
        b_ub[row] = 0.0
        row += 1

    # 2. sum_k p_{k, t} <= P_grid_max
    for t in range(T):
        for k in range(K):
            A_ub[row, k * T + t] = 1.0
        b_ub[row] = p_grid_max
        row += 1

    # 3. Energy delivery: -sum_t p_{k, t} <= -E_k / (eta * dt)
    for k in range(K):
        e_req = scenario.vehicles[k].energy_required_kwh
        for t in range(T):
            A_ub[row, k * T + t] = -1.0 * eta * dt
        b_ub[row] = -e_req
        row += 1

    # Bounds
    bounds = []
    for k in range(K):
        for t in range(T):
            upper = scenario.vehicles[k].max_charge_power_kw if avail[k, t] else 0.0
            bounds.append((0.0, upper))
    bounds.append((0.0, p_grid_max))

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        # Fallback to zero schedule
        p_sol = np.zeros((K, T), dtype=np.float64)
    else:
        p_sol = res.x[:K * T].reshape((K, T))
        p_sol = np.maximum(0.0, p_sol) * avail

    t1 = time.perf_counter()
    return evaluate_schedule(scenario, p_sol, method_name="LinearProgramming_HiGHS", exec_time_sec=t1 - t0)


def solve_ev_priority_greedy_heuristic(
    scenario: FleetDepotScenario,
) -> ScheduleEvaluationResult:
    """Method 3: Priority-Based Greedy Valley-Filling Heuristic.

    Sorts available time slots by Time-of-Use tariff ascending (cheapest overnight valley first).
    Allocates charging power to vehicles with earliest departures and highest energy deficits,
    respecting individual charger rates and remaining transformer capacity headroom.
    Runs in O(K T log T) time, making it ideal for real-time edge embedded controllers.
    """
    t0 = time.perf_counter()
    K = scenario.num_vehicles
    T = scenario.num_time_steps
    dt = scenario.time_step_hours
    eta = scenario.charger_efficiency
    p_grid_max = scenario.depot_transformer_limit_kw

    schedule = np.zeros((K, T), dtype=np.float64)
    remaining_grid_headroom = np.full(T, p_grid_max, dtype=np.float64)

    # Sort vehicles by urgency: departure hour ascending, then energy deficit descending
    def vehicle_urgency(v_tuple):
        k, v = v_tuple
        return (v.departure_hour, -v.energy_required_kwh)

    sorted_vehicles = sorted(enumerate(scenario.vehicles), key=vehicle_urgency)

    for k, veh in sorted_vehicles:
        e_needed = veh.energy_required_kwh
        # Find available slots for this vehicle
        avail_slots = [t for t in range(T) if scenario.is_available(veh, t)]
        # Sort available slots by electricity tariff (cheapest first)
        avail_slots.sort(key=lambda t: scenario.tou_tariff_per_kwh[t])

        for t in avail_slots:
            if e_needed <= 1e-6:
                break
            # Power limit is min of:
            # 1. Charger max power
            # 2. Remaining energy divided by (eta * dt)
            # 3. Remaining transformer capacity headroom in slot t
            p_alloc = min(
                veh.max_charge_power_kw,
                e_needed / (eta * dt),
                remaining_grid_headroom[t],
            )
            if p_alloc > 1e-4:
                schedule[k, t] = p_alloc
                remaining_grid_headroom[t] -= p_alloc
                e_needed -= p_alloc * eta * dt

    t1 = time.perf_counter()
    return evaluate_schedule(scenario, schedule, method_name="Priority_Greedy_ValleyFilling", exec_time_sec=t1 - t0)
