"""Track E: Main Real-Process Case Study (EV Fleet Smart Charging Scheduling)."""

from track_E_case_study.src.ev_charging_problem import (
    EVProfile,
    FleetDepotScenario,
    ScheduleEvaluationResult,
    evaluate_schedule,
    generate_synthetic_fleet,
    simulate_asap_baseline,
)
from track_E_case_study.src.ev_optimizers import (
    solve_ev_continuous_convex_slsqp,
    solve_ev_linear_programming_highs,
    solve_ev_priority_greedy_heuristic,
)

__all__ = [
    "EVProfile",
    "FleetDepotScenario",
    "ScheduleEvaluationResult",
    "evaluate_schedule",
    "generate_synthetic_fleet",
    "simulate_asap_baseline",
    "solve_ev_continuous_convex_slsqp",
    "solve_ev_linear_programming_highs",
    "solve_ev_priority_greedy_heuristic",
]
