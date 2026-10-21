"""Unit tests for Track E EV Fleet Smart Charging Case Study."""

import numpy as np
import pytest

from track_E_case_study.src.ev_charging_problem import (
    EVProfile,
    FleetDepotScenario,
    generate_synthetic_fleet,
    simulate_asap_baseline,
)
from track_E_case_study.src.ev_optimizers import (
    solve_ev_continuous_convex_slsqp,
    solve_ev_linear_programming_highs,
    solve_ev_priority_greedy_heuristic,
)


@pytest.fixture
def standard_fleet_scenario() -> FleetDepotScenario:
    """Fixture providing a standard 8-vehicle depot charging scenario."""
    return generate_synthetic_fleet(num_vehicles=8, seed=42)


def test_ev_profile_and_energy_calculations() -> None:
    """Verify energy requirement calculation for an individual EV."""
    veh = EVProfile(
        vehicle_id=1,
        battery_capacity_kwh=75.0,
        arrival_hour=17,
        departure_hour=7,
        initial_soc=0.20,
        target_soc=0.90,
        max_charge_power_kw=22.0,
    )
    # (0.90 - 0.20) * 75.0 = 52.5 kWh
    assert np.isclose(veh.energy_required_kwh, 52.5, atol=1e-5)


def test_availability_matrix_window(standard_fleet_scenario: FleetDepotScenario) -> None:
    """Verify that vehicles are only marked available during their dwell period."""
    avail = standard_fleet_scenario.availability_matrix()
    assert avail.shape == (8, 24)

    for k, veh in enumerate(standard_fleet_scenario.vehicles):
        # Overnight vehicles (e.g. arr=17, dep=7): available at 18:00 and 02:00, unavailable at 12:00
        arr = veh.arrival_hour
        dep = veh.departure_hour
        if arr > dep:
            assert avail[k, arr] is True or avail[k, arr] == 1
            assert avail[k, (dep - 1) % 24] is True or avail[k, (dep - 1) % 24] == 1
            assert avail[k, 12] is False or avail[k, 12] == 0


def test_asap_baseline_energy_delivery_and_peak_coincidence(standard_fleet_scenario: FleetDepotScenario) -> None:
    """Verify baseline delivers energy but creates high peak power coinciding with evening tariffs."""
    res_base = simulate_asap_baseline(standard_fleet_scenario)

    # Must deliver all required energy
    assert res_base.unmet_energy_kwh <= 1e-3
    assert res_base.total_energy_delivered_kwh >= res_base.total_energy_required_kwh - 1e-3
    # Baseline peak power should be substantial due to simultaneous plug-in
    assert res_base.peak_depot_power_kw > 40.0
    # Peak load occurs between 16:00 and 20:00
    agg_load = np.sum(res_base.power_schedule, axis=0)
    peak_hour = int(np.argmax(agg_load))
    assert 16 <= peak_hour <= 20


def test_continuous_convex_slsqp_optimization(standard_fleet_scenario: FleetDepotScenario) -> None:
    """Verify SLSQP convex optimizer reduces cost, peak load, and satisfies all constraints."""
    res_base = simulate_asap_baseline(standard_fleet_scenario)
    res_slsqp = solve_ev_continuous_convex_slsqp(standard_fleet_scenario, max_iter=200)

    # Strictly feasible
    assert res_slsqp.is_feasible is True
    assert res_slsqp.grid_power_violations_kw <= 1e-3
    assert res_slsqp.unmet_energy_kwh <= 1e-3
    assert res_slsqp.peak_depot_power_kw <= standard_fleet_scenario.depot_transformer_limit_kw + 1e-3

    # Cost reduction: SLSQP should achieve significantly lower total cost than unoptimized baseline
    assert res_slsqp.total_cost_dollars < res_base.total_cost_dollars
    # Peak power reduction
    assert res_slsqp.peak_depot_power_kw < res_base.peak_depot_power_kw


def test_linear_programming_highs_optimization(standard_fleet_scenario: FleetDepotScenario) -> None:
    """Verify HiGHS linear programming solver satisfies constraints and achieves optimal linear cost."""
    res_base = simulate_asap_baseline(standard_fleet_scenario)
    res_highs = solve_ev_linear_programming_highs(standard_fleet_scenario)

    assert res_highs.is_feasible is True
    assert res_highs.grid_power_violations_kw <= 1e-3
    assert res_highs.unmet_energy_kwh <= 1e-3
    assert res_highs.total_cost_dollars < res_base.total_cost_dollars
    assert res_highs.peak_depot_power_kw <= standard_fleet_scenario.depot_transformer_limit_kw + 1e-3


def test_priority_greedy_valley_filling_heuristic(standard_fleet_scenario: FleetDepotScenario) -> None:
    """Verify Priority Greedy Valley Filling allocates charging predominantly in overnight hours."""
    res_base = simulate_asap_baseline(standard_fleet_scenario)
    res_greedy = solve_ev_priority_greedy_heuristic(standard_fleet_scenario)

    assert res_greedy.is_feasible is True
    assert res_greedy.grid_power_violations_kw <= 1e-3
    assert res_greedy.unmet_energy_kwh <= 1e-3
    assert res_greedy.total_cost_dollars < res_base.total_cost_dollars

    # Check that majority of energy is delivered during off-peak overnight hours (00:00 - 06:00)
    agg_load = np.sum(res_greedy.power_schedule, axis=0)
    offpeak_energy = np.sum(agg_load[:6])
    total_energy = np.sum(agg_load)
    assert (offpeak_energy / total_energy) > 0.50


def test_solver_comparative_consistency(standard_fleet_scenario: FleetDepotScenario) -> None:
    """All optimization solvers should agree on delivery requirements and outperform baseline."""
    res_base = simulate_asap_baseline(standard_fleet_scenario)
    res_slsqp = solve_ev_continuous_convex_slsqp(standard_fleet_scenario)
    res_highs = solve_ev_linear_programming_highs(standard_fleet_scenario)
    res_greedy = solve_ev_priority_greedy_heuristic(standard_fleet_scenario)

    for res in [res_slsqp, res_highs, res_greedy]:
        # Savings must be positive (> 15% reduction vs baseline)
        cost_savings_pct = (res_base.total_cost_dollars - res.total_cost_dollars) / res_base.total_cost_dollars * 100
        assert cost_savings_pct > 15.0, f"{res.method_name} achieved only {cost_savings_pct:.1f}% savings"
