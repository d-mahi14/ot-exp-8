"""EV Fleet Smart Charging Problem Formulation for Track E.

Defines:
1. Fleet and Depot Grid specifications:
   - Vehicle battery capacity, initial/target SoC, arrival/departure time windows
   - Charger maximum rate and charging efficiency
   - Depot transformer maximum power limit
   - 24-hour Time-of-Use (ToU) electricity tariff and peak demand charge
2. Formal constraints:
   - Grid power limit: sum_k p_{k, t} <= P_grid_max
   - Charger power limit: 0 <= p_{k, t} <= P_max
   - Availability window: p_{k, t} = 0 when vehicle not plugged in
   - Target energy fulfillment: sum_t p_{k, t} * eta * dt >= E_k_req
3. Baseline Uncontrolled (ASAP) charging simulator.
4. Objective function evaluator and constraint validator.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EVProfile:
    """Individual Electric Vehicle specifications and operational schedule."""
    vehicle_id: int
    battery_capacity_kwh: float  # e.g., 60.0 kWh (commercial delivery van)
    arrival_hour: int            # e.g., 17 (17:00 arrival at depot)
    departure_hour: int          # e.g., 7 (07:00 departure next day; wrapped or 24+h)
    initial_soc: float           # e.g., 0.20 (20% SoC upon return from delivery route)
    target_soc: float            # e.g., 0.90 (90% SoC required before departure)
    max_charge_power_kw: float   # e.g., 22.0 kW (Level 2 AC commercial fast charger)

    @property
    def energy_required_kwh(self) -> float:
        """Net energy in kWh required to reach target SoC."""
        return max(0.0, (self.target_soc - self.initial_soc) * self.battery_capacity_kwh)


@dataclass
class FleetDepotScenario:
    """Complete depot operating environment over a discrete time horizon."""
    time_horizon_hours: int = 24
    time_step_hours: float = 1.0  # 1-hour resolution (T=24 time slots)
    depot_transformer_limit_kw: float = 120.0  # Maximum transformer capacity
    charger_efficiency: float = 0.92          # 92% AC-DC charging efficiency
    peak_demand_charge_per_kw: float = 8.50    # Monthly peak demand surcharge ($/kW)
    degradation_coeff: float = 0.001           # Penalty on high current heat dissipation

    # 24-Hour Time-of-Use (ToU) Electricity Tariffs ($/kWh)
    # Off-peak (00:00 - 06:00): $0.08 / kWh
    # Mid-peak (06:00 - 16:00, 21:00 - 24:00): $0.16 / kWh
    # On-peak (16:00 - 21:00 evening peak): $0.38 / kWh (4.75x off-peak!)
    tou_tariff_per_kwh: np.ndarray = field(default_factory=lambda: np.array([
        0.08, 0.08, 0.08, 0.08, 0.08, 0.08,  # 00:00 - 06:00 (Super Off-peak)
        0.16, 0.16, 0.16, 0.16, 0.16, 0.16,  # 06:00 - 12:00 (Mid-peak)
        0.16, 0.16, 0.16, 0.16,              # 12:00 - 16:00 (Mid-peak)
        0.38, 0.38, 0.38, 0.38, 0.38,        # 16:00 - 21:00 (On-peak Evening)
        0.16, 0.16, 0.16                     # 21:00 - 24:00 (Mid-peak)
    ], dtype=np.float64))

    vehicles: List[EVProfile] = field(default_factory=list)

    @property
    def num_time_steps(self) -> int:
        return int(round(self.time_horizon_hours / self.time_step_hours))

    @property
    def num_vehicles(self) -> int:
        return len(self.vehicles)

    def is_available(self, vehicle: EVProfile, time_step: int) -> bool:
        """Determine if vehicle is parked and plugged into depot charger at time_step."""
        # For overnight shifts (e.g. arrival 17:00, departure 07:00):
        # Vehicle is available during [arr, 24) and [0, dep)
        arr = vehicle.arrival_hour
        dep = vehicle.departure_hour
        if arr < dep:
            # Daytime charging
            return arr <= time_step < dep
        else:
            # Overnight charging across midnight
            return (time_step >= arr) or (time_step < dep)

    def availability_matrix(self) -> np.ndarray:
        """Return boolean matrix of shape (num_vehicles, num_time_steps)."""
        A = np.zeros((self.num_vehicles, self.num_time_steps), dtype=bool)
        for k, veh in enumerate(self.vehicles):
            for t in range(self.num_time_steps):
                A[k, t] = self.is_available(veh, t)
        return A


def generate_synthetic_fleet(
    num_vehicles: int = 10,
    seed: int = 42,
) -> FleetDepotScenario:
    """Generate a realistic commercial delivery EV fleet dataset."""
    rng = np.random.default_rng(seed)
    vehicles = []

    for k in range(num_vehicles):
        # Delivery vans return between 16:00 and 19:00
        arr = int(rng.integers(16, 19))
        # Depart between 06:00 and 08:00 next morning
        dep = int(rng.integers(6, 8))
        # Battery capacity 60 - 80 kWh
        cap = float(rng.choice([60.0, 75.0, 80.0]))
        # Returning SoC 15% - 35%
        soc_init = float(rng.uniform(0.15, 0.35))
        # Required target SoC 85% - 95%
        soc_target = float(rng.uniform(0.85, 0.95))
        # Level 2 charger: 11 kW or 22 kW
        p_max = float(rng.choice([11.0, 22.0]))

        vehicles.append(
            EVProfile(
                vehicle_id=k,
                battery_capacity_kwh=cap,
                arrival_hour=arr,
                departure_hour=dep,
                initial_soc=soc_init,
                target_soc=soc_target,
                max_charge_power_kw=p_max,
            )
        )

    # Scale depot transformer capacity proportionally: ~12-15 kW per vehicle
    transformer_kw = float(max(60.0, num_vehicles * 12.0))

    return FleetDepotScenario(
        depot_transformer_limit_kw=transformer_kw,
        vehicles=vehicles,
    )


@dataclass
class ScheduleEvaluationResult:
    is_feasible: bool
    total_cost_dollars: float
    tou_energy_cost_dollars: float
    peak_demand_charge_dollars: float
    degradation_cost_dollars: float
    peak_depot_power_kw: float
    grid_limit_kw: float
    total_energy_delivered_kwh: float
    total_energy_required_kwh: float
    unmet_energy_kwh: float
    grid_power_violations_kw: float
    method_name: str
    execution_time_sec: float
    power_schedule: np.ndarray  # shape (K, T)


def evaluate_schedule(
    scenario: FleetDepotScenario,
    power_schedule: np.ndarray,
    method_name: str,
    exec_time_sec: float = 0.0,
) -> ScheduleEvaluationResult:
    """Independently verify constraints and compute cost components of power schedule."""
    K = scenario.num_vehicles
    T = scenario.num_time_steps
    dt = scenario.time_step_hours
    eta = scenario.charger_efficiency

    assert power_schedule.shape == (K, T)

    # 1. Total aggregate load per time slot
    aggregate_power_kw = np.sum(power_schedule, axis=0)  # shape (T,)
    peak_depot_power_kw = float(np.max(aggregate_power_kw))

    # Grid capacity violations
    excess_grid_power = np.maximum(0.0, aggregate_power_kw - scenario.depot_transformer_limit_kw)
    total_grid_violation = float(np.sum(excess_grid_power))

    # 2. Vehicle-level energy delivery and window constraints
    avail_mask = scenario.availability_matrix()
    unplugged_charging = np.sum(power_schedule * (~avail_mask))

    total_energy_delivered = 0.0
    total_energy_required = 0.0
    unmet_energy = 0.0

    for k, veh in enumerate(scenario.vehicles):
        e_req = veh.energy_required_kwh
        total_energy_required += e_req
        e_delivered = float(np.sum(power_schedule[k, :]) * eta * dt)
        total_energy_delivered += e_delivered
        if e_delivered < e_req - 1e-3:
            unmet_energy += (e_req - e_delivered)

    # 3. Cost components
    tou_cost = float(np.sum(aggregate_power_kw * scenario.tou_tariff_per_kwh * dt))
    peak_charge = float(peak_depot_power_kw * scenario.peak_demand_charge_per_kw)
    deg_cost = float(scenario.degradation_coeff * np.sum(power_schedule ** 2) * dt)
    total_cost = tou_cost + peak_charge + deg_cost

    # Feasibility boolean
    is_feasible = bool(
        total_grid_violation <= 1e-3
        and unmet_energy <= 1e-3
        and unplugged_charging <= 1e-3
    )

    return ScheduleEvaluationResult(
        is_feasible=is_feasible,
        total_cost_dollars=total_cost,
        tou_energy_cost_dollars=tou_cost,
        peak_demand_charge_dollars=peak_charge,
        degradation_cost_dollars=deg_cost,
        peak_depot_power_kw=peak_depot_power_kw,
        grid_limit_kw=scenario.depot_transformer_limit_kw,
        total_energy_delivered_kwh=total_energy_delivered,
        total_energy_required_kwh=total_energy_required,
        unmet_energy_kwh=unmet_energy,
        grid_power_violations_kw=total_grid_violation,
        method_name=method_name,
        execution_time_sec=exec_time_sec,
        power_schedule=power_schedule,
    )


def simulate_asap_baseline(scenario: FleetDepotScenario) -> ScheduleEvaluationResult:
    """Baseline: Uncontrolled 'As-Soon-As-Possible' (ASAP) charging.

    Vehicles start charging at maximum power immediately upon arrival until target SoC is reached.
    Causes severe peak load coincidence during on-peak evening tariff hours (16:00 - 20:00).
    """
    import time
    t0 = time.perf_counter()

    K = scenario.num_vehicles
    T = scenario.num_time_steps
    dt = scenario.time_step_hours
    eta = scenario.charger_efficiency

    schedule = np.zeros((K, T), dtype=np.float64)

    for k, veh in enumerate(scenario.vehicles):
        e_remaining = veh.energy_required_kwh
        # Sequence of available time steps starting from arrival
        # If overnight, order is: arrival..23, then 0..departure-1
        arr = veh.arrival_hour
        dep = veh.departure_hour
        if arr < dep:
            time_slots = list(range(arr, dep))
        else:
            time_slots = list(range(arr, T)) + list(range(0, dep))

        for t in time_slots:
            if e_remaining <= 1e-6:
                break
            # Charge at max charger rate or whatever is needed to top off
            p_charge = min(veh.max_charge_power_kw, e_remaining / (eta * dt))
            schedule[k, t] = p_charge
            e_remaining -= p_charge * eta * dt

    t1 = time.perf_counter()
    return evaluate_schedule(scenario, schedule, method_name="Baseline_ASAP_Uncontrolled", exec_time_sec=t1 - t0)
