import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from track_E_case_study.src.ev_charging_problem import generate_synthetic_fleet

DATA_DIR = Path("track_E_case_study/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    scenario = generate_synthetic_fleet(num_vehicles=12, seed=42)
    records = []
    for v in scenario.vehicles:
        records.append({
            "vehicle_id": v.vehicle_id,
            "battery_capacity_kwh": v.battery_capacity_kwh,
            "arrival_hour": v.arrival_hour,
            "departure_hour": v.departure_hour,
            "initial_soc": v.initial_soc,
            "target_soc": v.target_soc,
            "energy_required_kwh": v.energy_required_kwh,
            "max_charge_power_kw": v.max_charge_power_kw,
        })

    df = pd.DataFrame(records)
    csv_path = DATA_DIR / "synthetic_fleet_dataset.csv"
    json_path = DATA_DIR / "synthetic_fleet_dataset.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "depot_transformer_limit_kw": scenario.depot_transformer_limit_kw,
            "charger_efficiency": scenario.charger_efficiency,
            "peak_demand_charge_per_kw": scenario.peak_demand_charge_per_kw,
            "tou_tariff_per_kwh": list(scenario.tou_tariff_per_kwh),
            "vehicles": records,
        }, f, indent=2)

    print(f"Exported dataset to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
