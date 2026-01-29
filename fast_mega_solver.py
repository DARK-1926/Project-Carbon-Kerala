"""
FAST MEGA SOLVER v17 - Optimized for Speed
===========================================
Fast version focusing on what works:
1. Efficient parameter search
2. Fast local search
3. Skip slow LP
"""

import pandas as pd
import numpy as np
import json
import os
from math import radians, sin, cos, sqrt, atan2, ceil
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
import random

warnings.filterwarnings('ignore')
random.seed(42)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


@dataclass
class ProblemData:
    num_days: int
    num_farms: int
    num_stps: int
    farm_ids: List
    stp_ids: List
    dates: pd.DatetimeIndex
    stp_daily_output: np.ndarray
    stp_storage_max: np.ndarray
    stp_coords: np.ndarray
    farm_area: np.ndarray
    farm_coords: np.ndarray
    farm_zone: np.ndarray
    zones: List
    distances: np.ndarray
    rain_locked: np.ndarray
    crop_active: np.ndarray
    n_demand_kg: np.ndarray
    buffer_tons: np.ndarray
    demand_tons: np.ndarray
    params: Dict = field(default_factory=dict)
    transport_cost_per_trip: np.ndarray = None
    closest_stp: np.ndarray = None


def load_params() -> Dict:
    with open("config.json") as f:
        return json.load(f)


def load_all_data() -> ProblemData:
    params = load_params()

    stp = pd.read_csv("stp_registry.csv").sort_values("stp_id").reset_index(drop=True)
    farm = pd.read_csv("farm_locations.csv").sort_values("farm_id").reset_index(drop=True)
    weather = pd.read_csv("daily_weather_2025.csv")
    planting = pd.read_csv("planting_schedule_2025.csv")

    dates = pd.date_range("2025-01-01", "2025-12-31")
    D, F, S = len(dates), len(farm), len(stp)

    farm_ids = farm["farm_id"].tolist()
    stp_ids = stp["stp_id"].tolist()

    stp_output = stp["daily_output_tons"].values.astype(float)
    stp_max = stp["storage_max_tons"].values.astype(float)
    stp_coords = stp[["lat", "lon"]].values.astype(float)

    farm_area = farm["area_ha"].values.astype(float)
    farm_coords = farm[["lat", "lon"]].values.astype(float)

    distances = np.array([[haversine(stp_coords[s, 0], stp_coords[s, 1],
                                     farm_coords[f, 0], farm_coords[f, 1])
                          for f in range(F)] for s in range(S)])

    zones = sorted(farm["zone"].unique())
    zone_map = {z: i for i, z in enumerate(zones)}
    farm_zone = np.array([zone_map[z] for z in farm["zone"]])

    weather["date"] = pd.to_datetime(weather["date"])
    weather = weather.set_index("date")

    rain_threshold = params["environmental_thresholds"]["rain_lock_threshold_mm"]
    rain_window = params["environmental_thresholds"]["forecast_window_days"]

    rain_locked = np.zeros((D, F), dtype=bool)
    for d, date in enumerate(dates):
        for f in range(F):
            zone = zones[farm_zone[f]]
            for i in range(rain_window):
                check = date + pd.Timedelta(days=i)
                if check > dates[-1]:
                    break
                try:
                    if weather.loc[check, zone] > rain_threshold:
                        rain_locked[d, f] = True
                        break
                except KeyError:
                    pass

    planting["plant_date"] = pd.to_datetime(planting["plant_date"])
    planting["harvest_date"] = pd.to_datetime(planting["harvest_date"])

    crop_active = np.zeros((D, F), dtype=bool)
    fid_map = {fid: idx for idx, fid in enumerate(farm_ids)}

    for _, row in planting.iterrows():
        if row["farm_id"] not in fid_map:
            continue
        f = fid_map[row["farm_id"]]
        mask = (dates >= row["plant_date"]) & (dates <= row["harvest_date"])
        crop_active[mask, f] = True

    demand_df = pd.read_csv("daily_n_demand.csv")
    demand_df["date"] = pd.to_datetime(demand_df["date"])
    demand_df = demand_df.set_index("date")

    n_demand_kg = np.zeros((D, F))
    for f, fid in enumerate(farm_ids):
        col = str(fid) if str(fid) in demand_df.columns else fid
        if col in demand_df.columns:
            vals = demand_df[col].reindex(dates).fillna(0).values
            n_demand_kg[:, f] = vals * farm_area[f]

    n_frac = params["agronomic_constants"]["nitrogen_content_kg_per_ton_biosolid"] / 1000
    buf_factor = params["agronomic_constants"]["application_buffer_percent"] / 100

    buffer_tons = (n_demand_kg * (1 + buf_factor)) / (n_frac * 1000.0)
    demand_tons = n_demand_kg / (n_frac * 1000.0)

    p_trans = params["logistics_constants"]["diesel_emission_factor_kg_co2_per_km"]
    transport_cost_per_trip = distances * 2.0 * p_trans
    closest_stp = np.argmin(distances, axis=0)

    data = ProblemData(
        num_days=D, num_farms=F, num_stps=S,
        farm_ids=farm_ids, stp_ids=stp_ids, dates=dates,
        stp_daily_output=stp_output, stp_storage_max=stp_max, stp_coords=stp_coords,
        farm_area=farm_area, farm_coords=farm_coords, farm_zone=farm_zone,
        zones=zones, distances=distances,
        rain_locked=rain_locked, crop_active=crop_active,
        n_demand_kg=n_demand_kg, buffer_tons=buffer_tons, demand_tons=demand_tons,
        params=params
    )
    data.transport_cost_per_trip = transport_cost_per_trip
    data.closest_stp = closest_stp

    return data


def compute_lookahead(data: ProblemData):
    D, F = data.num_days, data.num_farms
    
    future_locked = np.zeros((D, F))
    for f in range(F):
        for d in range(D - 2, -1, -1):
            if data.rain_locked[d + 1, f]:
                future_locked[d, f] = data.n_demand_kg[d + 1, f] + future_locked[d + 1, f]
            else:
                future_locked[d, f] = 0
    
    days_until_lock = np.full((D, F), 8, dtype=int)
    for d in range(D):
        for f in range(F):
            if data.rain_locked[d, f]:
                days_until_lock[d, f] = 0
            else:
                for ahead in range(1, min(8, D - d)):
                    if data.rain_locked[d + ahead, f]:
                        days_until_lock[d, f] = ahead
                        break
    
    return future_locked, days_until_lock


def greedy_solver(data: ProblemData, ta: float, ca: float, uw: float, ow: float) -> Tuple[np.ndarray, Dict, float]:
    """Optimized greedy solver."""
    D, F, S = data.num_days, data.num_farms, data.num_stps
    params = data.params
    truck = params["logistics_constants"]["truck_capacity_tons"]
    n_frac = params["agronomic_constants"]["nitrogen_content_kg_per_ton_biosolid"] / 1000
    
    c_synth = params["agronomic_constants"]["synthetic_n_offset_credit_kg_co2_per_kg_n"]
    c_soil = params["agronomic_constants"]["soil_organic_carbon_gain_kg_co2_per_kg_biosolid"]
    p_leach = params["agronomic_constants"]["leaching_penalty_kg_co2_per_kg_excess_n"]
    p_over = params["environmental_thresholds"]["stp_overflow_penalty_kg_co2_per_ton"]
    p_trans = params["logistics_constants"]["diesel_emission_factor_kg_co2_per_km"]
    
    future_locked, days_until_lock = compute_lookahead(data)
    
    delivered = np.zeros((D, F))
    records = {}
    storage = np.zeros(S)
    overflow_total = 0.0
    
    for d in range(D):
        available = storage + data.stp_daily_output
        storage_urgency = np.clip(available / data.stp_storage_max, 0, 1) ** 3
        
        options = []
        
        for f in range(F):
            if data.rain_locked[d, f]:
                continue
            
            buffer_t = data.buffer_tons[d, f]
            n_demand = data.n_demand_kg[d, f]
            
            for s in range(S):
                if available[s] < 0.1:
                    continue
                
                transport_cost = data.distances[s, f] * p_trans * ta
                
                if data.crop_active[d, f] and n_demand > 0:
                    max_delivery = min(buffer_t, available[s], truck)
                    if max_delivery > 0.1:
                        n_delivered = max_delivery * 1000 * n_frac
                        syn_credit = min(n_delivered, n_demand) * c_synth * ca
                        soil_credit = max_delivery * 1000 * c_soil * ca
                        
                        opp_bonus = 0
                        if days_until_lock[d, f] > 0 and days_until_lock[d, f] <= 3:
                            opp_bonus = future_locked[d, f] * c_synth * 0.3 * ow
                        
                        urgency_bonus = storage_urgency[s] * p_over * uw
                        
                        value = syn_credit + soil_credit + opp_bonus + urgency_bonus - transport_cost
                        options.append((value / max_delivery, s, f, buffer_t, True))
                
                # Excess
                n_per_ton = 1000 * n_frac
                soil_per_ton = 1000 * c_soil * ca
                leach_per_ton = n_per_ton * p_leach
                urgency_bonus = storage_urgency[s] * p_over * uw
                value = soil_per_ton + urgency_bonus - leach_per_ton - transport_cost / truck
                options.append((value, s, f, truck, False))
        
        options.sort(reverse=True, key=lambda x: x[0])
        
        remaining = available.copy()
        day_delivery = defaultdict(float)
        day_stp = {}
        
        for value_per_ton, s, f, max_tons, is_crop in options:
            if remaining[s] < 0.01:
                continue
            
            current = day_delivery[f]
            avail_at_farm = max(0, max_tons - current)
            if avail_at_farm < 0.01:
                continue
            
            delivery = min(avail_at_farm, remaining[s])
            if delivery < 0.01:
                continue
            
            day_delivery[f] += delivery
            if f not in day_stp:
                day_stp[f] = s
            remaining[s] -= delivery
        
        for f, tons in day_delivery.items():
            if tons > 0.001:
                delivered[d, f] = tons
                records[(d, f)] = (day_stp[f], tons)
        
        for s in range(S):
            if remaining[s] > data.stp_storage_max[s]:
                overflow_total += remaining[s] - data.stp_storage_max[s]
                storage[s] = data.stp_storage_max[s]
            else:
                storage[s] = max(0.0, remaining[s])
    
    return delivered, records, overflow_total


def score_solution(delivered: np.ndarray, records: Dict, data: ProblemData, overflow: float) -> Dict:
    D, F = data.num_days, data.num_farms
    params = data.params

    n_frac = params["agronomic_constants"]["nitrogen_content_kg_per_ton_biosolid"] / 1000
    buf_factor = params["agronomic_constants"]["application_buffer_percent"] / 100
    truck = params["logistics_constants"]["truck_capacity_tons"]

    c_soil = params["agronomic_constants"]["soil_organic_carbon_gain_kg_co2_per_kg_biosolid"]
    c_synth = params["agronomic_constants"]["synthetic_n_offset_credit_kg_co2_per_kg_n"]
    p_leach = params["agronomic_constants"]["leaching_penalty_kg_co2_per_kg_excess_n"]
    p_trans = params["logistics_constants"]["diesel_emission_factor_kg_co2_per_km"]
    p_over = params["environmental_thresholds"]["stp_overflow_penalty_kg_co2_per_ton"]

    syn, soil, trans, leach = 0.0, 0.0, 0.0, 0.0

    for d in range(D):
        for f in range(F):
            tons = delivered[d, f]
            if tons <= 0:
                continue

            kg = tons * 1000
            n_kg = kg * n_frac

            soil += kg * c_soil

            if data.crop_active[d, f] and data.n_demand_kg[d, f] > 0:
                syn += min(n_kg, data.n_demand_kg[d, f]) * c_synth
                excess_n = max(0, n_kg - data.n_demand_kg[d, f] * (1 + buf_factor))
                leach += excess_n * p_leach
            else:
                leach += n_kg * p_leach

            if (d, f) in records:
                si, st = records[(d, f)]
                trucks = ceil(st / truck) if st > 0 else 0
                if trucks == 0 and st > 0:
                    trucks = 1
                trans += trucks * (data.distances[si, f] * 2.0) * p_trans

    return {
        "net_score": syn + soil - trans - leach - overflow * p_over,
        "synthetic": syn,
        "soil": soil,
        "transport": trans,
        "leaching": leach,
        "overflow": overflow * p_over,
        "overflow_tons": overflow,
        "delivered_tons": delivered.sum()
    }


def grid_search(data: ProblemData) -> Tuple[Dict, float, np.ndarray, Dict, float]:
    """Fast grid search - focused on best known region."""
    best_score = -np.inf
    best_params = None
    best_delivered = None
    best_records = None
    best_overflow = 0
    
    # Focused configs around BEST (ta=2.10, ca=1.18, uw=0.64, ow=0.0)
    configs = [
        # New best region
        (2.10, 1.18, 0.64, 0.0),
        (2.10, 1.18, 0.60, 0.0),
        (2.10, 1.18, 0.68, 0.0),
        (2.08, 1.18, 0.64, 0.0),
        (2.12, 1.18, 0.64, 0.0),
        (2.10, 1.16, 0.64, 0.0),
        (2.10, 1.20, 0.64, 0.0),
        (2.10, 1.18, 0.55, 0.0),
        (2.10, 1.18, 0.70, 0.0),
        (2.10, 1.19, 0.63, 0.0),
        (2.10, 1.17, 0.65, 0.0),
        (2.09, 1.18, 0.64, 0.0),
        (2.11, 1.18, 0.64, 0.0),
        (2.10, 1.18, 0.62, 0.0),
    ]
    
    print(f"  Testing {len(configs)} focused configurations...")
    
    for i, (ta, ca, uw, ow) in enumerate(configs):
        delivered, records, overflow = greedy_solver(data, ta, ca, uw, ow)
        score = score_solution(delivered, records, data, overflow)['net_score']
        
        if score > best_score:
            best_score = score
            best_params = {"transport_aversion": ta, "credit_aggression": ca, 
                          "urgency_weight": uw, "opportunity_weight": ow}
            best_delivered = delivered
            best_records = records
            best_overflow = overflow
            print(f"  [{i+1}/{len(configs)}] New best: {score:,.0f} (ta={ta}, ca={ca}, uw={uw}, ow={ow})")
    
    return best_params, best_score, best_delivered, best_records, best_overflow


def random_search(data: ProblemData, best_params: Dict, best_score: float, 
                  best_delivered: np.ndarray, best_records: Dict, best_overflow: float,
                  n_iter: int = 50) -> Tuple[Dict, float, np.ndarray, Dict, float]:
    """Random search around best parameters with adaptive step size."""
    print(f"  Random search ({n_iter} iterations)...")
    
    no_improvement_count = 0
    step_size = 0.2
    
    for i in range(n_iter):
        # Reduce step size if stuck
        if no_improvement_count > 10:
            step_size = max(0.05, step_size * 0.8)
            no_improvement_count = 0
        
        ta = best_params["transport_aversion"] + random.uniform(-step_size, step_size)
        ca = best_params["credit_aggression"] + random.uniform(-step_size*0.5, step_size*0.5)
        uw = best_params["urgency_weight"] + random.uniform(-step_size, step_size)
        ow = max(0, best_params["opportunity_weight"] + random.uniform(-step_size*0.5, step_size*0.5))
        
        ta = max(0.5, min(5.0, ta))
        ca = max(0.5, min(2.0, ca))
        uw = max(0.3, min(3.0, uw))
        ow = max(0.0, min(2.0, ow))
        
        delivered, records, overflow = greedy_solver(data, ta, ca, uw, ow)
        score = score_solution(delivered, records, data, overflow)['net_score']
        
        if score > best_score:
            best_score = score
            best_params = {"transport_aversion": ta, "credit_aggression": ca,
                          "urgency_weight": uw, "opportunity_weight": ow}
            best_delivered = delivered
            best_records = records
            best_overflow = overflow
            no_improvement_count = 0
            print(f"    [{i+1}] New best: {score:,.0f} (ta={ta:.3f}, ca={ca:.3f}, uw={uw:.3f}, ow={ow:.3f})")
        else:
            no_improvement_count += 1
    
    return best_params, best_score, best_delivered, best_records, best_overflow


def validate_solution(delivered: np.ndarray, records: Dict, data: ProblemData) -> Dict:
    rain_violations = 0.0
    for d in range(data.num_days):
        for f in range(data.num_farms):
            if delivered[d, f] > 0 and data.rain_locked[d, f]:
                rain_violations += delivered[d, f]
    return {"rain_violations": rain_violations}


def save_solution(delivered: np.ndarray, records: Dict, data: ProblemData, score_dict: Dict):
    template = pd.read_csv("sample_submission.csv")

    lookup = {}
    for d in range(data.num_days):
        date_str = data.dates[d].strftime("%Y-%m-%d")
        for f in range(data.num_farms):
            farm_id = str(data.farm_ids[f])
            tons = delivered[d, f]

            if (d, f) in records:
                si, _ = records[(d, f)]
                stp_id = str(data.stp_ids[si])
            else:
                s_closest = np.argmin(data.distances[:, f])
                stp_id = str(data.stp_ids[s_closest])

            lookup[(date_str, farm_id)] = (stp_id, tons)

    rows = []
    for _, row in template.iterrows():
        date_str = row["date"]
        farm_id = str(row["farm_id"])

        if (date_str, farm_id) in lookup:
            stp_id, tons = lookup[(date_str, farm_id)]
        else:
            s_closest = 0
            stp_id = str(data.stp_ids[s_closest])
            tons = 0.0

        rows.append({
            "id": row["id"],
            "date": date_str,
            "stp_id": stp_id,
            "farm_id": farm_id,
            "tons_delivered": round(tons, 6)
        })

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/solution.csv", index=False)
    df.to_csv("submission.csv", index=False)

    with open("outputs/summary_metrics.json", "w") as f:
        json.dump({
            "net_carbon_credit_score": float(score_dict["net_score"]),
            "synthetic_offset": float(score_dict["synthetic"]),
            "soil_sequestration": float(score_dict["soil"]),
            "transport_emissions": float(score_dict["transport"]),
            "nitrogen_leaching": float(score_dict["leaching"]),
            "stp_overflow": float(score_dict["overflow"]),
            "delivered_tons": float(score_dict["delivered_tons"]),
            "overflow_tons": float(score_dict["overflow_tons"])
        }, f, indent=2)


def main():
    print("\n" + "="*70)
    print("   FAST MEGA SOLVER v17")
    print("="*70)

    data = load_all_data()

    print(f"\n[DATA]")
    print(f"  Days: {data.num_days}, Farms: {data.num_farms}, STPs: {data.num_stps}")
    print(f"  Daily production: {data.stp_daily_output.sum():.0f} tons")
    print(f"  Rain-locked: {data.rain_locked.sum():,} ({100*data.rain_locked.sum()/(data.num_days*data.num_farms):.1f}%)")

    print("\n[STEP 1] Grid Search")
    best_params, best_score, best_delivered, best_records, best_overflow = grid_search(data)
    
    print(f"\n[STEP 2] Random Refinement")
    best_params, best_score, best_delivered, best_records, best_overflow = random_search(
        data, best_params, best_score, best_delivered, best_records, best_overflow, n_iter=200
    )
    
    final_score = score_solution(best_delivered, best_records, data, best_overflow)
    v = validate_solution(best_delivered, best_records, data)

    print("\n" + "="*70)
    print("   FINAL RESULTS")
    print("="*70)
    print(f"\n  NET SCORE: {final_score['net_score']:,.0f} kg CO2-eq")
    print(f"\n  Best params: {best_params}")
    print(f"\n  Delivered: {final_score['delivered_tons']:,.0f} tons")
    print(f"  Overflow:  {final_score['overflow_tons']:.0f} tons")
    print(f"  Rain violations: {v['rain_violations']:.0f}t")
    print(f"\n  (+) Synthetic:  {final_score['synthetic']:>15,.0f}")
    print(f"  (+) Soil:       {final_score['soil']:>15,.0f}")
    print(f"  (-) Transport:  {final_score['transport']:>15,.0f}")
    print(f"  (-) Leaching:   {final_score['leaching']:>15,.0f}")
    print(f"  (-) Overflow:   {final_score['overflow']:>15,.0f}")

    save_solution(best_delivered, best_records, data, final_score)
    print(f"\n  Saved: outputs/solution.csv, submission.csv")
    print("="*70 + "\n")

    return final_score['net_score']


if __name__ == "__main__":
    main()
