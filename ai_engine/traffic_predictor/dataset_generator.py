"""
Historical Traffic Dataset Generator with Directional Tidal Flow
Generates high-fidelity time-series traffic datasets modeling directional commuter tidal flows.
"""

import os
import sys
import random
import datetime
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def generate_traffic_dataset(output_csv_path: str, num_days: int = 60):
    print(f"Generating high-fidelity traffic dataset ({num_days} days, 15-min intervals)...")
    
    start_date = datetime.datetime(2026, 6, 1, 0, 0)
    total_intervals = num_days * 24 * 4
    
    records = []
    
    prev_pcu_N = 6.0
    prev_pcu_S = 6.0
    prev_pcu_E = 6.0
    prev_pcu_W = 6.0

    for i in range(total_intervals):
        current_dt = start_date + datetime.timedelta(minutes=15 * i)
        
        hour = current_dt.hour
        minute = current_dt.minute
        day_of_week = current_dt.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Cyclical temporal encodings
        sin_hour = np.sin(2 * np.pi * (hour + minute / 60.0) / 24.0)
        cos_hour = np.cos(2 * np.pi * (hour + minute / 60.0) / 24.0)
        sin_day = np.sin(2 * np.pi * day_of_week / 7.0)
        cos_day = np.cos(2 * np.pi * day_of_week / 7.0)

        # Weather simulation
        weather_roll = random.random()
        if weather_roll < 0.82:
            weather = "Clear"
            weather_mult = 1.0
        elif weather_roll < 0.95:
            weather = "Rain"
            weather_mult = 1.25
        else:
            weather = "Heavy Rain"
            weather_mult = 1.50

        # --- DIRECTIONAL TIDAL FLOW MODELING ---
        is_morning_rush = (7 <= hour <= 9) and not is_weekend
        is_evening_rush = (16 <= hour <= 19) and not is_weekend
        is_midday = (11 <= hour <= 15)
        is_night = (hour >= 23 or hour <= 5)

        if is_weekend:
            if 10 <= hour <= 17:
                base_N = 45.0 + 8.0 * np.sin(hour)
                base_S = 45.0 + 8.0 * np.cos(hour)
                base_E = 40.0 + 6.0 * np.sin(hour)
                base_W = 40.0 + 6.0 * np.cos(hour)
            elif is_night:
                base_N = 6.0
                base_S = 6.0
                base_E = 5.0
                base_W = 5.0
            else:
                base_N = 22.0
                base_S = 22.0
                base_E = 20.0
                base_W = 20.0
        else:
            if is_morning_rush:
                # Inbound tidal wave: North & East heavy, South & West light
                peak_factor = 1.0 + 0.3 * (1.0 - abs(hour - 8.0))
                base_N = 110.0 * peak_factor
                base_E = 85.0 * peak_factor
                base_S = 18.0
                base_W = 18.0
            elif is_evening_rush:
                # Outbound tidal wave: South & West heavy, North & East light
                peak_factor = 1.0 + 0.3 * (1.0 - abs(hour - 17.5) / 1.5)
                base_S = 110.0 * peak_factor
                base_W = 85.0 * peak_factor
                base_N = 18.0
                base_E = 18.0
            elif is_midday:
                base_N = 40.0
                base_S = 40.0
                base_E = 35.0
                base_W = 35.0
            elif is_night:
                base_N = 6.0
                base_S = 6.0
                base_E = 5.0
                base_W = 5.0
            else:
                base_N = 25.0
                base_S = 25.0
                base_E = 22.0
                base_W = 22.0

        # Apply noise & weather multiplier
        flow_N = round(base_N * weather_mult * random.uniform(0.95, 1.05), 1)
        flow_S = round(base_S * weather_mult * random.uniform(0.95, 1.05), 1)
        flow_E = round(base_E * weather_mult * random.uniform(0.95, 1.05), 1)
        flow_W = round(base_W * weather_mult * random.uniform(0.95, 1.05), 1)

        pcu_N = round(flow_N * 0.22, 1)
        pcu_S = round(flow_S * 0.22, 1)
        pcu_E = round(flow_E * 0.22, 1)
        pcu_W = round(flow_W * 0.22, 1)

        total_pcu = pcu_N + pcu_S + pcu_E + pcu_W

        def get_cong_level(pcu):
            if pcu < 6.0: return "LOW"
            elif pcu < 14.0: return "MEDIUM"
            elif pcu < 24.0: return "HIGH"
            else: return "VERY_HIGH"

        pcu_dict = {'N': pcu_N, 'S': pcu_S, 'E': pcu_E, 'W': pcu_W}
        critical_direction = max(pcu_dict, key=pcu_dict.get)

        flow_NS = max(flow_N, flow_S) + (0.35 * min(flow_N, flow_S))
        flow_EW = max(flow_E, flow_W) + (0.35 * min(flow_E, flow_W))
        total_flow = flow_NS + flow_EW + 1e-5
        
        ratio_NS = flow_NS / total_flow
        ratio_EW = flow_EW / total_flow
        
        target_green_NS = max(8.0, min(42.0, round(52.0 * ratio_NS, 1)))
        target_green_EW = max(8.0, min(42.0, round(52.0 * ratio_EW, 1)))

        records.append({
            "timestamp": current_dt.strftime("%Y-%m-%d %H:%M"),
            "hour": hour,
            "minute": minute,
            "day_of_week": day_of_week,
            "sin_hour": round(sin_hour, 4),
            "cos_hour": round(cos_hour, 4),
            "sin_day": round(sin_day, 4),
            "cos_day": round(cos_day, 4),
            "is_weekend": is_weekend,
            "is_morning_rush": 1 if is_morning_rush else 0,
            "is_evening_rush": 1 if is_evening_rush else 0,
            "weather": weather,
            "weather_multiplier": weather_mult,
            "lag_pcu_N": prev_pcu_N,
            "lag_pcu_S": prev_pcu_S,
            "lag_pcu_E": prev_pcu_E,
            "lag_pcu_W": prev_pcu_W,
            "flow_north_vpm": flow_N,
            "flow_south_vpm": flow_S,
            "flow_east_vpm": flow_E,
            "flow_west_vpm": flow_W,
            "pcu_north": pcu_N,
            "pcu_south": pcu_S,
            "pcu_east": pcu_E,
            "pcu_west": pcu_W,
            "total_intersection_pcu": round(total_pcu, 1),
            "congestion_north": get_cong_level(pcu_N),
            "congestion_south": get_cong_level(pcu_S),
            "congestion_east": get_cong_level(pcu_E),
            "congestion_west": get_cong_level(pcu_W),
            "critical_approach": critical_direction,
            "target_green_NS": target_green_NS,
            "target_green_EW": target_green_EW
        })

        prev_pcu_N = pcu_N
        prev_pcu_S = pcu_S
        prev_pcu_E = pcu_E
        prev_pcu_W = pcu_W

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Dataset generated successfully with {len(df)} records saved to: {output_csv_path}")
    return df

if __name__ == "__main__":
    csv_path = os.path.join(
        PROJECT_ROOT, "data", "datasets", "historical_traffic_data.csv"
    )
    generate_traffic_dataset(csv_path, num_days=60)
