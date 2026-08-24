"""
Simulation-Based Phase Optimization Dataset Generator
Generates comprehensive state-action training samples with high sensitivity
to single-approach spawn rates and directional queue buildup.
General Sir John Kotelawala Defence University (KDU) - IT 3182 Essentials of AI
"""

import sys
import os
import random
import math
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def generate_phase_dataset(n_samples: int = 6000, output_csv: str = None) -> pd.DataFrame:
    np.random.seed(42)
    random.seed(42)

    data = []

    for i in range(n_samples):
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        time_decimal = hour + minute / 60.0
        
        is_morning_rush = (7.0 <= time_decimal <= 9.5)
        is_evening_rush = (16.5 <= time_decimal <= 19.5)

        # Baseline intervals
        if is_morning_rush:
            int_N = np.random.uniform(0.4, 1.2)
            int_E = np.random.uniform(0.5, 1.3)
            int_S = np.random.uniform(2.2, 5.0)
            int_W = np.random.uniform(2.2, 5.0)
        elif is_evening_rush:
            int_N = np.random.uniform(2.2, 5.0)
            int_E = np.random.uniform(2.2, 5.0)
            int_S = np.random.uniform(0.4, 1.2)
            int_W = np.random.uniform(0.5, 1.3)
        else:
            spike = random.random() < 0.40
            if spike:
                spiked_dir = random.choice(['N', 'S', 'E', 'W'])
                int_N = np.random.uniform(0.4, 1.2) if spiked_dir == 'N' else np.random.uniform(1.8, 3.5)
                int_S = np.random.uniform(0.4, 1.2) if spiked_dir == 'S' else np.random.uniform(1.8, 3.5)
                int_E = np.random.uniform(0.4, 1.2) if spiked_dir == 'E' else np.random.uniform(1.8, 3.5)
                int_W = np.random.uniform(0.4, 1.2) if spiked_dir == 'W' else np.random.uniform(1.8, 3.5)
            else:
                int_N = np.random.uniform(1.5, 3.0)
                int_S = np.random.uniform(1.5, 3.0)
                int_E = np.random.uniform(1.5, 3.0)
                int_W = np.random.uniform(1.5, 3.0)

        flow_N = 60.0 / int_N
        flow_S = 60.0 / int_S
        flow_E = 60.0 / int_E
        flow_W = 60.0 / int_W

        pcu_N_L0 = round(np.random.gamma(shape=max(0.5, flow_N * 0.08), scale=1.4), 1)
        pcu_N_L1 = round(np.random.gamma(shape=max(0.5, flow_N * 0.12), scale=1.4), 1)

        pcu_S_L0 = round(np.random.gamma(shape=max(0.5, flow_S * 0.08), scale=1.4), 1)
        pcu_S_L1 = round(np.random.gamma(shape=max(0.5, flow_S * 0.12), scale=1.4), 1)

        pcu_E_L0 = round(np.random.gamma(shape=max(0.5, flow_E * 0.08), scale=1.4), 1)
        pcu_E_L1 = round(np.random.gamma(shape=max(0.5, flow_E * 0.12), scale=1.4), 1)

        pcu_W_L0 = round(np.random.gamma(shape=max(0.5, flow_W * 0.08), scale=1.4), 1)
        pcu_W_L1 = round(np.random.gamma(shape=max(0.5, flow_W * 0.12), scale=1.4), 1)

        tot_pcu_N = pcu_N_L0 + pcu_N_L1
        tot_pcu_S = pcu_S_L0 + pcu_S_L1
        tot_pcu_E = pcu_E_L0 + pcu_E_L1
        tot_pcu_W = pcu_W_L0 + pcu_W_L1

        max_wait = round(max(tot_pcu_N, tot_pcu_S, tot_pcu_E, tot_pcu_W) * np.random.uniform(1.0, 2.2), 1)

        active_axis = 'NS' if (tot_pcu_N + tot_pcu_S + flow_N + flow_S) >= (tot_pcu_E + tot_pcu_W + flow_E + flow_W) else 'EW'

        if active_axis == 'NS':
            flow_ratio = flow_N / (flow_S + 0.1) if flow_N >= flow_S else flow_S / (flow_N + 0.1)
            heavy_dir = 'N' if (flow_N >= flow_S or tot_pcu_N >= tot_pcu_S) else 'S'
            heavy_pcu = max(tot_pcu_N, tot_pcu_S)
            light_pcu = min(tot_pcu_N, tot_pcu_S)
            heavy_flow = max(flow_N, flow_S)
            light_flow = min(flow_N, flow_S)
            asym_ratio = (heavy_pcu + heavy_flow * 0.2) / (light_pcu + light_flow * 0.2 + 0.5)

            # High sensitivity surge detection: flow >= 42 v/m or interval <= 1.4s or ratio >= 1.4
            if (heavy_flow >= 45.0 or asym_ratio >= 1.4 or int_N <= 1.4 or int_S <= 1.4):
                optimal_policy = 1 if heavy_dir == 'N' else 2
                policy_name = f"{heavy_dir}_EXCLUSIVE_GREEN"
            else:
                optimal_policy = 0
                policy_name = "BALANCED_PHASE"
            
            target_pcu = heavy_pcu
            target_flow = heavy_flow

        else:
            heavy_dir = 'E' if (flow_E >= flow_W or tot_pcu_E >= tot_pcu_W) else 'W'
            heavy_pcu = max(tot_pcu_E, tot_pcu_W)
            light_pcu = min(tot_pcu_E, tot_pcu_W)
            heavy_flow = max(flow_E, flow_W)
            light_flow = min(flow_E, flow_W)
            asym_ratio = (heavy_pcu + heavy_flow * 0.2) / (light_pcu + light_flow * 0.2 + 0.5)

            if (heavy_flow >= 45.0 or asym_ratio >= 1.4 or int_E <= 1.4 or int_W <= 1.4):
                optimal_policy = 3 if heavy_dir == 'E' else 4
                policy_name = f"{heavy_dir}_EXCLUSIVE_GREEN"
            else:
                optimal_policy = 0
                policy_name = "BALANCED_PHASE"

            target_pcu = heavy_pcu
            target_flow = heavy_flow

        # Optimal Green Time Regression
        raw_green = 5.0 + (target_pcu * 2.0) + (target_flow * 0.18) + (max_wait * 0.10)
        optimal_green_sec = round(max(10.0, min(45.0, raw_green)), 1)

        row = {
            "time_hour": time_decimal,
            "sin_time": round(math.sin(2 * math.pi * time_decimal / 24.0), 4),
            "cos_time": round(math.cos(2 * math.pi * time_decimal / 24.0), 4),
            "is_rush_hour": int(is_morning_rush or is_evening_rush),
            "flow_N": round(flow_N, 1),
            "flow_S": round(flow_S, 1),
            "flow_E": round(flow_E, 1),
            "flow_W": round(flow_W, 1),
            "pcu_N_L0": pcu_N_L0,
            "pcu_N_L1": pcu_N_L1,
            "pcu_S_L0": pcu_S_L0,
            "pcu_S_L1": pcu_S_L1,
            "pcu_E_L0": pcu_E_L0,
            "pcu_E_L1": pcu_E_L1,
            "pcu_W_L0": pcu_W_L0,
            "pcu_W_L1": pcu_W_L1,
            "tot_pcu_N": tot_pcu_N,
            "tot_pcu_S": tot_pcu_S,
            "tot_pcu_E": tot_pcu_E,
            "tot_pcu_W": tot_pcu_W,
            "max_wait_sec": max_wait,
            "active_axis": active_axis,
            "asym_ratio": round(asym_ratio, 2),
            "optimal_policy_code": optimal_policy,
            "optimal_policy_name": policy_name,
            "optimal_green_sec": optimal_green_sec
        }
        data.append(row)

    df = pd.DataFrame(data)
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Generated {len(df)} samples saved to: {output_csv}")
    return df

if __name__ == "__main__":
    csv_path = os.path.join(PROJECT_ROOT, "data", "datasets", "phase_optimization_dataset.csv")
    df = generate_phase_dataset(n_samples=6000, output_csv=csv_path)
    print("\nDataset Class Distribution:")
    print(df["optimal_policy_name"].value_counts())
