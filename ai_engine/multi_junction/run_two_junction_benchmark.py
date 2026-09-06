"""
5-Hour Multi-Scenario Scientific Benchmark for 2-Junction Connected System
General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI — Group 06

Compares 3 Traffic Management Methods across 5 distinct real-world traffic scenarios:
- Mode 1: Traditional Fixed-Time Controller (Baseline)
- Mode 2: Dynamic Fuzzy-ML Cycle Optimizer
- Mode 3: Deep Reinforcement Learning (DQN Agent)

Scenarios (1 Hour Each = 5 Simulated Hours / 18,000s Total):
- Hour 1: Balanced Moderate Baseline Flow (30 - 45 v/m)
- Hour 2: Morning Arterial Rush (Heavy Eastbound J1->J2 Surge, 100 - 120 v/m)
- Hour 3: Cross-Street Surge & Heavy Left-Turn Conflict (North/South Inflow, 80 - 100 v/m)
- Hour 4: Evening Reverse Commute (Heavy Westbound J2->J1 Surge, 110 - 130 v/m)
- Hour 5: Extreme Dynamic Spikes & Emergency Ambulance Injections
"""

import os
import sys
import math
import random
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['SDL_VIDEODRIVER'] = 'dummy'

from ai_engine.multi_junction.junction_node import JunctionNode
from ai_engine.multi_junction.two_junction_vehicle import TwoJunctionVehicle, VEHICLE_CONFIGS
from ai_engine.multi_junction.two_junction_coordinator import TwoJunctionCoordinator


class FastTwoJunctionBenchmark:
    def __init__(self, mode_id: int):
        self.mode_id = mode_id
        self.mode_name = {1: "Fixed-Time", 2: "Fuzzy-ML (I2I Coordinated)", 3: "Deep RL (I2I Coordinated)"}[mode_id]

        self.width = 1440
        self.height = 760

        self.j1 = JunctionNode(1, "J1: West", cx=400, cy=400)
        self.j2 = JunctionNode(2, "J2: East", cx=1040, cy=400)
        self.junctions = {1: self.j1, 2: self.j2}
        for j in self.junctions.values():
            j.set_mode(mode_id)

        self.coordinator = TwoJunctionCoordinator(self.junctions, enabled=(mode_id in (2, 3)))

        self.vehicles = []
        self.vehicle_id_counter = 1

        self.cleared_vehicles = 0
        self.cleared_pcu = 0.0
        self.cleared_waits = []

        self.inflow_keys = ["J1_N", "J1_S", "J1_W", "J2_N", "J2_S", "J2_E"]
        self.inflow_specs = {
            "J1_N": (1, 'S'),
            "J1_S": (1, 'N'),
            "J1_W": (1, 'E'),
            "J2_N": (2, 'S'),
            "J2_S": (2, 'N'),
            "J2_E": (2, 'W')
        }
        self.spawn_timers = {k: random.uniform(0.0, 1.5) for k in self.inflow_keys}

    def get_scenario_intervals(self, hour: int) -> dict:
        """Defines traffic arrival intervals (seconds between vehicles) per approach for each scenario."""
        if hour == 1:
            # Scenario 1: Balanced Moderate Baseline (30-45 v/m)
            return {
                "J1_N": 2.2, "J1_S": 2.2, "J1_W": 1.8,
                "J2_N": 2.2, "J2_S": 2.2, "J2_E": 1.8
            }
        elif hour == 2:
            # Scenario 2: Morning Arterial Rush (Heavy Eastbound J1->J2 Surge, ~110 v/m)
            return {
                "J1_N": 3.0, "J1_S": 3.0, "J1_W": 0.55,
                "J2_N": 3.2, "J2_S": 3.2, "J2_E": 2.8
            }
        elif hour == 3:
            # Scenario 3: Cross-Street Surge (Heavy N/S traffic on both junctions, ~90 v/m)
            return {
                "J1_N": 0.70, "J1_S": 0.75, "J1_W": 2.5,
                "J2_N": 0.70, "J2_S": 0.75, "J2_E": 2.5
            }
        elif hour == 4:
            # Scenario 4: Evening Reverse Commute (Heavy Westbound J2->J1 Surge, ~120 v/m)
            return {
                "J1_N": 3.2, "J1_S": 3.2, "J1_W": 2.8,
                "J2_N": 3.0, "J2_S": 3.0, "J2_E": 0.50
            }
        else:
            # Scenario 5: Extreme Asymmetric Dynamic Spikes & Emergency Ambulance Injections
            return {
                "J1_N": 0.85, "J1_S": 2.4, "J1_W": 0.65,
                "J2_N": 2.4,  "J2_S": 0.85, "J2_E": 1.2
            }

    def spawn_vehicle(self, inflow_key: str, is_ambulance: bool = False):
        j_id, direction = self.inflow_specs[inflow_key]
        j = self.junctions[j_id]

        lane_idx = 1 if is_ambulance else random.choices([0, 1], weights=[0.35, 0.65])[0]
        lane_offset = 20 if lane_idx == 0 else 60

        if direction == 'S': spawn_x, spawn_y = j.cx - lane_offset, -35.0
        elif direction == 'N': spawn_x, spawn_y = j.cx + lane_offset, self.height + 35.0
        elif direction == 'E': spawn_x, spawn_y = -35.0, j.cy + lane_offset
        elif direction == 'W': spawn_x, spawn_y = self.width + 35.0, j.cy - lane_offset
        else: spawn_x, spawn_y = 0, 0

        # Collision check with existing vehicles near entryway
        for ov in self.vehicles:
            if math.hypot(ov.x - spawn_x, ov.y - spawn_y) < 45.0:
                return

        v_type = "ambulance" if is_ambulance else random.choices(
            ["car", "van", "bus", "truck", "motorcycle"],
            weights=[0.60, 0.15, 0.08, 0.07, 0.10]
        )[0]

        v = TwoJunctionVehicle(
            vehicle_id=self.vehicle_id_counter,
            vehicle_type=v_type,
            direction=direction,
            lane_idx=lane_idx,
            spawn_pos=(spawn_x, spawn_y),
            target_j_id=j_id
        )
        self.vehicle_id_counter += 1
        self.vehicles.append(v)

    def step(self, dt: float, current_intervals: dict, inject_ambulance: bool = False):
        # 1. Spawn vehicles according to current scenario intervals
        for k in self.inflow_keys:
            self.spawn_timers[k] += dt
            if self.spawn_timers[k] >= current_intervals[k]:
                self.spawn_timers[k] = 0.0
                self.spawn_vehicle(k)

        if inject_ambulance:
            self.spawn_vehicle("J1_W", is_ambulance=True)

        # 2. Update Junctions
        for j in self.junctions.values():
            j.update(dt, self.vehicles)

        # 3. Inter-Intersection (I2I) Green Wave Coordinator
        if self.mode_id in (2, 3):
            self.coordinator.update(dt, self.vehicles)

        # 4. Arterial Corridor Handoff
        for v in self.vehicles:
            if v.has_cleared_intersection and not v.is_turning:
                if v.direction == 'E' and v.current_junction_id == 1 and v.x < self.j2.cx - 85:
                    v.current_junction_id = 2
                    v.has_cleared_intersection = False
                elif v.direction == 'W' and v.current_junction_id == 2 and v.x > self.j1.cx + 85:
                    v.current_junction_id = 1
                    v.has_cleared_intersection = False

        # 4. Update Vehicles with Lane Ordering
        lane_buckets = {}
        for v in self.vehicles:
            if not v.is_turning:
                key = (v.current_junction_id, v.direction, v.lane_idx)
                lane_buckets.setdefault(key, []).append(v)

        for key, v_list in lane_buckets.items():
            j_id, d, lane_idx = key
            if d == 'E': v_list.sort(key=lambda v: -v.x)
            elif d == 'W': v_list.sort(key=lambda v: v.x)
            elif d == 'S': v_list.sort(key=lambda v: -v.y)
            elif d == 'N': v_list.sort(key=lambda v: v.y)

            for i, v in enumerate(v_list):
                leading_v = v_list[i - 1] if i > 0 else None
                j = self.junctions.get(v.current_junction_id, None)
                sig_th, sig_lt = j.signals.get_signals_for_direction(v.direction) if j else ('GREEN', 'GREEN')
                v.update(dt, leading_v, sig_th, sig_lt, j)

        for v in self.vehicles:
            if v.is_turning:
                j = self.junctions.get(v.current_junction_id, None)
                v.update(dt, None, 'GREEN', 'GREEN', j)

        # 5. Remove off-screen vehicles
        for v in list(self.vehicles):
            if v.is_off_screen(self.width, self.height):
                self.cleared_vehicles += 1
                self.cleared_pcu += v.pcu
                self.cleared_waits.append(v.wait_time)
                self.vehicles.remove(v)


def run_5hour_benchmark():
    print("=" * 80)
    print("  5-HOUR 2-JUNCTION BENCHMARK: FIXED-TIME vs FUZZY-ML vs DEEP RL (DQN)")
    print("  KDU IT 3182 Essentials of AI — Group 06")
    print("=" * 80)

    # 5 simulated hours = 18,000 simulated seconds (300 minutes)
    total_sim_minutes = 300
    dt = 0.10  # 10Hz fast-forwarded physics simulation (equivalent to 1,800,000 micro-steps)
    steps_per_minute = int(60.0 / dt)

    modes = [1, 2, 3]
    mode_names = {1: "Fixed-Time", 2: "Fuzzy-ML", 3: "Deep RL"}
    all_results = {}
    time_series_data = []

    for mode_id in modes:
        mode_label = mode_names[mode_id]
        print(f"\n[BENCHMARK] Running 5-Hour Simulation for Mode {mode_id}: {mode_label}...")
        start_real_time = time.time()

        sim = FastTwoJunctionBenchmark(mode_id)
        minute_stats = []

        for minute in range(1, total_sim_minutes + 1):
            hour = (minute - 1) // 60 + 1
            intervals = sim.get_scenario_intervals(hour)

            # Periodic ambulance injection in Scenario 5 (every 10 minutes)
            inject_ambulance = (hour == 5 and minute % 10 == 0)

            # Run 1 minute of simulation
            for step_idx in range(steps_per_minute):
                sim.step(dt, intervals, inject_ambulance=(inject_ambulance and step_idx == 0))

            # Record 1-minute metrics
            recent_waits = sim.cleared_waits[-150:] if sim.cleared_waits else [0.0]
            current_awt = float(np.mean(recent_waits)) if recent_waits else 0.0
            current_max_wait = float(max((v.wait_time for v in sim.vehicles), default=0.0))
            active_vehs = len(sim.vehicles)

            # Scenario description
            scenario_name = {
                1: "1: Balanced Baseline",
                2: "2: Morning Arterial Rush (EB)",
                3: "3: Cross-Street Surge (N/S)",
                4: "4: Evening Reverse Commute (WB)",
                5: "5: Dynamic Asymmetric + Emergency"
            }[hour]

            record = {
                "Minute": minute,
                "Hour": hour,
                "Scenario": scenario_name,
                "Mode_ID": mode_id,
                "Mode_Name": mode_label,
                "AWT_sec": round(current_awt, 2),
                "Max_Wait_sec": round(current_max_wait, 2),
                "Active_Vehicles": active_vehs,
                "Cumulative_Cleared_Vehicles": sim.cleared_vehicles,
                "Cumulative_Cleared_PCU": round(sim.cleared_pcu, 1),
            }
            minute_stats.append(record)
            time_series_data.append(record)

            if minute % 60 == 0:
                print(f"  -> Hour {hour}/5 Complete | AWT: {current_awt:5.2f}s | MaxWait: {current_max_wait:5.2f}s | Cleared: {sim.cleared_vehicles:5d} veh")

        elapsed_wall = time.time() - start_real_time
        print(f"  [DONE] Mode {mode_label} completed in {elapsed_wall:.2f}s real time.")
        all_results[mode_id] = {
            "name": mode_label,
            "df": pd.DataFrame(minute_stats),
            "final_cleared_vehicles": sim.cleared_vehicles,
            "final_cleared_pcu": sim.cleared_pcu,
            "overall_awt": float(np.mean(sim.cleared_waits)) if sim.cleared_waits else 0.0,
            "overall_max_wait": float(max(sim.cleared_waits, default=0.0)),
        }

    # ── Summary DataFrames ──
    df_all = pd.DataFrame(time_series_data)

    # Hourly aggregation per mode
    hourly_summary = df_all.groupby(["Hour", "Scenario", "Mode_Name"]).agg(
        Hourly_Avg_AWT=("AWT_sec", "mean"),
        Hourly_Peak_Max_Wait=("Max_Wait_sec", "max"),
        Hourly_Avg_Active_Queue=("Active_Vehicles", "mean"),
        End_Cumulative_Cleared=("Cumulative_Cleared_Vehicles", "last"),
        End_Cumulative_PCU=("Cumulative_Cleared_PCU", "last")
    ).reset_index()

    # Executive Summary per mode
    exec_summary = []
    for mode_id in modes:
        res = all_results[mode_id]
        m_df = df_all[df_all["Mode_ID"] == mode_id]
        exec_summary.append({
            "Control Method": res["name"],
            "Overall AWT (sec)": round(m_df["AWT_sec"].mean(), 2),
            "Peak Max Wait (sec)": round(m_df["Max_Wait_sec"].max(), 2),
            "Average Active Queue (veh)": round(m_df["Active_Vehicles"].mean(), 1),
            "Total Vehicles Cleared": res["final_cleared_vehicles"],
            "Total PCU Cleared": round(res["final_cleared_pcu"], 1),
            "Throughput Gain vs Fixed-Time": f"{((res['final_cleared_vehicles'] - all_results[1]['final_cleared_vehicles']) / all_results[1]['final_cleared_vehicles'] * 100):+.1f}%",
            "AWT Reduction vs Fixed-Time": f"{((all_results[1]['df']['AWT_sec'].mean() - m_df['AWT_sec'].mean()) / all_results[1]['df']['AWT_sec'].mean() * 100):+.1f}%"
        })
    df_exec = pd.DataFrame(exec_summary)

    # ── Save Excel Report ──
    report_dir = os.path.join(PROJECT_ROOT, "data", "simulation_reports", "two_junction_reports")
    os.makedirs(report_dir, exist_ok=True)
    figures_dir = os.path.join(report_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    excel_path = os.path.join(report_dir, "5hour_2junction_3methods_comparison.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_exec.to_excel(writer, sheet_name="Executive Summary", index=False)
        hourly_summary.to_excel(writer, sheet_name="Hourly Scenario Breakdown", index=False)
        for mode_id in modes:
            all_results[mode_id]["df"].to_excel(writer, sheet_name=f"Minute Log - {mode_names[mode_id]}", index=False)

    print(f"\n[EXPORT] Excel Report successfully saved to: {excel_path}")

    # ── Generate Publication-Quality Figures ──
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: 5-Hour AWT Trajectory Comparison
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    colors = {1: '#E67E22', 2: '#27AE60', 3: '#2980B9'}

    for mode_id in modes:
        m_df = df_all[df_all["Mode_ID"] == mode_id]
        # 5-minute rolling average for clarity
        rolling_awt = m_df["AWT_sec"].rolling(window=5, min_periods=1).mean()
        ax.plot(m_df["Minute"], rolling_awt, label=mode_names[mode_id], color=colors[mode_id], linewidth=2.4)

    # Add Scenario Bands
    scenario_labels = [
        (0, 60, "Scenario 1:\nBalanced Baseline", "#f8f9fa"),
        (60, 120, "Scenario 2:\nMorning Arterial Rush (EB)", "#edf2f7"),
        (120, 180, "Scenario 3:\nCross-Street Surge (N/S)", "#f8f9fa"),
        (180, 240, "Scenario 4:\nEvening Reverse Commute (WB)", "#edf2f7"),
        (240, 300, "Scenario 5:\nDynamic Spikes + Emergency", "#f8f9fa")
    ]
    for x_start, x_end, label, bg in scenario_labels:
        ax.axvspan(x_start, x_end, color=bg, alpha=0.6, zorder=0)
        ax.text((x_start + x_end)/2, ax.get_ylim()[1]*0.88 if ax.get_ylim()[1] > 0 else 25, label,
                ha='center', va='center', fontsize=8.5, fontweight='bold', color='#4a5568')

    ax.set_title("5-Hour Average Wait Time (AWT) Trajectory — 2-Junction Connected System", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Simulation Elapsed Time (Minutes)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Average Wait Time (Seconds)", fontsize=11, fontweight='bold')
    ax.set_xlim(1, 300)
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left', fontsize=10)
    fig.tight_layout()
    fig1_path = os.path.join(figures_dir, "figure1_two_junction_5hour_awt.png")
    fig.savefig(fig1_path)
    plt.close(fig)

    # Figure 2: Max Wait Time Ceiling (Anti-Starvation)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    hourly_max = hourly_summary.pivot(index="Hour", columns="Mode_Name", values="Hourly_Peak_Max_Wait")
    x = np.arange(1, 6)
    width = 0.25
    ax.bar(x - width, hourly_max["Fixed-Time"], width, label="Fixed-Time", color='#E67E22', edgecolor='black', alpha=0.85)
    ax.bar(x, hourly_max["Fuzzy-ML"], width, label="Fuzzy-ML", color='#27AE60', edgecolor='black', alpha=0.85)
    ax.bar(x + width, hourly_max["Deep RL"], width, label="Deep RL", color='#2980B9', edgecolor='black', alpha=0.85)

    ax.axhline(55.0, color='red', linestyle='--', linewidth=1.8, label='Max Wait Safety Threshold (55s)')
    ax.set_title("Hourly Maximum Wait Time Ceiling (Anti-Starvation Performance)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Simulation Hour (Scenario)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Peak Maximum Wait (Seconds)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"H1: Base", f"H2: Rush EB", f"H3: N/S Surge", f"H4: Rush WB", f"H5: Dynamic"])
    ax.legend(frameon=True, loc='upper left', fontsize=9.5)
    fig.tight_layout()
    fig2_path = os.path.join(figures_dir, "figure2_two_junction_5hour_max_wait.png")
    fig.savefig(fig2_path)
    plt.close(fig)

    # Figure 3: Total Throughput & Capacity Comparison
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    totals = [res["final_cleared_vehicles"] for res in all_results.values()]
    names = [res["name"] for res in all_results.values()]
    bar_cols = ['#E67E22', '#27AE60', '#2980B9']

    bars = ax.bar(names, totals, color=bar_cols, width=0.55, edgecolor='black')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 150, f"{int(h):,} veh", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title("5-Hour Total Vehicle Clearance & Network Throughput Capacity", fontsize=13, fontweight='bold')
    ax.set_ylabel("Total Cleared Vehicles across 2 Junctions", fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(totals) * 1.15)
    fig.tight_layout()
    fig3_path = os.path.join(figures_dir, "figure3_two_junction_5hour_throughput.png")
    fig.savefig(fig3_path)
    plt.close(fig)

    print(f"[EXPORT] Generated Figures:\n  1. {fig1_path}\n  2. {fig2_path}\n  3. {fig3_path}")
    return df_exec, hourly_summary, excel_path


if __name__ == "__main__":
    df_exec, hourly_summary, excel_path = run_5hour_benchmark()
    print("\n--- 5-HOUR BENCHMARK EXECUTIVE SUMMARY ---")
    print(df_exec.to_string(index=False))
