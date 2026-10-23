"""
5-Hour Multi-Scenario Scientific Benchmark for 4-Junction 2×2 Grid System
AI Traffic Light Management System

Compares 3 Signal Control Methods across 5 Real-World Scenarios (1 Hour Each = 5 Simulated Hours / 18,000s Total):
  - Mode 1: Traditional Fixed-Time Controller (Baseline)
  - Mode 2: Dynamic Fuzzy-ML Cycle Optimizer + I2I Green Wave + Segment Metering
  - Mode 3: Deep Reinforcement Learning (Dueling DDQN) + I2I Green Wave + Segment Metering

Generates:
  - Multi-Tab Comparative Excel Workbook (.xlsx)
  - 4 High-Resolution Publication Figures (.png)
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['SDL_VIDEODRIVER'] = 'dummy'

from ai_engine.multi_junction.junction_node import JunctionNode
from ai_engine.multi_junction.two_junction_vehicle import TwoJunctionVehicle, VEHICLE_CONFIGS
from ai_engine.multi_junction.artery_segment import ArterySegment
from ai_engine.multi_junction.four_junction_coordinator import FourJunctionCoordinator
from ai_engine.multi_junction.collision_manager import CollisionManager


# Routing dictionary for 2x2 grid
CONNECTIONS = {
    (1, 'E'): 2,   # J1 -> J2 via Seg_TOP
    (1, 'S'): 3,   # J1 -> J3 via Seg_LEFT
    (2, 'W'): 1,   # J2 -> J1 via Seg_TOP
    (2, 'S'): 4,   # J2 -> J4 via Seg_RIGHT
    (3, 'N'): 1,   # J3 -> J1 via Seg_LEFT
    (3, 'E'): 4,   # J3 -> J4 via Seg_BOT
    (4, 'N'): 2,   # J4 -> J2 via Seg_RIGHT
    (4, 'W'): 3,   # J4 -> J3 via Seg_BOT
}


class FastFourJunctionBenchmark:
    WIDTH = 1440
    HEIGHT = 860
    STOP = 85

    def __init__(self, mode_id: int):
        self.mode_id = mode_id
        self.mode_name = {
            1: "Fixed-Time",
            2: "Fuzzy-ML (I2I Coordinated)",
            3: "Deep RL (I2I Coordinated)"
        }[mode_id]

        # 4 Junctions in 2x2 grid
        self.j1 = JunctionNode(1, "J1 NW", cx=400,  cy=230)
        self.j2 = JunctionNode(2, "J2 NE", cx=1040, cy=230)
        self.j3 = JunctionNode(3, "J3 SW", cx=400,  cy=630)
        self.j4 = JunctionNode(4, "J4 SE", cx=1040, cy=630)
        self.junctions = {1: self.j1, 2: self.j2, 3: self.j3, 4: self.j4}

        for j in self.junctions.values():
            j.set_mode(mode_id)

        # 4 Artery Segments
        self.seg_top   = ArterySegment("TOP",   self.j1, self.j2, orientation='EW', max_vehicles=12)
        self.seg_bot   = ArterySegment("BOT",   self.j3, self.j4, orientation='EW', max_vehicles=12)
        self.seg_left  = ArterySegment("LEFT",  self.j1, self.j3, orientation='NS', max_vehicles=6)
        self.seg_right = ArterySegment("RIGHT", self.j2, self.j4, orientation='NS', max_vehicles=6)
        self.segments  = [self.seg_top, self.seg_bot, self.seg_left, self.seg_right]

        # I2I Coordinator & Collision Manager
        self.coordinator = FourJunctionCoordinator(self.junctions, self.segments, enabled=(mode_id in (2, 3)))
        self.collision_mgr = CollisionManager()

        self.vehicles = []
        self.vehicle_id_counter = 1

        self.cleared_vehicles = 0
        self.cleared_pcu = 0.0
        self.cleared_waits = []
        self.hold_events_count = 0

        # 8 Inflow points
        self.inflow_points = [
            ("J1_W", 1, 'E'),
            ("J2_E", 2, 'W'),
            ("J3_W", 3, 'E'),
            ("J4_E", 4, 'W'),
            ("J1_N", 1, 'S'),
            ("J2_N", 2, 'S'),
            ("J3_S", 3, 'N'),
            ("J4_S", 4, 'N'),
        ]
        self.spawn_timers = {p[0]: random.uniform(0.0, 1.5) for p in self.inflow_points}

    def get_scenario_intervals(self, hour: int) -> dict:
        """Defines inflow intervals (seconds per vehicle) across 5 scenarios."""
        if hour == 1:
            # Scenario 1: Balanced Baseline (~35-45 veh/min)
            return {
                "J1_W": 2.2, "J2_E": 2.2, "J3_W": 2.2, "J4_E": 2.2,
                "J1_N": 2.6, "J2_N": 2.6, "J3_S": 2.6, "J4_S": 2.6
            }
        elif hour == 2:
            # Scenario 2: Morning Inbound Arterial Rush (Heavy Eastbound & Southbound)
            return {
                "J1_W": 0.65, "J2_E": 2.8, "J3_W": 0.85, "J4_E": 2.8,
                "J1_N": 0.90, "J2_N": 1.1, "J3_S": 3.2,  "J4_S": 3.2
            }
        elif hour == 3:
            # Scenario 3: Dense Cross-Grid Surge (Heavy North/South on both corridors)
            return {
                "J1_W": 2.0,  "J2_E": 2.0,  "J3_W": 2.0,  "J4_E": 2.0,
                "J1_N": 0.75, "J2_N": 0.75, "J3_S": 0.85, "J4_S": 0.85
            }
        elif hour == 4:
            # Scenario 4: Evening Reverse Commute (Heavy Westbound & Northbound)
            return {
                "J1_W": 2.8, "J2_E": 0.65, "J3_W": 2.8, "J4_E": 0.80,
                "J1_N": 3.2, "J2_N": 3.2,  "J3_S": 0.85, "J4_S": 0.90
            }
        else:
            # Scenario 5: Dynamic Asymmetric Spikes + Emergency Injections
            return {
                "J1_W": 0.85, "J2_E": 1.4, "J3_W": 0.95, "J4_E": 1.8,
                "J1_N": 1.1,  "J2_N": 2.4, "J3_S": 1.2,  "J4_S": 2.0
            }

    def _spawn_pos(self, j: JunctionNode, direction: str, lane_offset: int) -> tuple:
        if   direction == 'S': return (j.cx - lane_offset,            -35.0)
        elif direction == 'N': return (j.cx + lane_offset,  self.HEIGHT + 35.0)
        elif direction == 'E': return (-35.0,                j.cy + lane_offset)
        elif direction == 'W': return (self.WIDTH + 35.0,   j.cy - lane_offset)
        return (0.0, 0.0)

    def spawn_vehicle(self, inflow_key: str, j_id: int, direction: str, is_ambulance: bool = False):
        j = self.junctions[j_id]
        lane_idx = 1 if is_ambulance else random.choices([0, 1], weights=[0.35, 0.65])[0]
        lane_offset = 20 if lane_idx == 0 else 60
        sx, sy = self._spawn_pos(j, direction, lane_offset)

        # Collision guard at entry
        for ov in self.vehicles:
            if math.hypot(ov.x - sx, ov.y - sy) < (ov.length + 65.0):
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
            spawn_pos=(sx, sy),
            target_j_id=j_id
        )
        self.vehicle_id_counter += 1
        self.vehicles.append(v)

    def step(self, dt: float, current_intervals: dict, inject_ambulance: bool = False):
        # 1. Spawning
        for key, j_id, direction in self.inflow_points:
            self.spawn_timers[key] += dt
            if self.spawn_timers[key] >= current_intervals[key]:
                self.spawn_timers[key] = 0.0
                self.spawn_vehicle(key, j_id, direction)

        if inject_ambulance:
            self.spawn_vehicle("J1_W", 1, 'E', is_ambulance=True)

        # 2. Update Junction Signals
        for j in self.junctions.values():
            j.update(dt, self.vehicles)

        # 3. Update Segment Counts & Apply Metering Holds
        for seg in self.segments:
            seg.update(self.vehicles)
            if seg.is_draining():
                self.hold_events_count += 1

        # 4. I2I Coordinator
        if self.mode_id in (2, 3):
            self.coordinator.update(dt, self.vehicles)

        # 5. Vehicle Handoff
        for v in self.vehicles:
            if not v.has_cleared_intersection or v.is_turning:
                continue
            nxt_id = CONNECTIONS.get((v.current_junction_id, v.direction))
            if nxt_id is None:
                continue
            cur_j = self.junctions[v.current_junction_id]
            cleared_box = False
            if   v.direction == 'E' and v.x > cur_j.cx + self.STOP: cleared_box = True
            elif v.direction == 'W' and v.x < cur_j.cx - self.STOP: cleared_box = True
            elif v.direction == 'S' and v.y > cur_j.cy + self.STOP: cleared_box = True
            elif v.direction == 'N' and v.y < cur_j.cy - self.STOP: cleared_box = True
            if cleared_box:
                v.current_junction_id = nxt_id
                v.has_cleared_intersection = False

        # 6. Continuous Corridor Car-Following
        corridor_buckets = {}
        for v in self.vehicles:
            if not v.is_turning:
                if v.direction in ('E', 'W'):
                    road = 'TOP' if abs(v.y - 230) < abs(v.y - 630) else 'BOT'
                else:
                    road = 'LEFT' if abs(v.x - 400) < abs(v.x - 1040) else 'RIGHT'
                key = (road, v.direction, v.lane_idx)
                corridor_buckets.setdefault(key, []).append(v)

        for (road, d, _), v_list in corridor_buckets.items():
            if   d == 'E': v_list.sort(key=lambda v: -v.x)
            elif d == 'W': v_list.sort(key=lambda v:  v.x)
            elif d == 'S': v_list.sort(key=lambda v: -v.y)
            elif d == 'N': v_list.sort(key=lambda v:  v.y)

            for i, v in enumerate(v_list):
                leading = v_list[i - 1] if i > 0 else None
                j = self.junctions.get(v.current_junction_id)
                if j is not None:
                    sig_th, sig_lt = j.get_signals_metered(v.direction)
                    can_enter = self.collision_mgr.is_box_clear_for_entry(j.cx, j.cy, v.direction, self.vehicles, v.id)
                else:
                    sig_th, sig_lt = 'GREEN', 'GREEN'
                    can_enter = True
                v.update(dt, leading, sig_th, sig_lt, j, can_enter_box=can_enter)

        # Turning vehicles
        for v in self.vehicles:
            if v.is_turning:
                j = self.junctions.get(v.current_junction_id)
                rad_a = math.radians(v.heading_angle)
                cos_a, sin_a = math.cos(rad_a), math.sin(rad_a)
                path_blocked = False
                for ov in self.vehicles:
                    if ov.id == v.id: continue
                    dx = ov.x - v.x
                    dy = ov.y - v.y
                    dist = math.hypot(dx, dy)
                    if dist < (v.length + ov.length) * 0.55 + 10.0:
                        forward_dot = dx * cos_a + dy * sin_a
                        if forward_dot > 0:
                            path_blocked = True
                            break
                if path_blocked:
                    v.speed = 0.0
                v.update(dt, None, 'GREEN', 'GREEN', j)

        # 7. Collision Detection & Zero-Overlap Enforcement
        self.collision_mgr.check_and_resolve_collisions(self.vehicles, dt)

        # 8. Remove Off-Screen Vehicles
        for v in list(self.vehicles):
            if v.is_off_screen(self.WIDTH, self.HEIGHT):
                self.cleared_vehicles += 1
                self.cleared_pcu += v.pcu
                self.cleared_waits.append(v.wait_time)
                self.vehicles.remove(v)


def run_5hour_four_junction_benchmark():
    print("=" * 80)
    print("  5-HOUR 4-JUNCTION 2x2 GRID BENCHMARK: FIXED-TIME vs FUZZY-ML vs DEEP RL")
    
    print("=" * 80)

    total_sim_minutes = 300
    dt = 0.10  # 10Hz fast-forward
    steps_per_minute = int(60.0 / dt)

    modes = [1, 2, 3]
    mode_names = {1: "Fixed-Time", 2: "Fuzzy-ML (I2I Coordinated)", 3: "Deep RL (I2I Coordinated)"}
    all_results = {}
    time_series_data = []

    for mode_id in modes:
        mode_label = mode_names[mode_id]
        print(f"\n[BENCHMARK] Running 5-Hour Grid Simulation for Mode {mode_id}: {mode_label}...")
        start_real = time.time()

        sim = FastFourJunctionBenchmark(mode_id)
        minute_stats = []

        for minute in range(1, total_sim_minutes + 1):
            hour = (minute - 1) // 60 + 1
            intervals = sim.get_scenario_intervals(hour)
            inject_ambulance = (hour == 5 and minute % 10 == 0)

            for step_idx in range(steps_per_minute):
                sim.step(dt, intervals, inject_ambulance=(inject_ambulance and step_idx == 0))

            recent_waits = sim.cleared_waits[-200:] if sim.cleared_waits else [0.0]
            current_awt = float(np.mean(recent_waits)) if recent_waits else 0.0
            current_max_wait = float(max((v.wait_time for v in sim.vehicles), default=0.0))
            active_vehs = len(sim.vehicles)
            accidents = sim.collision_mgr.accident_count

            scenario_name = {
                1: "1: Balanced Baseline",
                2: "2: Morning Inbound Arterial (EB/SB)",
                3: "3: Dense Cross-Grid Surge (N/S)",
                4: "4: Evening Reverse Commute (WB/NB)",
                5: "5: Dynamic Spikes + Emergency"
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
                "Accidents": accidents,
                "Cumulative_Cleared_Vehicles": sim.cleared_vehicles,
                "Cumulative_Cleared_PCU": round(sim.cleared_pcu, 1),
                "I2I_Waves_Fired": sim.coordinator.green_waves_triggered,
                "Capacity_Hold_Ticks": sim.hold_events_count
            }
            minute_stats.append(record)
            time_series_data.append(record)

            if minute % 60 == 0:
                print(f"  -> Hour {hour}/5 Complete | AWT: {current_awt:5.2f}s | MaxWait: {current_max_wait:5.2f}s | Cleared: {sim.cleared_vehicles:6d} veh | Accidents: {accidents}")

        elapsed_wall = time.time() - start_real
        print(f"  [DONE] Mode {mode_label} completed in {elapsed_wall:.1f}s real time.")
        all_results[mode_id] = {
            "name": mode_label,
            "df": pd.DataFrame(minute_stats),
            "final_cleared_vehicles": sim.cleared_vehicles,
            "final_cleared_pcu": sim.cleared_pcu,
            "overall_awt": float(np.mean(sim.cleared_waits)) if sim.cleared_waits else 0.0,
            "overall_max_wait": float(max(sim.cleared_waits, default=0.0)),
            "accidents": sim.collision_mgr.accident_count,
            "i2I_waves": sim.coordinator.green_waves_triggered
        }

    df_all = pd.DataFrame(time_series_data)

    hourly_summary = df_all.groupby(["Hour", "Scenario", "Mode_Name"]).agg(
        Hourly_Avg_AWT=("AWT_sec", "mean"),
        Hourly_Peak_Max_Wait=("Max_Wait_sec", "max"),
        Hourly_Avg_Active_Queue=("Active_Vehicles", "mean"),
        Total_Accidents=("Accidents", "max"),
        End_Cumulative_Cleared=("Cumulative_Cleared_Vehicles", "last"),
        End_Cumulative_PCU=("Cumulative_Cleared_PCU", "last")
    ).reset_index()

    exec_summary = []
    base_cleared = all_results[1]["final_cleared_vehicles"]
    base_awt = df_all[df_all["Mode_ID"] == 1]["AWT_sec"].mean()

    for mode_id in modes:
        res = all_results[mode_id]
        m_df = df_all[df_all["Mode_ID"] == mode_id]
        cur_cleared = res["final_cleared_vehicles"]
        cur_awt = m_df["AWT_sec"].mean()
        exec_summary.append({
            "Control Method": res["name"],
            "Overall AWT (sec)": round(cur_awt, 2),
            "Peak Max Wait (sec)": round(m_df["Max_Wait_sec"].max(), 2),
            "Average Active Queue (veh)": round(m_df["Active_Vehicles"].mean(), 1),
            "Total Vehicles Cleared": cur_cleared,
            "Total PCU Cleared": round(res["final_cleared_pcu"], 1),
            "I2I Green Waves Triggered": res["i2I_waves"],
            "Total Accidents": res["accidents"],
            "Throughput Gain vs Fixed-Time": f"{((cur_cleared - base_cleared) / max(1, base_cleared) * 100):+.1f}%",
            "AWT Reduction vs Fixed-Time": f"{((base_awt - cur_awt) / max(0.1, base_awt) * 100):+.1f}%"
        })
    df_exec = pd.DataFrame(exec_summary)

    report_dir = os.path.join(PROJECT_ROOT, "data", "simulation_reports", "3_2026-09-17_Four_Junction_Grid")
    os.makedirs(report_dir, exist_ok=True)
    figures_dir = os.path.join(report_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    excel_path = os.path.join(report_dir, "5hour_4junction_3methods_comparison.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_exec.to_excel(writer, sheet_name="Executive Summary", index=False)
        hourly_summary.to_excel(writer, sheet_name="Hourly Scenario Breakdown", index=False)
        for mode_id in modes:
            all_results[mode_id]["df"].to_excel(writer, sheet_name=f"Minute Log - Mode {mode_id}", index=False)

    print(f"\n[EXPORT] Excel Report saved to: {excel_path}")

    # Generate Figures
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    colors = {1: '#E67E22', 2: '#27AE60', 3: '#2980B9'}

    # Figure 1: 5-Hour AWT Trajectory
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    for mode_id in modes:
        m_df = df_all[df_all["Mode_ID"] == mode_id]
        rolling_awt = m_df["AWT_sec"].rolling(window=5, min_periods=1).mean()
        ax.plot(m_df["Minute"], rolling_awt, label=mode_names[mode_id], color=colors[mode_id], linewidth=2.4)

    scenario_labels = [
        (0, 60, "Scenario 1:\nBalanced Baseline", "#f8f9fa"),
        (60, 120, "Scenario 2:\nMorning Arterial Peak", "#edf2f7"),
        (120, 180, "Scenario 3:\nCross-Grid Surge", "#f8f9fa"),
        (180, 240, "Scenario 4:\nEvening Reverse Commute", "#edf2f7"),
        (240, 300, "Scenario 5:\nDynamic Spikes + Emergency", "#f8f9fa")
    ]
    for x_start, x_end, label, bg in scenario_labels:
        ax.axvspan(x_start, x_end, color=bg, alpha=0.55, zorder=0)

    ax.set_title("5-Hour Average Wait Time (AWT) Trajectory — 4-Junction 2×2 Grid System", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Simulation Time (Minutes)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Average Wait Time (Seconds)", fontsize=11, fontweight='bold')
    ax.set_xlim(1, 300)
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left', fontsize=10)
    fig.tight_layout()
    fig1_path = os.path.join(figures_dir, "figure1_four_junction_5hour_awt.png")
    fig.savefig(fig1_path)
    plt.close(fig)

    # Figure 2: Max Wait Time Ceiling
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    hourly_max = hourly_summary.pivot(index="Hour", columns="Mode_Name", values="Hourly_Peak_Max_Wait")
    x = np.arange(1, 6)
    width = 0.25
    ax.bar(x - width, hourly_max.get("Fixed-Time", [0]*5), width, label="Fixed-Time", color='#E67E22', edgecolor='black', alpha=0.85)
    ax.bar(x, hourly_max.get("Fuzzy-ML (I2I Coordinated)", [0]*5), width, label="Fuzzy-ML (I2I)", color='#27AE60', edgecolor='black', alpha=0.85)
    ax.bar(x + width, hourly_max.get("Deep RL (I2I Coordinated)", [0]*5), width, label="Deep RL (I2I)", color='#2980B9', edgecolor='black', alpha=0.85)

    ax.axhline(55.0, color='red', linestyle='--', linewidth=1.8, label='Safety Delay Threshold (55s)')
    ax.set_title("Hourly Maximum Wait Time Ceiling (Anti-Starvation Performance in 2×2 Grid)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Simulation Hour (Scenario)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Peak Maximum Wait (Seconds)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(["H1: Base", "H2: Inbound Rush", "H3: Grid Surge", "H4: Reverse Commute", "H5: Dynamic"])
    ax.legend(frameon=True, loc='upper left', fontsize=9.5)
    fig.tight_layout()
    fig2_path = os.path.join(figures_dir, "figure2_four_junction_5hour_max_wait.png")
    fig.savefig(fig2_path)
    plt.close(fig)

    # Figure 3: Total Throughput
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    totals = [res["final_cleared_vehicles"] for res in all_results.values()]
    names = ["Fixed-Time", "Fuzzy-ML (I2I)", "Deep RL (I2I)"]
    bars = ax.bar(names, totals, color=['#E67E22', '#27AE60', '#2980B9'], width=0.55, edgecolor='black')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 300, f"{int(h):,} veh", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title("5-Hour Total Vehicle Clearance — 4-Junction Network Throughput", fontsize=13, fontweight='bold')
    ax.set_ylabel("Total Cleared Vehicles across 4 Junctions", fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(totals) * 1.15)
    fig.tight_layout()
    fig3_path = os.path.join(figures_dir, "figure3_four_junction_5hour_throughput.png")
    fig.savefig(fig3_path)
    plt.close(fig)

    print(f"[EXPORT] Generated Figures in {figures_dir}")
    return df_exec, hourly_summary, excel_path


if __name__ == "__main__":
    df_exec, hourly_summary, excel_path = run_5hour_four_junction_benchmark()
    print("\n--- 4-JUNCTION 5-HOUR BENCHMARK EXECUTIVE SUMMARY ---")
    print(df_exec.to_string(index=False))
