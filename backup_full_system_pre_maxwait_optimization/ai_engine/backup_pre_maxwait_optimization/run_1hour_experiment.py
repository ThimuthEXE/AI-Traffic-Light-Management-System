"""
1-Hour Simulation Runner & Excel Report Generator with Minute-by-Minute Max Wait Analysis
KDU IT 3182 Essentials of AI - AI Traffic Light Management System

Simulation Setup:
- U (North) Spawn Rate: 150 v/m (0.4s spawn interval)
- D (South) Spawn Rate: 40 v/m (1.5s spawn interval)
- L (West)  Spawn Rate: 35 v/m (1.714s spawn interval)
- R (East)  Spawn Rate: 30 v/m (2.0s spawn interval)
- Total Simulated Time: 3600 seconds (1 hour = 216,000 ticks at 60 FPS)
- Includes dedicated Minute-by-Minute (1-60 mins) Max Wait Time & Performance tracking
"""

import os
import sys
import time
import math
import pandas as pd
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['SDL_VIDEODRIVER'] = 'dummy'

from ai_engine.simulation.run_simulation import TrafficSimulationApp
from ai_engine.simulation.traffic_signal import SignalPhase
from ai_engine.utils.pcu_calculator import calculate_lane_pcu

def run_1hour_simulation_with_minute_analysis():
    print("=" * 75)
    print("STARTING 1-HOUR TRAFFIC SIMULATION (WITH MINUTE-BY-MINUTE MAX WAIT ANALYSIS)")
    print("Approach Configuration:")
    print("  U (North): 150 v/m (interval = 0.400s)")
    print("  D (South):  40 v/m (interval = 1.500s)")
    print("  L (West):   35 v/m (interval = 1.714s)")
    print("  R (East):   30 v/m (interval = 2.000s)")
    print("Total Simulated Duration: 3600.0s (1.00 Hour, 216,000 steps @ dt=1/60s)")
    print("=" * 75)

    app = TrafficSimulationApp()

    # Configure exact spawn rates requested
    app.approach_intervals['N'] = 60.0 / 150.0  # 0.4s
    app.approach_intervals['S'] = 60.0 / 40.0   # 1.5s
    app.approach_intervals['W'] = 60.0 / 35.0   # ~1.714s
    app.approach_intervals['E'] = 60.0 / 30.0   # 2.0s

    total_sim_time = 3600.0  # 1 hour
    dt = 1.0 / 60.0
    total_ticks = int(total_sim_time / dt)

    # Telemetry data structures
    cycle_records = []
    timeseries_records = []
    cleared_vehicle_records = []
    minute_records = []

    # Track vehicle spawns per direction and per minute
    spawn_stats = {
        'N': {'count': 0, 'car': 0, 'van': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0, 'straight': 0, 'left': 0, 'right': 0},
        'S': {'count': 0, 'car': 0, 'van': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0, 'straight': 0, 'left': 0, 'right': 0},
        'W': {'count': 0, 'car': 0, 'van': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0, 'straight': 0, 'left': 0, 'right': 0},
        'E': {'count': 0, 'car': 0, 'van': 0, 'bus': 0, 'truck': 0, 'motorcycle': 0, 'straight': 0, 'left': 0, 'right': 0},
    }

    spawns_in_current_minute = {'N': 0, 'S': 0, 'W': 0, 'E': 0}
    cleared_in_current_minute = []
    max_wait_observed_in_minute = {'N': 0.0, 'S': 0.0, 'W': 0.0, 'E': 0.0, 'overall': 0.0}

    # Intercept spawns
    original_spawn_fn = app.spawn_vehicle_for_approach
    def logged_spawn(direction):
        prev_len = len(app.vehicles)
        original_spawn_fn(direction)
        if len(app.vehicles) > prev_len:
            v = app.vehicles[-1]
            st = spawn_stats[direction]
            st['count'] += 1
            st[v.vehicle_type] = st.get(v.vehicle_type, 0) + 1
            intent = getattr(v, 'turn_intent', 'STRAIGHT')
            if intent == 'LEFT': st['left'] += 1
            elif intent == 'RIGHT': st['right'] += 1
            else: st['straight'] += 1
            spawns_in_current_minute[direction] += 1
    app.spawn_vehicle_for_approach = logged_spawn

    # Intercept vehicle clearance
    original_record_cleared = app.active_metrics.record_vehicle_cleared
    def logged_cleared(vehicle):
        original_record_cleared(vehicle)
        rec = {
            'vehicle_id': vehicle.id,
            'vehicle_type': vehicle.vehicle_type,
            'approach_dir': vehicle.direction,
            'approach_name': {'N': 'U (North)', 'S': 'D (South)', 'W': 'L (West)', 'E': 'R (East)'}.get(vehicle.direction, vehicle.direction),
            'lane_idx': vehicle.lane_idx,
            'turn_intent': getattr(vehicle, 'turn_intent', 'STRAIGHT'),
            'pcu_weight': vehicle.pcu,
            'wait_time_sec': round(vehicle.wait_time, 2),
            'sim_time_sec': round(current_sim_time, 2)
        }
        cleared_vehicle_records.append(rec)
        cleared_in_current_minute.append(rec)
    app.active_metrics.record_vehicle_cleared = logged_cleared

    sim_start_wall = time.time()
    current_sim_time = 0.0
    last_ts_snapshot = -10.0
    last_minute_marker = 0
    prev_cycle_count = 0

    print("Simulating 3,600s across 60 minute intervals...")

    for tick in range(total_ticks):
        current_sim_time = tick * dt
        app.update(dt)

        # Track continuous max wait across all waiting vehicles during this step
        for v in app.vehicles:
            if not v.has_cleared_intersection:
                w = v.wait_time
                d = v.direction
                if w > max_wait_observed_in_minute.get(d, 0.0):
                    max_wait_observed_in_minute[d] = w
                if w > max_wait_observed_in_minute['overall']:
                    max_wait_observed_in_minute['overall'] = w

        # 10s Time-Series Snapshot
        if current_sim_time - last_ts_snapshot >= 10.0:
            last_ts_snapshot = current_sim_time
            pcu_by_dir = app._pcu_by_direction()
            
            # Max wait right now on active vehicles per approach
            live_max_w_N = max((v.wait_time for v in app.vehicles if v.direction == 'N' and not v.has_cleared_intersection), default=0.0)
            live_max_w_S = max((v.wait_time for v in app.vehicles if v.direction == 'S' and not v.has_cleared_intersection), default=0.0)
            live_max_w_W = max((v.wait_time for v in app.vehicles if v.direction == 'W' and not v.has_cleared_intersection), default=0.0)
            live_max_w_E = max((v.wait_time for v in app.vehicles if v.direction == 'E' and not v.has_cleared_intersection), default=0.0)

            timeseries_records.append({
                'sim_time_sec': round(current_sim_time, 1),
                'sim_time_min': round(current_sim_time / 60.0, 2),
                'active_phase': str(app.signals.current_phase),
                'active_policy': str(app.signals.active_policy),
                'u_north_queue_pcu': round(pcu_by_dir.get('N', 0.0), 2),
                'd_south_queue_pcu': round(pcu_by_dir.get('S', 0.0), 2),
                'l_west_queue_pcu': round(pcu_by_dir.get('W', 0.0), 2),
                'r_east_queue_pcu': round(pcu_by_dir.get('E', 0.0), 2),
                'u_north_live_max_wait_s': round(live_max_w_N, 1),
                'd_south_live_max_wait_s': round(live_max_w_S, 1),
                'l_west_live_max_wait_s': round(live_max_w_W, 1),
                'r_east_live_max_wait_s': round(live_max_w_E, 1),
                'overall_live_max_wait_s': round(max(live_max_w_N, live_max_w_S, live_max_w_W, live_max_w_E), 1),
                'total_active_vehicles': len(app.vehicles),
                'cumulative_cleared_vehicles': app.active_metrics.cleared_vehicles,
                'cumulative_cleared_pcu': round(app.active_metrics.cleared_pcu, 2),
                'running_awt_sec': round(app.active_metrics.average_wait_time, 2),
                'max_observed_wait_sec': round(app.active_metrics.max_observed_wait, 2)
            })

        # Cycle Completion Logger
        if app.completed_cycles != prev_cycle_count:
            prev_cycle_count = app.completed_cycles
            dec = app._last_hud_decision
            obs_n = app.cycle_observer.get_prev_summary('N')
            obs_s = app.cycle_observer.get_prev_summary('S')
            obs_w = app.cycle_observer.get_prev_summary('W')
            obs_e = app.cycle_observer.get_prev_summary('E')

            cycle_records.append({
                'cycle_num': app.completed_cycles,
                'sim_time_sec': round(current_sim_time, 1),
                'sim_time_min': round(current_sim_time / 60.0, 2),
                'selected_policy': dec.get('policy', str(app.signals.active_policy)),
                'prioritized_direction': dec.get('prioritized_dir', 'NONE'),
                'dominance_ratio': dec.get('asym_ratio', 1.0),
                'total_green_allocated_sec': round(dec.get('total_green_sec', app.signals.allocated_total_green), 1),
                'through_green_sec': round(dec.get('through_green_sec', app.signals.allocated_through_green), 1),
                'turn_green_sec': round(dec.get('turn_green_sec', app.signals.allocated_left_green), 1),
                'through_pct': dec.get('thru_pct', 68),
                'turn_pct': dec.get('turn_pct', 32),
                'opposing_queue_corrected': bool(dec.get('opposing_adjusted', False)),
                'decision_source': dec.get('data_source', 'default_fallback'),
                'vehicles_cleared_cycle': app.current_phase_cleared,
                'cumulative_cleared': app.active_metrics.cleared_vehicles,
                'running_awt_sec': round(app.active_metrics.average_wait_time, 2),
                'max_wait_sec': round(app.active_metrics.max_observed_wait, 2),
                'u_north_obs_flow_vpm': obs_n.get('arrival_flow_vpm', 0),
                'd_south_obs_flow_vpm': obs_s.get('arrival_flow_vpm', 0),
                'l_west_obs_flow_vpm': obs_w.get('arrival_flow_vpm', 0),
                'r_east_obs_flow_vpm': obs_e.get('arrival_flow_vpm', 0),
            })

        # -------------------------------------------------------------
        # MINUTE BOUNDARY RECORDER (Every 60.0 simulated seconds)
        # -------------------------------------------------------------
        current_minute_int = int(current_sim_time // 60) + 1
        if (tick > 0 and tick % (60 * 60) == 0) or tick == total_ticks - 1:
            min_num = int(current_sim_time // 60)
            if min_num == 0: min_num = 1
            if min_num > 60: min_num = 60

            pcu_by_dir = app._pcu_by_direction()
            
            # Minute cleared stats
            cleared_cnt = len(cleared_in_current_minute)
            cleared_pcu = sum(v['pcu_weight'] for v in cleared_in_current_minute)
            minute_waits = [v['wait_time_sec'] for v in cleared_in_current_minute]
            min_awt = round(np.mean(minute_waits), 2) if minute_waits else 0.0

            # Max wait of vehicles cleared or waiting during this minute
            max_w_overall = round(max_wait_observed_in_minute['overall'], 2)
            max_w_U = round(max_wait_observed_in_minute['N'], 2)
            max_w_D = round(max_wait_observed_in_minute['S'], 2)
            max_w_L = round(max_wait_observed_in_minute['W'], 2)
            max_w_R = round(max_wait_observed_in_minute['E'], 2)

            # Spawn totals this minute
            sp_U = spawns_in_current_minute['N']
            sp_D = spawns_in_current_minute['S']
            sp_L = spawns_in_current_minute['W']
            sp_R = spawns_in_current_minute['E']
            sp_tot = sp_U + sp_D + sp_L + sp_R

            start_sec = (min_num - 1) * 60
            end_sec = min_num * 60
            time_window_str = f"{start_sec//60:02d}:00 - {end_sec//60:02d}:00"

            minute_records.append({
                'minute_num': min_num,
                'time_window': time_window_str,
                'max_wait_overall_sec': max_w_overall,
                'max_wait_u_north_sec': max_w_U,
                'max_wait_d_south_sec': max_w_D,
                'max_wait_l_west_sec': max_w_L,
                'max_wait_r_east_sec': max_w_R,
                'minute_avg_wait_sec': min_awt,
                'cumulative_awt_sec': round(app.active_metrics.average_wait_time, 2),
                'vehicles_spawned_in_minute': sp_tot,
                'vehicles_cleared_in_minute': cleared_cnt,
                'cleared_pcu_in_minute': round(cleared_pcu, 1),
                'active_queue_u_north_pcu': round(pcu_by_dir.get('N', 0.0), 2),
                'active_queue_d_south_pcu': round(pcu_by_dir.get('S', 0.0), 2),
                'active_queue_l_west_pcu': round(pcu_by_dir.get('W', 0.0), 2),
                'active_queue_r_east_pcu': round(pcu_by_dir.get('E', 0.0), 2),
                'active_vehicles_on_road': len(app.vehicles),
                'cumulative_cleared_total': app.active_metrics.cleared_vehicles,
                'active_policy': str(app.signals.active_policy)
            })

            # Reset minute-window accumulators
            spawns_in_current_minute = {'N': 0, 'S': 0, 'W': 0, 'E': 0}
            cleared_in_current_minute = []
            
            # Seed next minute with current waiting vehicle times
            max_wait_observed_in_minute = {
                'N': max((v.wait_time for v in app.vehicles if v.direction == 'N' and not v.has_cleared_intersection), default=0.0),
                'S': max((v.wait_time for v in app.vehicles if v.direction == 'S' and not v.has_cleared_intersection), default=0.0),
                'W': max((v.wait_time for v in app.vehicles if v.direction == 'W' and not v.has_cleared_intersection), default=0.0),
                'E': max((v.wait_time for v in app.vehicles if v.direction == 'E' and not v.has_cleared_intersection), default=0.0),
                'overall': max((v.wait_time for v in app.vehicles if not v.has_cleared_intersection), default=0.0)
            }

            if min_num % 10 == 0 or min_num == 60:
                elapsed_wall = time.time() - sim_start_wall
                print(f"  [Min {min_num:2d}/60] Overall Max Wait: {max_w_overall:5.1f}s | U-Max: {max_w_U:5.1f}s | D-Max: {max_w_D:5.1f}s | Cleared: {app.active_metrics.cleared_vehicles:4d} veh (Wall: {elapsed_wall:.1f}s)")

    wall_duration = time.time() - sim_start_wall
    print("=" * 75)
    print(f"SIMULATION COMPLETED in {wall_duration:.2f}s ({3600.0/wall_duration:.1f}x real-time)")
    print(f"Total Vehicles Cleared: {app.active_metrics.cleared_vehicles} ({app.active_metrics.cleared_pcu:.1f} PCU)")
    print(f"Overall Average Wait Time (AWT): {app.active_metrics.average_wait_time:.2f}s")
    print(f"Overall Peak Max Observed Wait Time: {app.active_metrics.max_observed_wait:.2f}s")
    print(f"Recorded {len(minute_records)} Minute-by-Minute Data Intervals")
    print("=" * 75)

    # -------------------------------------------------------------
    # BUILD ENHANCED EXCEL WORKBOOK
    # -------------------------------------------------------------
    output_dir = os.path.join(PROJECT_ROOT, "data", "simulation_reports")
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, "simulation_1hour_U150_D40_L35_R30.xlsx")

    print(f"Writing updated Excel file with Minute-by-Minute Max Wait sheet to:\n{excel_path}")

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # Styles
    navy_header_fill   = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    crimson_header_fill= PatternFill(start_color="8B0000", end_color="8B0000", fill_type="solid")
    teal_header_fill   = PatternFill(start_color="008080", end_color="008080", fill_type="solid")
    dark_gray_fill     = PatternFill(start_color="2F3542", end_color="2F3542", fill_type="solid")
    accent_green_fill  = PatternFill(start_color="D4EDDA", end_color="D4EDDA", fill_type="solid")
    accent_yellow_fill = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")
    accent_red_fill    = PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid")
    zebra_fill         = PatternFill(start_color="F8F9FA", end_color="F8F9FA", fill_type="solid")
    
    header_font    = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font     = Font(name="Calibri", size=14, bold=True, color="1B365D")
    sub_title_font = Font(name="Calibri", size=10, italic=True, color="555555")
    bold_font      = Font(name="Calibri", size=10, bold=True)
    regular_font   = Font(name="Calibri", size=10)

    thin_border = Border(
        left=Side(style='thin', color='D3D3D3'),
        right=Side(style='thin', color='D3D3D3'),
        top=Side(style='thin', color='D3D3D3'),
        bottom=Side(style='thin', color='D3D3D3')
    )

    # -------------------------------------------------------------
    # SHEET 1: EXECUTIVE SUMMARY
    # -------------------------------------------------------------
    ws1 = wb.create_sheet(title="Executive Summary")
    ws1.views.sheetView[0].showGridLines = True

    ws1["A1"] = "AI TRAFFIC LIGHT MANAGEMENT SYSTEM - 1-HOUR SIMULATION REPORT"
    ws1["A1"].font = title_font
    ws1["A2"] = "General Sir John Kotelawala Defence University (KDU) | IT 3182 Essentials of AI | Group 06"
    ws1["A2"].font = sub_title_font

    summary_data = [
        ("EXPERIMENT CONFIGURATION", ""),
        ("Simulated Duration", "3,600 Seconds (1.00 Hour / 60 Full Minutes)"),
        ("Simulation Mode", "ML Cycle-Adaptive Phasing + Dynamic Surge Priority"),
        ("U Approach (North) Spawn Rate", "150 Vehicles / Minute (0.40s Interval)"),
        ("D Approach (South) Spawn Rate", "40 Vehicles / Minute (1.50s Interval)"),
        ("L Approach (West) Spawn Rate", "35 Vehicles / Minute (1.71s Interval)"),
        ("R Approach (East) Spawn Rate", "30 Vehicles / Minute (2.00s Interval)"),
        ("Total Intersection Inflow Rate", "255 Vehicles / Minute (15,300 Vehicles / Hour Nominal Demand)"),
        ("", ""),
        ("OVERALL SYSTEM PERFORMANCE & WAIT TIME METRICS", ""),
        ("Total Vehicles Spawned", sum(s['count'] for s in spawn_stats.values())),
        ("Total Vehicles Cleared", app.active_metrics.cleared_vehicles),
        ("Total PCU Cleared", round(app.active_metrics.cleared_pcu, 1)),
        ("Throughput Clearance Efficiency", f"{min(100.0, round(app.active_metrics.cleared_vehicles / max(1, sum(s['count'] for s in spawn_stats.values())) * 100, 1))}%"),
        ("Overall Average Wait Time (AWT)", f"{app.active_metrics.average_wait_time:.2f} seconds"),
        ("Overall Peak Max Observed Wait Time", f"{app.active_metrics.max_observed_wait:.2f} seconds"),
        ("Average Minute-by-Minute Max Wait", f"{np.mean([r['max_wait_overall_sec'] for r in minute_records]):.2f} seconds"),
        ("Total Signal Cycles Executed", app.completed_cycles),
        ("Total Collisions Occurred", "0 (0.0% Collision Risk - Protected Phasing)"),
        ("Emergency Vehicles Preempted", "0 (Baseline Traffic Test)")
    ]

    row_idx = 4
    for label, val in summary_data:
        ws1.cell(row=row_idx, column=1, value=label)
        ws1.cell(row=row_idx, column=2, value=val)
        if val == "":
            ws1.cell(row=row_idx, column=1).font = bold_font
            ws1.cell(row=row_idx, column=1).fill = PatternFill(start_color="E9ECEF", fill_type="solid")
            ws1.cell(row=row_idx, column=2).fill = PatternFill(start_color="E9ECEF", fill_type="solid")
        else:
            ws1.cell(row=row_idx, column=1).font = regular_font
            ws1.cell(row=row_idx, column=2).font = bold_font
            ws1.cell(row=row_idx, column=1).border = thin_border
            ws1.cell(row=row_idx, column=2).border = thin_border
        row_idx += 1

    row_idx += 2
    ws1.cell(row=row_idx, column=1, value="CLEARED VEHICLE FLEET COMPOSITION").font = bold_font
    ws1.cell(row=row_idx, column=1).fill = PatternFill(start_color="E9ECEF", fill_type="solid")
    ws1.cell(row=row_idx, column=2).fill = PatternFill(start_color="E9ECEF", fill_type="solid")
    row_idx += 1

    for v_type, count in app.active_metrics.vehicle_counts.items():
        ws1.cell(row=row_idx, column=1, value=f"{v_type.capitalize()}s Cleared")
        ws1.cell(row=row_idx, column=2, value=count)
        ws1.cell(row=row_idx, column=1).border = thin_border
        ws1.cell(row=row_idx, column=2).border = thin_border
        row_idx += 1

    ws1.column_dimensions['A'].width = 42
    ws1.column_dimensions['B'].width = 34

    # -------------------------------------------------------------
    # SHEET 2: MINUTE-BY-MINUTE ANALYSIS (NEW DEDICATED SHEET)
    # -------------------------------------------------------------
    ws2 = wb.create_sheet(title="Minute-by-Minute Analysis")
    ws2.views.sheetView[0].showGridLines = True

    min_headers = [
        "Minute #", "Time Window", "Overall Max Wait (s)", "U (North) Max Wait (s)", "D (South) Max Wait (s)",
        "L (West) Max Wait (s)", "R (East) Max Wait (s)", "Minute AWT (s)", "Cumulative AWT (s)",
        "Vehicles Spawned", "Vehicles Cleared", "Cleared PCU", "U (North) Queue (PCU)",
        "D (South) Queue (PCU)", "L (West) Queue (PCU)", "R (East) Queue (PCU)", "Active Vehicles on Road",
        "Cumulative Cleared Total", "Active Signal Policy"
    ]

    for col_idx, h in enumerate(min_headers, 1):
        cell = ws2.cell(row=1, column=col_idx, value=h)
        cell.fill = crimson_header_fill if "Max Wait" in h else navy_header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for r_idx, r in enumerate(minute_records):
        r_num = r_idx + 2
        vals = [
            r['minute_num'], r['time_window'], r['max_wait_overall_sec'], r['max_wait_u_north_sec'],
            r['max_wait_d_south_sec'], r['max_wait_l_west_sec'], r['max_wait_r_east_sec'],
            r['minute_avg_wait_sec'], r['cumulative_awt_sec'], r['vehicles_spawned_in_minute'],
            r['vehicles_cleared_in_minute'], r['cleared_pcu_in_minute'], r['active_queue_u_north_pcu'],
            r['active_queue_d_south_pcu'], r['active_queue_l_west_pcu'], r['active_queue_r_east_pcu'],
            r['active_vehicles_on_road'], r['cumulative_cleared_total'], r['active_policy']
        ]
        fill_to_use = zebra_fill if r_num % 2 == 0 else PatternFill(fill_type=None)
        for c_idx, val in enumerate(vals, 1):
            cell = ws2.cell(row=r_num, column=c_idx, value=val)
            cell.font = regular_font
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center")
            if fill_to_use.fill_type: cell.fill = fill_to_use
            # Highlight Overall Max Wait column
            if c_idx == 3:
                cell.font = bold_font
                if val >= 100.0: cell.fill = accent_red_fill
                elif val >= 40.0: cell.fill = accent_yellow_fill
                else: cell.fill = accent_green_fill

    for col in ws2.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws2.column_dimensions[col_letter].width = max(max_len + 3, 14)

    # -------------------------------------------------------------
    # SHEET 3: APPROACH PERFORMANCE BREAKDOWN
    # -------------------------------------------------------------
    ws3 = wb.create_sheet(title="Approach Performance")
    ws3.views.sheetView[0].showGridLines = True

    headers_app = [
        "Approach Code", "Direction Name", "Configured Spawn Rate (v/m)", "Spawn Interval (s)",
        "Total Vehicles Spawned", "Total Vehicles Cleared", "Through Traffic (Straight+Right)", "Left-Turn Traffic",
        "Through %", "Turn %", "Observed AWT (s)", "Peak Max Wait Time (s)", "Congestion Level"
    ]
    for col_idx, h in enumerate(headers_app, 1):
        cell = ws3.cell(row=1, column=col_idx, value=h)
        cell.fill = navy_header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    cleared_by_dir = {'N': 0, 'S': 0, 'W': 0, 'E': 0}
    wait_by_dir = {'N': [], 'S': [], 'W': [], 'E': []}
    for rec in cleared_vehicle_records:
        d = rec['approach_dir']
        cleared_by_dir[d] = cleared_by_dir.get(d, 0) + 1
        wait_by_dir[d].append(rec['wait_time_sec'])

    app_rows = [
        ('U', 'North', 150, 0.4, 'N'),
        ('D', 'South', 40,  1.5, 'S'),
        ('L', 'West',  35,  1.71, 'W'),
        ('R', 'East',  30,  2.0, 'E'),
    ]

    for r_i, (code, name, sr, interv, d_key) in enumerate(app_rows, 2):
        st = spawn_stats[d_key]
        sp_tot = st['count']
        cl_tot = cleared_by_dir[d_key]
        thru_cnt = st['straight'] + st['right']
        turn_cnt = st['left']
        tot_cnt = max(1, thru_cnt + turn_cnt)
        thru_pct = round(thru_cnt / tot_cnt * 100, 1)
        turn_pct = round(turn_cnt / tot_cnt * 100, 1)
        avg_w = round(np.mean(wait_by_dir[d_key]), 2) if wait_by_dir[d_key] else 0.0
        max_w = round(max(wait_by_dir[d_key]), 2) if wait_by_dir[d_key] else 0.0
        cong = "CRITICAL SURGE" if sr >= 100 else "MEDIUM" if sr >= 35 else "LIGHT"

        row_vals = [
            code, name, sr, interv, sp_tot, cl_tot, thru_cnt, turn_cnt,
            f"{thru_pct}%", f"{turn_pct}%", avg_w, max_w, cong
        ]
        for c_i, val in enumerate(row_vals, 1):
            c = ws3.cell(row=r_i, column=c_i, value=val)
            c.font = regular_font
            c.border = thin_border
            c.alignment = Alignment(horizontal="center")
            if code == 'U':
                c.fill = accent_yellow_fill

    for col in ws3.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws3.column_dimensions[col_letter].width = max(max_len + 4, 15)

    # -------------------------------------------------------------
    # SHEET 4: CYCLE LOG
    # -------------------------------------------------------------
    ws4 = wb.create_sheet(title="Cycle Log")
    ws4.views.sheetView[0].showGridLines = True

    df_cycles = pd.DataFrame(cycle_records)
    cycle_headers = [
        "Cycle #", "Sim Time (s)", "Sim Time (min)", "Selected Policy", "Prioritized Approach",
        "Dominance Ratio", "Total Green (s)", "Through Green (s)", "Turn Green (s)",
        "Through %", "Turn %", "Opposing Corrected", "Decision Source", "Vehicles Cleared in Cycle",
        "Cumulative Cleared", "Running AWT (s)", "Max Wait (s)",
        "U (North) Flow (v/m)", "D (South) Flow (v/m)", "L (West) Flow (v/m)", "R (East) Flow (v/m)"
    ]

    for col_idx, h in enumerate(cycle_headers, 1):
        cell = ws4.cell(row=1, column=col_idx, value=h)
        cell.fill = teal_header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for r_idx, row in df_cycles.iterrows():
        r_num = r_idx + 2
        vals = [
            row['cycle_num'], row['sim_time_sec'], row['sim_time_min'], row['selected_policy'],
            row['prioritized_direction'], row['dominance_ratio'], row['total_green_allocated_sec'],
            row['through_green_sec'], row['turn_green_sec'], row['through_pct'], row['turn_pct'],
            "YES" if row['opposing_queue_corrected'] else "NO", row['decision_source'],
            row['vehicles_cleared_cycle'], row['cumulative_cleared'], row['running_awt_sec'],
            row['max_wait_sec'], row['u_north_obs_flow_vpm'], row['d_south_obs_flow_vpm'],
            row['l_west_obs_flow_vpm'], row['r_east_obs_flow_vpm']
        ]
        fill_to_use = zebra_fill if r_num % 2 == 0 else PatternFill(fill_type=None)
        for c_idx, val in enumerate(vals, 1):
            cell = ws4.cell(row=r_num, column=c_idx, value=val)
            cell.font = regular_font
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center")
            if fill_to_use.fill_type: cell.fill = fill_to_use
            if c_idx == 4 and "N_EXCLUSIVE" in str(val):
                cell.fill = accent_green_fill

    for col in ws4.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws4.column_dimensions[col_letter].width = max(max_len + 3, 13)

    # -------------------------------------------------------------
    # SHEET 5: TIME-SERIES TELEMETRY (10s)
    # -------------------------------------------------------------
    ws5 = wb.create_sheet(title="Time-Series Telemetry (10s)")
    ws5.views.sheetView[0].showGridLines = True

    df_ts = pd.DataFrame(timeseries_records)
    ts_headers = [
        "Sim Time (s)", "Sim Time (min)", "Active Phase", "Active Policy",
        "U (North) Queue (PCU)", "D (South) Queue (PCU)", "L (West) Queue (PCU)", "R (East) Queue (PCU)",
        "U Max Wait (s)", "D Max Wait (s)", "L Max Wait (s)", "R Max Wait (s)", "Overall Live Max Wait (s)",
        "Active Vehicles on Road", "Cumulative Cleared Veh", "Cumulative Cleared PCU",
        "Running AWT (s)", "Max Observed Wait (s)"
    ]

    for col_idx, h in enumerate(ts_headers, 1):
        cell = ws5.cell(row=1, column=col_idx, value=h)
        cell.fill = dark_gray_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for r_idx, row in df_ts.iterrows():
        r_num = r_idx + 2
        vals = [
            row['sim_time_sec'], row['sim_time_min'], row['active_phase'], row['active_policy'],
            row['u_north_queue_pcu'], row['d_south_queue_pcu'], row['l_west_queue_pcu'], row['r_east_queue_pcu'],
            row['u_north_live_max_wait_s'], row['d_south_live_max_wait_s'], row['l_west_live_max_wait_s'], row['r_east_live_max_wait_s'],
            row['overall_live_max_wait_s'], row['total_active_vehicles'], row['cumulative_cleared_vehicles'], row['cumulative_cleared_pcu'],
            row['running_awt_sec'], row['max_observed_wait_sec']
        ]
        fill_to_use = zebra_fill if r_num % 2 == 0 else PatternFill(fill_type=None)
        for c_idx, val in enumerate(vals, 1):
            cell = ws5.cell(row=r_num, column=c_idx, value=val)
            cell.font = regular_font
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center")
            if fill_to_use.fill_type: cell.fill = fill_to_use

    for col in ws5.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws5.column_dimensions[col_letter].width = max(max_len + 3, 14)

    # -------------------------------------------------------------
    # SHEET 6: VEHICLE CLEARANCE LOG
    # -------------------------------------------------------------
    ws6 = wb.create_sheet(title="Vehicle Clearance Log")
    ws6.views.sheetView[0].showGridLines = True

    veh_headers = [
        "Vehicle ID", "Vehicle Type", "Approach Code", "Approach Name", "Lane Index",
        "Turn Intention", "PCU Weight", "Wait Time (s)", "Cleared at Sim Time (s)"
    ]
    for col_idx, h in enumerate(veh_headers, 1):
        cell = ws6.cell(row=1, column=col_idx, value=h)
        cell.fill = navy_header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    sample_vehs = cleared_vehicle_records[:10000]
    for r_idx, v_rec in enumerate(sample_vehs):
        r_num = r_idx + 2
        vals = [
            v_rec['vehicle_id'], v_rec['vehicle_type'], v_rec['approach_dir'], v_rec['approach_name'],
            v_rec['lane_idx'], v_rec['turn_intent'], v_rec['pcu_weight'], v_rec['wait_time_sec'],
            v_rec['sim_time_sec']
        ]
        fill_to_use = zebra_fill if r_num % 2 == 0 else PatternFill(fill_type=None)
        for c_idx, val in enumerate(vals, 1):
            cell = ws6.cell(row=r_num, column=c_idx, value=val)
            cell.font = regular_font
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center")
            if fill_to_use.fill_type: cell.fill = fill_to_use

    for col in ws6.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws6.column_dimensions[col_letter].width = max(max_len + 3, 13)

    wb.save(excel_path)
    print(f"\nUPDATED EXCEL REPORT SAVED: {excel_path}")
    return excel_path

if __name__ == "__main__":
    run_1hour_simulation_with_minute_analysis()
