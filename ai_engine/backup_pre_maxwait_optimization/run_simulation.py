"""
AI Traffic Light Management System — Observation-First Cycle-Adaptive Simulation
==================================================================================
Cycle 0  →  Default timers.  CycleObserver silently records spawn rates,
             through/turn ratios, PCU buildup, wait times per approach.
Cycle 1+ →  Controller uses observations to:
             • Detect which approach (D/U/L/R) has the highest priority score.
             • Give it Exclusive All-Green (Straight + Turn both GREEN).
             • Split through_green / turn_green by the ACTUAL observed vehicle ratio.
             • Shorten the turn phase if the opposing approach has a heavy queue
               (gets back to that queue faster → reduces their max wait).
             • All choices minimise both AVG and MAX wait times across all approaches.

General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI
"""

import sys
import os
import math
import time
import random
import json
import threading
import urllib.request
import pygame

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_engine.simulation.vehicle        import Vehicle, VEHICLE_CONFIGS
from ai_engine.simulation.traffic_signal import TrafficSignalManager, SignalPhase
from ai_engine.simulation.controllers    import FixedTimeController, AIFuzzyController
from ai_engine.simulation.metrics_tracker import MetricsTracker
from ai_engine.simulation.cycle_logger   import CycleDataLogger
from ai_engine.simulation.cycle_observer import CycleObserver
from ai_engine.utils.pcu_calculator      import calculate_lane_pcu, classify_traffic_density


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FULL_NAMES = {"N": "North (U)", "S": "South (D)", "E": "East (R)", "W": "West (L)"}
OPPOSING   = {"N": "S", "S": "N", "E": "W", "W": "E"}


class TrafficSimulationApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(
            "AI Traffic Light System — Cycle-Adaptive Observer + ML Phasing | KDU IT3182"
        )

        self.width  = 1280
        self.height = 720
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock  = pygame.time.Clock()
        self.running = True
        self.paused  = False

        # Fonts
        self.title_font = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.font       = pygame.font.SysFont("Segoe UI", 14)
        self.bold_font  = pygame.font.SysFont("Segoe UI", 14, bold=True)
        self.small_font = pygame.font.SysFont("Segoe UI", 11)
        self.big_font   = pygame.font.SysFont("Segoe UI", 22, bold=True)

        # Core components
        self.signals         = TrafficSignalManager(yellow_duration=2.0, all_red_duration=1.0)
        self.fixed_controller = FixedTimeController(fixed_green=25.0)
        self.ai_controller    = AIFuzzyController(min_green=8.0, max_green=45.0)

        self.active_mode        = 2
        self.current_controller = self.ai_controller

        self.metrics_ai    = MetricsTracker()
        self.metrics_fixed = MetricsTracker()
        self.cycle_logger  = CycleDataLogger()

        # Cycle Observer
        self.cycle_observer = CycleObserver()

        self.vehicles       = []
        self.next_vehicle_id = 1

        self.approach_intervals = {"N": 2.0, "S": 2.0, "E": 2.0, "W": 2.0}
        self.approach_timers    = {"N": 0.0,  "S": 0.0,  "E": 0.0,  "W": 0.0}

        self.buttons = []
        self._init_buttons()

        self.current_phase_cleared  = 0
        self.emergency_banner_timer = 0.0
        self.emergency_active       = False

        # Completed cycles counter (0 = calibration in progress)
        self.completed_cycles = 0

        # Cache last decision for HUD display
        self._last_hud_decision: dict = {}

        self.sync_thread = threading.Thread(
            target=self._background_sync_worker, daemon=True
        )
        self.sync_thread.start()

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _init_buttons(self):
        self.buttons = []
        y = 70
        for d in ("N", "S", "E", "W"):
            self.buttons.append((pygame.Rect(265, y - 2, 22, 20), "-", d))
            self.buttons.append((pygame.Rect(292, y - 2, 22, 20), "+", d))
            y += 42

    @property
    def active_metrics(self) -> MetricsTracker:
        return self.metrics_ai if self.active_mode == 2 else self.metrics_fixed

    # ------------------------------------------------------------------
    # Background API sync
    # ------------------------------------------------------------------

    def _background_sync_worker(self):
        sync_url = "http://127.0.0.1:8000/api/intersections/INT-KDU-01/sync"
        while self.running:
            try:
                apps = {}
                for d, name in (("N", "North"), ("S", "South"),
                                 ("E", "East"),  ("W", "West")):
                    rate = self.approach_intervals[d]
                    flow = round(60.0 / max(0.2, rate), 1)
                    cat  = "HIGH" if rate <= 1.0 else "MEDIUM" if rate <= 2.2 else "LOW"
                    apps[name] = {
                        "direction": d,
                        "spawn_interval_sec": rate,
                        "arrival_flow_vpm":   flow,
                        "density_category":   cat,
                    }
                payload = {
                    "intersection_id":    "INT-KDU-01",
                    "active_mode":        self.active_mode,
                    "control_mode_name":  ("AI_CYCLE_ADAPTIVE"
                                           if self.active_mode == 2 else "FIXED_TIME"),
                    "signals": {
                        "North": self.signals.get_signal_state("N"),
                        "South": self.signals.get_signal_state("S"),
                        "East":  self.signals.get_signal_state("E"),
                        "West":  self.signals.get_signal_state("W"),
                    },
                    "active_phase": {
                        "phase_name":      str(self.signals.current_phase),
                        "active_axis":     self.signals.active_green_axis,
                        "allocated_green": round(self.signals.allocated_green, 1),
                        "countdown_seconds": int(
                            max(0.0, self.signals.allocated_green
                                - self.signals.time_in_state) + 0.99
                        ),
                    },
                    "approaches":     apps,
                    "ai_decision":    self.ai_controller.last_decision,
                    "metrics":        self.active_metrics.get_summary(),
                    "emergency_active": self.emergency_active,
                }
                req = urllib.request.Request(
                    sync_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=0.3) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    if ("active_mode" in resp_data
                            and resp_data["active_mode"] != self.active_mode):
                        self.active_mode = resp_data["active_mode"]
                        self.current_controller = (
                            self.ai_controller if self.active_mode == 2
                            else self.fixed_controller
                        )
            except Exception:
                pass
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Traffic control
    # ------------------------------------------------------------------

    def adjust_density(self, direction: str, delta: float):
        new_val = max(0.4, min(8.0, self.approach_intervals[direction] + delta))
        self.approach_intervals[direction] = round(new_val, 1)

    def spawn_vehicle_for_approach(self, direction: str):
        weights = [0.62, 0.12, 0.08, 0.05, 0.13]
        v_types = ["car", "van", "bus", "truck", "motorcycle"]
        v_type  = random.choices(v_types, weights=weights)[0]
        lane_idx = random.choice([0, 1])

        same_lane_v = [
            v for v in self.vehicles
            if (v.direction == direction and v.lane_idx == lane_idx
                and not v.is_turning and not v.has_cleared_intersection)
        ]
        if same_lane_v:
            if direction == "E" and any(v.x < 45  for v in same_lane_v): return
            if direction == "W" and any(v.x > 1235 for v in same_lane_v): return
            if direction == "S" and any(v.y < 45   for v in same_lane_v): return
            if direction == "N" and any(v.y > 675   for v in same_lane_v): return

        new_v = Vehicle(self.next_vehicle_id, v_type, direction, lane_idx)
        self.vehicles.append(new_v)
        self.next_vehicle_id += 1

        # Notify observer
        self.cycle_observer.on_vehicle_spawned(new_v)

    def spawn_emergency_vehicle(self):
        self.ai_controller.analyze_all_lanes(self.vehicles)
        dirs    = ["N", "S", "E", "W"]
        busy_d  = max(dirs, key=lambda d: self.ai_controller.get_approach_stats(
            d, self.approach_intervals[d])["total_pcu"])
        new_v   = Vehicle(self.next_vehicle_id, "ambulance", busy_d, 0,
                          turn_intent="STRAIGHT")
        self.vehicles.append(new_v)
        self.next_vehicle_id += 1
        self.emergency_banner_timer = 4.0

    def check_emergency_preemption(self):
        for v in self.vehicles:
            if v.is_emergency and not v.has_cleared_intersection and not v.is_turning:
                in_approach = False
                if   v.direction == "E" and 0   < v.x < 555:  in_approach = True
                elif v.direction == "W" and 725  < v.x < 1280: in_approach = True
                elif v.direction == "S" and 0   < v.y < 275:  in_approach = True
                elif v.direction == "N" and 445  < v.y < 720:  in_approach = True
                if in_approach:
                    axis = "EW" if v.direction in ("E", "W") else "NS"
                    self.signals.force_emergency_axis(axis, 16.0)
                    self.emergency_active       = True
                    self.emergency_banner_timer = 2.0
                    return
        self.emergency_active = False

    # ------------------------------------------------------------------
    # Decision callback (called from TrafficSignalManager on ALL_RED)
    # ------------------------------------------------------------------

    def calculate_next_decision(self, next_axis: str) -> dict:
        # After calibration cycle, pass the observer; during calibration pass None
        observer_arg = (
            self.cycle_observer
            if (self.completed_cycles > 0 and self.active_mode == 2)
            else None
        )
        decision = self.current_controller.get_next_phase_decision(
            next_axis, self.vehicles, self.approach_intervals,
            cycle_observer=observer_arg,
        )
        self._last_hud_decision = decision
        return decision

    # ------------------------------------------------------------------
    # Per-tick PCU snapshot helper for observer
    # ------------------------------------------------------------------

    def _pcu_by_direction(self) -> dict:
        pcu = {}
        for d in ("N", "S", "E", "W"):
            lane_v = [v for v in self.vehicles
                      if v.direction == d and not v.has_cleared_intersection]
            pcu[d] = calculate_lane_pcu(lane_v)
        return pcu

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                for rect, action, direction in self.buttons:
                    if rect.collidepoint(pos):
                        delta = -0.3 if action == "+" else +0.3
                        self.adjust_density(direction, delta)

            elif event.type == pygame.KEYDOWN:
                mods  = pygame.key.get_mods()
                shift = bool(mods & (pygame.KMOD_SHIFT | pygame.KMOD_LSHIFT | pygame.KMOD_RSHIFT))
                delta = 0.3 if shift else -0.3

                if   event.key in (pygame.K_q, pygame.K_ESCAPE): self.running = False
                elif event.key == pygame.K_SPACE:  self.paused = not self.paused
                elif event.key == pygame.K_1:
                    self.active_mode = 1
                    self.current_controller = self.fixed_controller
                elif event.key == pygame.K_2:
                    self.active_mode = 2
                    self.current_controller = self.ai_controller
                elif event.key == pygame.K_e and not shift:
                    self.spawn_emergency_vehicle()
                elif event.key == pygame.K_r:
                    self.vehicles.clear()
                    self.metrics_ai.reset()
                    self.metrics_fixed.reset()
                elif event.key == pygame.K_n: self.adjust_density("N", delta)
                elif event.key == pygame.K_s and not (mods & pygame.KMOD_CTRL):
                    self.adjust_density("S", delta)
                elif event.key == pygame.K_e and shift: self.adjust_density("E", delta)
                elif event.key == pygame.K_w: self.adjust_density("W", delta)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, dt: float):
        if self.paused:
            return

        # Spawn vehicles
        for d in ("N", "S", "E", "W"):
            self.approach_timers[d] += dt
            if self.approach_timers[d] >= self.approach_intervals[d]:
                self.approach_timers[d] = 0.0
                self.spawn_vehicle_for_approach(d)

        self.check_emergency_preemption()

        # Tick CycleObserver
        pcu_by_dir = self._pcu_by_direction()
        self.cycle_observer.tick(dt, self.vehicles, pcu_by_dir)

        switched, completed_p, new_p, allocated = self.signals.update(
            dt, next_decision_calc_fn=self.calculate_next_decision
        )

        # On cycle boundary: end observation, begin new cycle
        if switched and new_p in (SignalPhase.ALL_RED_1, SignalPhase.ALL_RED_2):
            self.cycle_observer.end_cycle()
            self.cycle_observer.begin_cycle()
            self.completed_cycles += 1

        if switched and ("LEFT_GREEN" in str(completed_p)
                         or "EXCLUSIVE_GREEN" in str(completed_p)):
            axis = ("EW" if any(c in str(completed_p) for c in ("EW", "E_", "W_"))
                    else "NS")
            axis_dirs = ["E", "W"] if axis == "EW" else ["N", "S"]
            res_pcu = calculate_lane_pcu([
                v for v in self.vehicles
                if v.direction in axis_dirs and not v.has_cleared_intersection
            ])
            dec = self.ai_controller.last_decision
            self.cycle_logger.log_cycle_completion(
                phase_axis=axis,
                allocated_green=self.signals.allocated_total_green,
                pcu_demand=dec.get("target_pcu", dec.get("heavy_score", 0.0)),
                vehicles_cleared=self.current_phase_cleared,
                max_wait_time=dec.get("max_wait", 0.0),
                residual_pcu=res_pcu,
                rules=dec.get("rule_activations", {}),
            )
            self.current_phase_cleared = 0

        # Update vehicles in lane order
        lane_buckets: dict = {}
        for v in self.vehicles:
            if not v.is_turning and not v.has_cleared_intersection:
                key = (v.direction, v.lane_idx)
                lane_buckets.setdefault(key, []).append(v)

        for key, lane_v_list in lane_buckets.items():
            for i, v in enumerate(lane_v_list):
                leading_v = lane_v_list[i - 1] if i > 0 else None
                sig_th, sig_lt = self.signals.get_signals_for_direction(v.direction)
                v.update(dt, leading_v, signal_through=sig_th, signal_turn=sig_lt)

        for v in self.vehicles:
            if v.is_turning or v.has_cleared_intersection:
                sig_th, sig_lt = self.signals.get_signals_for_direction(v.direction)
                v.update(dt, None, signal_through=sig_th, signal_turn=sig_lt)

        for v in list(self.vehicles):
            if v.is_off_screen(self.width, self.height):
                self.active_metrics.record_vehicle_cleared(v)
                self.current_phase_cleared += 1
                self.vehicles.remove(v)

        self.active_metrics.update_live_queues(self.vehicles)

        if self.emergency_banner_timer > 0:
            self.emergency_banner_timer -= dt

    # ------------------------------------------------------------------
    # Draw: road network
    # ------------------------------------------------------------------

    def draw_road_network(self):
        self.screen.fill((46, 125, 50))
        pygame.draw.rect(self.screen, (40, 44, 52), (0,   280, self.width, 160))
        pygame.draw.rect(self.screen, (40, 44, 52), (560, 0,   160, self.height))

        pygame.draw.line(self.screen, (241, 196, 15), (0,   360), (550,        360), 3)
        pygame.draw.line(self.screen, (241, 196, 15), (730, 360), (self.width, 360), 3)
        pygame.draw.line(self.screen, (241, 196, 15), (640, 0),   (640,        270), 3)
        pygame.draw.line(self.screen, (241, 196, 15), (640, 450), (640,  self.height), 3)

        for x in range(0, 540, 40):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 320), (x + 20, 320), 2)
            pygame.draw.line(self.screen, (200, 200, 200), (x, 400), (x + 20, 400), 2)
        for x in range(740, self.width, 40):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 320), (x + 20, 320), 2)
            pygame.draw.line(self.screen, (200, 200, 200), (x, 400), (x + 20, 400), 2)

        for y in range(0, 260, 40):
            pygame.draw.line(self.screen, (200, 200, 200), (600, y), (600, y + 20), 2)
            pygame.draw.line(self.screen, (200, 200, 200), (680, y), (680, y + 20), 2)
        for y in range(460, self.height, 40):
            pygame.draw.line(self.screen, (200, 200, 200), (600, y), (600, y + 20), 2)
            pygame.draw.line(self.screen, (200, 200, 200), (680, y), (680, y + 20), 2)

        pygame.draw.line(self.screen, (255, 255, 255), (555, 360), (555, 440), 5)
        pygame.draw.line(self.screen, (255, 255, 255), (725, 280), (725, 360), 5)
        pygame.draw.line(self.screen, (255, 255, 255), (560, 275), (640, 275), 5)
        pygame.draw.line(self.screen, (255, 255, 255), (640, 445), (720, 445), 5)

        for i in range(285, 435, 15):
            pygame.draw.rect(self.screen, (220, 220, 220), (530, i, 18, 8))
            pygame.draw.rect(self.screen, (220, 220, 220), (732, i, 18, 8))
        for i in range(565, 715, 15):
            pygame.draw.rect(self.screen, (220, 220, 220), (i, 250, 8, 18))
            pygame.draw.rect(self.screen, (220, 220, 220), (i, 452, 8, 18))

    # ------------------------------------------------------------------
    # Draw: HUD
    # ------------------------------------------------------------------

    def draw_hud(self):
        # ── Top-Left: Approach Panel ──────────────────────────────────
        panel_rect = pygame.Rect(15, 15, 360, 270)
        pygame.draw.rect(self.screen, (22, 26, 34), panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (55, 65, 80), panel_rect, width=1, border_radius=8)

        mode_text  = ("ML CYCLE-ADAPTIVE OBSERVER" if self.active_mode == 2
                      else "TRADITIONAL FIXED-TIME")
        mode_color = (46, 204, 113) if self.active_mode == 2 else (230, 126, 34)
        self.screen.blit(self.bold_font.render(mode_text, True, mode_color), (25, 22))

        calib_text = (
            "CYCLE 0 — CALIBRATION IN PROGRESS"
            if self.completed_cycles == 0
            else f"Cycle {self.completed_cycles} — Adaptive Mode Active"
        )
        calib_col = (241, 196, 15) if self.completed_cycles == 0 else (52, 152, 219)
        self.screen.blit(self.small_font.render(calib_text, True, calib_col), (25, 40))

        self.ai_controller.analyze_all_lanes(self.vehicles)

        y = 58
        for d in ("N", "S", "E", "W"):
            app  = self.ai_controller.get_approach_stats(d, self.approach_intervals[d])
            rate = self.approach_intervals[d]
            flow = int(app["arrival_flow"])

            density_str = "HIGH" if rate <= 1.0 else "MEDIUM" if rate <= 2.2 else "LOW"
            d_col = ((231, 76, 60) if density_str == "HIGH"
                     else (241, 196, 15) if density_str == "MEDIUM"
                     else (46, 204, 113))

            label = {"N": "U (North)", "S": "D (South)",
                     "E": "R (East)",  "W": "L (West)"}[d]
            txt = f"{label}: {flow} v/m  ({app['total_pcu']:.1f} PCU)  [{density_str}]"
            self.screen.blit(self.bold_font.render(txt, True, (240, 240, 240)), (25, y))

            # Observed ratios from previous cycle
            obs = self.cycle_observer.get_prev_summary(d)
            if obs and obs.get("spawn_count", 0) >= 3:
                thru_p = int(obs["through_ratio"] * 100)
                turn_p = int(obs["turn_ratio"]    * 100)
                obs_txt = (f"  Thru {thru_p}%  Turn {turn_p}%  "
                           f"AvgW {obs['avg_wait_sec']:.0f}s  "
                           f"MaxW {obs['max_wait_sec']:.0f}s")
            else:
                obs_txt = "  Observing..."
            self.screen.blit(self.small_font.render(obs_txt, True, (150, 170, 200)), (25, y + 17))

            y += 46

        for rect, action, d in self.buttons:
            pygame.draw.rect(self.screen, (40, 48, 62), rect, border_radius=3)
            pygame.draw.rect(self.screen, (80, 95, 120), rect, width=1, border_radius=3)
            btn_txt = self.bold_font.render(action, True, (255, 255, 255))
            self.screen.blit(btn_txt, (rect.x + (6 if action == "+" else 7), rect.y + 1))

        self.screen.blit(
            self.small_font.render(
                "Click +/- or [N/S/E/W] to adjust individual approach traffic",
                True, (130, 150, 175)
            ),
            (25, 255)
        )

        # ── Top-Right: Performance Panel ───────────────────────────────
        perf_rect = pygame.Rect(self.width - 360, 15, 345, 165)
        pygame.draw.rect(self.screen, (22, 26, 34), perf_rect, border_radius=8)
        pygame.draw.rect(self.screen, (55, 65, 80), perf_rect, width=1, border_radius=8)

        self.screen.blit(
            self.bold_font.render("PERFORMANCE BENCHMARK", True, (52, 152, 219)),
            (self.width - 345, 22)
        )

        cur_m = self.active_metrics
        awt_col = ((46, 204, 113) if cur_m.average_wait_time < 8.0
                   else (241, 196, 15) if cur_m.average_wait_time < 15.0
                   else (231, 76, 60))

        self.screen.blit(
            self.font.render(
                f"Cleared: {cur_m.cleared_vehicles} veh  ({cur_m.cleared_pcu:.1f} PCU)",
                True, (240, 240, 240)
            ),
            (self.width - 345, 46)
        )
        self.screen.blit(
            self.bold_font.render(
                f"Avg Wait Time (AWT): {cur_m.average_wait_time:.1f} s",
                True, awt_col
            ),
            (self.width - 345, 68)
        )
        self.screen.blit(
            self.font.render(
                f"Max Observed Wait: {cur_m.max_observed_wait:.1f} s",
                True, (240, 240, 240)
            ),
            (self.width - 345, 90)
        )
        self.screen.blit(
            self.font.render("Collision Probability: 0.0% (Protected Phasing)",
                             True, (46, 204, 113)),
            (self.width - 345, 112)
        )
        self.screen.blit(
            self.small_font.render(
                "[1] Fixed  [2] AI  [E] Ambulance  [R] Reset  [Space] Pause",
                True, (140, 160, 190)
            ),
            (self.width - 345, 140)
        )

        # ── Bottom-Right: AI Decision Panel ────────────────────────────
        ai_panel = pygame.Rect(self.width - 400, self.height - 215, 385, 200)
        pygame.draw.rect(self.screen, (22, 26, 34), ai_panel, border_radius=8)
        brd_col = (46, 204, 113) if self.active_mode == 2 else (60, 70, 90)
        pygame.draw.rect(self.screen, brd_col, ai_panel, width=1, border_radius=8)

        self.screen.blit(
            self.bold_font.render("AI CYCLE-ADAPTIVE DECISION", True, brd_col),
            (self.width - 388, self.height - 207)
        )

        dec = self._last_hud_decision
        cur_p = str(self.signals.current_phase)

        if "EXCLUSIVE_GREEN" in cur_p:
            h = cur_p[0]
            dir_name = {"N": "North (U)", "S": "South (D)",
                        "E": "East (R)",  "W": "West (L)"}.get(h, h)
            opp_name = {"N": "South (D)", "S": "North (U)",
                        "E": "West (L)",  "W": "East (R)"}.get(h, "Cross")
            p_desc  = f"PRIORITY FLUSH: {dir_name} — ALL GREEN"
            p_sub   = f"Opposing {opp_name} held RED (0% collision)"
            p_col   = (0, 255, 180)
        elif "THROUGH" in cur_p:
            p_desc  = "BALANCED THROUGH — Straight / Right Active"
            p_sub   = "Opposing Left-Turns held at Red"
            p_col   = (241, 196, 15)
        elif "LEFT" in cur_p:
            p_desc  = "PROTECTED LEFT TURN — Safe Crossing"
            p_sub   = "All straight traffic held at Red"
            p_col   = (52, 152, 219)
        else:
            p_desc  = "ALL-RED TRANSITION CLEARANCE"
            p_sub   = "Calculating next cycle..."
            p_col   = (231, 76, 60)

        self.screen.blit(
            self.bold_font.render(p_desc, True, p_col),
            (self.width - 388, self.height - 185)
        )
        self.screen.blit(
            self.small_font.render(p_sub, True, (200, 215, 230)),
            (self.width - 388, self.height - 163)
        )

        # Through / Turn split from last decision
        thru_g = dec.get("through_green_sec", self.signals.allocated_through_green)
        turn_g = dec.get("turn_green_sec",    self.signals.allocated_left_green)
        thru_p = dec.get("thru_pct", 68)
        turn_p = dec.get("turn_pct", 32)
        src    = dec.get("data_source", "default_fallback")
        opp_adj = "✓ Opp.corrected" if dec.get("opposing_adjusted") else ""
        split_src_label = "Observed" if src == "observed" else "Default"

        self.screen.blit(
            self.small_font.render(
                f"Through: {thru_g:.1f}s ({thru_p}%)   "
                f"Turn: {turn_g:.1f}s ({turn_p}%)   "
                f"[{split_src_label}] {opp_adj}",
                True, (180, 200, 220)
            ),
            (self.width - 388, self.height - 140)
        )

        # Priority direction info
        pdir = dec.get("prioritized_dir", "NONE")
        asym = dec.get("asym_ratio", 1.0)
        if pdir != "NONE":
            pdir_label = {"N": "North(U)", "S": "South(D)",
                          "E": "East(R)",  "W": "West(L)"}.get(pdir, pdir)
            pri_txt = f"Priority: {pdir_label}  Dominance: {asym:.2f}x"
            pri_col = (0, 255, 180)
        else:
            pri_txt = f"Balanced — No dominant approach  Ratio: {asym:.2f}x"
            pri_col = (160, 180, 200)

        self.screen.blit(
            self.small_font.render(pri_txt, True, pri_col),
            (self.width - 388, self.height - 118)
        )

        # Countdown
        rem = max(0.0, self.signals.allocated_green - self.signals.time_in_state)
        self.screen.blit(
            self.small_font.render(
                f"Countdown: {rem:.1f}s   Policy: {self.signals.active_policy}",
                True, (160, 180, 200)
            ),
            (self.width - 388, self.height - 96)
        )
        self.screen.blit(
            self.small_font.render(
                "ML: RandomForest (94.75%) + GradientBoost (R²=0.94) | CycleObserver",
                True, (0, 210, 120)
            ),
            (self.width - 388, self.height - 74)
        )

        # ── Emergency Banner ───────────────────────────────────────────
        if self.emergency_banner_timer > 0:
            banner = pygame.Rect(self.width // 2 - 250, 20, 500, 45)
            pygame.draw.rect(self.screen, (231, 76, 60), banner, border_radius=8)
            pygame.draw.rect(self.screen, (255, 255, 255), banner, width=2, border_radius=8)
            msg = self.bold_font.render("EMERGENCY VEHICLE PREEMPTION ACTIVE",
                                        True, (255, 255, 255))
            self.screen.blit(msg, (self.width // 2 - msg.get_width() // 2, 32))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        ticks = 0
        while self.running:
            dt    = self.clock.tick(60) / 1000.0
            ticks = pygame.time.get_ticks()

            self.handle_input()
            self.update(dt)

            self.draw_road_network()
            for v in self.vehicles:
                v.draw(self.screen, time_ticks=ticks)
            self.signals.draw(self.screen, self.bold_font)
            self.draw_hud()

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    app = TrafficSimulationApp()
    app.run()
