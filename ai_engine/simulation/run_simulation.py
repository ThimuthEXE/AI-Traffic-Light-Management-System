"""
AI Traffic Light Management System - Per-Approach Density Control & Granular Lane Simulation
General Sir John Kotelawala Defence University (KDU) - IT 3182 Essentials of AI
"""

import sys
import os
import math
import random
import pygame

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_engine.simulation.vehicle import Vehicle, VEHICLE_CONFIGS
from ai_engine.simulation.traffic_signal import TrafficSignalManager, SignalPhase
from ai_engine.simulation.controllers import FixedTimeController, AIFuzzyController
from ai_engine.simulation.metrics_tracker import MetricsTracker
from ai_engine.simulation.cycle_logger import CycleDataLogger
from ai_engine.utils.pcu_calculator import calculate_lane_pcu, classify_traffic_density


class TrafficSimulationApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("AI Traffic Light System - Flow Proportional & Lane-Aware Control | KDU IT3182")
        
        self.width = 1280
        self.height = 720
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False

        # Fonts
        self.title_font = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.font = pygame.font.SysFont("Segoe UI", 14)
        self.bold_font = pygame.font.SysFont("Segoe UI", 14, bold=True)
        self.small_font = pygame.font.SysFont("Segoe UI", 11)
        self.big_font = pygame.font.SysFont("Segoe UI", 22, bold=True)

        # Core Components
        self.signals = TrafficSignalManager(yellow_duration=2.5, all_red_duration=1.0)
        self.fixed_controller = FixedTimeController(fixed_green=25.0)
        self.ai_controller = AIFuzzyController(min_green=8.0, max_green=45.0)
        
        # Mode: 1 = Fixed-Time Baseline, 2 = AI Cycle Adaptive
        self.active_mode = 2
        self.current_controller = self.ai_controller

        # Metrics and Historical Cycle Logger
        self.metrics_ai = MetricsTracker()
        self.metrics_fixed = MetricsTracker()
        self.cycle_logger = CycleDataLogger()

        # Vehicles collection
        self.vehicles = []
        self.next_vehicle_id = 1

        # --- SEPARATE PER-APPROACH SPAWN DENSITIES ---
        # Lower interval = higher vehicle spawn density / arrival rate
        self.approach_intervals = {
            'N': 2.0,  # North Approach
            'S': 2.0,  # South Approach
            'E': 2.0,  # East Approach
            'W': 2.0   # West Approach
        }
        self.approach_timers = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}

        # UI Button rectangles
        self.buttons = []
        self._init_buttons()

        self.current_phase_cleared = 0

        # Emergency vehicle alert
        self.emergency_banner_timer = 0.0
        self.emergency_active = False

    def _init_buttons(self):
        self.buttons = []
        y = 70
        for d in ['N', 'S', 'E', 'W']:
            minus_rect = pygame.Rect(265, y - 2, 22, 20)
            plus_rect = pygame.Rect(292, y - 2, 22, 20)
            self.buttons.append((minus_rect, "-", d))
            self.buttons.append((plus_rect, "+", d))
            y += 42

    @property
    def active_metrics(self) -> MetricsTracker:
        return self.metrics_ai if self.active_mode == 2 else self.metrics_fixed

    def adjust_density(self, direction: str, delta: float):
        new_val = max(0.4, min(8.0, self.approach_intervals[direction] + delta))
        self.approach_intervals[direction] = round(new_val, 1)

    def spawn_vehicle_for_approach(self, direction: str):
        weights = [0.62, 0.12, 0.08, 0.05, 0.13]
        v_types = ["car", "van", "bus", "truck", "motorcycle"]
        v_type = random.choices(v_types, weights=weights)[0]

        lane_idx = random.choice([0, 1])

        # Precise bumper clearance check to allow dense convoys
        same_lane_v = [
            v for v in self.vehicles 
            if v.direction == direction and v.lane_idx == lane_idx
        ]
        if same_lane_v:
            if direction == 'E' and any(v.x < 45 for v in same_lane_v): return
            elif direction == 'W' and any(v.x > 1235 for v in same_lane_v): return
            elif direction == 'S' and any(v.y < 45 for v in same_lane_v): return
            elif direction == 'N' and any(v.y > 675 for v in same_lane_v): return

        new_v = Vehicle(self.next_vehicle_id, v_type, direction, lane_idx)
        self.vehicles.append(new_v)
        self.next_vehicle_id += 1

    def spawn_emergency_vehicle(self):
        self.ai_controller.analyze_all_lanes(self.vehicles)
        dirs = ['N', 'S', 'E', 'W']
        busy_dir = max(dirs, key=lambda d: self.ai_controller.get_approach_stats(d, self.approach_intervals[d])["total_pcu"])
        new_v = Vehicle(self.next_vehicle_id, "ambulance", busy_dir, random.choice([0, 1]))
        self.vehicles.append(new_v)
        self.next_vehicle_id += 1
        self.emergency_banner_timer = 4.0

    def check_emergency_preemption(self):
        for v in self.vehicles:
            if v.is_emergency and not v.has_cleared_intersection:
                in_approach = False
                if v.direction == 'E' and 0 < v.x < 555: in_approach = True
                elif v.direction == 'W' and 725 < v.x < 1280: in_approach = True
                elif v.direction == 'S' and 0 < v.y < 275: in_approach = True
                elif v.direction == 'N' and 445 < v.y < 720: in_approach = True
                
                if in_approach:
                    emergency_axis = 'EW' if v.direction in ['E', 'W'] else 'NS'
                    self.signals.force_emergency_axis(emergency_axis, emergency_green_time=15.0)
                    self.emergency_active = True
                    self.emergency_banner_timer = 2.0
                    return
        self.emergency_active = False

    def calculate_next_green(self, next_axis: str) -> float:
        return self.current_controller.get_next_green_duration(
            next_axis, self.vehicles, self.approach_intervals
        )

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                for rect, action, direction in self.buttons:
                    if rect.collidepoint(pos):
                        if action == "+":
                            self.adjust_density(direction, -0.3)  # Denser traffic
                        elif action == "-":
                            self.adjust_density(direction, +0.3)  # Lighter traffic
            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                shift = (mods & pygame.KMOD_SHIFT) or (mods & pygame.KMOD_LSHIFT) or (mods & pygame.KMOD_RSHIFT)
                delta = 0.3 if shift else -0.3

                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
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
                elif event.key == pygame.K_n:
                    self.adjust_density('N', delta)
                elif event.key == pygame.K_s and not (mods & pygame.KMOD_CTRL):
                    self.adjust_density('S', delta)
                elif event.key == pygame.K_e and shift:
                    self.adjust_density('E', delta)
                elif event.key == pygame.K_w:
                    self.adjust_density('W', delta)

    def update(self, dt: float):
        if self.paused:
            return

        # 1. Independent Spawning for each approach
        for d in ['N', 'S', 'E', 'W']:
            self.approach_timers[d] += dt
            if self.approach_timers[d] >= self.approach_intervals[d]:
                self.approach_timers[d] = 0.0
                self.spawn_vehicle_for_approach(d)

        # 2. Emergency Preemption
        self.check_emergency_preemption()

        # 3. Update Signal Clock with Flow Proportional Analysis
        switched, completed_p, new_p, allocated = self.signals.update(
            dt, next_green_duration_calc_fn=self.calculate_next_green
        )

        if switched and completed_p in [SignalPhase.EW_GREEN, SignalPhase.NS_GREEN]:
            axis = 'EW' if completed_p == SignalPhase.EW_GREEN else 'NS'
            axis_dirs = ['E', 'W'] if axis == 'EW' else ['N', 'S']
            res_pcu = calculate_lane_pcu([v for v in self.vehicles if v.direction in axis_dirs and not v.has_cleared_intersection])
            
            decision = getattr(self.ai_controller, "last_decision", {})
            self.cycle_logger.log_cycle_completion(
                phase_axis=axis,
                allocated_green=self.signals.allocated_green,
                pcu_demand=decision.get("target_pcu", 0.0),
                vehicles_cleared=self.current_phase_cleared,
                max_wait_time=decision.get("max_wait", 0.0),
                residual_pcu=res_pcu,
                rules=decision.get("rule_activations", {})
            )
            self.current_phase_cleared = 0

        # 4. Vehicle Physics Update
        self.vehicles.sort(key=lambda v: (
            v.x if v.direction == 'E' else -v.x if v.direction == 'W' else
            v.y if v.direction == 'S' else -v.y
        ), reverse=True)

        lane_buckets = {}
        for v in self.vehicles:
            key = (v.direction, v.lane_idx)
            if key not in lane_buckets:
                lane_buckets[key] = []
            lane_buckets[key].append(v)

        for key, lane_v_list in lane_buckets.items():
            for i, v in enumerate(lane_v_list):
                leading_v = lane_v_list[i - 1] if i > 0 else None
                sig_state = self.signals.get_signal_state(v.direction)
                v.update(dt, leading_v, sig_state)

        # 5. Record cleared vehicles
        for v in list(self.vehicles):
            if v.is_off_screen(self.width, self.height):
                self.active_metrics.record_vehicle_cleared(v)
                self.current_phase_cleared += 1
                self.vehicles.remove(v)

        self.active_metrics.update_live_queues(self.vehicles)

        if self.emergency_banner_timer > 0:
            self.emergency_banner_timer -= dt

    def draw_road_network(self):
        self.screen.fill((46, 125, 50))

        # Main Asphalt Surfaces
        pygame.draw.rect(self.screen, (40, 44, 52), (0, 280, self.width, 160))
        pygame.draw.rect(self.screen, (40, 44, 52), (560, 0, 160, self.height))

        # Center Lines
        pygame.draw.line(self.screen, (241, 196, 15), (0, 360), (550, 360), 3)
        pygame.draw.line(self.screen, (241, 196, 15), (730, 360), (self.width, 360), 3)
        pygame.draw.line(self.screen, (241, 196, 15), (640, 0), (640, 270), 3)
        pygame.draw.line(self.screen, (241, 196, 15), (640, 450), (640, self.height), 3)

        # Lane Markings
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

        # Stop Lines
        pygame.draw.line(self.screen, (255, 255, 255), (555, 360), (555, 440), 5)
        pygame.draw.line(self.screen, (255, 255, 255), (725, 280), (725, 360), 5)
        pygame.draw.line(self.screen, (255, 255, 255), (560, 275), (640, 275), 5)
        pygame.draw.line(self.screen, (255, 255, 255), (640, 445), (720, 445), 5)

        # Crosswalks
        for i in range(285, 435, 15):
            pygame.draw.rect(self.screen, (220, 220, 220), (530, i, 18, 8))
            pygame.draw.rect(self.screen, (220, 220, 220), (732, i, 18, 8))
        for i in range(565, 715, 15):
            pygame.draw.rect(self.screen, (220, 220, 220), (i, 250, 8, 18))
            pygame.draw.rect(self.screen, (220, 220, 220), (i, 452, 8, 18))

    def draw_hud(self):
        # Top-Left: Approach & Granular Lane Density Controller
        panel_rect = pygame.Rect(15, 15, 360, 245)
        pygame.draw.rect(self.screen, (22, 26, 34, 240), panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (55, 65, 80), panel_rect, width=1, border_radius=8)

        mode_text = "AI PROPORTIONAL FLOW CONTROL" if self.active_mode == 2 else "TRADITIONAL FIXED-TIME CONTROL"
        mode_color = (46, 204, 113) if self.active_mode == 2 else (230, 126, 34)
        self.screen.blit(self.bold_font.render(mode_text, True, mode_color), (25, 25))

        self.ai_controller.analyze_all_lanes(self.vehicles)

        y = 52
        dir_names = [('North', 'N'), ('South', 'S'), ('East', 'E'), ('West', 'W')]
        for name, d in dir_names:
            app = self.ai_controller.get_approach_stats(d, self.approach_intervals[d])
            rate = self.approach_intervals[d]
            flow_vpm = int(app["arrival_flow"])
            
            density_str = "HIGH" if rate <= 1.0 else "MEDIUM" if rate <= 2.2 else "LOW"
            d_col = (231, 76, 60) if density_str == "HIGH" else (241, 196, 15) if density_str == "MEDIUM" else (46, 204, 113)

            txt = f"{name}: {flow_vpm} v/m ({app['total_pcu']:.1f}p on road) [{density_str}]"
            self.screen.blit(self.bold_font.render(txt, True, (240, 240, 240)), (25, y))

            l0 = app["lane_0"]
            l1 = app["lane_1"]
            lane_sub = f"  L0: {l0['count']}v | L1: {l1['count']}v | Spwn: 1v/{rate:.1f}s"
            self.screen.blit(self.small_font.render(lane_sub, True, (170, 185, 200)), (25, y + 18))

            y += 42

        for rect, action, d in self.buttons:
            pygame.draw.rect(self.screen, (40, 48, 62), rect, border_radius=3)
            pygame.draw.rect(self.screen, (80, 95, 120), rect, width=1, border_radius=3)
            btn_txt = self.bold_font.render(action, True, (255, 255, 255))
            self.screen.blit(btn_txt, (rect.x + (6 if action == "+" else 7), rect.y + 1))

        self.screen.blit(self.small_font.render("Click +/- or press [N/S/E/W] to adjust individual approach traffic", True, (130, 150, 175)), (25, 230))

        # Top-Right: Performance Benchmark Card
        perf_rect = pygame.Rect(self.width - 360, 15, 345, 245)
        pygame.draw.rect(self.screen, (22, 26, 34, 240), perf_rect, border_radius=8)
        pygame.draw.rect(self.screen, (55, 65, 80), perf_rect, width=1, border_radius=8)

        self.screen.blit(self.bold_font.render("PERFORMANCE BENCHMARK (H1 vs H0)", True, (52, 152, 219)), (self.width - 345, 25))

        cur_m = self.active_metrics
        awt_color = (46, 204, 113) if cur_m.average_wait_time < 8.0 else (241, 196, 15) if cur_m.average_wait_time < 15.0 else (231, 76, 60)
        
        self.screen.blit(self.font.render(f"Cleared Vehicles: {cur_m.cleared_vehicles} ({cur_m.cleared_pcu:.1f} PCU)", True, (240, 240, 240)), (self.width - 345, 55))
        self.screen.blit(self.bold_font.render(f"Avg Wait Time (AWT): {cur_m.average_wait_time:.1f} sec", True, awt_color), (self.width - 345, 80))
        self.screen.blit(self.font.render(f"Max Waiting Time: {cur_m.max_observed_wait:.1f} sec", True, (240, 240, 240)), (self.width - 345, 105))
        self.screen.blit(self.font.render(f"Peak Observed Queue: {cur_m.max_observed_queue} vehicles", True, (240, 240, 240)), (self.width - 345, 130))
        
        summary_txt = f"Cars: {cur_m.vehicle_counts['car']} | Buses: {cur_m.vehicle_counts['bus']} | Trucks: {cur_m.vehicle_counts['truck']} | Bikes: {cur_m.vehicle_counts['motorcycle']}"
        self.screen.blit(self.small_font.render(summary_txt, True, (170, 170, 170)), (self.width - 345, 160))

        controls_txt = "[1] Fixed-Time  [2] AI Adaptive  [E] Ambulance  [R] Reset"
        self.screen.blit(self.small_font.render(controls_txt, True, (140, 160, 190)), (self.width - 345, 225))

        # Bottom-Right: Flow Proportional Decision Inspector
        ai_panel = pygame.Rect(self.width - 360, self.height - 170, 345, 155)
        pygame.draw.rect(self.screen, (22, 26, 34, 240), ai_panel, border_radius=8)
        pygame.draw.rect(self.screen, (46, 204, 113) if self.active_mode == 2 else (60, 70, 90), ai_panel, width=1, border_radius=8)

        self.screen.blit(self.bold_font.render("FLOW PROPORTIONAL DECISION ENGINE", True, (46, 204, 113) if self.active_mode == 2 else (230, 126, 34)), (self.width - 345, self.height - 160))
        
        dec = self.ai_controller.last_decision
        if dec:
            tgt_ax = dec.get("target_axis", "EW")
            g_time = dec.get("green_duration", 20.0)
            share = dec.get("target_ratio", 50.0)

            line1 = f"Active Phase {tgt_ax}: Demand Share = {share}% of Traffic"
            self.screen.blit(self.font.render(line1, True, (220, 220, 220)), (self.width - 345, self.height - 132))
            
            line2 = f"Allocated Green Countdown: {self.signals.allocated_green:.1f}s"
            self.screen.blit(self.bold_font.render(line2, True, (241, 196, 15)), (self.width - 345, self.height - 108))

            flow_line = f"Demand Weights -> NS: {dec.get('demand_NS', 0.0)} | EW: {dec.get('demand_EW', 0.0)}"
            self.screen.blit(self.small_font.render(flow_line, True, (160, 180, 200)), (self.width - 345, self.height - 84))

        self.screen.blit(self.small_font.render("Mode: Flow Arrival Rate + PCU Queue Proportional", True, (0, 230, 115)), (self.width - 345, self.height - 58))

        # Emergency Banner
        if self.emergency_banner_timer > 0:
            banner = pygame.Rect(self.width // 2 - 250, 20, 500, 45)
            pygame.draw.rect(self.screen, (231, 76, 60), banner, border_radius=8)
            pygame.draw.rect(self.screen, (255, 255, 255), banner, width=2, border_radius=8)
            msg = "EMERGENCY VEHICLE PREEMPTION ACTIVE"
            txt = self.bold_font.render(msg, True, (255, 255, 255))
            self.screen.blit(txt, (self.width // 2 - txt.get_width() // 2, 32))

    def run(self):
        ticks = 0
        while self.running:
            dt = self.clock.tick(60) / 1000.0
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
