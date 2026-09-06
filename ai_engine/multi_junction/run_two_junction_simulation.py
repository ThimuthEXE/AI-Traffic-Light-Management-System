"""
2-Junction Connected Arterial Network Simulation
General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI — Group 06

Features:
- 2 Interconnected 4-way Junctions (Junction 1: West, Junction 2: East)
- Full AI Phasing Optimization (Mode 1: Fixed-Time, Mode 2: Fuzzy-ML, Mode 3: Deep RL)
- Complete Protected Phasing Signals (Left-Turn + Through dual-heads)
- Seamless arterial corridor vehicle handoff
- Real-time HUD with live metrics per junction & network-wide stats
"""

import os
import sys
import math
import random
import pygame

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_engine.multi_junction.junction_node import JunctionNode
from ai_engine.multi_junction.two_junction_vehicle import TwoJunctionVehicle, VEHICLE_CONFIGS
from ai_engine.multi_junction.two_junction_coordinator import TwoJunctionCoordinator


class TwoJunctionSimulationApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(
            "AI 2-Junction Traffic Simulation — Deep RL & Fuzzy-ML | KDU IT3182"
        )

        self.width = 1440
        self.height = 760
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False

        # Fonts
        self.title_font = pygame.font.SysFont("Segoe UI", 18, bold=True)
        self.font       = pygame.font.SysFont("Segoe UI", 13)
        self.bold_font  = pygame.font.SysFont("Segoe UI", 13, bold=True)
        self.small_font = pygame.font.SysFont("Segoe UI", 11)
        self.big_font   = pygame.font.SysFont("Segoe UI", 22, bold=True)
        self.tiny_font  = pygame.font.SysFont("Segoe UI", 9, bold=True)

        # 2 Connected Junctions
        self.j1 = JunctionNode(1, "Junction 1 (West)", cx=400, cy=400)
        self.j2 = JunctionNode(2, "Junction 2 (East)", cx=1040, cy=400)
        self.junctions = {1: self.j1, 2: self.j2}
        self.coordinator = TwoJunctionCoordinator(self.junctions, enabled=True)

        # Active Mode
        self.active_mode = 2  # 1: Fixed, 2: Fuzzy-ML, 3: Deep RL
        self.set_global_mode(2)

        # Vehicles
        self.vehicles = []
        self.vehicle_id_counter = 1

        # Global Network Metrics
        self.network_cleared_count = 0
        self.network_cleared_waits = []

        # 6 Outer Perimeter Entry Inflows
        self.inflow_points = [
            ("J1_N", 1, 'S'),  # North entrance to J1
            ("J1_S", 1, 'N'),  # South entrance to J1
            ("J1_W", 1, 'E'),  # West entrance to J1
            ("J2_N", 2, 'S'),  # North entrance to J2
            ("J2_S", 2, 'N'),  # South entrance to J2
            ("J2_E", 2, 'W'),  # East entrance to J2
        ]
        self.spawn_intervals = {p[0]: 2.2 for p in self.inflow_points}
        self.spawn_timers = {p[0]: random.uniform(0.0, 2.0) for p in self.inflow_points}

    def set_global_mode(self, mode_id: int):
        self.active_mode = mode_id
        for j in self.junctions.values():
            j.set_mode(mode_id)

    def spawn_vehicle(self, inflow_key: str, j_id: int, direction: str, v_type: str = None):
        j = self.junctions[j_id]
        lane_idx = random.choices([0, 1], weights=[0.35, 0.65])[0]
        lane_offset = 20 if lane_idx == 0 else 60

        # Determine exact spawn position outside screen edge
        if direction == 'S':
            spawn_x = j.cx - lane_offset
            spawn_y = -35.0
        elif direction == 'N':
            spawn_x = j.cx + lane_offset
            spawn_y = self.height + 35.0
        elif direction == 'E':
            spawn_x = -35.0
            spawn_y = j.cy + lane_offset
        elif direction == 'W':
            spawn_x = self.width + 35.0
            spawn_y = j.cy - lane_offset
        else:
            spawn_x, spawn_y = 0, 0

        # Collision guard at entryway
        for ov in self.vehicles:
            if math.hypot(ov.x - spawn_x, ov.y - spawn_y) < 55.0:
                return  # Entryway occupied; skip spawn

        if v_type is None:
            v_type = random.choices(
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

    def spawn_ambulance(self):
        v = TwoJunctionVehicle(
            vehicle_id=self.vehicle_id_counter,
            vehicle_type="ambulance",
            direction="E",
            lane_idx=1,
            spawn_pos=(-35.0, self.j1.cy + 60),
            target_j_id=1
        )
        self.vehicle_id_counter += 1
        self.vehicles.append(v)

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_1:
                    self.set_global_mode(1)
                elif event.key == pygame.K_2:
                    self.set_global_mode(2)
                elif event.key == pygame.K_3:
                    self.set_global_mode(3)
                elif event.key == pygame.K_e:
                    self.spawn_ambulance()
                elif event.key == pygame.K_r:
                    self.vehicles.clear()
                    self.network_cleared_count = 0
                    self.network_cleared_waits.clear()

    def update(self, dt: float):
        if self.paused:
            return

        # 1. Spawn vehicles at outer borders
        for inflow_key, j_id, direction in self.inflow_points:
            self.spawn_timers[inflow_key] += dt
            if self.spawn_timers[inflow_key] >= self.spawn_intervals[inflow_key]:
                self.spawn_timers[inflow_key] = 0.0
                self.spawn_vehicle(inflow_key, j_id, direction)

        # 2. Update Junction Signals & AI Controllers
        for j in self.junctions.values():
            j.update(dt, self.vehicles)

        # 3. Inter-Intersection (I2I) Green Wave Coordinator (for AI modes 2 and 3)
        if self.active_mode in (2, 3):
            self.coordinator.update(dt, self.vehicles)

        # 4. Arterial Corridor Handoff between J1 and J2
        for v in self.vehicles:
            if v.has_cleared_intersection and not v.is_turning:
                # Eastbound vehicle cleared J1 -> hand off to J2
                if v.direction == 'E' and v.current_junction_id == 1 and v.x < self.j2.cx - 85:
                    v.current_junction_id = 2
                    v.has_cleared_intersection = False
                # Westbound vehicle cleared J2 -> hand off to J1
                elif v.direction == 'W' and v.current_junction_id == 2 and v.x > self.j1.cx + 85:
                    v.current_junction_id = 1
                    v.has_cleared_intersection = False

        # 4. Update Vehicles by lane order (lane-based queuing)
        lane_buckets = {}
        for v in self.vehicles:
            if not v.is_turning:
                # Group by junction, direction, and lane index
                key = (v.current_junction_id, v.direction, v.lane_idx)
                lane_buckets.setdefault(key, []).append(v)

        # Sort lane vehicles so leading vehicle is processed first
        for key, v_list in lane_buckets.items():
            j_id, d, lane_idx = key
            if d == 'E': v_list.sort(key=lambda v: -v.x)
            elif d == 'W': v_list.sort(key=lambda v: v.x)
            elif d == 'S': v_list.sort(key=lambda v: -v.y)
            elif d == 'N': v_list.sort(key=lambda v: v.y)

            for i, v in enumerate(v_list):
                leading_v = v_list[i - 1] if i > 0 else None
                j = self.junctions.get(v.current_junction_id, None)
                if j is not None:
                    sig_th, sig_lt = j.signals.get_signals_for_direction(v.direction)
                else:
                    sig_th, sig_lt = 'GREEN', 'GREEN'

                v.update(dt, leading_v, sig_th, sig_lt, j)

        # Update turning vehicles
        for v in self.vehicles:
            if v.is_turning:
                j = self.junctions.get(v.current_junction_id, None)
                v.update(dt, None, 'GREEN', 'GREEN', j)

        # 5. Remove off-screen vehicles & record network metrics
        for v in list(self.vehicles):
            if v.is_off_screen(self.width, self.height):
                self.network_cleared_count += 1
                self.network_cleared_waits.append(v.wait_time)
                if v.current_junction_id in self.junctions:
                    self.junctions[v.current_junction_id].active_metrics.record_vehicle_cleared(v)
                self.vehicles.remove(v)

        for j in self.junctions.values():
            j.active_metrics.update_live_queues(self.vehicles)

    def draw_roads(self):
        self.screen.fill((46, 125, 50))  # Green terrain
        road_col = (40, 44, 52)
        rw = 160
        hrw = rw // 2

        # ── Horizontal Arterial Road (Connecting J1 <-> J2) ──
        pygame.draw.rect(self.screen, road_col, (0, 400 - hrw, self.width, rw))

        # ── Vertical Roads at J1 and J2 ──
        pygame.draw.rect(self.screen, road_col, (self.j1.cx - hrw, 0, rw, self.height))
        pygame.draw.rect(self.screen, road_col, (self.j2.cx - hrw, 0, rw, self.height))

        # ── Center Yellow Lines ──
        # Horizontal East-West center line
        pygame.draw.line(self.screen, (241, 196, 15), (0, 400), (self.j1.cx - hrw, 400), 3)
        pygame.draw.line(self.screen, (241, 196, 15), (self.j1.cx + hrw, 400), (self.j2.cx - hrw, 400), 3)
        pygame.draw.line(self.screen, (241, 196, 15), (self.j2.cx + hrw, 400), (self.width, 400), 3)

        # Vertical center lines
        for j in (self.j1, self.j2):
            pygame.draw.line(self.screen, (241, 196, 15), (j.cx, 0), (j.cx, 400 - hrw), 3)
            pygame.draw.line(self.screen, (241, 196, 15), (j.cx, 400 + hrw), (j.cx, self.height), 3)

        # ── White Dashed Lane Separators ──
        for y_dash in (400 - 40, 400 + 40):
            for x in range(0, self.width, 32):
                # Skip inside junction boxes
                if (self.j1.cx - hrw <= x <= self.j1.cx + hrw) or (self.j2.cx - hrw <= x <= self.j2.cx + hrw):
                    continue
                pygame.draw.line(self.screen, (200, 200, 200), (x, y_dash), (x + 16, y_dash), 2)

        for j in (self.j1, self.j2):
            for x_dash in (j.cx - 40, j.cx + 40):
                for y in range(0, self.height, 32):
                    if 400 - hrw <= y <= 400 + hrw:
                        continue
                    pygame.draw.line(self.screen, (200, 200, 200), (x_dash, y), (x_dash, y + 16), 2)

        # ── Stop Lines for both Junctions ──
        for j in (self.j1, self.j2):
            # West entrance (Eastbound): x = cx - 85, y in [400, 480]
            pygame.draw.line(self.screen, (255, 255, 255), (j.cx - 85, 400), (j.cx - 85, 400 + hrw), 5)
            # East entrance (Westbound): x = cx + 85, y in [320, 400]
            pygame.draw.line(self.screen, (255, 255, 255), (j.cx + 85, 400 - hrw), (j.cx + 85, 400), 5)
            # North entrance (Southbound): y = 400 - 85, x in [cx - hrw, cx]
            pygame.draw.line(self.screen, (255, 255, 255), (j.cx - hrw, 400 - 85), (j.cx, 400 - 85), 5)
            # South entrance (Northbound): y = 400 + 85, x in [cx, cx + hrw]
            pygame.draw.line(self.screen, (255, 255, 255), (j.cx, 400 + 85), (j.cx + hrw, 400 + 85), 5)

            # Junction Box outline
            pygame.draw.rect(self.screen, (60, 65, 75), (j.cx - hrw, 400 - hrw, rw, rw), width=1)

    def draw_traffic_lights(self):
        for j in (self.j1, self.j2):
            signal_specs = [
                # Direction, Center Box Position (px, py)
                ('S', (j.cx - 55, 400 - 105)),  # North entrance (Southbound)
                ('N', (j.cx + 55, 400 + 105)),  # South entrance (Northbound)
                ('E', (j.cx - 105, 400 + 55)),  # West entrance (Eastbound)
                ('W', (j.cx + 105, 400 - 55)),  # East entrance (Westbound)
            ]

            for d, (px, py) in signal_specs:
                sig_th, sig_lt = j.signals.get_signals_for_direction(d)

                box_w, box_h = 36, 42
                box_rect = pygame.Rect(px - box_w // 2, py - box_h // 2, box_w, box_h)
                pygame.draw.rect(self.screen, (20, 24, 30), box_rect, border_radius=4)
                pygame.draw.rect(self.screen, (70, 80, 95), box_rect, width=1, border_radius=4)

                turn_col = (231, 76, 60) if sig_lt == 'RED' else (241, 196, 15) if sig_lt == 'YELLOW' else (46, 204, 113)
                th_col   = (231, 76, 60) if sig_th == 'RED' else (241, 196, 15) if sig_th == 'YELLOW' else (46, 204, 113)

                pygame.draw.circle(self.screen, turn_col, (px - 8, py - 4), 5)
                pygame.draw.circle(self.screen, th_col,   (px + 8, py - 4), 5)

                self.screen.blit(self.tiny_font.render("LT", True, (160, 180, 200)), (px - 14, py + 6))
                self.screen.blit(self.tiny_font.render("TH", True, (160, 180, 200)), (px + 3,  py + 6))

    def draw_hud(self):
        # ── Top Control & Metrics Panel ────────────────────────────────
        top_rect = pygame.Rect(15, 10, self.width - 30, 80)
        pygame.draw.rect(self.screen, (20, 24, 32), top_rect, border_radius=8)
        pygame.draw.rect(self.screen, (55, 65, 80), top_rect, width=1, border_radius=8)

        # Mode Text
        if self.active_mode == 3:
            m_txt, m_col = "DEEP REINFORCEMENT LEARNING (DQN CONTROLLERS)", (0, 240, 255)
        elif self.active_mode == 2:
            m_txt, m_col = "FUZZY-ML DYNAMIC CYCLE OPTIMIZER", (46, 204, 113)
        else:
            m_txt, m_col = "TRADITIONAL FIXED-TIME CONTROLLER", (230, 126, 34)

        self.screen.blit(self.big_font.render(m_txt, True, m_col), (30, 16))
        self.screen.blit(self.small_font.render("2-Junction Connected Arterial System | KDU IT3182 Essentials of AI", True, (160, 180, 200)), (30, 44))

        # Network Stats
        awt = float(sum(self.network_cleared_waits) / len(self.network_cleared_waits)) if self.network_cleared_waits else 0.0
        max_w = max((v.wait_time for v in self.vehicles), default=0.0)

        self.screen.blit(self.bold_font.render(f"Network Cleared: {self.network_cleared_count} veh", True, (240, 240, 240)), (580, 18))
        self.screen.blit(self.bold_font.render(f"Global AWT: {awt:.1f} s", True, (46, 204, 113) if awt < 15 else (241, 196, 15)), (580, 42))
        self.screen.blit(self.bold_font.render(f"Max Network Wait: {max_w:.1f} s", True, (240, 240, 240)), (780, 18))
        self.screen.blit(self.bold_font.render(f"Active Vehicles: {len(self.vehicles)} veh", True, (240, 240, 240)), (780, 42))

        # Shortcuts
        self.screen.blit(self.small_font.render("[1] Fixed-Time  [2] Fuzzy-ML  [3] Deep RL  [E] Ambulance  [Space] Pause  [R] Reset", True, (180, 200, 220)), (1000, 44))

        # ── Label each Junction Box & Show Local Signal Status ─────────
        for j in (self.j1, self.j2):
            # Junction Label
            j_lbl = self.bold_font.render(j.name, True, (255, 255, 255))
            self.screen.blit(j_lbl, (j.cx - 65, 400 - 145))

            # Phase & Timer Badge
            p_name = str(j.signals.current_phase).split('.')[-1]
            timer_txt = f"{p_name} | {j.signals.time_in_state:.1f}s"
            t_surf = self.small_font.render(timer_txt, True, (0, 240, 255))
            self.screen.blit(t_surf, (j.cx - 80, 400 + 130))

    def run(self):
        dt = 1.0 / 60.0
        while self.running:
            self.handle_input()
            self.update(dt)

            self.draw_roads()
            time_ticks = pygame.time.get_ticks()
            for v in self.vehicles:
                v.draw(self.screen, time_ticks=time_ticks)
            self.draw_traffic_lights()
            self.draw_hud()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    app = TwoJunctionSimulationApp()
    app.run()
