"""
4-Junction 2×2 Grid Traffic Simulation
General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI — Group 06

Layout:
    J1 (NW, cx=400, cy=230) ─── Seg_TOP ─── J2 (NE, cx=1040, cy=230)
          │                                         │
       Seg_LEFT                                Seg_RIGHT
          │                                         │
    J3 (SW, cx=400, cy=630) ─── Seg_BOT ─── J4 (SE, cx=1040, cy=630)

Each junction connects to exactly 2 others.
Vehicles can turn at any junction and enter a connecting road.
Lanes between junctions are capacity-managed: when a segment fills, the upstream
junction's departing direction is held RED until the segment drains.

Hotkeys:
    [1] Fixed-Time   [2] Fuzzy-ML (I2I on)   [3] Deep RL (I2I on)
    [E] Ambulance    [Space] Pause            [R] Reset    [Q]/[Esc] Quit
"""

import os
import sys
import math
import random
import pygame

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_engine.multi_junction.junction_node         import JunctionNode
from ai_engine.multi_junction.two_junction_vehicle  import TwoJunctionVehicle
from ai_engine.multi_junction.artery_segment        import ArterySegment
from ai_engine.multi_junction.four_junction_coordinator import FourJunctionCoordinator
from ai_engine.multi_junction.collision_manager     import CollisionManager


# ── Colours ───────────────────────────────────────────────────────────────────
COL_BG       = (46,  125,  50)
COL_ROAD     = (40,   44,  52)
COL_PANEL    = (20,   24,  32)
COL_PANEL_BD = (55,   65,  80)
COL_WHITE    = (255, 255, 255)
COL_YELLOW   = (241, 196,  15)
COL_DASH     = (200, 200, 200)
COL_CYAN     = (0,   240, 255)
COL_GREEN    = (46,  204, 113)
COL_RED      = (231,  76,  60)
COL_ORANGE   = (230, 126,  34)
COL_TEXT     = (240, 240, 240)
COL_SUBTEXT  = (160, 180, 200)
COL_BAR_OK   = (46,  204, 113)
COL_BAR_WARN = (241, 196,  15)
COL_BAR_FULL = (231,  76,  60)


# ── Network routing table ─────────────────────────────────────────────────────
# (current_junction_id, vehicle_direction) -> next_junction_id
CONNECTIONS = {
    (1, 'E'): 2,   # J1 → J2 via Seg_TOP
    (1, 'S'): 3,   # J1 → J3 via Seg_LEFT
    (2, 'W'): 1,   # J2 → J1 via Seg_TOP
    (2, 'S'): 4,   # J2 → J4 via Seg_RIGHT
    (3, 'N'): 1,   # J3 → J1 via Seg_LEFT
    (3, 'E'): 4,   # J3 → J4 via Seg_BOT
    (4, 'N'): 2,   # J4 → J2 via Seg_RIGHT
    (4, 'W'): 3,   # J4 → J3 via Seg_BOT
}


class FourJunctionGridApp:
    # ── Canvas & road geometry ────────────────────────────────────────
    WIDTH  = 1440
    HEIGHT = 860
    ROAD_W = 160
    HALF_W = 80
    STOP   = 85    # stop-line offset from junction centre

    # ── Junction positions ────────────────────────────────────────────
    J_POS = {
        1: (400,  230),   # NW
        2: (1040, 230),   # NE
        3: (400,  630),   # SW
        4: (1040, 630),   # SE
    }
    J_NAMES = {1: "J1 NW", 2: "J2 NE", 3: "J3 SW", 4: "J4 SE"}

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(
            "AI 4-Junction 2×2 Grid Traffic Simulation | KDU IT3182 — Group 06"
        )
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock  = pygame.time.Clock()
        self.running = True
        self.paused  = False

        # Fonts
        self.big_font   = pygame.font.SysFont("Segoe UI", 20, bold=True)
        self.bold_font  = pygame.font.SysFont("Segoe UI", 13, bold=True)
        self.font       = pygame.font.SysFont("Segoe UI", 13)
        self.small_font = pygame.font.SysFont("Segoe UI", 11)
        self.tiny_font  = pygame.font.SysFont("Segoe UI",  9, bold=True)

        # ── 4 Junction Nodes ─────────────────────────────────────────
        self.junctions: dict[int, JunctionNode] = {}
        for j_id, (cx, cy) in self.J_POS.items():
            self.junctions[j_id] = JunctionNode(
                j_id, self.J_NAMES[j_id], cx=cx, cy=cy
            )
        j1, j2, j3, j4 = (self.junctions[i] for i in [1, 2, 3, 4])

        # ── 4 Artery Segments ────────────────────────────────────────
        #   Horizontal cap ≈ (1040-400-170)/40 ≈ 12 vehicles
        #   Vertical   cap ≈ (630-230-170)/40  ≈  6 vehicles
        self.seg_top   = ArterySegment("TOP",   j1, j2, orientation='EW', max_vehicles=12)
        self.seg_bot   = ArterySegment("BOT",   j3, j4, orientation='EW', max_vehicles=12)
        self.seg_left  = ArterySegment("LEFT",  j1, j3, orientation='NS', max_vehicles=6)
        self.seg_right = ArterySegment("RIGHT", j2, j4, orientation='NS', max_vehicles=6)
        self.segments  = [self.seg_top, self.seg_bot, self.seg_left, self.seg_right]

        # ── I2I Coordinator ──────────────────────────────────────────
        self.coordinator = FourJunctionCoordinator(
            junctions=self.junctions,
            segments=self.segments,
            enabled=True
        )

        # ── Mode ─────────────────────────────────────────────────────
        self.active_mode = 2
        self.set_global_mode(2)

        # ── Vehicles ─────────────────────────────────────────────────
        self.vehicles: list[TwoJunctionVehicle] = []
        self.vehicle_id_counter = 1

        # ── Network metrics ──────────────────────────────────────────
        self.network_cleared_count = 0
        self.network_cleared_waits: list[float] = []

        # ── Collision & Accident Manager (Strict Zero-Overlap Rule) ───
        self.collision_mgr = CollisionManager()

        # ── Spawn inflow points (8 outer edges) ──────────────────────
        # (key, junction_id, direction)
        self.inflow_points = [
            ("J1_W", 1, 'E'),   # EB enters from West  on top road
            ("J2_E", 2, 'W'),   # WB enters from East  on top road
            ("J3_W", 3, 'E'),   # EB enters from West  on bottom road
            ("J4_E", 4, 'W'),   # WB enters from East  on bottom road
            ("J1_N", 1, 'S'),   # SB enters from North on left road
            ("J2_N", 2, 'S'),   # SB enters from North on right road
            ("J3_S", 3, 'N'),   # NB enters from South on left road
            ("J4_S", 4, 'N'),   # NB enters from South on right road
        ]
        self.spawn_intervals = {p[0]: 2.6 for p in self.inflow_points}
        self.spawn_timers    = {p[0]: random.uniform(0.0, 2.4) for p in self.inflow_points}

    # ── Mode control ──────────────────────────────────────────────────

    def set_global_mode(self, mode_id: int):
        self.active_mode = mode_id
        for j in self.junctions.values():
            j.set_mode(mode_id)
        self.coordinator.enabled = (mode_id in (2, 3))

    # ── Spawning ──────────────────────────────────────────────────────

    def _spawn_pos(self, j: JunctionNode, direction: str, lane_offset: int) -> tuple:
        """Returns (x, y) spawn position just outside the screen edge."""
        if   direction == 'S': return (j.cx - lane_offset,            -35.0)
        elif direction == 'N': return (j.cx + lane_offset,  self.HEIGHT + 35.0)
        elif direction == 'E': return (-35.0,                j.cy + lane_offset)
        elif direction == 'W': return (self.WIDTH + 35.0,   j.cy - lane_offset)
        return (0.0, 0.0)

    def spawn_vehicle(self, inflow_key: str, j_id: int, direction: str,
                      v_type: str = None):
        j = self.junctions[j_id]
        lane_idx    = random.choices([0, 1], weights=[0.35, 0.65])[0]
        lane_offset = 20 if lane_idx == 0 else 60
        sx, sy = self._spawn_pos(j, direction, lane_offset)

        # Strict Collision Guard at Entryway: Ensure entire spawn approach is clear
        for ov in self.vehicles:
            if math.hypot(ov.x - sx, ov.y - sy) < (ov.length + 65.0):
                return

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
            spawn_pos=(sx, sy),
            target_j_id=j_id
        )
        self.vehicle_id_counter += 1
        self.vehicles.append(v)

    def spawn_ambulance(self):
        """Spawn an EB ambulance entering from the west on the top road (J1)."""
        j1 = self.junctions[1]
        v = TwoJunctionVehicle(
            vehicle_id=self.vehicle_id_counter,
            vehicle_type="ambulance",
            direction="E",
            lane_idx=1,
            spawn_pos=(-35.0, j1.cy + 60),
            target_j_id=1
        )
        self.vehicle_id_counter += 1
        self.vehicles.append(v)

    # ── Input ─────────────────────────────────────────────────────────

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if   event.key in (pygame.K_q, pygame.K_ESCAPE): self.running = False
                elif event.key == pygame.K_SPACE: self.paused = not self.paused
                elif event.key == pygame.K_1:     self.set_global_mode(1)
                elif event.key == pygame.K_2:     self.set_global_mode(2)
                elif event.key == pygame.K_3:     self.set_global_mode(3)
                elif event.key == pygame.K_e:     self.spawn_ambulance()
                elif event.key == pygame.K_r:
                    self.vehicles.clear()
                    self.network_cleared_count = 0
                    self.network_cleared_waits.clear()
                    self.collision_mgr.accident_count = 0
                    self.collision_mgr.accident_log.clear()
                    self.collision_mgr.active_accident_flash = 0.0
                    for j in self.junctions.values():
                        j.capacity_hold_directions.clear()
                        j.green_wave_active = False

    # ── Update Loop ───────────────────────────────────────────────────

    def update(self, dt: float):
        if self.paused:
            return

        # 1. Spawn vehicles
        for key, j_id, direction in self.inflow_points:
            self.spawn_timers[key] += dt
            if self.spawn_timers[key] >= self.spawn_intervals[key]:
                self.spawn_timers[key] = 0.0
                self.spawn_vehicle(key, j_id, direction)

        # 2. Update junction signals (AI / fixed-time controllers)
        for j in self.junctions.values():
            j.update(dt, self.vehicles)

        # 3. Update segment counts → apply / release capacity holds
        for seg in self.segments:
            seg.update(self.vehicles)

        # 4. I2I green-wave coordinator
        self.coordinator.update(dt, self.vehicles)

        # 5. Vehicle handoff using CONNECTIONS map
        #    As soon as a vehicle physically clears its current junction box,
        #    it immediately targets the next downstream junction on the corridor.
        for v in self.vehicles:
            if not v.has_cleared_intersection or v.is_turning:
                continue
            nxt_id = CONNECTIONS.get((v.current_junction_id, v.direction))
            if nxt_id is None:
                continue   # Direction exits the network; let is_off_screen handle it
            cur_j = self.junctions[v.current_junction_id]
            cleared_box = False
            if   v.direction == 'E' and v.x > cur_j.cx + self.STOP: cleared_box = True
            elif v.direction == 'W' and v.x < cur_j.cx - self.STOP: cleared_box = True
            elif v.direction == 'S' and v.y > cur_j.cy + self.STOP: cleared_box = True
            elif v.direction == 'N' and v.y < cur_j.cy - self.STOP: cleared_box = True

            if cleared_box:
                v.current_junction_id     = nxt_id
                v.has_cleared_intersection = False

        # 6. Continuous Corridor Car-Following:
        #    Group vehicles along continuous road corridors so vehicles traveling in the
        #    same direction on the same road ALWAYS maintain car-following headway,
        #    even across junction boundaries.
        corridor_buckets: dict = {}
        for v in self.vehicles:
            if not v.is_turning:
                if v.direction in ('E', 'W'):
                    road = 'TOP' if abs(v.y - 230) < abs(v.y - 630) else 'BOT'
                else:
                    road = 'LEFT' if abs(v.x - 400) < abs(v.x - 1040) else 'RIGHT'
                key = (road, v.direction, v.lane_idx)
                corridor_buckets.setdefault(key, []).append(v)

        for (road, d, _), v_list in corridor_buckets.items():
            # Sort so the leading vehicle along travel axis is first
            if   d == 'E': v_list.sort(key=lambda v: -v.x)
            elif d == 'W': v_list.sort(key=lambda v:  v.x)
            elif d == 'S': v_list.sort(key=lambda v: -v.y)
            elif d == 'N': v_list.sort(key=lambda v:  v.y)

            for i, v in enumerate(v_list):
                leading = v_list[i - 1] if i > 0 else None
                j = self.junctions.get(v.current_junction_id)
                if j is not None:
                    sig_th, sig_lt = j.get_signals_metered(v.direction)
                    can_enter = self.collision_mgr.is_box_clear_for_entry(
                        j.cx, j.cy, v.direction, self.vehicles, v.id
                    )
                else:
                    sig_th, sig_lt = 'GREEN', 'GREEN'
                    can_enter = True
                v.update(dt, leading, sig_th, sig_lt, j, can_enter_box=can_enter)

        # Turning vehicles: speed modulation if path ahead is obstructed
        for v in self.vehicles:
            if v.is_turning:
                j = self.junctions.get(v.current_junction_id)
                rad_a = math.radians(v.heading_angle)
                cos_a, sin_a = math.cos(rad_a), math.sin(rad_a)
                path_blocked = False
                for ov in self.vehicles:
                    if ov.id == v.id:
                        continue
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

        # 7. Collision Detection & Strict Zero-Overlap Invariant Enforcement
        self.collision_mgr.check_and_resolve_collisions(self.vehicles, dt)

        # 7. Remove off-screen vehicles
        for v in list(self.vehicles):
            if v.is_off_screen(self.WIDTH, self.HEIGHT):
                self.network_cleared_count += 1
                self.network_cleared_waits.append(v.wait_time)
                j = self.junctions.get(v.current_junction_id)
                if j:
                    j.active_metrics.record_vehicle_cleared(v)
                self.vehicles.remove(v)

        for j in self.junctions.values():
            j.active_metrics.update_live_queues(self.vehicles)

    # ── Drawing ───────────────────────────────────────────────────────

    def draw_roads(self):
        self.screen.fill(COL_BG)

        j1, j2, j3, j4 = (self.junctions[i] for i in [1, 2, 3, 4])
        cy_top = j1.cy
        cy_bot = j3.cy
        cx_lft = j1.cx
        cx_rgt = j2.cx

        # ── Road rectangles ──────────────────────────────────────────
        # Horizontal: top road, bottom road
        pygame.draw.rect(self.screen, COL_ROAD,
                         (0, cy_top - self.HALF_W, self.WIDTH, self.ROAD_W))
        pygame.draw.rect(self.screen, COL_ROAD,
                         (0, cy_bot - self.HALF_W, self.WIDTH, self.ROAD_W))
        # Vertical: left road, right road
        pygame.draw.rect(self.screen, COL_ROAD,
                         (cx_lft - self.HALF_W, 0, self.ROAD_W, self.HEIGHT))
        pygame.draw.rect(self.screen, COL_ROAD,
                         (cx_rgt - self.HALF_W, 0, self.ROAD_W, self.HEIGHT))

        # ── Capacity utilization overlays on road surface ─────────────
        self._draw_segment_overlays()

        # ── Yellow centre lines ───────────────────────────────────────
        def hline_segments(y, cx_pairs):
            """Draw horizontal centre line broken by junction boxes."""
            prev = 0
            for cx in sorted(cx_pairs):
                pygame.draw.line(self.screen, COL_YELLOW, (prev, y), (cx - self.HALF_W, y), 3)
                prev = cx + self.HALF_W
            pygame.draw.line(self.screen, COL_YELLOW, (prev, y), (self.WIDTH, y), 3)

        def vline_segments(x, cy_pairs):
            """Draw vertical centre line broken by junction boxes."""
            prev = 0
            for cy in sorted(cy_pairs):
                pygame.draw.line(self.screen, COL_YELLOW, (x, prev), (x, cy - self.HALF_W), 3)
                prev = cy + self.HALF_W
            pygame.draw.line(self.screen, COL_YELLOW, (x, prev), (x, self.HEIGHT), 3)

        hline_segments(cy_top, [cx_lft, cx_rgt])
        hline_segments(cy_bot, [cx_lft, cx_rgt])
        vline_segments(cx_lft, [cy_top, cy_bot])
        vline_segments(cx_rgt, [cy_top, cy_bot])

        # ── Dashed lane dividers ──────────────────────────────────────
        # Horizontal roads
        for cy in (cy_top, cy_bot):
            for y_off in (-40, 40):
                y = cy + y_off
                for x in range(0, self.WIDTH, 32):
                    in_jbox = any(
                        j.cx - self.HALF_W <= x <= j.cx + self.HALF_W
                        for j in self.junctions.values()
                    )
                    if not in_jbox:
                        pygame.draw.line(self.screen, COL_DASH, (x, y), (x + 16, y), 2)
        # Vertical roads
        for cx in (cx_lft, cx_rgt):
            for x_off in (-40, 40):
                x = cx + x_off
                for y in range(0, self.HEIGHT, 32):
                    in_jbox = any(
                        j.cy - self.HALF_W <= y <= j.cy + self.HALF_W
                        for j in self.junctions.values()
                    )
                    if not in_jbox:
                        pygame.draw.line(self.screen, COL_DASH, (x, y), (x, y + 16), 2)

        # ── Stop lines & junction boxes ───────────────────────────────
        for j in self.junctions.values():
            cx, cy = j.cx, j.cy
            pygame.draw.line(self.screen, COL_WHITE,
                             (cx - self.STOP, cy),             (cx - self.STOP, cy + self.HALF_W), 5)
            pygame.draw.line(self.screen, COL_WHITE,
                             (cx + self.STOP, cy - self.HALF_W), (cx + self.STOP, cy), 5)
            pygame.draw.line(self.screen, COL_WHITE,
                             (cx - self.HALF_W, cy - self.STOP), (cx, cy - self.STOP), 5)
            pygame.draw.line(self.screen, COL_WHITE,
                             (cx, cy + self.STOP),              (cx + self.HALF_W, cy + self.STOP), 5)
            pygame.draw.rect(self.screen, (60, 65, 75),
                             (cx - self.HALF_W, cy - self.HALF_W, self.ROAD_W, self.ROAD_W), width=1)

    def _draw_segment_overlays(self):
        """Translucent capacity-fill colour on each road segment."""
        for seg in self.segments:
            util = min(1.0, seg.utilization())
            if util < 0.02:
                continue
            if util < 0.50:  alpha = 35; col = COL_BAR_OK
            elif util < 0.80: alpha = 50; col = COL_BAR_WARN
            else:             alpha = 65; col = COL_BAR_FULL

            if seg.orientation == 'EW':
                fill_w = max(2, int((seg.pos_end - seg.pos_start) * util))
                surf   = pygame.Surface((fill_w, self.ROAD_W - 4), pygame.SRCALPHA)
                surf.fill((*col, alpha))
                self.screen.blit(surf, (seg.pos_start, seg.road_ref - self.HALF_W + 2))
            else:  # 'NS'
                fill_h = max(2, int((seg.pos_end - seg.pos_start) * util))
                surf   = pygame.Surface((self.ROAD_W - 4, fill_h), pygame.SRCALPHA)
                surf.fill((*col, alpha))
                self.screen.blit(surf, (seg.road_ref - self.HALF_W + 2, seg.pos_start))

    def draw_traffic_lights(self):
        for j in self.junctions.values():
            cx, cy = j.cx, j.cy
            specs = [
                ('S', (cx - 55, cy - 105)),   # North entrance (SB traffic)
                ('N', (cx + 55, cy + 105)),   # South entrance (NB traffic)
                ('E', (cx - 105, cy + 55)),   # West entrance  (EB traffic)
                ('W', (cx + 105, cy - 55)),   # East entrance  (WB traffic)
            ]
            for d, (px, py) in specs:
                sig_th, sig_lt = j.get_signals_metered(d)

                # Signal box
                bw, bh = 36, 42
                box = pygame.Rect(px - bw // 2, py - bh // 2, bw, bh)
                pygame.draw.rect(self.screen, (20, 24, 30), box, border_radius=4)
                pygame.draw.rect(self.screen, (70, 80, 95), box, width=1, border_radius=4)

                def col(s):
                    return (231, 76, 60) if s == 'RED' else \
                           (241, 196, 15) if s == 'YELLOW' else (46, 204, 113)

                pygame.draw.circle(self.screen, col(sig_lt), (px - 8, py - 4), 5)
                pygame.draw.circle(self.screen, col(sig_th), (px + 8, py - 4), 5)
                self.screen.blit(self.tiny_font.render("LT", True, (160, 180, 200)), (px - 14, py + 6))
                self.screen.blit(self.tiny_font.render("TH", True, (160, 180, 200)), (px + 3,  py + 6))

                if d in j.capacity_hold_directions:
                    self.screen.blit(
                        self.tiny_font.render("HOLD", True, COL_RED), (px - 14, py + 18)
                    )

    def draw_hud(self):
        # ── Top panel ────────────────────────────────────────────────
        top = pygame.Rect(10, 8, self.WIDTH - 20, 76)
        pygame.draw.rect(self.screen, COL_PANEL, top, border_radius=8)
        pygame.draw.rect(self.screen, COL_PANEL_BD, top, width=1, border_radius=8)

        MODE_INFO = {
            1: ("TRADITIONAL FIXED-TIME CONTROLLER",       COL_ORANGE),
            2: ("FUZZY-ML DYNAMIC CYCLE OPTIMIZER",        COL_GREEN),
            3: ("DEEP REINFORCEMENT LEARNING (DQN)",       COL_CYAN),
        }
        m_txt, m_col = MODE_INFO[self.active_mode]
        self.screen.blit(self.big_font.render(m_txt, True, m_col), (22, 13))
        self.screen.blit(self.small_font.render(
            "4-Junction 2×2 Grid  |  KDU IT3182 Essentials of AI — Group 06",
            True, COL_SUBTEXT), (22, 42))

        awt = (sum(self.network_cleared_waits) / len(self.network_cleared_waits)
               if self.network_cleared_waits else 0.0)
        max_w = max((v.wait_time for v in self.vehicles), default=0.0)
        awt_col = COL_GREEN if awt < 15 else COL_YELLOW if awt < 30 else COL_RED

        col_x = 520
        acc_cnt = self.collision_mgr.accident_count
        acc_col = COL_GREEN if acc_cnt == 0 else COL_RED

        for txt, c in [
            (f"Cleared: {self.network_cleared_count} veh",        COL_TEXT),
            (f"AWT: {awt:.1f} s",                                 awt_col),
            (f"Active: {len(self.vehicles)} veh",                 COL_TEXT),
            (f"I2I Waves: {self.coordinator.green_waves_triggered}", COL_CYAN),
            (f"Accidents: {acc_cnt}",                             acc_col),
        ]:
            self.screen.blit(self.bold_font.render(txt, True, c), (col_x, 18))
            col_x += 145

        rule_badge = "Zero-Overlap Rule: ENFORCED (0 Collisions)" if acc_cnt == 0 else f"⚠️ ACCIDENTS: {acc_cnt} RESOLVED"
        rule_badge_col = COL_CYAN if acc_cnt == 0 else COL_RED
        self.screen.blit(self.small_font.render(rule_badge, True, rule_badge_col), (520, 48))
        self.screen.blit(self.small_font.render(
            "[1] Fixed  [2] Fuzzy-ML  [3] Deep RL  [E] Ambulance  [Space] Pause  [R] Reset  [Q] Quit",
            True, COL_SUBTEXT), (780, 48))

        # ── Collision Alert Banner ────────────────────────────────────
        if self.collision_mgr.active_accident_flash > 0:
            banner_rect = pygame.Rect(self.WIDTH // 2 - 300, 92, 600, 32)
            pygame.draw.rect(self.screen, (180, 20, 20), banner_rect, border_radius=6)
            pygame.draw.rect(self.screen, (255, 255, 255), banner_rect, width=2, border_radius=6)
            alert_msg = f"⚠️ ACCIDENT PREVENTED! Emergency Separation Applied | Count: {acc_cnt}"
            self.screen.blit(self.bold_font.render(alert_msg, True, (255, 255, 255)), (banner_rect.x + 20, banner_rect.y + 6))

        # ── Junction labels & phase timers ────────────────────────────
        for j in self.junctions.values():
            lbl = self.bold_font.render(j.name, True, COL_WHITE)
            self.screen.blit(lbl, (j.cx - 42, j.cy - 160))
            phase_str = str(j.signals.current_phase).split('.')[-1]
            timer_str = f"{phase_str} | {j.signals.time_in_state:.1f}s"
            self.screen.blit(
                self.small_font.render(timer_str, True, COL_CYAN),
                (j.cx - 60, j.cy + 145)
            )

        # ── Segment utilization strip ────────────────────────────────
        self._draw_segment_hud()

    def _draw_segment_hud(self):
        """Draw one compact info bar per segment, floating near the segment midpoint."""
        for seg in self.segments:
            util    = min(1.0, seg.utilization())
            hold_up = seg.upstream_depart_dir   in seg.upstream_j.capacity_hold_directions
            hold_dn = seg.downstream_depart_dir in seg.downstream_j.capacity_hold_directions
            held    = hold_up or hold_dn
            bar_col = COL_BAR_OK if util < 0.50 else (COL_BAR_WARN if util < 0.80 else COL_BAR_FULL)

            if seg.orientation == 'EW':
                mid_x   = (seg.pos_start + seg.pos_end) // 2
                mid_y   = seg.road_ref
                bar_w   = min(140, seg.pos_end - seg.pos_start)
                bar_h   = 12
                bx      = mid_x - bar_w // 2
                by      = mid_y - 22          # just above road centre line
                fill_w  = max(2, int(bar_w * util))
            else:  # 'NS'
                mid_x   = seg.road_ref
                mid_y   = (seg.pos_start + seg.pos_end) // 2
                bar_w   = 12
                bar_h   = min(100, seg.pos_end - seg.pos_start)
                bx      = mid_x - 35           # left of road
                by      = mid_y - bar_h // 2
                fill_w  = bar_w                # full-width for vertical; vary height
                # override: draw a horizontal mini bar rotated to vertical conceptually
                bar_h   = max(2, int(100 * util)) if seg.pos_end > seg.pos_start else bar_h

            # Draw background + fill
            bg_rect = pygame.Rect(bx - 1, by - 1, bar_w + 2, 14)
            pygame.draw.rect(self.screen, (30, 35, 45), bg_rect, border_radius=3)
            fill_rect = pygame.Rect(bx, by, fill_w, 12)
            pygame.draw.rect(self.screen, bar_col, fill_rect, border_radius=3)
            pygame.draw.rect(self.screen, COL_PANEL_BD, bg_rect, width=1, border_radius=3)

            # Label
            hold_str = " ⛔" if held else ""
            label = f"Seg {seg.seg_id}: {seg.current_count}/{seg.max_vehicles}{hold_str}"
            ls = self.tiny_font.render(label, True, COL_TEXT)
            if seg.orientation == 'EW':
                self.screen.blit(ls, (bx, by + 16))
            else:
                self.screen.blit(ls, (bx - 30, by + bar_h // 2 - 6))

    # ── Main Loop ─────────────────────────────────────────────────────

    def run(self):
        dt = 1.0 / 60.0
        while self.running:
            self.handle_input()
            self.update(dt)

            self.draw_roads()
            ticks = pygame.time.get_ticks()
            for v in self.vehicles:
                v.draw(self.screen, time_ticks=ticks)
            self.draw_traffic_lights()
            self.draw_hud()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    app = FourJunctionGridApp()
    app.run()
