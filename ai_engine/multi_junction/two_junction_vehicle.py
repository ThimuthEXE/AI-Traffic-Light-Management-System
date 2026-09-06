"""
Vehicle Kinematics & Multi-Junction Navigation Module
General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI — Group 06

Features:
- Individual vehicle dynamics, PCU properties, braking, queuing
- Smooth Bézier curve turns at any junction (J1 or J2)
- Strict Protected Phasing signal compliance
- Collision-free car-following & arterial corridor handoffs
"""

import math
import random
import pygame
from ai_engine.utils.pcu_calculator import get_pcu_weight

VEHICLE_CONFIGS = {
    "car": {
        "length": 34, "width": 18, "max_speed": 4.5, "accel": 0.22, "decel": 0.38,
        "color": (52, 152, 219),  # Blue
        "pcu": 1.0
    },
    "van": {
        "length": 42, "width": 20, "max_speed": 4.0, "accel": 0.19, "decel": 0.32,
        "color": (155, 89, 182),  # Purple
        "pcu": 1.2
    },
    "bus": {
        "length": 65, "width": 24, "max_speed": 3.4, "accel": 0.15, "decel": 0.26,
        "color": (241, 196, 15),  # Yellow
        "pcu": 2.5
    },
    "truck": {
        "length": 75, "width": 24, "max_speed": 3.0, "accel": 0.12, "decel": 0.22,
        "color": (230, 126, 34),  # Orange
        "pcu": 3.0
    },
    "motorcycle": {
        "length": 22, "width": 10, "max_speed": 5.0, "accel": 0.28, "decel": 0.42,
        "color": (46, 204, 113),  # Green
        "pcu": 0.5
    },
    "ambulance": {
        "length": 48, "width": 22, "max_speed": 5.5, "accel": 0.35, "decel": 0.45,
        "color": (231, 76, 60),   # Red
        "pcu": 1.0
    }
}


def get_turn_path(cx: float, cy: float, direction: str, turn_intent: str) -> dict:
    """Computes Bézier turning control points relative to junction center (cx, cy)."""
    if direction == 'E' and turn_intent == 'LEFT':
        return {'p0': (cx - 85, cy + 20), 'p1': (cx + 20, cy + 20), 'p2': (cx + 20, cy - 85), 'exit_dir': 'N', 'length': 155.0}
    elif direction == 'E' and turn_intent == 'RIGHT':
        return {'p0': (cx - 85, cy + 55), 'p1': (cx - 55, cy + 55), 'p2': (cx - 55, cy + 85), 'exit_dir': 'S', 'length': 75.0}
    elif direction == 'W' and turn_intent == 'LEFT':
        return {'p0': (cx + 85, cy - 20), 'p1': (cx - 20, cy - 20), 'p2': (cx - 20, cy + 85), 'exit_dir': 'S', 'length': 155.0}
    elif direction == 'W' and turn_intent == 'RIGHT':
        return {'p0': (cx + 85, cy - 55), 'p1': (cx + 55, cy - 55), 'p2': (cx + 55, cy - 85), 'exit_dir': 'N', 'length': 75.0}
    elif direction == 'S' and turn_intent == 'LEFT':
        return {'p0': (cx - 20, cy - 85), 'p1': (cx - 20, cy + 20), 'p2': (cx + 85, cy + 20), 'exit_dir': 'E', 'length': 155.0}
    elif direction == 'S' and turn_intent == 'RIGHT':
        return {'p0': (cx - 55, cy - 85), 'p1': (cx - 55, cy - 55), 'p2': (cx - 85, cy - 55), 'exit_dir': 'W', 'length': 75.0}
    elif direction == 'N' and turn_intent == 'LEFT':
        return {'p0': (cx + 20, cy + 85), 'p1': (cx + 20, cy - 20), 'p2': (cx - 85, cy - 20), 'exit_dir': 'W', 'length': 155.0}
    elif direction == 'N' and turn_intent == 'RIGHT':
        return {'p0': (cx + 55, cy + 85), 'p1': (cx + 55, cy + 55), 'p2': (cx + 85, cy + 55), 'exit_dir': 'E', 'length': 75.0}
    return None


class TwoJunctionVehicle:
    def __init__(self, vehicle_id: int, vehicle_type: str, direction: str, lane_idx: int, spawn_pos: tuple, target_j_id: int):
        self.id = vehicle_id
        self.vehicle_type = vehicle_type
        self.direction = direction
        self.lane_idx = lane_idx
        self.current_junction_id = target_j_id

        # Turn Intent
        if self.vehicle_type == "ambulance":
            self.turn_intent = "STRAIGHT"
        elif lane_idx == 0:
            self.turn_intent = random.choices(["LEFT", "STRAIGHT"], weights=[0.85, 0.15])[0]
        else:
            self.turn_intent = random.choices(["STRAIGHT", "RIGHT"], weights=[0.75, 0.25])[0]

        cfg = VEHICLE_CONFIGS.get(vehicle_type, VEHICLE_CONFIGS["car"])
        self.length = cfg["length"]
        self.width = cfg["width"]
        self.max_speed = cfg["max_speed"]
        self.accel = cfg["accel"]
        self.decel = cfg["decel"]
        self.base_color = cfg["color"]
        self.pcu = cfg["pcu"]
        self.is_emergency = (vehicle_type == "ambulance")

        self.speed = self.max_speed * random.uniform(0.92, 1.0)
        self.wait_time = 0.0
        self.has_cleared_intersection = False

        # Initial Position
        self.x, self.y = spawn_pos
        self.heading_angle = 0.0
        self._init_heading()

        # Turning state
        self.is_turning = False
        self.turn_progress = 0.0
        self.turn_path_data = None

    def _init_heading(self):
        if self.direction == 'E': self.heading_angle = 0.0
        elif self.direction == 'W': self.heading_angle = 180.0
        elif self.direction == 'S': self.heading_angle = 90.0
        elif self.direction == 'N': self.heading_angle = 270.0

    @property
    def has_cleared_junction(self) -> bool:
        return self.has_cleared_intersection

    def update(self, dt: float, leading_vehicle, signal_through: str, signal_turn: str, junction):
        # 1. Determine relevant signal
        relevant_signal = signal_turn if self.turn_intent == "LEFT" else signal_through

        # 2. Check stop line for approaching junction
        stop_line = None
        if not self.has_cleared_intersection and junction is not None:
            stop_line = junction.get_stop_line(self.direction)

        # 3. Can cross check
        if stop_line is not None and not self.has_cleared_intersection and not self.is_turning:
            can_cross = (relevant_signal == 'GREEN')
            crossed = False
            if self.direction == 'E' and self.x >= stop_line: crossed = True
            elif self.direction == 'W' and self.x <= stop_line: crossed = True
            elif self.direction == 'S' and self.y >= stop_line: crossed = True
            elif self.direction == 'N' and self.y <= stop_line: crossed = True

            if crossed:
                if can_cross:
                    self.has_cleared_intersection = True
                    if self.turn_intent in ["LEFT", "RIGHT"]:
                        self._setup_turn(junction.cx, junction.cy)
                else:
                    # Clamp at stop line
                    if self.direction == 'E': self.x = stop_line - self.length / 2 - 2
                    elif self.direction == 'W': self.x = stop_line + self.length / 2 + 2
                    elif self.direction == 'S': self.y = stop_line - self.length / 2 - 2
                    elif self.direction == 'N': self.y = stop_line + self.length / 2 + 2
                    self.speed = 0.0

        # 4. Car-Following Safety Distance
        safe_gap = 14.0
        target_distance = 9999.0

        if leading_vehicle is not None and not self.is_turning:
            if self.direction == 'E':
                dist = (leading_vehicle.x - leading_vehicle.length / 2) - (self.x + self.length / 2)
            elif self.direction == 'W':
                dist = (self.x - self.length / 2) - (leading_vehicle.x + leading_vehicle.length / 2)
            elif self.direction == 'S':
                dist = (leading_vehicle.y - leading_vehicle.length / 2) - (self.y + self.length / 2)
            elif self.direction == 'N':
                dist = (self.y - self.length / 2) - (leading_vehicle.y + leading_vehicle.length / 2)
            else:
                dist = 9999.0

            if dist > 0:
                target_distance = min(target_distance, dist)

        if stop_line is not None and not self.has_cleared_intersection and not self.is_turning and relevant_signal in ['RED', 'YELLOW']:
            if self.direction == 'E': dist_to_stop = stop_line - (self.x + self.length / 2)
            elif self.direction == 'W': dist_to_stop = (self.x - self.length / 2) - stop_line
            elif self.direction == 'S': dist_to_stop = stop_line - (self.y + self.length / 2)
            elif self.direction == 'N': dist_to_stop = (self.y - self.length / 2) - stop_line
            else: dist_to_stop = 9999.0

            if dist_to_stop >= 0:
                target_distance = min(target_distance, dist_to_stop)

        # 5. Kinematic Speed Update
        top_speed = self.max_speed * (0.75 if self.is_turning else 1.0)

        if target_distance < safe_gap:
            self.speed = max(0.0, self.speed - self.decel * 2.5)
        elif target_distance < 80.0:
            desired = (target_distance - safe_gap) / 80.0 * top_speed
            if self.speed > desired:
                self.speed = max(0.0, self.speed - self.decel)
            else:
                self.speed = min(top_speed, self.speed + self.accel)
        else:
            self.speed = min(top_speed, self.speed + self.accel)

        move_dist = self.speed * (dt * 60.0)

        # 6. Position Advancement
        if self.is_turning:
            self.turn_progress += move_dist / self.turn_path_data['length']
            if self.turn_progress >= 1.0:
                self.is_turning = False
                self.direction = self.turn_path_data['exit_dir']
                self.x, self.y = self.turn_path_data['p2']
                self._init_heading()
            else:
                u = self.turn_progress
                p0 = self.turn_path_data['p0']
                p1 = self.turn_path_data['p1']
                p2 = self.turn_path_data['p2']
                self.x = (1 - u)**2 * p0[0] + 2 * (1 - u) * u * p1[0] + u**2 * p2[0]
                self.y = (1 - u)**2 * p0[1] + 2 * (1 - u) * u * p1[1] + u**2 * p2[1]
                dx = 2 * (1 - u) * (p1[0] - p0[0]) + 2 * u * (p2[0] - p1[0])
                dy = 2 * (1 - u) * (p1[1] - p0[1]) + 2 * u * (p2[1] - p1[1])
                self.heading_angle = math.degrees(math.atan2(dy, dx))
        else:
            if self.direction == 'E': self.x += move_dist
            elif self.direction == 'W': self.x -= move_dist
            elif self.direction == 'S': self.y += move_dist
            elif self.direction == 'N': self.y -= move_dist

        # 7. Wait Time Accumulation
        if self.speed < 0.2 and not self.has_cleared_intersection:
            self.wait_time += dt

    def _setup_turn(self, cx: float, cy: float):
        path_data = get_turn_path(cx, cy, self.direction, self.turn_intent)
        if path_data:
            self.is_turning = True
            self.turn_progress = 0.0
            self.turn_path_data = path_data

    def is_off_screen(self, screen_w: int, screen_h: int) -> bool:
        margin = 60
        return (self.x < -margin or self.x > screen_w + margin or self.y < -margin or self.y > screen_h + margin)

    def draw(self, surface: pygame.Surface, time_ticks: int = 0):
        veh_surf = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        color = self.base_color
        if self.is_emergency:
            flash = (time_ticks // 150) % 2
            color = (255, 50, 50) if flash else (255, 255, 255)

        body_rect = pygame.Rect(0, 0, self.length, self.width)
        pygame.draw.rect(veh_surf, color, body_rect, border_radius=4)
        pygame.draw.rect(veh_surf, (20, 20, 20), body_rect, width=1, border_radius=4)

        roof_rect = pygame.Rect(int(self.length * 0.28), int(self.width * 0.15), int(self.length * 0.44), int(self.width * 0.70))
        pygame.draw.rect(veh_surf, (40, 44, 52), roof_rect, border_radius=2)

        # Headlights
        pygame.draw.rect(veh_surf, (255, 240, 180), (self.length - 3, 2, 3, 3))
        pygame.draw.rect(veh_surf, (255, 240, 180), (self.length - 3, self.width - 5, 3, 3))

        # Blinkers
        if self.turn_intent in ["LEFT", "RIGHT"] and not self.has_cleared_intersection and not self.is_turning:
            blinker_on = (time_ticks // 220) % 2 == 0
            if blinker_on:
                by = 1 if self.turn_intent == "LEFT" else self.width - 4
                pygame.draw.circle(veh_surf, (255, 165, 0), (self.length - 4, by + 2), 3)

        rotated_surf = pygame.transform.rotate(veh_surf, -self.heading_angle)
        new_rect = rotated_surf.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated_surf, new_rect.topleft)
