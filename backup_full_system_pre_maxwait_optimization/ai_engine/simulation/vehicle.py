"""
Vehicle Kinematics & Turning Model Module
Simulates individual vehicle dynamics, PCU properties, braking, queuing,
and smooth multi-directional turns with strict Protected Phasing signal compliance.
General Sir John Kotelawala Defence University (KDU) - IT 3182 Essentials of AI
"""

import math
import random
import pygame
from ai_engine.utils.pcu_calculator import get_pcu_weight

VEHICLE_CONFIGS = {
    "car": {
        "length": 34, "width": 18, "max_speed": 5.2, "accel": 0.28, "decel": 0.40,
        "color": (52, 152, 219),  # Blue
        "pcu": 1.0
    },
    "van": {
        "length": 42, "width": 20, "max_speed": 4.8, "accel": 0.24, "decel": 0.35,
        "color": (155, 89, 182),  # Purple
        "pcu": 1.2
    },
    "bus": {
        "length": 65, "width": 24, "max_speed": 4.0, "accel": 0.18, "decel": 0.30,
        "color": (241, 196, 15),  # Yellow
        "pcu": 2.5
    },
    "truck": {
        "length": 75, "width": 24, "max_speed": 3.6, "accel": 0.15, "decel": 0.25,
        "color": (230, 126, 34),  # Orange
        "pcu": 3.0
    },
    "motorcycle": {
        "length": 22, "width": 10, "max_speed": 5.8, "accel": 0.35, "decel": 0.45,
        "color": (46, 204, 113),  # Green
        "pcu": 0.5
    },
    "ambulance": {
        "length": 48, "width": 22, "max_speed": 6.5, "accel": 0.45, "decel": 0.50,
        "color": (231, 76, 60),   # Red / White
        "pcu": 1.0
    }
}

# Bézier Turn Curves for 4 approaches (P0: entry stop line, P1: intersection control corner, P2: exit departure lane)
TURN_PATHS = {
    # Eastbound entry (moving +X)
    ('E', 'LEFT'):   {'p0': (555, 380), 'p1': (660, 380), 'p2': (660, 275), 'exit_dir': 'N', 'length': 155.0},
    ('E', 'RIGHT'):  {'p0': (555, 415), 'p1': (585, 415), 'p2': (585, 445), 'exit_dir': 'S', 'length': 75.0},

    # Westbound entry (moving -X)
    ('W', 'LEFT'):   {'p0': (725, 340), 'p1': (620, 340), 'p2': (620, 445), 'exit_dir': 'S', 'length': 155.0},
    ('W', 'RIGHT'):  {'p0': (725, 305), 'p1': (695, 305), 'p2': (695, 275), 'exit_dir': 'N', 'length': 75.0},

    # Southbound entry (moving +Y)
    ('S', 'LEFT'):   {'p0': (620, 275), 'p1': (620, 380), 'p2': (725, 380), 'exit_dir': 'E', 'length': 155.0},
    ('S', 'RIGHT'):  {'p0': (585, 275), 'p1': (585, 305), 'p2': (555, 305), 'exit_dir': 'W', 'length': 75.0},

    # Northbound entry (moving -Y)
    ('N', 'LEFT'):   {'p0': (660, 445), 'p1': (660, 340), 'p2': (555, 340), 'exit_dir': 'W', 'length': 155.0},
    ('N', 'RIGHT'):  {'p0': (695, 445), 'p1': (695, 415), 'p2': (725, 415), 'exit_dir': 'E', 'length': 75.0},
}


class Vehicle:
    def __init__(self, vehicle_id: int, vehicle_type: str, direction: str, lane_idx: int = 0, turn_intent: str = None):
        self.id = vehicle_id
        self.vehicle_type = vehicle_type
        self.direction = direction
        self.lane_idx = lane_idx
        
        if turn_intent is None:
            if self.vehicle_type == "ambulance":
                self.turn_intent = "STRAIGHT"
            elif lane_idx == 0:
                # Inner Lane: Straight (60%) or Protected Left Turn (40%)
                self.turn_intent = random.choices(["STRAIGHT", "LEFT"], weights=[0.60, 0.40])[0]
            else:
                # Outer Lane: Straight (65%) or Protected Right Turn (35%)
                self.turn_intent = random.choices(["STRAIGHT", "RIGHT"], weights=[0.65, 0.35])[0]
        else:
            self.turn_intent = turn_intent

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

        # Turning state
        self.is_turning = False
        self.turn_progress = 0.0
        self.turn_cfg = TURN_PATHS.get((self.direction, self.turn_intent))
        
        self.heading_angle = 0.0
        self._init_position()

    def _init_position(self):
        if self.direction == 'E':
            self.x = -self.length - random.uniform(0, 20)
            self.y = 380 if self.lane_idx == 0 else 415
            self.stop_line = 555
            self.heading_angle = 0.0
        elif self.direction == 'W':
            self.x = 1280 + self.length + random.uniform(0, 20)
            self.y = 340 if self.lane_idx == 0 else 305
            self.stop_line = 725
            self.heading_angle = 180.0
        elif self.direction == 'S':
            self.x = 620 if self.lane_idx == 0 else 585
            self.y = -self.length - random.uniform(0, 20)
            self.stop_line = 275
            self.heading_angle = 90.0
        elif self.direction == 'N':
            self.x = 660 if self.lane_idx == 0 else 695
            self.y = 720 + self.length + random.uniform(0, 20)
            self.stop_line = 445
            self.heading_angle = 270.0

    def update(self, dt: float, leading_vehicle, signal_through: str = 'RED', signal_turn: str = 'RED'):
        # Left-turn vehicles obey signal_turn; Straight/Right obey signal_through
        relevant_signal = signal_turn if self.turn_intent == "LEFT" else signal_through
        
        target_stop_pos = None

        # 1. Stop line check & intersection crossing
        if not self.has_cleared_intersection and not self.is_turning:
            can_cross = (relevant_signal == 'GREEN')
            
            if can_cross:
                crossed = False
                if self.direction == 'E' and self.x >= self.stop_line: crossed = True
                elif self.direction == 'W' and self.x <= self.stop_line: crossed = True
                elif self.direction == 'S' and self.y >= self.stop_line: crossed = True
                elif self.direction == 'N' and self.y <= self.stop_line: crossed = True

                if crossed:
                    if self.turn_intent != "STRAIGHT" and self.turn_cfg is not None:
                        self.is_turning = True
                        self.turn_progress = 0.0
                    else:
                        self.has_cleared_intersection = True
            else:
                # Strictly stop before the stop line on RED / YELLOW
                target_stop_pos = self.stop_line

        # 2. Car-following safety distance check
        safe_gap = 10.0
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
            target_distance = min(target_distance, dist)

        if target_stop_pos is not None:
            if self.direction == 'E': dist_to_stop = self.stop_line - (self.x + self.length / 2)
            elif self.direction == 'W': dist_to_stop = (self.x - self.length / 2) - self.stop_line
            elif self.direction == 'S': dist_to_stop = self.stop_line - (self.y + self.length / 2)
            elif self.direction == 'N': dist_to_stop = (self.y - self.length / 2) - self.stop_line
            
            if dist_to_stop >= 0:
                target_distance = min(target_distance, dist_to_stop)

        # 3. Kinematic speed update
        top_speed = self.max_speed * (0.75 if self.is_turning else 1.2 if self.has_cleared_intersection else 1.0)

        if target_distance < safe_gap:
            self.speed = max(0.0, self.speed - self.decel * 3.0)
        elif target_distance < 75.0:
            desired = (target_distance - safe_gap) / 75.0 * top_speed
            if self.speed > desired:
                self.speed = max(0.0, self.speed - self.decel)
            else:
                self.speed = min(top_speed, self.speed + self.accel)
        else:
            self.speed = min(top_speed, self.speed + self.accel)

        move_dist = self.speed * (dt * 60.0)

        # 4. Position advancement & Strict Stop Line Clamping on Red
        if not self.has_cleared_intersection and not self.is_turning and relevant_signal in ['RED', 'YELLOW']:
            if self.direction == 'E' and (self.x + self.length / 2 + move_dist) > self.stop_line:
                self.x = self.stop_line - self.length / 2 - 2
                self.speed = 0.0
                move_dist = 0.0
            elif self.direction == 'W' and (self.x - self.length / 2 - move_dist) < self.stop_line:
                self.x = self.stop_line + self.length / 2 + 2
                self.speed = 0.0
                move_dist = 0.0
            elif self.direction == 'S' and (self.y + self.length / 2 + move_dist) > self.stop_line:
                self.y = self.stop_line - self.length / 2 - 2
                self.speed = 0.0
                move_dist = 0.0
            elif self.direction == 'N' and (self.y - self.length / 2 - move_dist) < self.stop_line:
                self.y = self.stop_line + self.length / 2 + 2
                self.speed = 0.0
                move_dist = 0.0

        if self.is_turning and self.turn_cfg is not None:
            curve_len = self.turn_cfg['length']
            self.turn_progress += move_dist / curve_len

            if self.turn_progress >= 1.0:
                self.is_turning = False
                self.has_cleared_intersection = True
                self.direction = self.turn_cfg['exit_dir']
                p2 = self.turn_cfg['p2']
                self.x, self.y = p2[0], p2[1]
                
                if self.direction == 'E': self.heading_angle = 0.0
                elif self.direction == 'W': self.heading_angle = 180.0
                elif self.direction == 'S': self.heading_angle = 90.0
                elif self.direction == 'N': self.heading_angle = 270.0
            else:
                u = self.turn_progress
                p0 = self.turn_cfg['p0']
                p1 = self.turn_cfg['p1']
                p2 = self.turn_cfg['p2']

                self.x = (1 - u)**2 * p0[0] + 2 * (1 - u) * u * p1[0] + u**2 * p2[0]
                self.y = (1 - u)**2 * p0[1] + 2 * (1 - u) * u * p1[1] + u**2 * p2[1]

                dx = 2 * (1 - u) * (p1[0] - p0[0]) + 2 * u * (p2[0] - p1[0])
                dy = 2 * (1 - u) * (p1[1] - p0[1]) + 2 * u * (p2[1] - p1[1])
                self.heading_angle = math.degrees(math.atan2(dy, dx))
        else:
            if self.direction == 'E':
                self.x += move_dist
                self.heading_angle = 0.0
            elif self.direction == 'W':
                self.x -= move_dist
                self.heading_angle = 180.0
            elif self.direction == 'S':
                self.y += move_dist
                self.heading_angle = 90.0
            elif self.direction == 'N':
                self.y -= move_dist
                self.heading_angle = 270.0

        if self.speed < 0.3 and not self.has_cleared_intersection and not self.is_turning:
            self.wait_time += dt

    def is_off_screen(self, screen_w=1280, screen_h=720) -> bool:
        margin = 120
        return (self.x < -margin or self.x > screen_w + margin or
                self.y < -margin or self.y > screen_h + margin)

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

        # Flashing Turn Blinker
        if self.turn_intent in ["LEFT", "RIGHT"] and not self.has_cleared_intersection:
            blinker_on = (time_ticks // 220) % 2 == 0
            if blinker_on:
                by = 1 if self.turn_intent == "LEFT" else self.width - 4
                pygame.draw.circle(veh_surf, (255, 165, 0), (self.length - 4, by + 2), 3)

        if self.is_emergency:
            pygame.draw.circle(veh_surf, (255, 0, 0), (self.length // 2, self.width // 2), 4)

        rotated_surf = pygame.transform.rotate(veh_surf, -self.heading_angle)
        new_rect = rotated_surf.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated_surf, new_rect.topleft)
