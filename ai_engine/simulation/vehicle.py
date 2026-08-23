"""
Vehicle Kinematics & Modeling Module
Simulates individual vehicle dynamics, PCU properties, braking distance, and queuing.
"""

import pygame
import random
import math
from ai_engine.utils.pcu_calculator import get_pcu_weight

# Optimized vehicle kinematics for realistic saturation discharge rates
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
        "color": (231, 76, 60),   # Red / White with siren
        "pcu": 1.0
    }
}

class Vehicle:
    def __init__(self, vehicle_id: int, vehicle_type: str, direction: str, lane_idx: int = 0):
        self.id = vehicle_id
        self.vehicle_type = vehicle_type
        self.direction = direction  # 'N', 'S', 'E', 'W'
        self.lane_idx = lane_idx
        
        cfg = VEHICLE_CONFIGS.get(vehicle_type, VEHICLE_CONFIGS["car"])
        self.length = cfg["length"]
        self.width = cfg["width"]
        self.max_speed = cfg["max_speed"]
        self.accel = cfg["accel"]
        self.decel = cfg["decel"]
        self.base_color = cfg["color"]
        self.pcu = cfg["pcu"]
        self.is_emergency = (vehicle_type == "ambulance")
        
        self.speed = self.max_speed * random.uniform(0.9, 1.0)
        self.wait_time = 0.0  # seconds spent stationary in queue
        self.has_cleared_intersection = False

        self._init_position()

    def _init_position(self):
        if self.direction == 'E':
            self.x = -self.length - random.uniform(0, 20)
            self.y = 380 if self.lane_idx == 0 else 415
            self.stop_line = 555
        elif self.direction == 'W':
            self.x = 1280 + self.length + random.uniform(0, 20)
            self.y = 305 if self.lane_idx == 0 else 340
            self.stop_line = 725
        elif self.direction == 'S':
            self.x = 585 if self.lane_idx == 0 else 620
            self.y = -self.length - random.uniform(0, 20)
            self.stop_line = 275
        elif self.direction == 'N':
            self.x = 660 if self.lane_idx == 0 else 695
            self.y = 720 + self.length + random.uniform(0, 20)
            self.stop_line = 445

    def update(self, dt: float, leading_vehicle, signal_state: str):
        """
        Update vehicle position, speed, and queuing state using car-following model.
        """
        target_stop_pos = None
        
        # Check stop line status
        if not self.has_cleared_intersection:
            crossed = False
            if self.direction == 'E' and self.x >= self.stop_line:
                crossed = True
            elif self.direction == 'W' and self.x <= self.stop_line:
                crossed = True
            elif self.direction == 'S' and self.y >= self.stop_line:
                crossed = True
            elif self.direction == 'N' and self.y <= self.stop_line:
                crossed = True
                
            if crossed:
                self.has_cleared_intersection = True
            elif signal_state in ['RED', 'YELLOW']:
                target_stop_pos = self.stop_line

        # Check leading vehicle distance
        safe_gap = 10.0
        target_distance = 9999.0

        if leading_vehicle is not None:
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
            if self.direction == 'E':
                dist_to_stop = self.stop_line - (self.x + self.length / 2)
            elif self.direction == 'W':
                dist_to_stop = (self.x - self.length / 2) - self.stop_line
            elif self.direction == 'S':
                dist_to_stop = self.stop_line - (self.y + self.length / 2)
            elif self.direction == 'N':
                dist_to_stop = (self.y - self.length / 2) - self.stop_line
                
            if dist_to_stop >= 0:
                target_distance = min(target_distance, dist_to_stop)

        # Kinematic acceleration / deceleration
        top_speed = self.max_speed * (1.2 if self.has_cleared_intersection else 1.0)

        if target_distance < safe_gap:
            self.speed = max(0.0, self.speed - self.decel * 2.5)
        elif target_distance < 70.0:
            desired_speed = (target_distance - safe_gap) / 70.0 * top_speed
            if self.speed > desired_speed:
                self.speed = max(0.0, self.speed - self.decel)
            else:
                self.speed = min(top_speed, self.speed + self.accel)
        else:
            self.speed = min(top_speed, self.speed + self.accel)

        # Advance position
        move_amount = self.speed * (dt * 60.0)
        if self.direction == 'E':
            self.x += move_amount
        elif self.direction == 'W':
            self.x -= move_amount
        elif self.direction == 'S':
            self.y += move_amount
        elif self.direction == 'N':
            self.y -= move_amount

        # Accurately increment wait time only while delayed in queue before clearing stop line
        if self.speed < 0.3 and not self.has_cleared_intersection:
            self.wait_time += dt

    def is_off_screen(self, screen_w=1280, screen_h=720) -> bool:
        margin = 100
        return (self.x < -margin or self.x > screen_w + margin or
                self.y < -margin or self.y > screen_h + margin)

    def draw(self, surface: pygame.Surface, time_ticks: int = 0):
        if self.direction in ['E', 'W']:
            w, h = self.length, self.width
        else:
            w, h = self.width, self.length

        rect = pygame.Rect(0, 0, w, h)
        rect.center = (int(self.x), int(self.y))

        color = self.base_color
        if self.is_emergency:
            flash = (time_ticks // 150) % 2
            color = (255, 50, 50) if flash else (255, 255, 255)

        pygame.draw.rect(surface, color, rect, border_radius=4)
        pygame.draw.rect(surface, (20, 20, 20), rect, width=1, border_radius=4)

        if self.direction in ['E', 'W']:
            roof = pygame.Rect(0, 0, int(w * 0.4), int(h * 0.7))
        else:
            roof = pygame.Rect(0, 0, int(w * 0.7), int(h * 0.4))
        roof.center = rect.center
        pygame.draw.rect(surface, (40, 40, 40), roof, border_radius=2)

        if self.is_emergency:
            pygame.draw.circle(surface, (255, 0, 0), rect.center, 5)
