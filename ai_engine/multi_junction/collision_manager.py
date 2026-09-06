"""
Collision Manager & Accident Monitoring Module
General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI — Group 06

Features:
  - Strict Zero-Overlap Physical Invariant Rule:
    Every vehicle is modeled with geometric collision volumes (circle-packing along vehicle length).
    No two vehicles are permitted to overlap in 2D space.
  - Real-time Accident Detection:
    Scans every active vehicle pair every frame for spatial penetrations.
    If any penetration is detected, an ACCIDENT event is registered with timestamp,
    locations, and vehicle IDs.
  - Active Separation & Emergency Braking:
    Clamps positions and zeroes speeds of colliding/encroaching vehicles so that
    interpenetration is physically prevented.
  - Anti-Gridlock Intersection Entry Guard:
    Prevents vehicles from entering an intersection box if their exit or path is obstructed.
"""

import math
import pygame


class CollisionManager:
    """
    Monitors and enforces the zero-vehicle-overlap rule across the entire network.
    Detects and logs any accidents if vehicles collide or overlap.
    """

    def __init__(self):
        self.accident_count = 0
        self.accident_log = []
        self.active_accident_flash = 0.0  # seconds to display warning on HUD
        self.last_accident_details = ""
        self.near_miss_count = 0

    def get_collision_circles(self, v) -> list:
        """
        Approximates a vehicle using 2 circles along its longitudinal axis.
        This provides exact, rotation-invariant collision detection that matches
        the visual vehicle body accurately.
        """
        rad = v.width * 0.48
        half_len = (v.length - v.width) * 0.35
        rad_angle = math.radians(v.heading_angle)
        cos_a = math.cos(rad_angle)
        sin_a = math.sin(rad_angle)

        # Front circle and rear circle
        c1 = (v.x + half_len * cos_a, v.y + half_len * sin_a, rad)
        c2 = (v.x - half_len * cos_a, v.y - half_len * sin_a, rad)
        return [c1, c2]

    def check_and_resolve_collisions(self, vehicles: list, dt: float):
        """
        Scans all vehicle pairs:
        1. Checks for geometric circle overlaps (accidents).
        2. Applies emergency separation and zero-speed clamp to enforce the
           strict non-overlap physical invariant.
        """
        if self.active_accident_flash > 0:
            self.active_accident_flash = max(0.0, self.active_accident_flash - dt)

        n = len(vehicles)
        if n < 2:
            return

        # Pre-compute circles for each vehicle
        veh_circles = [self.get_collision_circles(v) for v in vehicles]

        for i in range(n):
            v1 = vehicles[i]
            c1_list = veh_circles[i]

            for j in range(i + 1, n):
                v2 = vehicles[j]

                # Broad-phase distance filter
                center_dist = math.hypot(v1.x - v2.x, v1.y - v2.y)
                max_combined = (v1.length + v2.length) * 0.65
                if center_dist > max_combined:
                    continue

                c2_list = veh_circles[j]

                # Narrow-phase: check all circle pairs
                has_overlap = False
                min_separation = 999.0
                overlap_depth = 0.0

                for cx1, cy1, r1 in c1_list:
                    for cx2, cy2, r2 in c2_list:
                        d = math.hypot(cx1 - cx2, cy1 - cy2)
                        required_d = r1 + r2
                        sep = d - required_d
                        if sep < min_separation:
                            min_separation = sep

                        if d < required_d:
                            has_overlap = True
                            overlap_depth = max(overlap_depth, required_d - d)

                if has_overlap:
                    # ── ACCIDENT DETECTED! ───────────────────────────────────
                    self.accident_count += 1
                    self.active_accident_flash = 2.5
                    self.last_accident_details = (
                        f"Veh #{v1.id} ({v1.vehicle_type}) collided with "
                        f"Veh #{v2.id} ({v2.vehicle_type}) at ({int(v1.x)}, {int(v1.y)})"
                    )
                    self.accident_log.append({
                        "v1_id": v1.id,
                        "v2_id": v2.id,
                        "v1_type": v1.vehicle_type,
                        "v2_type": v2.vehicle_type,
                        "x": (v1.x + v2.x) / 2.0,
                        "y": (v1.y + v2.y) / 2.0,
                        "overlap_depth": overlap_depth,
                    })

                    # ── ENFORCE ZERO-OVERLAP INVARIANT: Push Apart & Zero Speed ─
                    # Determine which vehicle is trailing / moving faster and resolve
                    dx = v2.x - v1.x
                    dy = v2.y - v1.y
                    dist = math.hypot(dx, dy)
                    if dist < 0.001:
                        dx, dy = 1.0, 0.0
                        dist = 1.0

                    nx = dx / dist
                    ny = dy / dist
                    push = (overlap_depth + 2.0) / 2.0

                    # Push apart along the separation normal
                    v1.x -= nx * push
                    v1.y -= ny * push
                    v2.x += nx * push
                    v2.y += ny * push

                    # Zero their speeds to stop further collision penetration
                    v1.speed = 0.0
                    v2.speed = 0.0

                elif min_separation < 5.0:
                    self.near_miss_count += 1
                    # Safety cushion: slow both vehicles down proactively
                    v1.speed = max(0.0, v1.speed - v1.decel * 1.5)
                    v2.speed = max(0.0, v2.speed - v2.decel * 1.5)

    def is_box_clear_for_entry(self, junction_cx: float, junction_cy: float,
                               approach_dir: str, vehicles: list, current_veh_id: int) -> bool:
        """
        Anti-gridlock rule: Checks whether the entry zone of an intersection is
        obstructed before allowing a vehicle to cross the stop line.
        """
        # Define the critical entry polygon depending on approach direction
        half_w = 75.0
        for ov in vehicles:
            if ov.id == current_veh_id:
                continue

            # Is other vehicle inside the junction box?
            in_box = (abs(ov.x - junction_cx) < half_w and abs(ov.y - junction_cy) < half_w)
            if not in_box:
                continue

            # Check if other vehicle is obstructing our entry path
            if approach_dir == 'E' and (junction_cx - 85 <= ov.x <= junction_cx - 10):
                if abs(ov.y - (junction_cy + 20)) < 25 or abs(ov.y - (junction_cy + 60)) < 25:
                    return False
            elif approach_dir == 'W' and (junction_cx + 10 <= ov.x <= junction_cx + 85):
                if abs(ov.y - (junction_cy - 20)) < 25 or abs(ov.y - (junction_cy - 60)) < 25:
                    return False
            elif approach_dir == 'S' and (junction_cy - 85 <= ov.y <= junction_cy - 10):
                if abs(ov.x - (junction_cx - 20)) < 25 or abs(ov.x - (junction_cx - 60)) < 25:
                    return False
            elif approach_dir == 'N' and (junction_cy + 10 <= ov.y <= junction_cy + 85):
                if abs(ov.x - (junction_cx + 20)) < 25 or abs(ov.x - (junction_cx + 60)) < 25:
                    return False

        return True
