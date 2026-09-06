"""
2-Junction Inter-Intersection (I2I) Green Wave Synchronization Coordinator
General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI — Group 06

Features:
- Real-time Platoon Tracking along the arterial corridor between Junction 1 & Junction 2
- Downstream ETA Calculation: ETA = Distance / Speed
- Dynamic Green Wave Phase Preemption: Coordinates phase transitions so platoons glide through both junctions without stopping
"""

import math


class TwoJunctionCoordinator:
    def __init__(self, junctions: dict, enabled: bool = True):
        self.junctions = junctions
        self.enabled = enabled
        self.green_waves_triggered = 0
        self.active_platoons = []

        # Arterial link connecting J1 (West) <-> J2 (East)
        self.j1 = junctions[1]
        self.j2 = junctions[2]

    def update(self, dt: float, vehicles: list):
        if not self.enabled:
            return

        # 1. Track Eastbound Platoons (J1 -> J2)
        eb_platoon = [
            v for v in vehicles
            if v.direction == 'E'
            and self.j1.cx - 20 <= v.x <= self.j2.cx - 85
            and not v.is_turning
        ]

        if len(eb_platoon) >= 3:
            # Sort to find lead vehicle closest to J2
            eb_platoon.sort(key=lambda v: -v.x)
            lead_v = eb_platoon[0]
            dist_to_j2 = (self.j2.cx - 85) - (lead_v.x + lead_v.length / 2)
            speed_px_s = max(1.5, lead_v.speed * 60.0)
            eta_sec = dist_to_j2 / speed_px_s

            # Trigger green wave preemption at J2 when platoon is within 2.0s - 7.0s of arrival
            if 1.5 <= eta_sec <= 7.0:
                if not getattr(self.j2, 'green_wave_active', False):
                    self.j2.trigger_green_wave_preemption("EW", lead_time_sec=eta_sec)
                    self.green_waves_triggered += 1

        # 2. Track Westbound Platoons (J2 -> J1)
        wb_platoon = [
            v for v in vehicles
            if v.direction == 'W'
            and self.j1.cx + 85 <= v.x <= self.j2.cx + 20
            and not v.is_turning
        ]

        if len(wb_platoon) >= 3:
            # Sort to find lead vehicle closest to J1
            wb_platoon.sort(key=lambda v: v.x)
            lead_v = wb_platoon[0]
            dist_to_j1 = (lead_v.x - lead_v.length / 2) - (self.j1.cx + 85)
            speed_px_s = max(1.5, lead_v.speed * 60.0)
            eta_sec = dist_to_j1 / speed_px_s

            # Trigger green wave preemption at J1 when platoon is within 2.0s - 7.0s of arrival
            if 1.5 <= eta_sec <= 7.0:
                if not getattr(self.j1, 'green_wave_active', False):
                    self.j1.trigger_green_wave_preemption("EW", lead_time_sec=eta_sec)
                    self.green_waves_triggered += 1
