"""
Multi-Lane ROI & Directional Counting Engine
Supports both:
1. 4-Way Intersection Overview Mode (Bird's eye 4-branch view)
2. Single-Approach Corridor Mode (Straight-on CCTV / Dashcam view of a 1-3 lane roadway)
"""

import cv2
import numpy as np
from collections import defaultdict
from ai_engine.utils.pcu_calculator import get_pcu_weight, classify_traffic_density

# Default 4-Way Approach ROIs for standard 1280x720 top-down intersection views
DEFAULT_4WAY_ZONES = {
    "East Approach": {
        "direction": "E",
        "polygon": np.array([[0, 360], [555, 360], [555, 440], [0, 440]], dtype=np.int32),
        "color": (255, 100, 100)
    },
    "West Approach": {
        "direction": "W",
        "polygon": np.array([[725, 280], [1280, 280], [1280, 360], [725, 360]], dtype=np.int32),
        "color": (100, 255, 100)
    },
    "South Approach": {
        "direction": "S",
        "polygon": np.array([[560, 0], [640, 0], [640, 275], [560, 275]], dtype=np.int32),
        "color": (100, 200, 255)
    },
    "North Approach": {
        "direction": "N",
        "polygon": np.array([[640, 445], [720, 445], [720, 720], [640, 720]], dtype=np.int32),
        "color": (200, 100, 255)
    }
}

def create_single_approach_zones(width: int = 1280, height: int = 720, direction: str = "N", num_lanes: int = 2) -> dict:
    """Creates full-frame lane segmentation zones for a single-approach camera."""
    dir_name_map = {"N": "North Approach", "S": "South Approach", "E": "East Approach", "W": "West Approach"}
    base_name = dir_name_map.get(direction.upper(), "Main Approach")
    
    zones = {}
    lane_width = width // num_lanes
    colors = [(255, 140, 0), (0, 215, 255), (147, 112, 219)]

    for i in range(num_lanes):
        x1 = i * lane_width
        x2 = (i + 1) * lane_width if i < num_lanes - 1 else width
        
        poly = np.array([
            [x1, int(height * 0.15)],
            [x2, int(height * 0.15)],
            [x2, height],
            [x1, height]
        ], dtype=np.int32)

        zones[f"{base_name} - Lane {i+1}"] = {
            "direction": direction.upper(),
            "polygon": poly,
            "color": colors[i % len(colors)]
        }

    return zones


class LaneTrafficAnalyzer:
    def __init__(self, zones: dict = None, mode: str = "4way", width: int = 1280, height: int = 720, direction: str = "N"):
        self.mode = mode  # '4way' or 'single'
        self.width = width
        self.height = height
        self.direction = direction.upper()
        
        if zones is not None:
            self.zones = zones
        elif mode == "single":
            self.zones = create_single_approach_zones(width, height, direction=self.direction, num_lanes=2)
        else:
            self.zones = DEFAULT_4WAY_ZONES

        self.total_counted_ids = set()
        self.lifetime_vehicle_counts = defaultdict(int)

    def set_mode(self, mode: str, width: int = None, height: int = None, direction: str = None):
        """Switches dynamically between 4-way intersection and single approach mode."""
        self.mode = mode
        if width: self.width = width
        if height: self.height = height
        if direction: self.direction = direction.upper()

        if self.mode == "single":
            self.zones = create_single_approach_zones(self.width, self.height, direction=self.direction, num_lanes=2)
        else:
            self.zones = DEFAULT_4WAY_ZONES

    def is_point_in_zone(self, point: tuple, polygon: np.ndarray) -> bool:
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    def analyze_lanes(self, tracked_vehicles: list) -> dict:
        results = {}

        for zone_name, zone_cfg in self.zones.items():
            poly = zone_cfg["polygon"]
            direction = zone_cfg["direction"]

            zone_vehicles = []
            breakdown = defaultdict(int)
            total_pcu = 0.0

            for tv in tracked_vehicles:
                if self.is_point_in_zone(tv.center, poly):
                    tv.assigned_lane = zone_name
                    zone_vehicles.append(tv)
                    breakdown[tv.class_name] += 1
                    total_pcu += get_pcu_weight(tv.class_name)

                    if tv.track_id not in self.total_counted_ids:
                        self.total_counted_ids.add(tv.track_id)
                        self.lifetime_vehicle_counts[tv.class_name] += 1

            results[zone_name] = {
                "direction": direction,
                "vehicle_count": len(zone_vehicles),
                "pcu_load": round(total_pcu, 2),
                "density_level": classify_traffic_density(total_pcu),
                "breakdown": dict(breakdown),
                "vehicles": zone_vehicles
            }

        return results

    def draw_zone_overlays(self, frame: np.ndarray, lane_results: dict):
        overlay = frame.copy()
        
        for zone_name, zone_cfg in self.zones.items():
            poly = zone_cfg["polygon"]
            color = zone_cfg.get("color", (0, 255, 255))
            
            # Semi-transparent lane fill
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)

            res = lane_results.get(zone_name, {})
            pcu = res.get("pcu_load", 0.0)
            cnt = res.get("vehicle_count", 0)
            density = res.get("density_level", "LOW")

            M = cv2.moments(poly)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = poly[0]

            label = f"{zone_name}: {cnt}v ({pcu:.1f}p)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (cx - tw // 2 - 4, cy - th - 6), (cx + tw // 2 + 4, cy + 4), (15, 20, 25), -1)
            cv2.rectangle(frame, (cx - tw // 2 - 4, cy - th - 6), (cx + tw // 2 + 4, cy + 4), color, 1)
            cv2.putText(frame, label, (cx - tw // 2, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend overlay
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
