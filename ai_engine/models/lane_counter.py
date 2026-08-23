"""
Multi-Lane ROI & Directional Counting Engine
Segments video frame into approach zones, maps detected vehicles into lanes,
and computes real-time PCU densities.
"""

import cv2
import numpy as np
from collections import defaultdict
from ai_engine.utils.pcu_calculator import get_pcu_weight, classify_traffic_density

# Default 4-Way Approach ROIs for standard 1280x720 intersection views
# Center (640, 360), Road edges at X:[560, 720], Y:[280, 440]
DEFAULT_APPROACH_ZONES = {
    "East Approach": {
        "direction": "E",
        "polygon": np.array([[0, 360], [555, 360], [555, 440], [0, 440]], dtype=np.int32),
        "color": (255, 100, 100)  # Light Blue
    },
    "West Approach": {
        "direction": "W",
        "polygon": np.array([[725, 280], [1280, 280], [1280, 360], [725, 360]], dtype=np.int32),
        "color": (100, 255, 100)  # Light Green
    },
    "South Approach": {
        "direction": "S",
        "polygon": np.array([[560, 0], [640, 0], [640, 275], [560, 275]], dtype=np.int32),
        "color": (100, 200, 255)  # Light Orange
    },
    "North Approach": {
        "direction": "N",
        "polygon": np.array([[640, 445], [720, 445], [720, 720], [640, 720]], dtype=np.int32),
        "color": (200, 100, 255)  # Light Magenta
    }
}

class LaneTrafficAnalyzer:
    def __init__(self, zones: dict = None):
        self.zones = zones if zones is not None else DEFAULT_APPROACH_ZONES
        self.total_counted_ids = set()
        self.lifetime_vehicle_counts = defaultdict(int)

    def is_point_in_zone(self, point: tuple, polygon: np.ndarray) -> bool:
        """Point-in-polygon test using OpenCV."""
        return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

    def analyze_lanes(self, tracked_vehicles: list) -> dict:
        """
        Groups active tracked vehicles into their respective approach lanes,
        calculates live vehicle counts, PCU loads, and density classification.
        
        Returns:
            dict of approach metrics:
                'East Approach': {
                    'direction': 'E',
                    'vehicle_count': int,
                    'pcu_load': float,
                    'density_level': 'LOW' | 'MEDIUM' | 'HIGH' | 'VERY_HIGH',
                    'breakdown': {'car': int, 'bus': int, ...},
                    'vehicles': list of TrackedVehicle
                }
        """
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
        """Draws semi-transparent polygon masks and zone labels over camera frame."""
        overlay = frame.copy()
        
        for zone_name, zone_cfg in self.zones.items():
            poly = zone_cfg["polygon"]
            color = zone_cfg.get("color", (0, 255, 255))
            
            # Fill polygon on overlay
            cv2.fillPoly(overlay, [poly], color)
            
            # Outline
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)

            # Zone label text
            res = lane_results.get(zone_name, {})
            pcu = res.get("pcu_load", 0.0)
            cnt = res.get("vehicle_count", 0)
            density = res.get("density_level", "LOW")

            # Calculate centroid of polygon for label
            M = cv2.moments(poly)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = poly[0][0], poly[0][1]

            label = f"{zone_name}: {cnt} veh ({pcu:.1f} PCU) [{density}]"
            cv2.putText(frame, label, (cx - 110, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Blend overlay (alpha=0.18)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
