"""
Vision-to-Fuzzy Decision Bridge
Directly connects YOLO vehicle detections & lane PCU loads with the Proportional Flow & Fuzzy Controller.
"""

from ai_engine.fuzzy_controller.fuzzy_engine import FuzzyTrafficController

class VisionFuzzyBridge:
    def __init__(self, min_green: float = 8.0, max_green: float = 42.0):
        self.fuzzy_controller = FuzzyTrafficController(min_green=min_green, max_green=max_green)
        self.min_green = min_green
        self.max_green = max_green
        self.last_decision = {}

    def decide_phase_timing(self, target_axis: str, lane_results: dict) -> dict:
        """
        Computes dynamic green timing based on computer vision lane analytics.
        """
        ew_pcu = 0.0
        ns_pcu = 0.0
        ew_vehicles = []
        ns_vehicles = []

        for zone_name, data in lane_results.items():
            direction = data.get("direction", "")
            if direction in ['E', 'W']:
                ew_pcu += data.get("pcu_load", 0.0)
                ew_vehicles.extend(data.get("vehicles", []))
            elif direction in ['N', 'S']:
                ns_pcu += data.get("pcu_load", 0.0)
                ns_vehicles.extend(data.get("vehicles", []))

        total_pcu = ew_pcu + ns_pcu + 1e-5
        ratio_ns = ns_pcu / total_pcu
        ratio_ew = ew_pcu / total_pcu

        if target_axis == 'EW':
            target_pcu = ew_pcu
            cross_pcu = ns_pcu
            target_vehicles = ew_vehicles
            target_ratio = ratio_ew
        else:
            target_pcu = ns_pcu
            cross_pcu = ew_pcu
            target_vehicles = ns_vehicles
            target_ratio = ratio_ns

        # Max stationary frames converted to approximate seconds (assuming ~30 FPS)
        max_stationary_frames = 0
        if target_vehicles:
            max_stationary_frames = max((getattr(v, "frames_stationary", 0) for v in target_vehicles), default=0)
        approx_wait_sec = max_stationary_frames / 30.0

        # Fuzzy evaluation
        fuzzy_eval = self.fuzzy_controller.compute_green_duration(
            current_pcu=target_pcu,
            max_wait_time=approx_wait_sec,
            cross_pcu=cross_pcu
        )

        # Proportional flow allocation from 52s green cycle pool
        proportional_sec = 52.0 * target_ratio
        combined_green = (0.70 * proportional_sec) + (0.30 * fuzzy_eval["green_duration"])
        final_green = max(self.min_green, min(self.max_green, round(combined_green, 1)))

        result = {
            "target_axis": target_axis,
            "green_duration": final_green,
            "target_ratio": round(target_ratio * 100, 1),
            "target_pcu": round(target_pcu, 1),
            "cross_pcu": round(cross_pcu, 1),
            "approx_wait_sec": round(approx_wait_sec, 1),
            "active_rules": {f"TrafficShare_{int(target_ratio*100)}%": round(target_ratio, 2)}
        }
        self.last_decision = result
        return result
