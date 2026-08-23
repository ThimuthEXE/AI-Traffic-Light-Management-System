"""
Intersection Control Strategies Module with Arrival Flow & Lane-by-Lane Synthesis
Implements Webster-Fuzzy Hybrid Adaptive Control:
Allocates green times strictly proportional to each approach's vehicle arrival density,
instantaneous lane PCU queues, and waiting times.
"""

from collections import defaultdict
from ai_engine.fuzzy_controller.fuzzy_engine import FuzzyTrafficController
from ai_engine.utils.pcu_calculator import calculate_lane_pcu, classify_traffic_density

class BaseController:
    def get_next_green_duration(self, target_axis: str, vehicles: list, approach_intervals: dict = None) -> float:
        raise NotImplementedError

class FixedTimeController(BaseController):
    def __init__(self, fixed_green: float = 25.0):
        self.fixed_green = fixed_green
        self.mode_name = "Fixed-Time (Traditional)"

    def get_next_green_duration(self, target_axis: str, vehicles: list, approach_intervals: dict = None) -> float:
        return self.fixed_green


class AIFuzzyController(BaseController):
    """
    Webster-Fuzzy Adaptive Controller.
    1. Measures real-time arrival flow rate (veh/min) from each approach's spawn density.
    2. Measures physical lane queues (PCU) and longest wait times.
    3. Calculates proportional flow demand ratio.
    4. Computes high-dynamic-range green allocations (8.0s to 45.0s).
    """
    def __init__(self, min_green: float = 8.0, max_green: float = 45.0):
        self.fuzzy_engine = FuzzyTrafficController(min_green=min_green, max_green=max_green)
        self.min_green = min_green
        self.max_green = max_green
        self.mode_name = "AI Cycle-Adaptive (Flow & Lane-Aware)"
        self.last_decision = {}
        self.lane_stats = {}

    def analyze_all_lanes(self, vehicles: list) -> dict:
        stats = {}
        for d in ['N', 'S', 'E', 'W']:
            for l_idx in [0, 1]:
                lane_v = [
                    v for v in vehicles 
                    if v.direction == d and v.lane_idx == l_idx and not v.has_cleared_intersection
                ]
                pcu = calculate_lane_pcu(lane_v)
                max_w = max((v.wait_time for v in lane_v), default=0.0) if lane_v else 0.0
                stats[(d, l_idx)] = {
                    "direction": d,
                    "lane_idx": l_idx,
                    "count": len(lane_v),
                    "pcu": pcu,
                    "max_wait": round(max_w, 1),
                    "density": classify_traffic_density(pcu),
                    "vehicles": lane_v
                }
        self.lane_stats = stats
        return stats

    def get_approach_stats(self, direction: str, interval: float = 2.0) -> dict:
        l0 = self.lane_stats.get((direction, 0), {"pcu": 0.0, "count": 0, "max_wait": 0.0})
        l1 = self.lane_stats.get((direction, 1), {"pcu": 0.0, "count": 0, "max_wait": 0.0})
        
        total_pcu = l0["pcu"] + l1["pcu"]
        total_count = l0["count"] + l1["count"]
        max_wait = max(l0["max_wait"], l1["max_wait"])
        
        # Arrival flow rate in vehicles per minute
        arrival_flow = (60.0 / max(0.2, interval))
        
        return {
            "direction": direction,
            "lane_0": l0,
            "lane_1": l1,
            "total_pcu": round(total_pcu, 2),
            "total_count": total_count,
            "max_wait": round(max_wait, 1),
            "arrival_flow": round(arrival_flow, 1),
            "density": classify_traffic_density(total_pcu)
        }

    def get_next_green_duration(self, target_axis: str, vehicles: list, approach_intervals: dict = None) -> float:
        """
        Calculates green light duration proportional to approach arrival density & lane PCU.
        """
        if approach_intervals is None:
            approach_intervals = {'N': 2.0, 'S': 2.0, 'E': 2.0, 'W': 2.0}

        self.analyze_all_lanes(vehicles)

        stats_N = self.get_approach_stats('N', approach_intervals.get('N', 2.0))
        stats_S = self.get_approach_stats('S', approach_intervals.get('S', 2.0))
        stats_E = self.get_approach_stats('E', approach_intervals.get('E', 2.0))
        stats_W = self.get_approach_stats('W', approach_intervals.get('W', 2.0))

        # 1. Flow Rate Calculations (vehicles per second)
        flow_N = 1.0 / max(0.3, approach_intervals.get('N', 2.0))
        flow_S = 1.0 / max(0.3, approach_intervals.get('S', 2.0))
        flow_E = 1.0 / max(0.3, approach_intervals.get('E', 2.0))
        flow_W = 1.0 / max(0.3, approach_intervals.get('W', 2.0))

        # Critical Approach Flow Synthesis
        # Heaviest branch takes priority
        flow_NS = max(flow_N, flow_S) + (0.35 * min(flow_N, flow_S))
        flow_EW = max(flow_E, flow_W) + (0.35 * min(flow_E, flow_W))

        # Add physical queue buildup bias (PCU on road)
        queue_NS = (stats_N["total_pcu"] + stats_S["total_pcu"])
        queue_EW = (stats_E["total_pcu"] + stats_W["total_pcu"])

        # Combined Demand metric (Flow Arrival Rate + Active Queue Weight)
        demand_NS = (flow_NS * 10.0) + (queue_NS * 0.8)
        demand_EW = (flow_EW * 10.0) + (queue_EW * 0.8)

        total_demand = demand_NS + demand_EW + 1e-5
        ratio_NS = demand_NS / total_demand
        ratio_EW = demand_EW / total_demand

        # Green Time Allocation Pool (between 16s and 80s cycle budget)
        cycle_green_pool = 52.0  # Total available green seconds per cycle

        if target_axis == 'NS':
            target_ratio = ratio_NS
            target_pcu = queue_NS
            cross_pcu = queue_EW
            max_wait = max(stats_N["max_wait"], stats_S["max_wait"])
            allocated_sec = cycle_green_pool * target_ratio
        else:
            target_ratio = ratio_EW
            target_pcu = queue_EW
            cross_pcu = queue_NS
            max_wait = max(stats_E["max_wait"], stats_W["max_wait"])
            allocated_sec = cycle_green_pool * target_ratio

        # Blend with Fuzzy Inference to account for wait times & safety bounds
        fuzzy_eval = self.fuzzy_engine.compute_green_duration(
            current_pcu=target_pcu,
            max_wait_time=max_wait,
            cross_pcu=cross_pcu
        )

        # Proportional flow weighting has primary influence (70% flow ratio + 30% fuzzy queue)
        combined_green = (0.75 * allocated_sec) + (0.25 * fuzzy_eval["green_duration"])
        final_green = max(self.min_green, min(self.max_green, round(combined_green, 1)))

        decision = {
            "target_axis": target_axis,
            "green_duration": final_green,
            "target_ratio": round(target_ratio * 100, 1),
            "demand_NS": round(demand_NS, 1),
            "demand_EW": round(demand_EW, 1),
            "target_pcu": round(target_pcu, 1),
            "cross_pcu": round(cross_pcu, 1),
            "max_wait": round(max_wait, 1),
            "rule_activations": {f"FlowShare_{int(target_ratio*100)}%": round(target_ratio, 2)}
        }
        self.last_decision = decision
        return final_green
