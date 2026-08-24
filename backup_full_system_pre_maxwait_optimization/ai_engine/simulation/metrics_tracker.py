"""
Simulation Metrics & Benchmarking Tracker
Records live traffic performance to rigorously validate Hypothesis H1 vs H0.
"""

from collections import defaultdict
from ai_engine.utils.pcu_calculator import calculate_lane_pcu

class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistical counters."""
        self.cleared_vehicles = 0
        self.cleared_pcu = 0.0
        self.total_wait_time = 0.0
        self.max_observed_wait = 0.0
        self.max_observed_queue = 0
        
        # Vehicle type breakdown
        self.vehicle_counts = defaultdict(int)
        
        # Directional breakdown
        self.dir_wait_times = defaultdict(list)
        self.dir_cleared = defaultdict(int)

    def record_vehicle_cleared(self, vehicle):
        """Record statistics for a vehicle that has crossed and cleared the intersection."""
        self.cleared_vehicles += 1
        self.cleared_pcu += vehicle.pcu
        self.total_wait_time += vehicle.wait_time
        self.max_observed_wait = max(self.max_observed_wait, vehicle.wait_time)
        
        self.vehicle_counts[vehicle.vehicle_type] += 1
        self.dir_cleared[vehicle.direction] += 1
        self.dir_wait_times[vehicle.direction].append(vehicle.wait_time)

    def update_live_queues(self, active_vehicles: list):
        """Update maximum queue observation from currently active queuing vehicles."""
        queuing_count = sum(1 for v in active_vehicles if v.speed < 0.5 and not v.has_cleared_intersection)
        self.max_observed_queue = max(self.max_observed_queue, queuing_count)

    @property
    def average_wait_time(self) -> float:
        """Returns Average Waiting Time (AWT) in seconds."""
        if self.cleared_vehicles == 0:
            return 0.0
        return round(self.total_wait_time / self.cleared_vehicles, 2)

    def get_summary(self) -> dict:
        """Returns structured metrics summary."""
        return {
            "cleared_vehicles": self.cleared_vehicles,
            "cleared_pcu": round(self.cleared_pcu, 2),
            "average_wait_time": self.average_wait_time,
            "max_wait_time": round(self.max_observed_wait, 2),
            "max_queue_length": self.max_observed_queue,
            "vehicle_breakdown": dict(self.vehicle_counts)
        }
