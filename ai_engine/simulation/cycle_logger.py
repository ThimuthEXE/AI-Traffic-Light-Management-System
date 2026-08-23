"""
Cycle Data Logger & Historical Record Store
Stores detailed per-cycle traffic metrics, queue buildup, allocated green times,
and throughput for training ML predictors and feeding backend analytics.
"""

import time
import json
import os
from collections import deque

class CycleDataLogger:
    def __init__(self, max_history=50):
        self.cycle_count = 0
        self.history = deque(maxlen=max_history)
        self.log_file = os.path.join(
            os.path.dirname(__file__), "../../data/traffic_cycle_logs.json"
        )
        self.current_cycle_data = {}

    def log_cycle_completion(self, phase_axis: str, allocated_green: float, pcu_demand: float,
                             vehicles_cleared: int, max_wait_time: float, residual_pcu: float, rules: dict):
        """
        Record a completed green cycle.
        """
        self.cycle_count += 1
        record = {
            "cycle_id": self.cycle_count,
            "timestamp": round(time.time(), 2),
            "phase_axis": phase_axis,
            "allocated_green": round(allocated_green, 1),
            "pcu_demand": round(pcu_demand, 2),
            "vehicles_cleared": vehicles_cleared,
            "max_wait_time": round(max_wait_time, 1),
            "residual_pcu": round(residual_pcu, 2),
            "fuzzy_rules": rules
        }
        self.history.append(record)
        return record

    def get_recent_history(self, count=5):
        """Get the most recent N logged cycles."""
        return list(self.history)[-count:]

    def export_to_json(self):
        """Persists all logged cycles to disk for Machine Learning training."""
        os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
        with open(self.log_file, "w") as f:
            json.dump(list(self.history), f, indent=2)
