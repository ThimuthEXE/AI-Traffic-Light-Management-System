"""
Junction Node Module — Encapsulates Signal Management, AI Controllers, and Metrics for one Junction
General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI — Group 06
"""

import sys
import os
import pygame

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_engine.simulation.traffic_signal import TrafficSignalManager, SignalPhase
from ai_engine.simulation.controllers import FixedTimeController, AIFuzzyController, DQNAdaptiveController
from ai_engine.simulation.metrics_tracker import MetricsTracker
from ai_engine.simulation.cycle_logger import CycleDataLogger
from ai_engine.simulation.cycle_observer import CycleObserver
from ai_engine.utils.pcu_calculator import calculate_lane_pcu, classify_traffic_density


class JunctionNode:
    def __init__(self, j_id: int, name: str, cx: float, cy: float):
        self.id = j_id
        self.name = name
        self.cx = cx
        self.cy = cy

        # Controllers & Signal Manager
        self.signals = TrafficSignalManager(yellow_duration=2.0, all_red_duration=1.0)
        self.fixed_controller = FixedTimeController(fixed_green=25.0)
        self.ai_controller = AIFuzzyController(min_green=8.0, max_green=38.0)
        self.dqn_controller = DQNAdaptiveController()

        self.active_mode = 2  # 1: Fixed, 2: Fuzzy-ML, 3: Deep RL
        self.current_controller = self.ai_controller

        self.metrics_ai = MetricsTracker()
        self.metrics_fixed = MetricsTracker()
        self.metrics_dqn = MetricsTracker()
        self.cycle_logger = CycleDataLogger()
        self.cycle_observer = CycleObserver()

        self.completed_cycles = 0
        self.current_phase_cleared = 0
        self.last_hud_decision = {}

        # Stop lines (calculated from center cx, cy)
        # Road width is 160px (half_w = 80px, stop line offset = 85px)
        self.stop_lines = {
            'E': self.cx - 85,  # Eastbound vehicles (moving +X) stop at West entrance
            'W': self.cx + 85,  # Westbound vehicles (moving -X) stop at East entrance
            'S': self.cy - 85,  # Southbound vehicles (moving +Y) stop at North entrance
            'N': self.cy + 85   # Northbound vehicles (moving -Y) stop at South entrance
        }

    @property
    def active_metrics(self) -> MetricsTracker:
        if self.active_mode == 1: return self.metrics_fixed
        elif self.active_mode == 2: return self.metrics_ai
        else: return self.metrics_dqn

    def set_mode(self, mode_id: int):
        self.active_mode = mode_id
        if mode_id == 1: self.current_controller = self.fixed_controller
        elif mode_id == 2: self.current_controller = self.ai_controller
        elif mode_id == 3: self.current_controller = self.dqn_controller

    def get_stop_line(self, direction: str) -> float:
        return self.stop_lines.get(direction, None)

    def _get_active_pcu(self, vehicles: list) -> float:
        cur_p = str(self.signals.current_phase)
        local_vehs = [v for v in vehicles if v.current_junction_id == self.id and not v.has_cleared_intersection]
        if "N_EXCLUSIVE" in cur_p: return calculate_lane_pcu([v for v in local_vehs if v.direction == 'N'])
        if "S_EXCLUSIVE" in cur_p: return calculate_lane_pcu([v for v in local_vehs if v.direction == 'S'])
        if "E_EXCLUSIVE" in cur_p: return calculate_lane_pcu([v for v in local_vehs if v.direction == 'E'])
        if "W_EXCLUSIVE" in cur_p: return calculate_lane_pcu([v for v in local_vehs if v.direction == 'W'])
        if "NS_THROUGH" in cur_p: return calculate_lane_pcu([v for v in local_vehs if v.direction in ('N', 'S')])
        if "EW_THROUGH" in cur_p: return calculate_lane_pcu([v for v in local_vehs if v.direction in ('E', 'W')])
        return 5.0

    def _get_pcu_by_direction(self, vehicles: list) -> dict:
        local_vehs = [v for v in vehicles if v.current_junction_id == self.id and not v.has_cleared_intersection]
        return {
            "N": calculate_lane_pcu([v for v in local_vehs if v.direction == 'N']),
            "S": calculate_lane_pcu([v for v in local_vehs if v.direction == 'S']),
            "E": calculate_lane_pcu([v for v in local_vehs if v.direction == 'E']),
            "W": calculate_lane_pcu([v for v in local_vehs if v.direction == 'W']),
        }

    def _calc_decision(self, axis: str, vehicles: list) -> dict:
        local_vehs = [v for v in vehicles if v.current_junction_id == self.id and not v.has_cleared_intersection]
        dec = self.current_controller.get_next_phase_decision(
            target_axis=axis,
            vehicles=local_vehs,
            cycle_observer=self.cycle_observer
        )
        self.last_hud_decision = dec
        return dec

    def update(self, dt: float, vehicles: list):
        local_vehs = [v for v in vehicles if v.current_junction_id == self.id and not v.has_cleared_intersection]
        pcu_map = self._get_pcu_by_direction(vehicles)
        active_pcu = self._get_active_pcu(vehicles)

        self.cycle_observer.tick(dt, local_vehs, pcu_map)

        phase_switched, completed_phase, new_phase, new_green = self.signals.update(
            dt,
            next_decision_calc_fn=lambda axis: self._calc_decision(axis, vehicles),
            active_queue_pcu=active_pcu
        )

        if phase_switched and new_phase == SignalPhase.ALL_RED_1:
            self.completed_cycles += 1
            self.cycle_observer.end_cycle()
            self.cycle_observer.begin_cycle()

    def trigger_green_wave_preemption(self, target_axis: str = "EW", lead_time_sec: float = 4.0):
        """Coordinated I2I Green Wave preemption: advances phase to EW Green in time for incoming platoon."""
        cur = self.signals.current_phase
        if cur == SignalPhase.EW_THROUGH_GREEN:
            # Extend green to ensure full platoon passes smoothly
            self.signals.allocated_through_green = max(
                self.signals.allocated_through_green,
                self.signals.time_in_state + lead_time_sec + 8.0
            )
            self.green_wave_active = True
        elif cur in (SignalPhase.NS_THROUGH_GREEN, SignalPhase.N_EXCLUSIVE_GREEN, SignalPhase.S_EXCLUSIVE_GREEN):
            # Only preempt if minimum safe green (6.0s) has already been served to cross-traffic
            if self.signals.time_in_state >= 6.0:
                self.signals.time_in_state = self.signals.allocated_through_green
                self.green_wave_active = True
        elif cur in (SignalPhase.ALL_RED_1, SignalPhase.ALL_RED_2):
            self.green_wave_active = True
