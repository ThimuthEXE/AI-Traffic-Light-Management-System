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

        # Capacity metering: directions held RED by upstream segment saturation
        self.capacity_hold_directions: set = set()
        # I2I green wave state (set True when a preemption is in progress)
        self.green_wave_active: bool = False

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

    def set_capacity_hold(self, direction: str, active: bool):
        """Called by ArterySegment when downstream link is at/near capacity.
        When active=True, the departing direction is forced RED (metering).
        When active=False, the hold is released.
        """
        if active:
            self.capacity_hold_directions.add(direction)
        else:
            self.capacity_hold_directions.discard(direction)

    def get_signals_metered(self, direction: str) -> tuple:
        """Returns (through_signal, turn_signal) with capacity-hold override applied."""
        if direction in self.capacity_hold_directions:
            return ('RED', 'RED')
        return self.signals.get_signals_for_direction(direction)

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
        """Coordinated I2I Green Wave preemption.

        target_axis='EW'  → tries to hold/advance to EW green for approaching EB/WB platoon.
        target_axis='NS'  → tries to hold/advance to NS green for approaching SB/NB platoon.

        Logic mirrors _get_active_pcu's string-match pattern so it stays compatible
        with whatever phase names the TrafficSignalManager uses.
        """
        cur_p = str(self.signals.current_phase)

        if target_axis == "EW":
            serving_target = "EW_THROUGH" in cur_p or "E_EXCLUSIVE" in cur_p or "W_EXCLUSIVE" in cur_p
            serving_cross  = "NS_THROUGH" in cur_p or "N_EXCLUSIVE" in cur_p or "S_EXCLUSIVE" in cur_p
        else:  # "NS"
            serving_target = "NS_THROUGH" in cur_p or "N_EXCLUSIVE" in cur_p or "S_EXCLUSIVE" in cur_p
            serving_cross  = "EW_THROUGH" in cur_p or "E_EXCLUSIVE" in cur_p or "W_EXCLUSIVE" in cur_p

        if serving_target:
            # Already green for the arriving platoon — extend it
            self.signals.allocated_through_green = max(
                self.signals.allocated_through_green,
                self.signals.time_in_state + lead_time_sec + 8.0
            )
            self.green_wave_active = True

        elif serving_cross:
            # Cross-traffic is green — only cut it short if min safe time (6 s) has been served
            if self.signals.time_in_state >= 6.0:
                self.signals.time_in_state = self.signals.allocated_through_green
                self.green_wave_active = True

        elif "ALL_RED" in cur_p:
            # In an all-red clearance — just mark wave active so the next green
            # phase (chosen by the signal manager) proceeds without interruption
            self.green_wave_active = True

