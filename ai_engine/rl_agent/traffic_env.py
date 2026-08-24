"""
Reinforcement Learning Simulation Environment Wrapper (Gym-Style)
General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI

Wraps the traffic simulation into a standardized MDP Environment:
- State Space: 18 continuous features (Lane PCUs, Approach Max Waits, Arrival Rates, Active Phase)
- Action Space: 9 discrete actions (Extend Green, NS-Through, NS-Left, EW-Through, EW-Left, N/S/E/W Exclusive)
- Reward: Multi-objective penalty minimizing Average Wait Time, Max Wait Time, and Queue Length
"""

import os
import sys
import math
import random
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['SDL_VIDEODRIVER'] = 'dummy'

from ai_engine.simulation.run_simulation import TrafficSimulationApp
from ai_engine.simulation.traffic_signal import SignalPhase
from ai_engine.utils.pcu_calculator import calculate_lane_pcu

ACTION_MAP = {
    0: "EXTEND_CURRENT_GREEN",
    1: SignalPhase.NS_THROUGH_GREEN,
    2: SignalPhase.NS_LEFT_GREEN,
    3: SignalPhase.EW_THROUGH_GREEN,
    4: SignalPhase.EW_LEFT_GREEN,
    5: SignalPhase.N_EXCLUSIVE_GREEN,
    6: SignalPhase.S_EXCLUSIVE_GREEN,
    7: SignalPhase.E_EXCLUSIVE_GREEN,
    8: SignalPhase.W_EXCLUSIVE_GREEN,
}

ACTION_NAMES = [
    "Extend Green", "NS Through", "NS Left Turn", "EW Through", "EW Left Turn",
    "N Exclusive (All-Green)", "S Exclusive (All-Green)", "E Exclusive (All-Green)", "W Exclusive (All-Green)"
]

class TrafficRLEnvironment:
    """
    Fast, headless Reinforcement Learning Environment for Traffic Light Control.
    """
    def __init__(self, step_duration_sec: float = 3.0):
        self.step_duration = step_duration_sec
        self.dt = 1.0 / 60.0
        self.ticks_per_step = int(self.step_duration / self.dt)
        self.state_dim = 18
        self.action_dim = 9
        self.app = None
        self.current_sim_time = 0.0
        self.prev_phase = None
        self.reset()

    def reset(self, scenario: str = "random") -> np.ndarray:
        """Reset simulation and configure traffic scenario."""
        self.app = TrafficSimulationApp()
        self.app.paused = False
        self.current_sim_time = 0.0
        self.prev_phase = self.app.signals.current_phase

        # Set diverse scenarios for comprehensive training
        if scenario == "morning_rush":
            self.app.approach_intervals['N'] = random.uniform(0.4, 0.8)   # U Heavy (75-150 v/m)
            self.app.approach_intervals['E'] = random.uniform(0.5, 1.0)   # R Moderate (60-120 v/m)
            self.app.approach_intervals['S'] = random.uniform(1.8, 3.5)
            self.app.approach_intervals['W'] = random.uniform(1.8, 3.5)
        elif scenario == "evening_rush":
            self.app.approach_intervals['S'] = random.uniform(0.4, 0.8)   # D Heavy
            self.app.approach_intervals['W'] = random.uniform(0.5, 1.0)   # L Moderate
            self.app.approach_intervals['N'] = random.uniform(1.8, 3.5)
            self.app.approach_intervals['E'] = random.uniform(1.8, 3.5)
        elif scenario == "dual_surge":
            self.app.approach_intervals['W'] = random.uniform(0.4, 0.7)   # L Heavy
            self.app.approach_intervals['S'] = random.uniform(0.5, 0.8)   # D Heavy
            self.app.approach_intervals['N'] = random.uniform(2.0, 3.5)
            self.app.approach_intervals['E'] = random.uniform(2.0, 3.5)
        elif scenario == "user_experiment":
            self.app.approach_intervals['N'] = 60.0 / 150.0  # U=150
            self.app.approach_intervals['S'] = 60.0 / 40.0   # D=40
            self.app.approach_intervals['W'] = 60.0 / 35.0   # L=35
            self.app.approach_intervals['E'] = 60.0 / 30.0   # R=30
        else:
            # Random uniform or asymmetric traffic
            self.app.approach_intervals['N'] = random.uniform(0.4, 3.0)
            self.app.approach_intervals['S'] = random.uniform(0.5, 3.0)
            self.app.approach_intervals['E'] = random.uniform(0.5, 3.0)
            self.app.approach_intervals['W'] = random.uniform(0.5, 3.0)

        # Warm up simulation for 10 seconds
        for _ in range(int(10.0 / self.dt)):
            self.app.update(self.dt)

        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Extract 18-dimensional normalized state vector S_t.
        """
        # 1. 8 Lane Queues (PCU normalized by 15.0)
        lane_pcus = []
        for d in ('N', 'S', 'E', 'W'):
            for l_idx in (0, 1):
                lane_v = [v for v in self.app.vehicles if v.direction == d and v.lane_idx == l_idx and not v.has_cleared_intersection]
                pcu = calculate_lane_pcu(lane_v)
                lane_pcus.append(min(2.0, pcu / 12.0))

        # 2. 4 Approach Max Waiting Times (normalized by 45.0s)
        max_waits = []
        for d in ('N', 'S', 'E', 'W'):
            v_d = [v for v in self.app.vehicles if v.direction == d and not v.has_cleared_intersection]
            w = max((v.wait_time for v in v_d), default=0.0)
            max_waits.append(min(3.0, w / 40.0))

        # 3. 4 Approach Arrival Rates (normalized by 150 v/m)
        flows = []
        for d in ('N', 'S', 'E', 'W'):
            rate = self.app.approach_intervals[d]
            flow = (60.0 / max(0.2, rate))
            flows.append(min(1.5, flow / 120.0))

        # 4. Active Phase Index and Elapsed Time in Phase
        cur_phase = str(self.app.signals.current_phase)
        axis_code = 1.0 if "NS" in cur_phase or "N_" in cur_phase or "S_" in cur_phase else 0.0
        elapsed_norm = min(2.0, self.app.signals.time_in_state / 20.0)

        state = np.array(lane_pcus + max_waits + flows + [axis_code, elapsed_norm], dtype=np.float32)
        return state

    def step(self, action: int) -> tuple:
        """
        Apply action, step simulation for 3.0s, and return (s', r, done, info).
        """
        chosen_action = ACTION_MAP.get(action, "EXTEND_CURRENT_GREEN")
        switch_cost = 0.0

        if chosen_action != "EXTEND_CURRENT_GREEN":
            target_phase = chosen_action
            if target_phase != self.app.signals.current_phase:
                self.app.signals.current_phase = target_phase
                self.app.signals.time_in_state = 0.0
                switch_cost = 0.8  # Slight penalty for light switching

        cleared_before = self.app.active_metrics.cleared_vehicles

        # Step simulation physics for 3 seconds
        for _ in range(self.ticks_per_step):
            self.app.update(self.dt)
            self.current_sim_time += self.dt

        cleared_now = self.app.active_metrics.cleared_vehicles
        newly_cleared = cleared_now - cleared_before

        next_state = self.get_state()

        # Multi-Objective Reward Calculation
        # R = - [ w1*AWT + w2*(MaxWait/20)^1.8 + w3*TotalQueue + w4*SwitchCost ] + w5*Cleared
        awt = self.app.active_metrics.average_wait_time
        max_w = max((v.wait_time for v in self.app.vehicles if not v.has_cleared_intersection), default=0.0)
        tot_queue = len([v for v in self.app.vehicles if not v.has_cleared_intersection])

        wait_penalty = math.pow(max_w / 25.0, 1.8) if max_w >= 25.0 else (max_w / 25.0)

        reward = - (
            0.30 * (awt / 15.0) +
            0.45 * wait_penalty +
            0.20 * (tot_queue / 15.0) +
            0.05 * switch_cost
        ) + (0.25 * newly_cleared)

        done = (self.current_sim_time >= 300.0)  # 5-minute training episode

        info = {
            'sim_time': self.current_sim_time,
            'cleared': self.app.active_metrics.cleared_vehicles,
            'awt': awt,
            'max_wait': max_w,
            'active_queue': tot_queue
        }

        return next_state, reward, done, info
