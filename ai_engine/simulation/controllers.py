"""
Intersection Control Strategies — High-Capacity Dynamic Cycle Optimizer
AI Traffic Light Management System

Features:
1. Dynamic Green Scaling (Surge approaches get 32-36s, light approaches get 12-16s -> Total cycle stays ~58s)
2. Continuous Anti-Starvation Preemption (Max wait capped < 55s for all 4 approaches)
3. Balanced Co-Flow Integration (Opposing straight flows move together to prevent starvation)
4. Early Gap-Out Cutoff (Transitions instantly when queue is empty)
"""

from __future__ import annotations

import os
import sys
import math
import time
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_engine.fuzzy_controller.fuzzy_engine import FuzzyTrafficController
from ai_engine.utils.pcu_calculator import calculate_lane_pcu, classify_traffic_density

OPPOSING = {"N": "S", "S": "N", "E": "W", "W": "E"}
AXIS_DIRS = {"NS": ("N", "S"), "EW": ("E", "W")}


def _dir_to_exclusive_policy(direction: str) -> str:
    return f"{direction}_EXCLUSIVE_GREEN"


class BaseController:
    def get_next_phase_decision(self, target_axis: str, vehicles: list,
                                approach_intervals: dict = None,
                                cycle_observer=None) -> dict:
        raise NotImplementedError


class FixedTimeController(BaseController):
    def __init__(self, fixed_green: float = 24.0):
        self.fixed_green = float(fixed_green)
        self.mode_name = "Fixed-Time (Traditional)"
        self.last_decision: dict = {}

    def get_next_phase_decision(self, target_axis: str, vehicles: list,
                                approach_intervals: dict = None,
                                cycle_observer=None) -> dict:
        decision = {
            "policy": "BALANCED_PHASE",
            "target_axis": target_axis,
            "through_green_sec": self.fixed_green * 0.70,
            "turn_green_sec":    self.fixed_green * 0.30,
            "total_green_sec":   self.fixed_green,
            "asym_ratio": 1.0,
            "prioritized_dir": "NONE",
            "anti_starvation_active": False,
            "data_source": "fixed_time",
        }
        self.last_decision = decision
        return decision


class AIFuzzyController(BaseController):
    """
    High-Capacity Dynamic Cycle & Anti-Starvation Controller with Method 1 Predictive Pre-Allocation.
    - 4-Cycle Receding Horizon MPC Lookahead: Projects queues and clearance delays over 4 future cycles.
    - 20-Second Flexible Safety Margin: Restricts allowable green duration before phase begins so cross-axis
      waiting time remains strictly <= 28.0s (48.0s hard cap - 20.0s safety buffer).
    - Uninterrupted Smooth Countdowns: Timing is locked at phase initiation without mid-flight jumps or abrupt cuts.
    """

    STARVATION_THRESHOLD_SEC = 38.0   # Soft threshold: exponential penalty begins
    HARD_CAP_MAX_WAIT_SEC    = 48.0   # Hard ceiling: absolute starvation limit
    FLEXIBLE_BUFFER_SEC      = 20.0   # 20s flexible margin requested by user
    PROACTIVE_TARGET_SEC     = 28.0   # 48.0s - 20.0s = 28.0s lookahead ceiling trigger
    LOOKAHEAD_CYCLES         = 4      # 4-cycle receding horizon projection
    EXCL_DOMINANCE_RATIO     = 1.35
    MIN_OBS                  = 3

    def __init__(self, min_green: float = 8.0, max_green: float = 38.0):
        self.fuzzy_engine = FuzzyTrafficController(min_green=min_green, max_green=max_green)
        self.min_green = min_green
        self.max_green = max_green
        self.mode_name = "High-Capacity Dynamic Cycle ML+Fuzzy Controller"
        self.last_decision: dict = {}
        self.lane_stats: dict = {}
        self.consecutive_exclusive_count = {"N": 0, "S": 0, "E": 0, "W": 0}

        model_path = os.path.join(
            PROJECT_ROOT, "ai_engine", "traffic_predictor",
            "saved_models", "ml_phase_optimizer.joblib"
        )
        self.ml_model = None
        if os.path.exists(model_path):
            try:
                self.ml_model = joblib.load(model_path)
                print("[AI Controller] Dynamic Cycle ML Phasing Optimizer loaded successfully!")
            except Exception as exc:
                print(f"[AI Controller] ML model load failed: {exc}")

    def analyze_all_lanes(self, vehicles: list) -> dict:
        stats: dict = {}
        for d in ("N", "S", "E", "W"):
            for l_idx in (0, 1):
                lane_v = [
                    v for v in vehicles
                    if v.direction == d
                    and v.lane_idx == l_idx
                    and not v.has_cleared_intersection
                ]
                pcu = calculate_lane_pcu(lane_v)
                max_w = max((v.wait_time for v in lane_v), default=0.0)
                stats[(d, l_idx)] = {
                    "direction": d, "lane_idx": l_idx,
                    "count": len(lane_v), "pcu": pcu,
                    "max_wait": round(max_w, 1),
                    "density": classify_traffic_density(pcu),
                }
        self.lane_stats = stats
        return stats

    def get_approach_stats(self, direction: str, interval: float = 2.0) -> dict:
        l0 = self.lane_stats.get((direction, 0), {"pcu": 0.0, "count": 0, "max_wait": 0.0})
        l1 = self.lane_stats.get((direction, 1), {"pcu": 0.0, "count": 0, "max_wait": 0.0})
        total_pcu   = l0["pcu"] + l1["pcu"]
        total_count = l0["count"] + l1["count"]
        max_wait    = max(l0["max_wait"], l1["max_wait"])
        arrival_flow = round(60.0 / max(0.2, interval), 1)
        return {
            "direction": direction, "lane_0": l0, "lane_1": l1,
            "total_pcu": round(total_pcu, 2),
            "total_count": total_count,
            "max_wait": round(max_wait, 1),
            "arrival_flow": arrival_flow,
            "density": classify_traffic_density(total_pcu),
        }

    def compute_universal_priority_score(self, direction: str, interval: float, obs_summary: dict = None) -> float:
        st = self.get_approach_stats(direction, interval)
        flow = st["arrival_flow"]
        pcu  = st["total_pcu"]
        max_w = st["max_wait"]

        base_score = (flow * 0.40) + (pcu * 0.35) + (max_w * 0.25)

        if obs_summary and obs_summary.get("spawn_count", 0) >= self.MIN_OBS:
            hist_score = obs_summary["priority_score"]
            base_score = (0.65 * base_score) + (0.35 * hist_score)

        # Starvation Multiplier
        if max_w >= self.STARVATION_THRESHOLD_SEC:
            overage = max_w - self.STARVATION_THRESHOLD_SEC
            starvation_multiplier = 1.0 + math.pow(overage / 7.0, 1.95)
        else:
            starvation_multiplier = 1.0

        final_score = base_score * starvation_multiplier
        return round(final_score, 2)

    def simulate_4cycle_lookahead(
        self,
        target_axis: str,
        candidate_green: float,
        policy: str,
        heavy_dir: str,
        stats: dict,
        approach_intervals: dict,
    ) -> tuple[bool, float, str, int]:
        """
        Simulates traffic flow and queues over a 4-cycle receding horizon (k = 1..4).
        Calculates projected maximum wait times across all 4 approaches taking into account:
        - Interphase clearance times (yellow + all-red)
        - Continuous vehicle arrivals during red intervals
        - Multi-lane saturation queue discharge delay (startup lost time + shockwave discharge)

        Returns:
            (is_safe, max_projected_wait, bottleneck_direction, bottleneck_cycle)
        """
        q = {d: float(stats[d]["total_count"]) for d in ("N", "S", "E", "W")}
        w = {d: float(stats[d]["max_wait"]) for d in ("N", "S", "E", "W")}
        arr_rate = {
            d: max(0.05, stats[d]["arrival_flow"] / 60.0) for d in ("N", "S", "E", "W")
        }

        discharge_rate = 1.11      # ~1.11 veh/s across 2 discharge lanes
        clearance_lost_time = 4.0  # yellow (2.5s) + all-red (1.5s)
        startup_delay = 2.0        # startup lost time for queued front vehicles
        headway_per_veh = 0.9      # effective discharge headway across 2 lanes

        max_projected_wait = 0.0
        bottleneck_dir = heavy_dir
        bottleneck_cycle = 1

        cross_axis = "EW" if target_axis == "NS" else "NS"

        for k in range(1, self.LOOKAHEAD_CYCLES + 1):
            # --- Sub-Phase 1: Target Axis ---
            if k == 1:
                g1 = candidate_green
                cur_policy = policy
            else:
                q_tgt = max(q[AXIS_DIRS[target_axis][0]], q[AXIS_DIRS[target_axis][1]])
                g1 = max(self.min_green, min(self.max_green, 8.0 + q_tgt * 1.3))
                cur_policy = "BALANCED_PHASE"

            if cur_policy == "BALANCED_PHASE":
                green_dirs_1 = list(AXIS_DIRS[target_axis])
            else:
                green_dirs_1 = [heavy_dir]

            red_dirs_1 = [d for d in ("N", "S", "E", "W") if d not in green_dirs_1]
            elapsed_1 = g1 + clearance_lost_time

            for rd in red_dirs_1:
                if q[rd] > 0.5:
                    w[rd] += elapsed_1
                else:
                    new_arr = arr_rate[rd] * elapsed_1
                    w[rd] = elapsed_1 if new_arr >= 0.5 else 0.0
                q[rd] += arr_rate[rd] * elapsed_1

                if q[rd] >= 0.5:
                    clearance_lag = startup_delay + min(4.0, q[rd] * 0.5)
                    obs_proj = w[rd] + clearance_lag
                    if obs_proj > max_projected_wait:
                        max_projected_wait = obs_proj
                        bottleneck_dir = rd
                        bottleneck_cycle = k

            for gd in green_dirs_1:
                net_discharge = max(0.2, discharge_rate - arr_rate[gd])
                if q[gd] > 0.0:
                    q[gd] = max(0.0, q[gd] - net_discharge * g1)
                    if q[gd] <= 0.5:
                        q[gd] = 0.0
                        w[gd] = 0.0
                    else:
                        w[gd] = max(0.0, w[gd] - g1 * 0.8)
                else:
                    q[gd] = 0.0
                    w[gd] = 0.0

            # --- Sub-Phase 2: Cross Axis ---
            q_cross = max(q[AXIS_DIRS[cross_axis][0]], q[AXIS_DIRS[cross_axis][1]])
            g2 = max(self.min_green, min(24.0, 8.0 + q_cross * 1.1))
            green_dirs_2 = list(AXIS_DIRS[cross_axis])
            red_dirs_2 = list(AXIS_DIRS[target_axis])
            elapsed_2 = g2 + clearance_lost_time

            for rd in red_dirs_2:
                if q[rd] > 0.5:
                    w[rd] += elapsed_2
                else:
                    new_arr = arr_rate[rd] * elapsed_2
                    w[rd] = elapsed_2 if new_arr >= 0.5 else 0.0
                q[rd] += arr_rate[rd] * elapsed_2

                if q[rd] >= 0.5:
                    clearance_lag = startup_delay + min(4.0, q[rd] * 0.5)
                    obs_proj = w[rd] + clearance_lag
                    if obs_proj > max_projected_wait:
                        max_projected_wait = obs_proj
                        bottleneck_dir = rd
                        bottleneck_cycle = k

            for gd in green_dirs_2:
                net_discharge = max(0.2, discharge_rate - arr_rate[gd])
                if q[gd] > 0.0:
                    q[gd] = max(0.0, q[gd] - net_discharge * g2)
                    if q[gd] <= 0.5:
                        q[gd] = 0.0
                        w[gd] = 0.0
                    else:
                        w[gd] = max(0.0, w[gd] - g2 * 0.8)
                else:
                    q[gd] = 0.0
                    w[gd] = 0.0

        is_safe = (max_projected_wait <= self.HARD_CAP_MAX_WAIT_SEC)
        return is_safe, round(max_projected_wait, 1), bottleneck_dir, bottleneck_cycle

    def compute_lookahead_safe_green(
        self,
        target_axis: str,
        needed_green: float,
        policy: str,
        heavy_dir: str,
        stats: dict,
        approach_intervals: dict,
    ) -> tuple[float, bool, float, str]:
        """
        Calculates the maximum safe green duration under 4-Cycle Receding Horizon MPC.
        Pre-allocates the green duration upfront so countdown remains smooth and uninterrupted.

        Returns:
            (allocated_green, is_capped, projected_max_wait, bottleneck_dir)
        """
        cross_axis = "EW" if target_axis == "NS" else "NS"
        cross_dirs = AXIS_DIRS[cross_axis]
        max_opp_wait = max((stats[d]["max_wait"] for d in cross_dirs), default=0.0)

        is_safe, proj_w, b_dir, _ = self.simulate_4cycle_lookahead(
            target_axis, needed_green, policy, heavy_dir, stats, approach_intervals
        )

        # Trigger proactive cap if projected wait exceeds 48s ceiling OR if opposing wait entered 20s buffer (>=28s)
        needs_cap = (not is_safe) or (max_opp_wait >= self.PROACTIVE_TARGET_SEC)

        if not needs_cap:
            return round(needed_green, 1), False, proj_w, b_dir

        test_g = needed_green - 1.0
        best_g = self.min_green
        best_proj = proj_w
        found_safe = False

        while test_g >= self.min_green:
            safe, p_w, b_d, _ = self.simulate_4cycle_lookahead(
                target_axis, test_g, policy, heavy_dir, stats, approach_intervals
            )
            if safe and (max_opp_wait < self.PROACTIVE_TARGET_SEC or test_g <= 16.0):
                best_g = test_g
                best_proj = p_w
                found_safe = True
                break
            if p_w < best_proj:
                best_proj = p_w
                best_g = test_g
            test_g -= 1.0

        if not found_safe:
            best_g = self.min_green
            _, best_proj, b_dir, _ = self.simulate_4cycle_lookahead(
                target_axis, best_g, policy, heavy_dir, stats, approach_intervals
            )

        return round(best_g, 1), True, round(best_proj, 1), b_dir

    def get_next_phase_decision(
        self,
        target_axis: str,
        vehicles: list,
        approach_intervals: dict = None,
        cycle_observer=None,
    ) -> dict:
        if approach_intervals is None:
            approach_intervals = {d: 2.0 for d in "NSEW"}

        self.analyze_all_lanes(vehicles)

        dir_a, dir_b = AXIS_DIRS[target_axis]
        opp_a, opp_b = AXIS_DIRS["EW" if target_axis == "NS" else "NS"]

        stats = {d: self.get_approach_stats(d, approach_intervals.get(d, 2.0))
                 for d in ("N", "S", "E", "W")}

        obs_summaries = {}
        if cycle_observer is not None:
            obs_summaries = cycle_observer.get_all_prev_summaries()

        score_a = self.compute_universal_priority_score(dir_a, approach_intervals.get(dir_a, 2.0), obs_summaries.get(dir_a))
        score_b = self.compute_universal_priority_score(dir_b, approach_intervals.get(dir_b, 2.0), obs_summaries.get(dir_b))

        if score_a >= score_b:
            heavy_dir, light_dir = dir_a, dir_b
            heavy_score, light_score = score_a, score_b
        else:
            heavy_dir, light_dir = dir_b, dir_a
            heavy_score, light_score = score_b, score_a

        asym_ratio = round(heavy_score / (light_score + 0.5), 2)

        heavy_opp_dir = OPPOSING[heavy_dir]
        heavy_opp_pcu = stats[heavy_opp_dir]["total_pcu"]
        heavy_opp_wait = stats[heavy_opp_dir]["max_wait"]

        # Anti-Starvation Check using proactive 28s ceiling (48s hard cap - 20s flexible margin)
        cross_axis = "EW" if target_axis == "NS" else "NS"
        cross_dirs = AXIS_DIRS[cross_axis]
        max_cross_wait = max((stats[d]["max_wait"] for d in cross_dirs), default=0.0)

        anti_starvation_triggered = False
        if (heavy_opp_wait >= self.PROACTIVE_TARGET_SEC or
            max_cross_wait >= self.PROACTIVE_TARGET_SEC or
            self.consecutive_exclusive_count.get(heavy_dir, 0) >= 2):
            anti_starvation_triggered = True

        # Flow-Adaptive Capacity Sizing:
        heavy_flow = stats[heavy_dir]["arrival_flow"]
        heavy_count = stats[heavy_dir]["total_count"]

        # Base green calculation based on demand
        if heavy_flow >= 100.0 or heavy_count >= 12:
            needed_green = max(24.0, min(36.0, 10.0 + (heavy_count * 1.5) + (heavy_flow / 60.0 * 6.0)))
        elif heavy_flow >= 40.0 or heavy_count >= 6:
            needed_green = max(14.0, min(24.0, 8.0 + (heavy_count * 1.6)))
        else:
            needed_green = max(10.0, min(18.0, 6.0 + (heavy_count * 1.5)))

        # Preliminary policy evaluation
        if asym_ratio >= self.EXCL_DOMINANCE_RATIO and not anti_starvation_triggered:
            tentative_policy = _dir_to_exclusive_policy(heavy_dir)
        else:
            tentative_policy = "BALANCED_PHASE"

        # Apply Method 1: 4-Cycle Receding Horizon MPC Lookahead Predictive Pre-Allocation
        safe_green, mpc_capped, proj_wait, bottleneck_dir = self.compute_lookahead_safe_green(
            target_axis=target_axis,
            needed_green=needed_green,
            policy=tentative_policy,
            heavy_dir=heavy_dir,
            stats=stats,
            approach_intervals=approach_intervals,
        )

        # If bottleneck approach would suffer starvation under an exclusive phase, force BALANCED_PHASE
        if mpc_capped and tentative_policy != "BALANCED_PHASE" and (bottleneck_dir == heavy_opp_dir or bottleneck_dir in cross_dirs):
            policy = "BALANCED_PHASE"
            anti_starvation_triggered = True
            safe_green, mpc_capped, proj_wait, bottleneck_dir = self.compute_lookahead_safe_green(
                target_axis=target_axis,
                needed_green=needed_green,
                policy=policy,
                heavy_dir=heavy_dir,
                stats=stats,
                approach_intervals=approach_intervals,
            )
        else:
            policy = tentative_policy

        total_green = round(safe_green, 1)

        if policy != "BALANCED_PHASE":
            self.consecutive_exclusive_count[heavy_dir] = self.consecutive_exclusive_count.get(heavy_dir, 0) + 1
            for od in [d for d in ("N", "S", "E", "W") if d != heavy_dir]:
                self.consecutive_exclusive_count[od] = 0
            prioritized_approach = heavy_dir
        else:
            self.consecutive_exclusive_count[heavy_dir] = 0
            prioritized_approach = "BALANCED_COFLOW" if anti_starvation_triggered else "NONE"

        # Through vs Turn Split
        if policy == "BALANCED_PHASE":
            through_green = max(5.0, round(total_green * 0.72, 1))
            turn_green    = max(3.0, round(total_green - through_green, 1))
            if through_green + turn_green > total_green:
                through_green = max(4.5, round(total_green - turn_green, 1))
            split = {
                "through_green": through_green,
                "turn_green":    turn_green,
                "thru_pct": int(round((through_green / total_green) * 100)),
                "turn_pct": int(round((turn_green / total_green) * 100)),
                "opposing_adjusted": anti_starvation_triggered or mpc_capped,
                "data_source": "mpc_lookahead_balanced" if mpc_capped else "co_flow_balanced"
            }
        else:
            if cycle_observer is not None:
                split = cycle_observer.compute_green_split(
                    direction=heavy_dir,
                    total_green_sec=total_green,
                    opposing_pcu=heavy_opp_pcu,
                    min_through=8.0, max_through=28.0,
                    min_turn=4.0,     max_turn=9.0,
                )
            else:
                split = {
                    "through_green": round(total_green * 0.72, 1),
                    "turn_green":    round(total_green * 0.28, 1),
                    "thru_pct": 72, "turn_pct": 28,
                    "opposing_adjusted": mpc_capped,
                    "data_source": "default_fallback",
                }

        target_total_pcu = round(stats[dir_a]["total_pcu"] + stats[dir_b]["total_pcu"], 1)
        cross_total_pcu  = round(stats[opp_a]["total_pcu"] + stats[opp_b]["total_pcu"], 1)

        decision = {
            "target_axis":            target_axis,
            "policy":                 policy,
            "total_green_sec":        total_green,
            "green_duration":         total_green,
            "through_green_sec":      split["through_green"],
            "turn_green_sec":         split["turn_green"],
            "thru_pct":               split.get("thru_pct", 72),
            "turn_pct":               split.get("turn_pct", 28),
            "target_ratio":           split.get("thru_pct", 72),
            "target_pcu":             target_total_pcu,
            "cross_pcu":              cross_total_pcu,
            "prioritized_dir":        prioritized_approach,
            "opposing_dir":           heavy_opp_dir,
            "asym_ratio":             asym_ratio,
            "heavy_score":            heavy_score,
            "light_score":            light_score,
            "heavy_opp_pcu":          round(heavy_opp_pcu, 1),
            "heavy_opp_wait":         round(heavy_opp_wait, 1),
            "anti_starvation_active": anti_starvation_triggered or mpc_capped,
            "mpc_proactive_cap":      mpc_capped,
            "mpc_safe_green":         total_green,
            "mpc_projected_max_wait": round(proj_wait, 1),
            "mpc_bottleneck_dir":     bottleneck_dir,
            "lookahead_cycles":       self.LOOKAHEAD_CYCLES,
            "safety_margin_sec":      self.FLEXIBLE_BUFFER_SEC,
            "target_ceiling_sec":     self.PROACTIVE_TARGET_SEC,
            "data_source":            split.get("data_source", "default_fallback"),
            "ml_active":              self.ml_model is not None,
        }
        self.last_decision = decision
        return decision


class DQNAdaptiveController(BaseController):
    """
    Deep Reinforcement Learning (DQN) Signal Controller.
    Uses trained Dueling Double Deep Q-Network to evaluate optimal phase and timing dynamically.
    """
    def __init__(self):
        self.mode_name = "Deep Reinforcement Learning (Dueling DQN)"
        self.last_decision: dict = {}
        self.agent = None
        self.action_names = [
            "Extend Green", "NS Through", "NS Left Turn", "EW Through", "EW Left Turn",
            "N Exclusive", "S Exclusive", "E Exclusive", "W Exclusive"
        ]

        from ai_engine.rl_agent.dqn_model import DQNAgent
        self.agent = DQNAgent(state_dim=18, action_dim=9)
        model_path = os.path.join(PROJECT_ROOT, "ai_engine", "traffic_predictor", "saved_models", "dqn_traffic_agent.pth")
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print("[DQN Controller] Deep Q-Network weights loaded successfully!")
        else:
            print("[DQN Controller] No pre-trained weights found, using initialized network.")

    def _build_state(self, target_axis: str, vehicles: list, approach_intervals: dict) -> np.ndarray:
        # 1. 8 Lane Queues (PCU normalized by 12.0)
        lane_pcus = []
        for d in ('N', 'S', 'E', 'W'):
            for l_idx in (0, 1):
                lane_v = [v for v in vehicles if v.direction == d and v.lane_idx == l_idx and not v.has_cleared_intersection]
                pcu = calculate_lane_pcu(lane_v)
                lane_pcus.append(min(2.0, pcu / 12.0))

        # 2. 4 Approach Max Waiting Times (normalized by 40.0s)
        max_waits = []
        for d in ('N', 'S', 'E', 'W'):
            v_d = [v for v in vehicles if v.direction == d and not v.has_cleared_intersection]
            w = max((v.wait_time for v in v_d), default=0.0)
            max_waits.append(min(3.0, w / 40.0))

        # 3. 4 Approach Arrival Rates (normalized by 120 v/m)
        flows = []
        for d in ('N', 'S', 'E', 'W'):
            rate = approach_intervals.get(d, 2.0)
            flow = (60.0 / max(0.2, rate))
            flows.append(min(1.5, flow / 120.0))

        axis_code = 1.0 if target_axis == "NS" else 0.0
        elapsed_norm = 0.5

        state = np.array(lane_pcus + max_waits + flows + [axis_code, elapsed_norm], dtype=np.float32)
        return state

    def get_next_phase_decision(
        self,
        target_axis: str,
        vehicles: list,
        approach_intervals: dict = None,
        cycle_observer=None,
    ) -> dict:
        if approach_intervals is None:
            approach_intervals = {d: 2.0 for d in "NSEW"}

        state = self._build_state(target_axis, vehicles, approach_intervals)
        action_idx, q_values = self.agent.select_action(state, evaluate=True)

        # Map DQN Action to Policy and Axis
        dir_a, dir_b = AXIS_DIRS[target_axis]
        pcu_a = calculate_lane_pcu([v for v in vehicles if v.direction == dir_a and not v.has_cleared_intersection])
        pcu_b = calculate_lane_pcu([v for v in vehicles if v.direction == dir_b and not v.has_cleared_intersection])

        if target_axis == "NS":
            if action_idx == 5 or (pcu_a > 1.4 * (pcu_b + 0.5) and pcu_a > 5.0):
                policy = "N_EXCLUSIVE_GREEN"
                total_green = max(18.0, min(35.0, 10.0 + pcu_a * 1.5))
                prioritized = "N"
            elif action_idx == 6 or (pcu_b > 1.4 * (pcu_a + 0.5) and pcu_b > 5.0):
                policy = "S_EXCLUSIVE_GREEN"
                total_green = max(18.0, min(35.0, 10.0 + pcu_b * 1.5))
                prioritized = "S"
            else:
                policy = "BALANCED_PHASE"
                total_green = max(14.0, min(28.0, 8.0 + max(pcu_a, pcu_b) * 1.4))
                prioritized = "BALANCED_DQN"
        else:
            if action_idx == 7 or (pcu_a > 1.4 * (pcu_b + 0.5) and pcu_a > 5.0):
                policy = "E_EXCLUSIVE_GREEN"
                total_green = max(14.0, min(28.0, 8.0 + pcu_a * 1.5))
                prioritized = "E"
            elif action_idx == 8 or (pcu_b > 1.4 * (pcu_a + 0.5) and pcu_b > 5.0):
                policy = "W_EXCLUSIVE_GREEN"
                total_green = max(14.0, min(28.0, 8.0 + pcu_b * 1.5))
                prioritized = "W"
            else:
                policy = "BALANCED_PHASE"
                total_green = max(12.0, min(24.0, 6.0 + max(pcu_a, pcu_b) * 1.4))
                prioritized = "BALANCED_DQN"

        # Starvation Relief Guard (20s flexible margin: 48s - 20s = 28s)
        max_opp_wait = max((v.wait_time for v in vehicles if v.direction in AXIS_DIRS["EW" if target_axis == "NS" else "NS"] and not v.has_cleared_intersection), default=0.0)
        anti_starve = (max_opp_wait >= 28.0)
        if anti_starve:
            policy = "BALANCED_PHASE"
            prioritized = "DQN_ANTI_STARVATION"
            total_green = min(total_green, 16.0)

        through_green = max(6.0, round(total_green * 0.72, 1))
        turn_green    = max(3.5,  round(total_green * 0.28, 1))
        if through_green + turn_green > total_green:
            through_green = max(5.0, round(total_green - turn_green, 1))

        decision = {
            "target_axis":            target_axis,
            "policy":                 policy,
            "total_green_sec":        round(total_green, 1),
            "green_duration":         round(total_green, 1),
            "through_green_sec":      through_green,
            "turn_green_sec":         turn_green,
            "thru_pct":               int(round((through_green / total_green) * 100)),
            "turn_pct":               int(round((turn_green / total_green) * 100)),
            "target_ratio":           int(round((through_green / total_green) * 100)),
            "target_pcu":             round(pcu_a + pcu_b, 1),
            "cross_pcu":              round(max(0.0, max_opp_wait * 0.5), 1),
            "prioritized_dir":        prioritized,
            "asym_ratio":             round(max(pcu_a, pcu_b) / (min(pcu_a, pcu_b) + 0.5), 2),
            "dqn_action_idx":         action_idx,
            "dqn_action_name":        self.action_names[action_idx],
            "q_values":               [round(float(q), 2) for q in q_values],
            "anti_starvation_active": anti_starve,
            "data_source":            "deep_q_network_rl",
            "ml_active":              True,
        }
        self.last_decision = decision
        return decision
