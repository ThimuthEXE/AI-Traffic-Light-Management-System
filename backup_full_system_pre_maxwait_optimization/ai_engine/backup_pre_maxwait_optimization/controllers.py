"""
Intersection Control Strategies — Observation-First Cycle-Adaptive Controller
==============================================================================
AIFuzzyController now works in two modes:

  Cycle 0  (calibration): returns default timers while CycleObserver learns.
  Cycle 1+ (adaptive):    uses previous cycle's observations to:
    1. Rank all 4 approaches by priority score (spawn rate, PCU, wait time).
    2. If the top approach is sufficiently dominant on its axis, trigger
       Exclusive All-Green for that direction (both THROUGH and TURN green).
    3. Compute through_green / turn_green from the ACTUAL observed
       through-vs-turn vehicle ratio for that approach.
    4. Apply opposing-approach correction — if the opposing approach has
       a big queue, shorten the turn phase to get back to it sooner.
    5. Return {policy, through_green_sec, turn_green_sec} to the signal manager.

General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPPOSING = {"N": "S", "S": "N", "E": "W", "W": "E"}
AXIS_DIRS = {"NS": ("N", "S"), "EW": ("E", "W")}


def _dir_to_exclusive_policy(direction: str) -> str:
    return f"{direction}_EXCLUSIVE_GREEN"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseController:
    def get_next_phase_decision(self, target_axis: str, vehicles: list,
                                approach_intervals: dict = None,
                                cycle_observer=None) -> dict:
        raise NotImplementedError


class FixedTimeController(BaseController):
    def __init__(self, fixed_green: float = 25.0):
        self.fixed_green = float(fixed_green)
        self.mode_name = "Fixed-Time (Traditional)"
        self.last_decision: dict = {}

    def get_next_phase_decision(self, target_axis: str, vehicles: list,
                                approach_intervals: dict = None,
                                cycle_observer=None) -> dict:
        decision = {
            "policy": "BALANCED_PHASE",
            "target_axis": target_axis,
            "through_green_sec": self.fixed_green * 0.68,
            "turn_green_sec":    self.fixed_green * 0.32,
            "total_green_sec":   self.fixed_green,
            "asym_ratio": 1.0,
            "prioritized_dir": "NONE",
            "data_source": "fixed_time",
        }
        self.last_decision = decision
        return decision


# ---------------------------------------------------------------------------
# AI Adaptive Controller
# ---------------------------------------------------------------------------

class AIFuzzyController(BaseController):
    """
    Observation-first, cycle-adaptive signal controller.

    Decision flow each time get_next_phase_decision() is called
    (once per ALL_RED transition):

        1.  Pull previous-cycle observations from CycleObserver.
        2.  Score all 4 approaches → pick heavy direction on this axis.
        3.  If dominance ratio ≥ EXCL_THRESHOLD → Exclusive All-Green.
        4.  Use fuzzy engine + ML to compute total_green budget.
        5.  Split budget with observed through/turn ratio.
        6.  Apply opposing-PCU correction on the turn slice.
        7.  Return full decision dict to TrafficSignalManager.
    """

    # Dominance threshold: heavy / light priority score to trigger exclusive
    EXCL_THRESHOLD = 1.35

    # Min observed vehicles before trusting ratios (else use defaults)
    MIN_OBS = 3

    def __init__(self, min_green: float = 8.0, max_green: float = 45.0):
        self.fuzzy_engine = FuzzyTrafficController(min_green=min_green, max_green=max_green)
        self.min_green = min_green
        self.max_green = max_green
        self.mode_name = "Observation-First Cycle-Adaptive ML+Fuzzy"
        self.last_decision: dict = {}
        self.lane_stats: dict = {}

        # Load ML Phase Optimizer
        model_path = os.path.join(
            PROJECT_ROOT, "ai_engine", "traffic_predictor",
            "saved_models", "ml_phase_optimizer.joblib"
        )
        self.ml_model = None
        if os.path.exists(model_path):
            try:
                self.ml_model = joblib.load(model_path)
                print("[AI Controller] ML Phasing & Green Optimizer loaded successfully!")
            except Exception as exc:
                print(f"[AI Controller] ML model load failed: {exc}")

    # ------------------------------------------------------------------ #
    # Live queue analysis (called every frame by run_simulation)          #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Main decision entry-point                                            #
    # ------------------------------------------------------------------ #

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

        dir_a, dir_b = AXIS_DIRS[target_axis]          # e.g. 'N','S'
        opp_a, opp_b = AXIS_DIRS["EW" if target_axis == "NS" else "NS"]

        stats = {d: self.get_approach_stats(d, approach_intervals.get(d, 2.0))
                 for d in ("N", "S", "E", "W")}

        # ---- 1. Build priority scores from observations ----
        obs_summaries: dict[str, dict] = {}
        if cycle_observer is not None:
            obs_summaries = cycle_observer.get_all_prev_summaries()

        def priority_score(direction: str) -> float:
            # Live stats always capture the current spawn interval
            st        = stats[direction]
            live_flow = st["arrival_flow"]          # = 60 / spawn_interval_sec
            live_pcu  = st["total_pcu"]
            live_wait = st["max_wait"]
            live_score = live_flow * 0.45 + live_pcu * 0.30 + live_wait * 0.25

            obs = obs_summaries.get(direction)
            if obs and obs.get("spawn_count", 0) >= self.MIN_OBS:
                # Blend: live score (60%) + historical score (40%)
                # This makes real-time spawn rate dominate while honouring history
                return 0.60 * live_score + 0.40 * obs["priority_score"]
            return live_score

        score_a = priority_score(dir_a)
        score_b = priority_score(dir_b)

        if score_a >= score_b:
            heavy_dir, light_dir = dir_a, dir_b
            heavy_score, light_score = score_a, score_b
        else:
            heavy_dir, light_dir = dir_b, dir_a
            heavy_score, light_score = score_b, score_a

        asym_ratio = round(heavy_score / (light_score + 0.5), 2)

        # Opposing axis PCU (used for opposing-correction inside split)
        opp_pcu = stats[opp_a]["total_pcu"] + stats[opp_b]["total_pcu"]
        heavy_opp_dir = OPPOSING[heavy_dir]
        heavy_opp_pcu = stats[heavy_opp_dir]["total_pcu"]

        # ---- 2. Compute total green budget (ML then Fuzzy fallback) ----
        total_pcu_axis  = stats[dir_a]["total_pcu"] + stats[dir_b]["total_pcu"]
        cross_pcu       = stats[opp_a]["total_pcu"] + stats[opp_b]["total_pcu"]
        all_max_wait    = max(s["max_wait"] for s in stats.values())

        total_green = self._compute_total_green(
            target_axis, stats, approach_intervals, asym_ratio,
            total_pcu_axis, cross_pcu, all_max_wait
        )

        # ---- 3. Determine policy ----
        is_exclusive = asym_ratio >= self.EXCL_THRESHOLD
        if is_exclusive:
            policy = _dir_to_exclusive_policy(heavy_dir)
        else:
            policy = "BALANCED_PHASE"

        # ---- 4. Compute through / turn split ----
        if cycle_observer is not None and is_exclusive:
            split = cycle_observer.compute_green_split(
                direction=heavy_dir,
                total_green_sec=total_green,
                opposing_pcu=heavy_opp_pcu,
                min_through=8.0, max_through=40.0,
                min_turn=4.5,    max_turn=20.0,
            )
        else:
            # Balanced phase: choose the heavier approach's observed ratio
            representative_dir = heavy_dir
            if cycle_observer is not None:
                split = cycle_observer.compute_green_split(
                    direction=representative_dir,
                    total_green_sec=total_green,
                    opposing_pcu=heavy_opp_pcu,
                    min_through=8.0, max_through=40.0,
                    min_turn=4.5,    max_turn=20.0,
                )
            else:
                split = {
                    "through_green": round(total_green * 0.68, 1),
                    "turn_green":    round(total_green * 0.32, 1),
                    "thru_pct": 68, "turn_pct": 32,
                    "opposing_adjusted": False,
                    "data_source": "default_fallback",
                }

        # ---- 5. Assemble final decision ----
        decision = {
            "target_axis":       target_axis,
            "policy":            policy,
            "total_green_sec":   total_green,
            "through_green_sec": split["through_green"],
            "turn_green_sec":    split["turn_green"],
            "thru_pct":          split.get("thru_pct", 68),
            "turn_pct":          split.get("turn_pct", 32),
            "prioritized_dir":   heavy_dir if is_exclusive else "NONE",
            "opposing_dir":      heavy_opp_dir,
            "asym_ratio":        asym_ratio,
            "heavy_score":       round(heavy_score, 1),
            "light_score":       round(light_score, 1),
            "heavy_opp_pcu":     round(heavy_opp_pcu, 1),
            "opposing_adjusted": split.get("opposing_adjusted", False),
            "data_source":       split.get("data_source", "default_fallback"),
            "ml_active":         self.ml_model is not None,
        }
        self.last_decision = decision
        return decision

    # ------------------------------------------------------------------ #
    # Total green budget computation (ML + Fuzzy)                         #
    # ------------------------------------------------------------------ #

    def _compute_total_green(
        self,
        target_axis: str,
        stats: dict,
        approach_intervals: dict,
        asym_ratio: float,
        total_pcu_axis: float,
        cross_pcu: float,
        all_max_wait: float,
    ) -> float:

        dir_a, dir_b = AXIS_DIRS[target_axis]
        flow_a = stats[dir_a]["arrival_flow"]
        flow_b = stats[dir_b]["arrival_flow"]

        t_now = time.localtime()
        time_dec = t_now.tm_hour + t_now.tm_min / 60.0
        sin_t = math.sin(2 * math.pi * time_dec / 24.0)
        cos_t = math.cos(2 * math.pi * time_dec / 24.0)
        is_rush = int((7.0 <= time_dec <= 9.5) or (16.5 <= time_dec <= 19.5))

        total_green = 22.0

        if self.ml_model is not None:
            pcu = {d: stats[d]["total_pcu"] for d in "NSEW"}
            feat = pd.DataFrame([{
                "sin_time": sin_t, "cos_time": cos_t, "is_rush_hour": is_rush,
                "flow_N": stats["N"]["arrival_flow"],
                "flow_S": stats["S"]["arrival_flow"],
                "flow_E": stats["E"]["arrival_flow"],
                "flow_W": stats["W"]["arrival_flow"],
                "pcu_N_L0": self.lane_stats.get(("N", 0), {}).get("pcu", 0.0),
                "pcu_N_L1": self.lane_stats.get(("N", 1), {}).get("pcu", 0.0),
                "pcu_S_L0": self.lane_stats.get(("S", 0), {}).get("pcu", 0.0),
                "pcu_S_L1": self.lane_stats.get(("S", 1), {}).get("pcu", 0.0),
                "pcu_E_L0": self.lane_stats.get(("E", 0), {}).get("pcu", 0.0),
                "pcu_E_L1": self.lane_stats.get(("E", 1), {}).get("pcu", 0.0),
                "pcu_W_L0": self.lane_stats.get(("W", 0), {}).get("pcu", 0.0),
                "pcu_W_L1": self.lane_stats.get(("W", 1), {}).get("pcu", 0.0),
                "tot_pcu_N": pcu["N"], "tot_pcu_S": pcu["S"],
                "tot_pcu_E": pcu["E"], "tot_pcu_W": pcu["W"],
                "max_wait_sec": all_max_wait,
                "asym_ratio": asym_ratio,
            }])
            try:
                reg = self.ml_model["regressor"]
                total_green = float(reg.predict(feat)[0])
            except Exception as exc:
                print(f"[ML Regressor Error] {exc}")

        # Fuzzy refinement
        try:
            fuzzy_eval = self.fuzzy_engine.compute_green_duration(
                current_pcu=total_pcu_axis,
                max_wait_time=all_max_wait,
                cross_pcu=cross_pcu,
            )
            fuzzy_g = fuzzy_eval["green_duration"]
            # Blend ML + Fuzzy (60/40 when ML active, else pure fuzzy)
            if self.ml_model is not None:
                total_green = round(0.60 * total_green + 0.40 * fuzzy_g, 1)
            else:
                total_green = fuzzy_g
        except Exception as exc:
            print(f"[Fuzzy Error] {exc}")

        return max(self.min_green, min(self.max_green, round(total_green, 1)))
