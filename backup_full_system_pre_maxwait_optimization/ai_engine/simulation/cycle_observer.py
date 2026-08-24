"""
Cycle Observer — Per-Approach Traffic Observation Engine
=========================================================
Runs during every signal cycle (default timers on Cycle 0, live timers after).
Records per direction:
  - Spawn count and arrival flow
  - Through / Turn vehicle ratio (based on actual turn_intent observed)
  - Queue PCU buildup
  - Average and maximum wait time
  - Opposing approach PCU snapshot at turn phase start

Provides a clean `get_cycle_summary(direction)` dict that the controller
uses to compute the next cycle's through_green / turn_green split and to
decide which approach, if any, receives Exclusive All-Green priority.

General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


@dataclass
class ApproachObservation:
    """Accumulates raw observations for one approach direction during one cycle."""
    direction: str

    # Spawn / flow
    spawn_count: int = 0
    cycle_duration_sec: float = 0.0          # time over which spawns were counted

    # Move intent counts (from vehicle.turn_intent)
    straight_count: int = 0                  # STRAIGHT
    right_count: int = 0                     # RIGHT (goes with through signal)
    left_count: int = 0                      # LEFT  (goes with turn signal)

    # Queue (sampled every sample_interval_sec)
    pcu_samples: List[float] = field(default_factory=list)

    # Wait times (snapshotted from active vehicles)
    wait_time_samples: List[float] = field(default_factory=list)

    def record_vehicle(self, vehicle) -> None:
        """Called when a new vehicle spawns on this approach."""
        self.spawn_count += 1
        intent = getattr(vehicle, "turn_intent", "STRAIGHT")
        if intent == "LEFT":
            self.left_count += 1
        elif intent == "RIGHT":
            self.right_count += 1
        else:
            self.straight_count += 1

    def sample_queue(self, pcu: float) -> None:
        self.pcu_samples.append(pcu)

    def sample_waits(self, wait_times: List[float]) -> None:
        self.wait_time_samples.extend(wait_times)

    # ------------------------------------------------------------------ #
    # Derived properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def total_intent_count(self) -> int:
        return self.straight_count + self.right_count + self.left_count

    @property
    def through_ratio(self) -> float:
        """Fraction of vehicles that use the THROUGH (straight / right) signal."""
        total = self.total_intent_count
        if total < 3:
            return 0.65           # safe default when sample too small
        return (self.straight_count + self.right_count) / total

    @property
    def turn_ratio(self) -> float:
        """Fraction of vehicles that use the TURN (protected left) signal."""
        total = self.total_intent_count
        if total < 3:
            return 0.35
        return self.left_count / total

    @property
    def arrival_flow_vpm(self) -> float:
        """Vehicles per minute (capped to avoid division by tiny durations)."""
        dur = max(self.cycle_duration_sec, 1.0)
        return round(self.spawn_count / dur * 60.0, 1)

    @property
    def avg_queue_pcu(self) -> float:
        if not self.pcu_samples:
            return 0.0
        return round(sum(self.pcu_samples) / len(self.pcu_samples), 2)

    @property
    def peak_queue_pcu(self) -> float:
        return round(max(self.pcu_samples, default=0.0), 2)

    @property
    def avg_wait_sec(self) -> float:
        if not self.wait_time_samples:
            return 0.0
        return round(sum(self.wait_time_samples) / len(self.wait_time_samples), 1)

    @property
    def max_wait_sec(self) -> float:
        return round(max(self.wait_time_samples, default=0.0), 1)

    @property
    def priority_score(self) -> float:
        """Composite score used to rank approaches for priority selection."""
        return (
            self.arrival_flow_vpm    * 0.40 +
            self.avg_queue_pcu       * 0.35 +
            self.avg_wait_sec        * 0.15 +
            self.max_wait_sec        * 0.10
        )

    def to_dict(self) -> dict:
        return {
            "direction":        self.direction,
            "spawn_count":      self.spawn_count,
            "arrival_flow_vpm": self.arrival_flow_vpm,
            "straight_count":   self.straight_count,
            "right_count":      self.right_count,
            "left_count":       self.left_count,
            "through_ratio":    round(self.through_ratio, 3),
            "turn_ratio":       round(self.turn_ratio, 3),
            "avg_queue_pcu":    self.avg_queue_pcu,
            "peak_queue_pcu":   self.peak_queue_pcu,
            "avg_wait_sec":     self.avg_wait_sec,
            "max_wait_sec":     self.max_wait_sec,
            "priority_score":   round(self.priority_score, 2),
        }


class CycleObserver:
    """
    Observes one full signal cycle (EW + NS) and produces per-approach summaries
    that drive the *next* cycle's timing decisions.

    Usage
    -----
    On every simulation tick:
        observer.tick(dt, vehicles, approach_pcu_by_dir)

    When a new vehicle spawns:
        observer.on_vehicle_spawned(vehicle)

    At the end of a complete EW+NS cycle:
        summaries = observer.end_cycle()   # dict[direction -> ApproachObservation]
        observer.begin_cycle()             # reset for next cycle
    """

    SAMPLE_INTERVAL = 2.0      # seconds between PCU / wait-time snapshots
    MIN_SAMPLE_COUNT = 3       # min observed vehicles before ratios are trusted

    def __init__(self):
        self.cycle_number: int = 0
        self.is_calibration: bool = True       # True during cycle 0
        self._observations: Dict[str, ApproachObservation] = {}
        self._prev_summaries: Dict[str, dict] = {}
        self._sample_timer: float = 0.0
        self._elapsed: float = 0.0
        self._begin_fresh()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _begin_fresh(self) -> None:
        for d in ("N", "S", "E", "W"):
            self._observations[d] = ApproachObservation(direction=d)
        self._sample_timer = 0.0
        self._elapsed = 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def on_vehicle_spawned(self, vehicle) -> None:
        """Call immediately after a new vehicle is added to the simulation."""
        d = vehicle.direction
        if d in self._observations:
            self._observations[d].record_vehicle(vehicle)

    def tick(self, dt: float, vehicles: list, pcu_by_dir: Dict[str, float]) -> None:
        """
        Called every simulation frame.

        Parameters
        ----------
        dt : float  — time step in seconds
        vehicles : list[Vehicle]  — all active vehicles on screen
        pcu_by_dir : dict[str, float]  — current PCU per direction (pre-computed by caller)
        """
        self._elapsed += dt
        for obs in self._observations.values():
            obs.cycle_duration_sec = self._elapsed

        self._sample_timer += dt
        if self._sample_timer >= self.SAMPLE_INTERVAL:
            self._sample_timer = 0.0
            # Snapshot PCU and wait times per direction
            for d, obs in self._observations.items():
                obs.sample_queue(pcu_by_dir.get(d, 0.0))
                waits = [
                    v.wait_time
                    for v in vehicles
                    if v.direction == d and not v.has_cleared_intersection
                ]
                if waits:
                    obs.sample_waits(waits)

    def end_cycle(self) -> Dict[str, dict]:
        """
        Finalise the current cycle observations and return summaries.
        Stores them as _prev_summaries for the controller to query.
        """
        summaries = {d: obs.to_dict() for d, obs in self._observations.items()}
        self._prev_summaries = summaries
        return summaries

    def begin_cycle(self) -> None:
        """Reset observer for the next cycle."""
        self.cycle_number += 1
        self.is_calibration = (self.cycle_number == 0)
        self._begin_fresh()

    def get_prev_summary(self, direction: str) -> dict:
        """Return the most recent completed cycle's summary for a direction."""
        return self._prev_summaries.get(direction, {})

    def get_all_prev_summaries(self) -> Dict[str, dict]:
        return dict(self._prev_summaries)

    # ------------------------------------------------------------------ #
    # Smart split computation (used by controller)                         #
    # ------------------------------------------------------------------ #

    def compute_green_split(
        self,
        direction: str,
        total_green_sec: float,
        opposing_pcu: float,
        min_through: float = 8.0,
        max_through: float = 40.0,
        min_turn: float = 4.5,
        max_turn: float = 20.0,
    ) -> dict:
        """
        Given the total green budget for this approach, split it into
        through_green and turn_green using the observed vehicle ratios,
        then apply an opposing-approach correction.

        Parameters
        ----------
        direction     : approach direction ('N','S','E','W')
        total_green_sec : total green time allocated for this approach
        opposing_pcu  : current PCU count of the opposing approach
        min/max_through : bounds for through green
        min/max_turn    : bounds for turn green

        Returns
        -------
        dict with keys: through_green, turn_green, thru_pct, turn_pct, opposing_adjusted
        """
        summary = self._prev_summaries.get(direction)
        if not summary or summary.get("spawn_count", 0) < self.MIN_SAMPLE_COUNT:
            # No reliable data yet → use safe defaults
            through_g = max(min_through, min(max_through, round(total_green_sec * 0.68, 1)))
            turn_g    = max(min_turn,    min(max_turn,    round(total_green_sec * 0.32, 1)))
            return {
                "through_green": through_g,
                "turn_green": turn_g,
                "thru_pct": 68,
                "turn_pct": 32,
                "opposing_adjusted": False,
                "data_source": "default_fallback",
            }

        thru_ratio = summary["through_ratio"]
        turn_ratio = summary["turn_ratio"]

        raw_thru = total_green_sec * thru_ratio
        raw_turn = total_green_sec * turn_ratio

        through_g = max(min_through, min(max_through, round(raw_thru, 1)))
        turn_g    = max(min_turn,    min(max_turn,    round(raw_turn, 1)))

        # Opposing-approach correction:
        # If opposing side has a significant queue, shorten turn phase
        # so the cycle ends sooner and opposing gets green faster.
        opposing_adjusted = False
        if opposing_pcu >= 4.0 and turn_g > min_turn:
            # Discount factor: heavier opposing → stronger reduction (max 40%)
            discount = min(0.40, (opposing_pcu - 4.0) / 20.0)
            adjusted_turn = max(min_turn, round(turn_g * (1.0 - discount), 1))
            if adjusted_turn < turn_g:
                turn_g = adjusted_turn
                opposing_adjusted = True

        return {
            "through_green": through_g,
            "turn_green": turn_g,
            "thru_pct": round(thru_ratio * 100),
            "turn_pct": round(turn_ratio * 100),
            "opposing_adjusted": opposing_adjusted,
            "data_source": "observed",
        }
