"""
4-Junction 2×2 Grid Inter-Intersection (I2I) Green Wave Synchronization Coordinator
General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI — Group 06

Handles 8 platoon-tracking channels:
  - EB + WB on each of 2 horizontal segments (Seg_TOP, Seg_BOT)
  - SB + NB on each of 2 vertical segments  (Seg_LEFT, Seg_RIGHT)

For each channel: detects a platoon of ≥ 3 vehicles on the segment, computes ETA to
the downstream junction's stop line, and fires trigger_green_wave_preemption() on that
junction with the correct target_axis ('EW' or 'NS').

A per-channel cooldown (8 s) prevents repeated firing on the same platoon.
"""


class FourJunctionCoordinator:
    PLATOON_MIN_SIZE = 3
    ETA_MIN          = 1.5   # seconds
    ETA_MAX          = 7.0   # seconds
    COOLDOWN_TIME    = 8.0   # seconds

    def __init__(self, junctions: dict, segments: list, enabled: bool = True):
        """
        Parameters
        ----------
        junctions : dict[int, JunctionNode]
            All 4 junctions keyed 1–4.
        segments : list[ArterySegment]
            All 4 artery segments (TOP, BOT, LEFT, RIGHT).
        enabled : bool
            Set False to disable I2I coordination (fixed-time mode ignores it).
        """
        self.junctions = junctions
        self.segments  = segments
        self.enabled   = enabled
        self.green_waves_triggered = 0
        self._cooldowns: dict = {}   # (seg_id, direction) -> remaining seconds

    # ------------------------------------------------------------------
    # Main Update (called once per frame)
    # ------------------------------------------------------------------

    def update(self, dt: float, vehicles: list):
        if not self.enabled:
            return

        # Tick down all active cooldowns
        for key in list(self._cooldowns):
            self._cooldowns[key] -= dt
            if self._cooldowns[key] <= 0:
                del self._cooldowns[key]

        for seg in self.segments:
            if seg.orientation == 'EW':
                self._check_platoon_ew(seg, 'E', vehicles)
                self._check_platoon_ew(seg, 'W', vehicles)
            else:
                self._check_platoon_ns(seg, 'S', vehicles)
                self._check_platoon_ns(seg, 'N', vehicles)

    # ------------------------------------------------------------------
    # Platoon Detection — Horizontal Segments
    # ------------------------------------------------------------------

    def _check_platoon_ew(self, seg, direction: str, vehicles: list):
        """Detect a platoon on a horizontal segment and fire EW green wave."""
        if direction == 'E':
            dst_j     = seg.downstream_j
            stop_line = dst_j.cx - 85          # EB stop line at downstream junction
            platoon   = [
                v for v in vehicles
                if v.direction == 'E'
                and not v.is_turning
                and seg.pos_start <= v.x <= seg.pos_end
                and abs(v.y - seg.road_ref) < 85
            ]
            if len(platoon) < self.PLATOON_MIN_SIZE:
                return
            platoon.sort(key=lambda v: -v.x)          # lead = highest x (closest to dst)
            lead = platoon[0]
            dist = stop_line - (lead.x + lead.length / 2)

        else:   # 'W'
            dst_j     = seg.upstream_j
            stop_line = dst_j.cx + 85          # WB stop line at upstream junction
            platoon   = [
                v for v in vehicles
                if v.direction == 'W'
                and not v.is_turning
                and seg.pos_start <= v.x <= seg.pos_end
                and abs(v.y - seg.road_ref) < 85
            ]
            if len(platoon) < self.PLATOON_MIN_SIZE:
                return
            platoon.sort(key=lambda v: v.x)           # lead = lowest x (closest to dst)
            lead = platoon[0]
            dist = (lead.x - lead.length / 2) - stop_line

        self._fire_if_valid(seg, direction, dst_j, dist, lead, target_axis='EW')

    # ------------------------------------------------------------------
    # Platoon Detection — Vertical Segments
    # ------------------------------------------------------------------

    def _check_platoon_ns(self, seg, direction: str, vehicles: list):
        """Detect a platoon on a vertical segment and fire NS green wave."""
        if direction == 'S':
            dst_j     = seg.downstream_j
            stop_line = dst_j.cy - 85          # SB stop line at downstream junction
            platoon   = [
                v for v in vehicles
                if v.direction == 'S'
                and not v.is_turning
                and seg.pos_start <= v.y <= seg.pos_end
                and abs(v.x - seg.road_ref) < 85
            ]
            if len(platoon) < self.PLATOON_MIN_SIZE:
                return
            platoon.sort(key=lambda v: -v.y)          # lead = highest y (closest to dst)
            lead = platoon[0]
            dist = stop_line - (lead.y + lead.length / 2)

        else:   # 'N'
            dst_j     = seg.upstream_j
            stop_line = dst_j.cy + 85          # NB stop line at upstream junction
            platoon   = [
                v for v in vehicles
                if v.direction == 'N'
                and not v.is_turning
                and seg.pos_start <= v.y <= seg.pos_end
                and abs(v.x - seg.road_ref) < 85
            ]
            if len(platoon) < self.PLATOON_MIN_SIZE:
                return
            platoon.sort(key=lambda v: v.y)           # lead = lowest y (closest to dst)
            lead = platoon[0]
            dist = (lead.y - lead.length / 2) - stop_line

        self._fire_if_valid(seg, direction, dst_j, dist, lead, target_axis='NS')

    # ------------------------------------------------------------------
    # Common Fire Logic
    # ------------------------------------------------------------------

    def _fire_if_valid(self, seg, direction, dst_j, dist, lead, target_axis):
        if dist <= 0:
            return
        speed_px_s = max(1.5, lead.speed * 60.0)
        eta        = dist / speed_px_s
        if not (self.ETA_MIN <= eta <= self.ETA_MAX):
            return
        key = (seg.seg_id, direction)
        if key in self._cooldowns:
            return
        dst_j.trigger_green_wave_preemption(target_axis, lead_time_sec=eta)
        self.green_waves_triggered += 1
        self._cooldowns[key] = self.COOLDOWN_TIME
