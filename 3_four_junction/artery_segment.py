"""
Artery Segment — Inter-Junction Road Link with Capacity Management
General Sir John Kotelawala Defence University (KDU) — IT 3182 Essentials of AI — Group 06

Supports both horizontal (EW) and vertical (NS) road segments between adjacent junctions.

Each ArterySegment:
  - Tracks how many vehicles are in-transit on the segment
  - Enforces a maximum capacity (hard cap)
  - Signals the upstream junction to meter (hold RED) the departing direction when the
    segment approaches saturation — analogous to a highway ramp-metering system

Orientation:
  'EW'  → upstream = west junction, downstream = east junction  (EB/WB traffic)
  'NS'  → upstream = north junction (lower cy), downstream = south junction (higher cy)  (SB/NB traffic)
"""

ROAD_HALF_W = 85   # half road-width used as cross-axis filter tolerance


class ArterySegment:
    """
    Represents the road link between two adjacent junction nodes.

    Parameters
    ----------
    seg_id : str
        Human-readable label, e.g. "TOP", "BOT", "LEFT", "RIGHT".
    upstream_j : JunctionNode
        For EW: the western junction.  For NS: the northern junction (lower cy).
    downstream_j : JunctionNode
        For EW: the eastern junction.  For NS: the southern junction (higher cy).
    orientation : str
        'EW' for horizontal segments, 'NS' for vertical segments.
    max_vehicles : int
        Hard capacity ceiling for the segment.
    drain_threshold : float
        Fraction of max_vehicles at which metering activates (default 0.80 = 80%).
    """

    def __init__(
        self,
        seg_id: str,
        upstream_j,
        downstream_j,
        orientation: str = 'EW',
        max_vehicles: int = 12,
        drain_threshold: float = 0.80,
    ):
        self.seg_id         = seg_id
        self.upstream_j     = upstream_j
        self.downstream_j   = downstream_j
        self.orientation    = orientation
        self.max_vehicles   = max_vehicles
        self.drain_threshold = drain_threshold

        if orientation == 'EW':
            # Horizontal: measured along the x-axis
            self.pos_start = upstream_j.cx + 85     # east exit of west junction
            self.pos_end   = downstream_j.cx - 85   # west entry of east junction
            self.road_ref  = upstream_j.cy           # y of the shared horizontal road
            self.upstream_depart_dir   = 'E'
            self.downstream_depart_dir = 'W'
        else:
            # Vertical: measured along the y-axis
            self.pos_start = upstream_j.cy + 85     # south exit of north junction
            self.pos_end   = downstream_j.cy - 85   # north entry of south junction
            self.road_ref  = upstream_j.cx           # x of the shared vertical road
            self.upstream_depart_dir   = 'S'
            self.downstream_depart_dir = 'N'

        self.current_count = 0
        self._history: list = []

    # ------------------------------------------------------------------
    # Per-Frame Update
    # ------------------------------------------------------------------

    def update(self, vehicles: list):
        """Count in-transit vehicles then apply or release capacity holds.
        Must be called once per simulation frame before vehicle physics updates.
        """
        if self.orientation == 'EW':
            self.current_count = sum(
                1 for v in vehicles
                if v.direction in ('E', 'W')
                and not v.is_turning
                and self.pos_start <= v.x <= self.pos_end
                and abs(v.y - self.road_ref) < ROAD_HALF_W
            )
        else:  # 'NS'
            self.current_count = sum(
                1 for v in vehicles
                if v.direction in ('S', 'N')
                and not v.is_turning
                and self.pos_start <= v.y <= self.pos_end
                and abs(v.x - self.road_ref) < ROAD_HALF_W
            )

        draining = self.is_draining()
        self.upstream_j.set_capacity_hold(self.upstream_depart_dir,   draining)
        self.downstream_j.set_capacity_hold(self.downstream_depart_dir, draining)

        self._history.append(self.current_count)
        if len(self._history) > 60:
            self._history.pop(0)

    # ------------------------------------------------------------------
    # Status Queries
    # ------------------------------------------------------------------

    def is_at_capacity(self) -> bool:
        return self.current_count >= self.max_vehicles

    def is_draining(self) -> bool:
        """True when metering should activate (≥ drain_threshold of capacity)."""
        return self.current_count >= self.max_vehicles * self.drain_threshold

    def utilization(self) -> float:
        """0.0 – 1.0+ fraction of max capacity currently in use."""
        return self.current_count / max(1, self.max_vehicles)

    def peak_recent(self) -> int:
        return max(self._history) if self._history else 0

    def __repr__(self) -> str:
        return (
            f"ArterySegment({self.seg_id} [{self.orientation}]: "
            f"J{self.upstream_j.id}↔J{self.downstream_j.id}, "
            f"{self.current_count}/{self.max_vehicles} veh, "
            f"util={self.utilization():.0%})"
        )
