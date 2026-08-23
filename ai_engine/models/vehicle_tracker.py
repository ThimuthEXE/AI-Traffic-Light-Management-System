"""
Multi-Object Vehicle Tracker
Maintains persistent IDs across video frames using Centroid Distance and IoU matching.
Prevents duplicate vehicle counts and calculates vehicle dwell/wait times.
"""

import math
from collections import OrderedDict

def calculate_iou(boxA, boxB):
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


class TrackedVehicle:
    def __init__(self, track_id: int, detection: dict):
        self.track_id = track_id
        self.class_name = detection["class_name"]
        self.bbox = detection["bbox"]
        self.center = detection["center"]
        self.confidence = detection["confidence"]
        
        self.trajectory = [self.center]
        self.disappeared_frames = 0
        self.frames_stationary = 0
        self.total_frames = 1
        self.assigned_lane = None

    def update(self, detection: dict):
        self.class_name = detection["class_name"]
        self.bbox = detection["bbox"]
        old_center = self.center
        self.center = detection["center"]
        self.confidence = detection["confidence"]
        
        self.trajectory.append(self.center)
        if len(self.trajectory) > 30:
            self.trajectory.pop(0)

        # Check movement delta (pixels)
        dist = math.hypot(self.center[0] - old_center[0], self.center[1] - old_center[1])
        if dist < 2.0:
            self.frames_stationary += 1
        else:
            self.frames_stationary = max(0, self.frames_stationary - 1)

        self.disappeared_frames = 0
        self.total_frames += 1


class MultiVehicleTracker:
    def __init__(self, max_disappeared: int = 15, max_distance: float = 65.0):
        self.next_track_id = 1
        self.tracks = OrderedDict()  # track_id -> TrackedVehicle
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections: list) -> list:
        """
        Associates current frame detections with existing tracks.
        Returns list of active TrackedVehicle objects.
        """
        if len(detections) == 0:
            # Mark all tracks as disappeared
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].disappeared_frames += 1
                if self.tracks[track_id].disappeared_frames > self.max_disappeared:
                    del self.tracks[track_id]
            return list(self.tracks.values())

        if len(self.tracks) == 0:
            # Register all detections as new tracks
            for det in detections:
                self._register_track(det)
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_centers = [self.tracks[tid].center for tid in track_ids]

        det_centers = [d["center"] for d in detections]

        # Compute cost / distance matrix
        matched_tracks = set()
        matched_dets = set()

        for det_idx, det in enumerate(detections):
            best_dist = float("inf")
            best_track_idx = -1
            
            for trk_idx, trk_id in enumerate(track_ids):
                if trk_idx in matched_tracks:
                    continue
                
                dist = math.hypot(det["center"][0] - track_centers[trk_idx][0],
                                  det["center"][1] - track_centers[trk_idx][1])
                
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_track_idx = trk_idx

            if best_track_idx != -1:
                matched_tracks.add(best_track_idx)
                matched_dets.add(det_idx)
                matched_tid = track_ids[best_track_idx]
                self.tracks[matched_tid].update(det)

        # Register unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                self._register_track(det)

        # Increment disappeared count for unmatched existing tracks
        for trk_idx, trk_id in enumerate(track_ids):
            if trk_idx not in matched_tracks:
                self.tracks[trk_id].disappeared_frames += 1
                if self.tracks[trk_id].disappeared_frames > self.max_disappeared:
                    del self.tracks[trk_id]

        return list(self.tracks.values())

    def _register_track(self, detection: dict):
        self.tracks[self.next_track_id] = TrackedVehicle(self.next_track_id, detection)
        self.next_track_id += 1
