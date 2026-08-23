"""
YOLOv8 Traffic Video Monitor & Real-Time PCU Analytics Visualizer
Processes traffic video streams, performs YOLOv8 detection, multi-object tracking,
lane ROI classification, and live Proportional Flow & Fuzzy Logic decision calculation.
"""

import sys
import os
import time
import argparse
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_engine.models.yolo_detector import YOLOVehicleDetector, CLASS_COLORS
from ai_engine.models.vehicle_tracker import MultiVehicleTracker
from ai_engine.models.lane_counter import LaneTrafficAnalyzer
from ai_engine.models.vision_fuzzy_bridge import VisionFuzzyBridge


def draw_hud(frame: np.ndarray, fps: float, lane_results: dict, total_active: int, lifetime_counted: int, fuzzy_decision: dict):
    """Draws sleek analytics telemetry HUD on the OpenCV camera frame."""
    h, w, _ = frame.shape
    
    # Top-Left Telemetry Card
    hud_bg = frame.copy()
    cv2.rectangle(hud_bg, (15, 15), (390, 230), (20, 24, 30), -1)
    cv2.addWeighted(hud_bg, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (15, 15), (390, 230), (70, 80, 95), 1)

    # Title & Live Stats
    cv2.putText(frame, "AI TRAFFIC VISION MONITOR (YOLOv8)", (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 115), 2, cv2.LINE_AA)
    
    stat_line = f"FPS: {fps:.1f} | Active: {total_active} | Cleared Total: {lifetime_counted}"
    cv2.putText(frame, stat_line, (25, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(frame, (25, 72), (380, 72), (60, 70, 80), 1)

    # Lane Breakdown
    y = 94
    for zone_name, data in lane_results.items():
        cnt = data.get("vehicle_count", 0)
        pcu = data.get("pcu_load", 0.0)
        density = data.get("density_level", "LOW")

        d_color = (0, 230, 115) if density == "LOW" else (0, 215, 255) if density == "MEDIUM" else (0, 100, 255)
        
        cv2.putText(frame, f"{zone_name}: {cnt} veh ({pcu:.1f} PCU)", (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(frame, f"[{density}]", (295, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, d_color, 1, cv2.LINE_AA)
        y += 24

    controls_legend = "[SPACE]: Pause  [S]: Snapshot  [Q]: Quit"
    cv2.putText(frame, controls_legend, (25, 218),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 160, 190), 1, cv2.LINE_AA)

    # Top-Right Proportional Decision Card
    ai_bg = frame.copy()
    cv2.rectangle(ai_bg, (w - 395, 15), (w - 15, 175), (20, 24, 30), -1)
    cv2.addWeighted(ai_bg, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (w - 395, 15), (w - 15, 175), (0, 230, 115), 1)

    cv2.putText(frame, "FLOW PROPORTIONAL SIGNAL TIMING", (w - 385, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 115), 2, cv2.LINE_AA)

    tgt_axis = fuzzy_decision.get("target_axis", "EW")
    g_time = fuzzy_decision.get("green_duration", 20.0)
    tgt_ratio = fuzzy_decision.get("target_ratio", 50.0)
    tgt_pcu = fuzzy_decision.get("target_pcu", 0.0)
    cross_pcu = fuzzy_decision.get("cross_pcu", 0.0)

    cv2.putText(frame, f"Active Phase: {tgt_axis} -> {tgt_ratio}% of Total Traffic", (w - 385, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)
    
    cv2.putText(frame, f"Allocated Green Countdown: {g_time:.1f}s", (w - 385, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Phase PCU: {tgt_pcu:.1f} | Cross PCU: {cross_pcu:.1f}", (w - 385, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    
    cv2.putText(frame, "Mode: Vision PCU + Proportional Demand", (w - 385, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 115), 1, cv2.LINE_AA)


def run_traffic_monitor(source: str, save_path: str = None, conf_thresh: float = 0.35, headless: bool = False, loop: bool = True):
    """Main processing pipeline for video / camera stream."""
    print(f"Initializing YOLOv8 Traffic Monitor on source: {source}")
    
    if not os.path.exists(source) and not source.isdigit():
        print(f"Error: Video file not found: {source}")
        return

    # Initialize components
    detector = YOLOVehicleDetector(conf_thresh=conf_thresh)
    tracker = MultiVehicleTracker(max_disappeared=20, max_distance=60.0)
    lane_analyzer = LaneTrafficAnalyzer()
    fuzzy_bridge = VisionFuzzyBridge()

    # Open Video Source
    is_cam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_cam else source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Video Writer if save requested
    out_writer = None
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(save_path, fourcc, fps_in, (width, height))
        print(f"Annotated video will be saved to: {save_path}")

    frame_count = 0
    start_time = time.time()
    current_axis = "EW"
    axis_timer = 0.0

    paused = False
    cv2_window_name = "AI Traffic Light System - YOLOv8 Detection & PCU Monitor"
    
    if not headless:
        cv2.namedWindow(cv2_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cv2_window_name, 1280, 720)

    try:
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if loop and not is_cam:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("End of video stream reached.")
                        break
                frame_count += 1

                # 1. YOLOv8 Vehicle Detection
                detections = detector.detect_vehicles(frame)

                # 2. Multi-Object Tracking
                tracked_vehicles = tracker.update(detections)

                # 3. Lane ROI & PCU Analysis
                lane_results = lane_analyzer.analyze_lanes(tracked_vehicles)

                # 4. Proportional & Fuzzy Timing Computation
                axis_timer += 1.0 / fps_in
                if axis_timer >= 12.0:
                    axis_timer = 0.0
                    current_axis = "NS" if current_axis == "EW" else "EW"

                fuzzy_decision = fuzzy_bridge.decide_phase_timing(current_axis, lane_results)

                # 5. Visual Annotations
                lane_analyzer.draw_zone_overlays(frame, lane_results)

                # Draw Vehicle Bounding Boxes & Trajectories
                for tv in tracked_vehicles:
                    x1, y1, x2, y2 = tv.bbox
                    color = CLASS_COLORS.get(tv.class_name, (255, 255, 255))
                    
                    # Bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Badge label
                    badge = f"#{tv.track_id} {tv.class_name.upper()} {tv.confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    cv2.rectangle(frame, (x1, y1 - 18), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(frame, badge, (x1 + 3, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 10, 10), 1, cv2.LINE_AA)

                    # Trajectory tail
                    if len(tv.trajectory) > 1:
                        pts = np.array(tv.trajectory, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

                # 6. Render HUD
                elapsed = time.time() - start_time
                live_fps = frame_count / elapsed if elapsed > 0 else 30.0
                lifetime_cnt = len(lane_analyzer.total_counted_ids)
                draw_hud(frame, live_fps, lane_results, len(tracked_vehicles), lifetime_cnt, fuzzy_decision)

                if out_writer is not None:
                    out_writer.write(frame)

            # Display Window
            if not headless:
                cv2.imshow(cv2_window_name, frame)
                key = cv2.waitKey(10 if not paused else 50) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord(' '):
                    paused = not paused
                elif key == ord('s'):
                    snap_path = os.path.join(PROJECT_ROOT, "data", f"snapshot_{int(time.time())}.jpg")
                    cv2.imwrite(snap_path, frame)
                    print(f"Snapshot saved to: {snap_path}")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
        if not headless:
            cv2.destroyAllWindows()
        print("Traffic monitor session concluded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 AI Traffic Monitor")
    default_vid = os.path.join(PROJECT_ROOT, "data", "sample_videos", "demo_intersection.mp4")
    parser.add_argument("--source", type=str, default=default_vid, help="Path to video file or camera index (e.g. 0)")
    parser.add_argument("--save", type=str, default=None, help="Output path for annotated video")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run without graphical window")
    parser.add_argument("--no-loop", dest="loop", action="store_false", help="Do not loop video")
    parser.set_defaults(loop=True)

    args = parser.parse_args()
    run_traffic_monitor(source=args.source, save_path=args.save, conf_thresh=args.conf, headless=args.headless, loop=args.loop)
