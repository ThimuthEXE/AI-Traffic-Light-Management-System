"""
YOLOv8 Traffic Video Monitor & Real-Time PCU Analytics Visualizer
Supports Portrait (9:16 phone / vertical CCTV) and Landscape (16:9) aspect ratios
with Automatic Letterboxing, Aspect-Ratio Preservation, and Side-Panel Telemetry HUDs.
General Sir John Kotelawala Defence University (KDU) - IT 3182 Essentials of AI
"""

import sys
import os
import time
import argparse
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_engine.models.yolo_detector import YOLOVehicleDetector, CLASS_COLORS
from ai_engine.models.vehicle_tracker import MultiVehicleTracker
from ai_engine.models.lane_counter import LaneTrafficAnalyzer
from ai_engine.models.vision_fuzzy_bridge import VisionFuzzyBridge


def draw_hud_overlay(canvas: np.ndarray, fps: float, lane_results: dict, total_active: int, lifetime_counted: int, fuzzy_decision: dict, source_name: str, mode: str, approach_dir: str, is_portrait: bool, x_offset: int, vid_w: int):
    """Draws sleek analytics telemetry HUDs adapted for both Portrait side-panels and Landscape overlays."""
    h, w, _ = canvas.shape
    dir_names = {"N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST"}
    dir_str = dir_names.get(approach_dir.upper(), "NORTH")
    mode_title = f"SINGLE-APPROACH [{dir_str}BOUND]" if mode == "single" else "4-WAY INTERSECTION"

    total_road_pcu = sum(data.get("pcu_load", 0.0) for data in lane_results.values())
    tgt_axis = fuzzy_decision.get("target_axis", "NS" if approach_dir in ["N", "S"] else "EW")
    g_time = fuzzy_decision.get("green_duration", 22.0)
    tgt_ratio = fuzzy_decision.get("target_ratio", 50.0)

    if is_portrait:
        # === PORTRAIT CONSOLE MODE: Left and Right Side-Panel Dashboards ===
        left_panel_w = max(340, x_offset - 25)
        right_panel_x = x_offset + vid_w + 15
        right_panel_w = w - right_panel_x - 15

        # 1. Left Telemetry Card
        cv2.rectangle(canvas, (15, 15), (15 + left_panel_w, h - 15), (20, 24, 30), -1)
        cv2.rectangle(canvas, (15, 15), (15 + left_panel_w, h - 15), (70, 80, 95), 1)

        cv2.putText(canvas, "AI TRAFFIC MONITOR", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 115), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Mode: {mode_title}", (25, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"FPS: {fps:.1f} | Active: {total_active} veh", (25, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Total Counted: {lifetime_counted} veh", (25, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        
        src_disp = (source_name[:24] + "...") if len(source_name) > 26 else source_name
        cv2.putText(canvas, f"File: {src_disp}", (25, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140, 160, 180), 1, cv2.LINE_AA)
        cv2.line(canvas, (25, 136), (15 + left_panel_w - 10, 136), (60, 70, 80), 1)

        # Lane Breakdown
        cv2.putText(canvas, "LANE DENSITY BREAKDOWN", (25, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        y = 188
        for zone_name, data in lane_results.items():
            cnt = data.get("vehicle_count", 0)
            pcu = data.get("pcu_load", 0.0)
            density = data.get("density_level", "LOW")
            d_color = (0, 230, 115) if density == "LOW" else (0, 215, 255) if density == "MEDIUM" else (0, 100, 255)
            
            z_short = zone_name.replace("North Approach - ", "").replace("South Approach - ", "")
            cv2.putText(canvas, f"{z_short}: {cnt}v ({pcu:.1f}p)", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"[{density}]", (15 + left_panel_w - 65, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, d_color, 1, cv2.LINE_AA)
            y += 26

        # Keybindings footer on left
        cv2.line(canvas, (25, h - 85), (15 + left_panel_w - 10, h - 85), (60, 70, 80), 1)
        cv2.putText(canvas, "[M] Mode  [A] Direction", (25, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 160, 190), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[SPACE] Pause  [O] Open", (25, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 160, 190), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[S] Snapshot  [Q] Exit", (25, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 160, 190), 1, cv2.LINE_AA)

        # 2. Right Decision Card
        if right_panel_w > 100:
            cv2.rectangle(canvas, (right_panel_x, 15), (w - 15, 230), (20, 24, 30), -1)
            cv2.rectangle(canvas, (right_panel_x, 15), (w - 15, 230), (0, 230, 115), 1)

            cv2.putText(canvas, "PROPORTIONAL TIMING", (right_panel_x + 15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 115), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Total Road Load: {total_road_pcu:.1f} PCU", (right_panel_x + 15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
            
            cv2.putText(canvas, "Allocated Green:", (right_panel_x + 15, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{g_time:.1f} Seconds", (right_panel_x + 15, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(canvas, f"Demand Share: {tgt_ratio}%", (right_panel_x + 15, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Mamdani + Webster Closed Loop", (right_panel_x + 15, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 230, 115), 1, cv2.LINE_AA)

    else:
        # === LANDSCAPE OVERLAY MODE ===
        hud_bg = canvas.copy()
        cv2.rectangle(hud_bg, (15, 15), (410, 240), (20, 24, 30), -1)
        cv2.addWeighted(hud_bg, 0.88, canvas, 0.12, 0, canvas)
        cv2.rectangle(canvas, (15, 15), (410, 240), (70, 80, 95), 1)

        cv2.putText(canvas, "AI TRAFFIC VISION MONITOR (YOLOv8)", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 115), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Mode: {mode_title} | FPS: {fps:.1f}", (25, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Active: {total_active} veh | Total Cleared: {lifetime_counted}", (25, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        
        src_disp = (source_name[:32] + "...") if len(source_name) > 35 else source_name
        cv2.putText(canvas, f"Source: {src_disp}", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 170, 190), 1, cv2.LINE_AA)
        cv2.line(canvas, (25, 98), (400, 98), (60, 70, 80), 1)

        y = 118
        for zone_name, data in lane_results.items():
            cnt = data.get("vehicle_count", 0)
            pcu = data.get("pcu_load", 0.0)
            density = data.get("density_level", "LOW")
            d_color = (0, 230, 115) if density == "LOW" else (0, 215, 255) if density == "MEDIUM" else (0, 100, 255)
            cv2.putText(canvas, f"{zone_name}: {cnt}v ({pcu:.1f}p)", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"[{density}]", (315, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, d_color, 1, cv2.LINE_AA)
            y += 22

        cv2.putText(canvas, "[M] Mode  [A] Direction  [SPACE] Pause  [O] Open  [Q] Exit", (25, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 150, 180), 1, cv2.LINE_AA)

        # Right Card
        ai_bg = canvas.copy()
        cv2.rectangle(ai_bg, (w - 395, 15), (w - 15, 175), (20, 24, 30), -1)
        cv2.addWeighted(ai_bg, 0.88, canvas, 0.12, 0, canvas)
        cv2.rectangle(canvas, (w - 395, 15), (w - 15, 175), (0, 230, 115), 1)

        cv2.putText(canvas, "FLOW PROPORTIONAL SIGNAL TIMING", (w - 385, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 115), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Approach Demand: {total_road_pcu:.1f} Total PCU", (w - 385, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Allocated Green Countdown: {g_time:.1f}s", (w - 385, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Phase: {tgt_axis} -> {tgt_ratio}% Demand Share", (w - 385, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Mode: Vision PCU + Fuzzy Demand", (w - 385, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 115), 1, cv2.LINE_AA)


def run_traffic_monitor(source: str, save_path: str = None, conf_thresh: float = 0.25, mode: str = "auto", approach_dir: str = "N", headless: bool = False, loop: bool = True):
    source_clean = source.strip('"').strip("'")
    print("=" * 65)
    print("INITIALIZING YOLOv8 TRAFFIC MONITOR")
    print(f"Input Video Source: {source_clean}")
    print("=" * 65)
    
    if not os.path.exists(source_clean) and not source_clean.isdigit():
        print(f"Error: Video file not found at: {source_clean}")
        return

    is_cam = source_clean.isdigit()
    cap = cv2.VideoCapture(int(source_clean) if is_cam else source_clean)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source_clean}")
        return

    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    is_portrait = raw_h > raw_w
    print(f"Detected Aspect Ratio: {'PORTRAIT (Vertical 9:16)' if is_portrait else 'LANDSCAPE (16:9)'} ({raw_w}x{raw_h})")

    # Standard Widescreen Display Target
    target_canvas_w = 1280
    target_canvas_h = 720

    current_mode = mode
    if current_mode == "auto":
        current_mode = "4way" if "demo_intersection" in source_clean else "single"

    current_dir = approach_dir.upper()

    detector = YOLOVehicleDetector(conf_thresh=conf_thresh)
    tracker = MultiVehicleTracker(max_disappeared=25, max_distance=75.0)
    lane_analyzer = LaneTrafficAnalyzer(mode=current_mode, width=raw_w, height=raw_h, direction=current_dir)
    fuzzy_bridge = VisionFuzzyBridge()

    out_writer = None
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(save_path, fourcc, fps_in, (target_canvas_w, target_canvas_h))
        print(f"Annotated video will be saved to: {save_path}")

    frame_count = 0
    start_time = time.time()
    paused = False
    source_basename = os.path.basename(source_clean) if not is_cam else f"Camera #{source_clean}"
    cv2_window_name = f"AI Traffic Monitor (YOLOv8) - [{source_basename}]"
    
    if not headless:
        cv2.namedWindow(cv2_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cv2_window_name, target_canvas_w, target_canvas_h)

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

                # 1. Detection & Tracking on native frame
                detections = detector.detect_vehicles(frame)
                tracked_vehicles = tracker.update(detections)

                # 2. Lane PCU Analysis
                lane_results = lane_analyzer.analyze_lanes(tracked_vehicles)

                # 3. Fuzzy Decision
                target_axis = "NS" if current_dir in ["N", "S"] else "EW"
                fuzzy_decision = fuzzy_bridge.decide_phase_timing(target_axis, lane_results)

                # 4. Draw overlays on raw frame
                lane_analyzer.draw_zone_overlays(frame, lane_results)

                for tv in tracked_vehicles:
                    x1, y1, x2, y2 = tv.bbox
                    color = CLASS_COLORS.get(tv.class_name, (255, 255, 255))
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    badge = f"#{tv.track_id} {tv.class_name.upper()} {tv.confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    cv2.rectangle(frame, (x1, y1 - 18), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(frame, badge, (x1 + 3, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 10, 10), 1, cv2.LINE_AA)

                    if len(tv.trajectory) > 1:
                        pts = np.array(tv.trajectory, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

                # 5. Composite Frame onto 1280x720 Canvas preserving true Aspect Ratio
                canvas = np.zeros((target_canvas_h, target_canvas_w, 3), dtype=np.uint8)
                canvas[:] = (18, 22, 28)  # Sleek dark background

                if is_portrait:
                    scale = target_canvas_h / float(raw_h)
                    scaled_w = int(raw_w * scale)
                    scaled_h = target_canvas_h
                    resized_video = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

                    x_offset = (target_canvas_w - scaled_w) // 2
                    canvas[:, x_offset:x_offset + scaled_w] = resized_video
                    cv2.rectangle(canvas, (x_offset, 0), (x_offset + scaled_w, scaled_h), (60, 75, 95), 2)
                else:
                    if raw_w != target_canvas_w or raw_h != target_canvas_h:
                        resized_video = cv2.resize(frame, (target_canvas_w, target_canvas_h), interpolation=cv2.INTER_AREA)
                    else:
                        resized_video = frame
                    canvas = resized_video
                    x_offset = 0
                    scaled_w = target_canvas_w

                # 6. Render Smart Adaptive HUD
                elapsed = time.time() - start_time
                live_fps = frame_count / elapsed if elapsed > 0 else 30.0
                lifetime_cnt = len(lane_analyzer.total_counted_ids)

                draw_hud_overlay(
                    canvas, live_fps, lane_results, len(tracked_vehicles), lifetime_cnt, 
                    fuzzy_decision, source_name=source_basename, mode=current_mode, 
                    approach_dir=current_dir, is_portrait=is_portrait, x_offset=x_offset, vid_w=scaled_w if is_portrait else target_canvas_w
                )

                if out_writer is not None:
                    out_writer.write(canvas)

            # Display Window
            if not headless:
                cv2.imshow(cv2_window_name, canvas)
                key = cv2.waitKey(10 if not paused else 50) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord(' '):
                    paused = not paused
                elif key == ord('s'):
                    snap_path = os.path.join(PROJECT_ROOT, "data", f"snapshot_{int(time.time())}.jpg")
                    cv2.imwrite(snap_path, canvas)
                    print(f"Snapshot saved to: {snap_path}")
                elif key == ord('m'):
                    current_mode = "single" if current_mode == "4way" else "4way"
                    lane_analyzer.set_mode(current_mode, width=raw_w, height=raw_h, direction=current_dir)
                    print(f"Switched surveillance mode to: {current_mode.upper()}")
                elif key == ord('a'):
                    dir_order = ["N", "E", "S", "W"]
                    next_idx = (dir_order.index(current_dir) + 1) % len(dir_order)
                    current_dir = dir_order[next_idx]
                    lane_analyzer.set_mode(current_mode, width=raw_w, height=raw_h, direction=current_dir)
                    print(f"Switched approach direction to: {current_dir}")
                elif key == ord('o'):
                    try:
                        import tkinter as tk
                        from tkinter import filedialog
                        root = tk.Tk()
                        root.withdraw()
                        selected_file = filedialog.askopenfilename(
                            title="Select Traffic Video File",
                            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
                        )
                        root.destroy()
                        if selected_file and os.path.exists(selected_file):
                            cap.release()
                            cv2.destroyAllWindows()
                            run_traffic_monitor(source=selected_file, save_path=save_path, conf_thresh=conf_thresh, 
                                                mode=mode, approach_dir=approach_dir, headless=headless, loop=loop)
                            return
                    except Exception as e:
                        print(f"File picker error: {e}")

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
    
    parser.add_argument("input_file", nargs="?", default=None, help="Path to video file (positional argument)")
    parser.add_argument("--source", type=str, default=None, help="Path to video file or camera index (e.g. 0)")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "single", "4way"], help="Surveillance mode: single or 4way")
    parser.add_argument("--approach", type=str, default="N", choices=["N", "S", "E", "W"], help="Approach direction for single mode")
    parser.add_argument("--save", type=str, default=None, help="Output path for annotated video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run without graphical window")
    parser.add_argument("--no-loop", dest="loop", action="store_false", help="Do not loop video")
    parser.set_defaults(loop=True)

    args = parser.parse_args()
    chosen_source = args.input_file or args.source or default_vid
    
    run_traffic_monitor(source=chosen_source, save_path=args.save, conf_thresh=args.conf, 
                        mode=args.mode, approach_dir=args.approach, headless=args.headless, loop=args.loop)
