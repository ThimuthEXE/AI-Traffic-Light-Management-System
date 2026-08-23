"""
Intersection Video Exporter
Generates synthetic traffic intersection video files (.mp4) using the simulation engine.
These videos can be used to benchmark and evaluate YOLO vehicle detection and PCU tracking.
"""

import sys
import os
import random
import cv2
import numpy as np
import pygame

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_engine.simulation.vehicle import Vehicle
from ai_engine.simulation.traffic_signal import TrafficSignalManager, SignalPhase
from ai_engine.simulation.controllers import AIFuzzyController

def generate_sample_traffic_video(output_path: str, duration_sec: int = 15, fps: int = 30):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    
    width, height = 1280, 720
    surface = pygame.Surface((width, height))
    font = pygame.font.SysFont("Segoe UI", 15, bold=True)

    signals = TrafficSignalManager()
    controller = AIFuzzyController()
    vehicles = []
    vehicle_id = 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    dt = 1.0 / fps
    
    # Asymmetrical spawn intervals (e.g. dense North/South, lighter East/West)
    approach_intervals = {'N': 1.0, 'S': 1.4, 'E': 3.5, 'W': 4.0}
    approach_timers = {'N': 0.0, 'S': 0.0, 'E': 0.0, 'W': 0.0}

    print(f"Generating {duration_sec}s traffic video ({total_frames} frames) to: {output_path}")

    def calc_next(next_axis):
        return controller.get_next_green_duration(next_axis, vehicles, approach_intervals)

    for frame_idx in range(total_frames):
        # Spawning logic
        for d in ['N', 'S', 'E', 'W']:
            approach_timers[d] += dt
            if approach_timers[d] >= approach_intervals[d]:
                approach_timers[d] = 0.0
                v_types = ["car", "van", "bus", "truck", "motorcycle"]
                v_type = random.choices(v_types, weights=[0.60, 0.12, 0.10, 0.06, 0.12])[0]
                lane_idx = random.choice([0, 1])
                new_v = Vehicle(vehicle_id, v_type, d, lane_idx)
                vehicles.append(new_v)
                vehicle_id += 1

        # Signal update
        signals.update(dt, next_green_duration_calc_fn=calc_next)

        # Vehicle updates
        vehicles.sort(key=lambda v: (
            v.x if v.direction == 'E' else -v.x if v.direction == 'W' else
            v.y if v.direction == 'S' else -v.y
        ), reverse=True)

        lane_buckets = {}
        for v in vehicles:
            key = (v.direction, v.lane_idx)
            if key not in lane_buckets:
                lane_buckets[key] = []
            lane_buckets[key].append(v)

        for key, lane_v_list in lane_buckets.items():
            for i, v in enumerate(lane_v_list):
                leading_v = lane_v_list[i - 1] if i > 0 else None
                sig_state = signals.get_signal_state(v.direction)
                v.update(dt, leading_v, sig_state)

        for v in list(vehicles):
            if v.is_off_screen(width, height):
                vehicles.remove(v)

        # Drawing
        surface.fill((46, 125, 50))
        pygame.draw.rect(surface, (40, 44, 52), (0, 280, width, 160))
        pygame.draw.rect(surface, (40, 44, 52), (560, 0, 160, height))

        # Lane markings
        pygame.draw.line(surface, (241, 196, 15), (0, 360), (550, 360), 3)
        pygame.draw.line(surface, (241, 196, 15), (730, 360), (width, 360), 3)
        pygame.draw.line(surface, (241, 196, 15), (640, 0), (640, 270), 3)
        pygame.draw.line(surface, (241, 196, 15), (640, 450), (640, height), 3)

        # Stop Lines
        pygame.draw.line(surface, (255, 255, 255), (555, 360), (555, 440), 5)
        pygame.draw.line(surface, (255, 255, 255), (725, 280), (725, 360), 5)
        pygame.draw.line(surface, (255, 255, 255), (560, 275), (640, 275), 5)
        pygame.draw.line(surface, (255, 255, 255), (640, 445), (720, 445), 5)

        for v in vehicles:
            v.draw(surface, time_ticks=int(frame_idx * dt * 1000))

        signals.draw(surface, font)

        # Convert Pygame surface to OpenCV frame
        view = pygame.surfarray.array3d(surface)
        view = view.transpose([1, 0, 2])
        bgr_frame = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

        out.write(bgr_frame)

    out.release()
    pygame.quit()
    print("Video generation complete!")

if __name__ == "__main__":
    out_file = os.path.join(PROJECT_ROOT, "data", "sample_videos", "demo_intersection.mp4")
    generate_sample_traffic_video(out_file, duration_sec=15, fps=30)
