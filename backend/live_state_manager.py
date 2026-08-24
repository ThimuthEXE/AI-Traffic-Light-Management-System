"""
Live State Manager & Real-Time Telemetry Hub with Protected Dual-Movement Phasing
Synchronizes physical Pygame simulation state with Web Dashboard WebSockets at 10 Hz.
General Sir John Kotelawala Defence University (KDU) - IT 3182 Essentials of AI
"""

import time
import math
from typing import Dict, Any, Set
from fastapi import WebSocket

from ai_engine.simulation.traffic_signal import TrafficSignalManager, SignalPhase
from ai_engine.simulation.controllers import FixedTimeController, AIFuzzyController
from ai_engine.simulation.metrics_tracker import MetricsTracker

class LiveIntersectionStateManager:
    def __init__(self, intersection_id: str = "INT-KDU-01", name: str = "KDU Main Campus 4-Way Junction"):
        self.intersection_id = intersection_id
        self.name = name
        
        # Dual-Movement Protected Phasing Signal Manager
        self.signals = TrafficSignalManager(yellow_duration=2.0, all_red_duration=1.0)
        
        # Controllers
        self.fixed_controller = FixedTimeController(fixed_green=25.0)
        self.ai_controller = AIFuzzyController(min_green=8.0, max_green=45.0)
        
        self.active_mode = 2  # Default to Mode 2: AI Adaptive
        self.current_controller = self.ai_controller
        
        # Metrics
        self.metrics = MetricsTracker()
        
        # Approach spawn intervals (seconds)
        self.approach_intervals = {
            "N": 2.0,
            "S": 2.0,
            "E": 2.0,
            "W": 2.0
        }
        
        # Live synchronized snapshot from running Pygame simulation
        self.last_sim_snapshot = None
        self.last_sim_sync_time = 0.0

        # Active connected WebSockets
        self.active_connections: Set[WebSocket] = set()
        self.last_tick_time = time.time()
        self.is_running = True

    async def connect_ws(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"[WS] Client connected. Total clients: {len(self.active_connections)}")

    def disconnect_ws(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WS] Client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast_state(self, telemetry: dict):
        if not self.active_connections:
            return
        dead_sockets = set()
        for ws in self.active_connections:
            try:
                await ws.send_json(telemetry)
            except Exception:
                dead_sockets.add(ws)
        self.active_connections -= dead_sockets

    def apply_simulation_sync(self, sim_data: dict):
        """Applies real-time frame data directly from the active Pygame simulation."""
        self.last_sim_snapshot = sim_data
        self.last_sim_sync_time = time.time()
        
        if "active_mode" in sim_data:
            self.active_mode = sim_data["active_mode"]
        if "approach_intervals" in sim_data:
            self.approach_intervals = sim_data["approach_intervals"]

    def get_snapshot(self) -> dict:
        """Returns structured JSON snapshot. If Pygame is actively syncing, returns exact Pygame frame."""
        now = time.time()
        if self.last_sim_snapshot and (now - self.last_sim_sync_time) < 2.5:
            snap = dict(self.last_sim_snapshot)
            snap["timestamp"] = round(now, 2)
            snap["sync_source"] = "LIVE_PYGAME_SIMULATION"
            return snap

        active_axis = self.signals.active_green_axis
        cur_phase = str(self.signals.current_phase)
        
        if "GREEN" in cur_phase:
            rem_time = max(0.0, self.signals.allocated_green - self.signals.time_in_state)
            current_state = "GREEN"
        elif "YELLOW" in cur_phase:
            rem_time = max(0.0, self.signals.yellow_duration - self.signals.time_in_state)
            current_state = "YELLOW"
        else:
            rem_time = max(0.0, self.signals.all_red_duration - self.signals.time_in_state)
            current_state = "ALL_RED"

        signals_map = {
            "North": self.signals.get_signal_state("N"),
            "South": self.signals.get_signal_state("S"),
            "East": self.signals.get_signal_state("E"),
            "West": self.signals.get_signal_state("W")
        }

        approach_stats = {}
        for d, name in [('N', 'North'), ('S', 'South'), ('E', 'East'), ('W', 'West')]:
            rate = self.approach_intervals[d]
            flow_vpm = round(60.0 / max(0.2, rate), 1)
            density_cat = "HIGH" if rate <= 1.0 else "MEDIUM" if rate <= 2.2 else "LOW"
            approach_stats[name] = {
                "direction": d,
                "spawn_interval_sec": rate,
                "arrival_flow_vpm": flow_vpm,
                "density_category": density_cat
            }

        decision = getattr(self.ai_controller, "last_decision", {})

        return {
            "intersection_id": self.intersection_id,
            "name": self.name,
            "timestamp": round(now, 2),
            "control_mode": "AI_CYCLE_ADAPTIVE" if self.active_mode == 2 else "FIXED_TIME",
            "active_phase": {
                "phase_name": self.signals.current_phase,
                "active_axis": active_axis,
                "state": current_state,
                "allocated_green": round(self.signals.allocated_green, 1),
                "countdown_seconds": int(rem_time + 0.99),
                "elapsed_in_state": round(self.signals.time_in_state, 1)
            },
            "signals": signals_map,
            "approaches": approach_stats,
            "metrics": self.metrics.get_summary(),
            "ai_decision": decision,
            "sync_source": "SERVER_STANDALONE"
        }

    def set_control_mode(self, mode: int):
        self.active_mode = mode
        self.current_controller = self.ai_controller if mode == 2 else self.fixed_controller

    def set_approach_density(self, direction: str, interval_sec: float):
        if direction in self.approach_intervals:
            self.approach_intervals[direction] = max(0.4, min(8.0, interval_sec))


state_manager = LiveIntersectionStateManager()
