"""
Intersection Management, Real-Time Control & Pygame Simulation Sync Routes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from backend.live_state_manager import state_manager

router = APIRouter(prefix="/api/intersections", tags=["Intersections"])

class ControlModeRequest(BaseModel):
    mode: int
    emergency_axis: Optional[str] = None

class DensityUpdateRequest(BaseModel):
    direction: str
    spawn_interval_sec: float

class SimulationSyncRequest(BaseModel):
    intersection_id: str = "INT-KDU-01"
    active_mode: int
    control_mode_name: str
    signals: Dict[str, str]
    active_phase: Dict[str, Any]
    approaches: Dict[str, Any]
    ai_decision: Dict[str, Any]
    metrics: Dict[str, Any]
    emergency_active: bool

@router.get("")
def list_intersections():
    return [
        {
            "id": state_manager.intersection_id,
            "name": state_manager.name,
            "branches": 4,
            "lanes_per_approach": 2,
            "active_controller": state_manager.current_controller.mode_name,
            "status": "ONLINE"
        }
    ]

@router.get("/{intersection_id}/live")
def get_live_state(intersection_id: str):
    return state_manager.get_snapshot()

@router.post("/{intersection_id}/control")
def update_control_mode(intersection_id: str, req: ControlModeRequest):
    if req.emergency_axis:
        state_manager.trigger_emergency(req.emergency_axis)
        return {"status": "success", "message": f"Emergency override activated for axis {req.emergency_axis}"}
    
    if req.mode in [1, 2]:
        state_manager.set_control_mode(req.mode)
        mode_str = "AI_CYCLE_ADAPTIVE" if req.mode == 2 else "FIXED_TIME"
        return {"status": "success", "active_mode": mode_str}
    
    raise HTTPException(status_code=400, detail="Invalid mode.")

@router.post("/{intersection_id}/density")
def update_approach_density(intersection_id: str, req: DensityUpdateRequest):
    d = req.direction.upper()
    if d not in ['N', 'S', 'E', 'W']:
        raise HTTPException(status_code=400, detail="Direction must be N, S, E, or W")
    
    state_manager.set_approach_interval(d, req.spawn_interval_sec)
    return {
        "status": "success",
        "direction": d,
        "new_interval_sec": state_manager.approach_intervals[d],
        "new_arrival_flow_vpm": round(60.0 / state_manager.approach_intervals[d], 1)
    }

@router.post("/{intersection_id}/sync")
async def sync_from_simulation(intersection_id: str, req: SimulationSyncRequest):
    """Receives live 1-to-1 frames from the Pygame simulation and mirrors to the Web Dashboard."""
    state_manager.apply_simulation_sync(req.dict())
    # Broadcast immediately to all connected web dashboard WebSockets
    snapshot = state_manager.get_snapshot()
    await state_manager.broadcast_state(snapshot)
    return {
        "status": "synced",
        "active_mode": state_manager.active_mode,
        "approach_intervals": state_manager.approach_intervals
    }
