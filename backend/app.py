"""
FastAPI Server Application with Integrated Web Dashboard
Provides REST API endpoints, Real-Time WebSockets, and hosts the Interactive Web Dashboard.
AI-Based Traffic Light Management System
"""

import sys
import os
import time
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.config import CORS_ORIGINS, HOST, PORT, DEBUG
from backend.live_state_manager import state_manager
from backend.db_service import db_service
from backend.routes import intersections, predictions, analytics

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# Background ticker task to advance intersection clock and broadcast WebSockets
async def background_telemetry_loop():
    print("[SERVER] Starting real-time telemetry broadcast loop (10 Hz)...")
    last_t = time.time()
    while True:
        now = time.time()
        dt = now - last_t
        last_t = now

        switched, completed_p, new_p, allocated = state_manager.signals.update(
            dt, next_decision_calc_fn=lambda axis: state_manager.current_controller.get_next_phase_decision(
                axis, [], state_manager.approach_intervals
            )
        )

        # Log completed cycles
        if switched and completed_p and "GREEN" in str(completed_p):
            axis = 'EW' if ('EW' in str(completed_p) or 'E_' in str(completed_p) or 'W_' in str(completed_p)) else 'NS'
            decision = getattr(state_manager.current_controller, "last_decision", {})
            db_service.save_cycle_log({
                "phase_axis": axis,
                "allocated_green": state_manager.signals.allocated_green,
                "pcu_demand": decision.get("target_pcu", 12.0),
                "vehicles_cleared": 6,
                "max_wait_time": decision.get("max_wait", 0.0),
                "rules": decision.get("rule_activations", {})
            })

        # Broadcast live snapshot to connected WebSocket dashboards
        snapshot = state_manager.get_snapshot()
        await state_manager.broadcast_state(snapshot)

        await asyncio.sleep(0.1)  # 10 Hz update rate


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(background_telemetry_loop())
    yield
    task.cancel()


app = FastAPI(
    title="AI Traffic Light Management System API",
    version="2.0.0",
    description="Real-time REST & WebSocket API for AI Traffic Light Management System",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Sub-Routers
app.include_router(intersections.router)
app.include_router(predictions.router)
app.include_router(analytics.router)

# Mount Frontend Dashboard Static Files
if os.path.exists(FRONTEND_DIR):
    app.mount("/dashboard", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


@app.get("/")
def root_redirect():
    """Redirects visitors directly to the interactive web dashboard."""
    return RedirectResponse(url="/dashboard")

@app.get("/api")
def api_info():
    return {
        "system": "AI-Based Traffic Light Management System",
        "version": "2.0.0",
        "api_docs": "http://localhost:8000/docs",
        "dashboard": "http://localhost:8000/dashboard",
        "websocket_endpoint": "ws://localhost:8000/ws/live-traffic",
        "status": "ONLINE"
    }

@app.get("/api/status")
def get_system_status():
    return {
        "status": "HEALTHY",
        "active_controller": state_manager.current_controller.mode_name,
        "database_mode": "MongoDB" if db_service.use_mongo else "Local JSON Fallback",
        "connected_ws_clients": len(state_manager.active_connections),
        "timestamp": time.time()
    }

@app.websocket("/ws/live-traffic")
async def websocket_traffic_endpoint(websocket: WebSocket):
    await state_manager.connect_ws(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        state_manager.disconnect_ws(websocket)


if __name__ == "__main__":
    import uvicorn
    print(f"Starting API Server & Dashboard on http://{HOST}:{PORT}/dashboard...")
    uvicorn.run("backend.app:app", host=HOST, port=PORT, reload=DEBUG)
