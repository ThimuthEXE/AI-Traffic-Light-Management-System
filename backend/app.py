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

        is_sim_active = (now - getattr(state_manager, "last_sim_sync_time", 0.0)) < 2.5

        if not is_sim_active:
            switched, completed_p, new_p, allocated = state_manager.signals.update(
                dt, next_decision_calc_fn=lambda axis: state_manager.current_controller.get_next_phase_decision(
                    axis, [], state_manager.approach_intervals
                )
            )

            # Log completed cycles with realistic dynamic traffic metrics
            if switched and completed_p and "GREEN" in str(completed_p):
                axis = 'EW' if ('EW' in str(completed_p) or 'E_' in str(completed_p) or 'W_' in str(completed_p)) else 'NS'
                decision = getattr(state_manager.current_controller, "last_decision", {})
                alloc_g = float(state_manager.signals.allocated_green or 20.0)
                
                # Derive realistic vehicle clearance and waiting metrics from approach intervals
                d1, d2 = (axis[0], axis[1])
                inv1 = state_manager.approach_intervals.get(d1, 2.0)
                inv2 = state_manager.approach_intervals.get(d2, 2.0)
                flow_vpm = 60.0 / max(0.4, (inv1 + inv2) / 2.0)
                
                # Saturated clearance during green: ~0.8-1.1 veh/sec for 2 active lanes
                cleared_veh = max(4, min(36, int(round((alloc_g * 0.85) * (flow_vpm / 45.0)))))
                pcu_dem = round(cleared_veh * 1.25, 1)
                max_w = round(11.0 + (flow_vpm * 0.14) + (alloc_g * 0.18), 1)

                db_service.save_cycle_log({
                    "phase_axis": axis,
                    "allocated_green": round(alloc_g, 1),
                    "pcu_demand": pcu_dem,
                    "vehicles_cleared": cleared_veh,
                    "max_wait_time": max_w,
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

# Dashboard Redirection & Static Asset Fallback Handlers
@app.get("/")
def root_redirect():
    """Redirects visitors directly to the interactive web dashboard."""
    return RedirectResponse(url="/dashboard/", status_code=307)


@app.get("/dashboard")
def dashboard_redirect():
    """Redirects /dashboard to /dashboard/ to ensure relative assets resolve properly."""
    return RedirectResponse(url="/dashboard/", status_code=307)


@app.get("/styles.css")
def get_root_styles():
    return FileResponse(os.path.join(FRONTEND_DIR, "styles.css"), media_type="text/css")


@app.get("/app.js")
def get_root_app_js():
    return FileResponse(os.path.join(FRONTEND_DIR, "app.js"), media_type="application/javascript")


# Mount Frontend Dashboard Static Files
if os.path.exists(FRONTEND_DIR):
    app.mount("/dashboard", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


@app.get("/api")
def api_info():
    return {
        "system": "AI-Based Traffic Light Management System",
        "version": "2.0.0",
        "api_docs": "http://localhost:8000/docs",
        "dashboard": "http://localhost:8000/dashboard/",
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
