@echo off
title AI Traffic Light - Full System Launcher
echo ===================================================================
echo Starting AI Traffic Light Management System (Backend + Web + Pygame)
echo General Sir John Kotelawala Defence University (KDU) - IT 3182
echo ===================================================================

echo [1/3] Starting FastAPI Backend & WebSockets Server...
start "AI Traffic Backend Server" cmd /k "python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload"

echo [2/3] Opening Web Dashboard in Default Browser...
timeout /t 2 /nobreak >nul
start "" http://localhost:8000/dashboard

echo [3/3] Launching Pygame 2D Traffic Simulation...
python "ai_engine\simulation\run_simulation.py"

pause
