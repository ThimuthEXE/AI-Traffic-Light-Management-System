@echo off
title AI Traffic Light - Full System Launcher
cd /d "%~dp0.."
echo ===================================================================
echo Starting AI Traffic Light Management System (Backend + Web + Pygame)
echo ===================================================================

netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if %errorlevel% equ 0 (
    echo [INFO] Backend is already running on port 8000.
) else (
    echo [1/3] Starting FastAPI Backend on port 8000...
    start "AI Traffic Backend Server (Port 8000)" cmd /k "python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload"
    echo [2/3] Waiting for server to initialize...
    timeout /t 3 /nobreak >nul
)

echo Opening Web Dashboard in default browser...
start "" http://localhost:8000/dashboard/

echo [3/3] Launching Pygame 2D Traffic Simulation...
python "ai_engine\simulation\run_simulation.py"

pause
