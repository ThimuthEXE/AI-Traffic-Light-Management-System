@echo off
title AI Traffic Light - Web Dashboard Launcher
cd /d "%~dp0.."
echo ===================================================================
echo Starting AI Traffic Light Backend and Web Dashboard...
echo ===================================================================

netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if %errorlevel% equ 0 (
    echo [INFO] Backend is already running on port 8000.
) else (
    echo [1/2] Starting FastAPI Backend on port 8000...
    start "AI Traffic Backend Server (Port 8000)" cmd /k "python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload"
    echo [2/2] Waiting for server to initialize...
    timeout /t 3 /nobreak >nul
)

echo Opening Web Dashboard in default browser...
start "" http://localhost:8000/dashboard/
