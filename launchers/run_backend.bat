@echo off
title AI Traffic Light - FastAPI REST & WebSocket Backend
cd /d "%~dp0.."
echo ===================================================================
echo Starting FastAPI REST and WebSockets Backend (Port 8000)...
echo ===================================================================

netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if %errorlevel% equ 0 (
    echo [WARNING] Port 8000 is already in use by another process.
    echo If another backend instance is running, open http://localhost:8000/dashboard/
    echo.
)
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
pause
