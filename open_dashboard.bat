@echo off
title AI Traffic Light - Web Dashboard Launcher
echo Starting AI Traffic Light Backend and Dashboard...
start "" http://localhost:8000/dashboard
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
pause
