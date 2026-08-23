@echo off
title AI Traffic Light - FastAPI REST & WebSocket Backend
echo Starting FastAPI & WebSockets Server on port 8000...
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
pause
