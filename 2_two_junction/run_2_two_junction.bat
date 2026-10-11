@echo off
title AI Traffic Light — 2-Junction Simulation
cd /d "%~dp0"

echo ============================================================
echo   AI Traffic Light Management System — 2 Connected Junctions
echo ============================================================
echo.

python run_two_junction_simulation.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Simulation exited with error code %ERRORLEVEL%.
    pause
)
