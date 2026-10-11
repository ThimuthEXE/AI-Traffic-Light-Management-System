@echo off
title AI Traffic Light — 1-Junction Simulation
cd /d "%~dp0"

echo ============================================================
echo   AI Traffic Light Management System — Single Junction
echo ============================================================
echo.

python run_single_junction_simulation.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Simulation exited with error code %ERRORLEVEL%.
    pause
)
