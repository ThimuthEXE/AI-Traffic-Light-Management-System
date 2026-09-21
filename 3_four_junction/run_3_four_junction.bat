@echo off
title AI Traffic Light — 4-Junction Grid Simulation
cd /d "%~dp0"

echo ============================================================
echo   AI Traffic Light Management System — 4-Junction 2x2 Grid
echo   General Sir John Kotelawala Defence University
echo   IT 3182 Essentials of AI — Group 06
echo ============================================================
echo.

python run_four_junction_simulation.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Simulation exited with error code %ERRORLEVEL%.
    pause
)
