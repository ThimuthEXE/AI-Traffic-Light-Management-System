@echo off
title AI 4-Junction Traffic Simulation
cd /d "%~dp0"

echo ============================================================
echo   AI 4-Junction Traffic Light Management System
echo ============================================================
echo.

python -m ai_engine.multi_junction.run_four_junction_simulation
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Simulation exited with an error. Please check the output above.
    pause
)
