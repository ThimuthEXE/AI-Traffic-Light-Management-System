@echo off
title AI 4-Junction Traffic Simulation — KDU IT3182
cd /d "%~dp0"

echo ============================================================
echo   AI 4-Junction Traffic Light Management System
echo   General Sir John Kotelawala Defence University
echo   IT 3182 Essentials of AI — Group 06
echo ============================================================
echo.

python -m ai_engine.multi_junction.run_four_junction_simulation
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Simulation exited with an error. Please check the output above.
    pause
)
