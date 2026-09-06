@echo off
title AI Traffic Light System - 2 Connected Junctions
cd /d "%~dp0"
echo ===============================================================================
echo   AI TRAFFIC LIGHT MANAGEMENT SYSTEM - 2 CONNECTED JUNCTIONS
echo   KDU IT 3182 Essentials of AI - Group 06
echo ===============================================================================
echo.
echo Starting 2-Junction Simulation (Junction 1: West, Junction 2: East)...
python ai_engine\multi_junction\run_two_junction_simulation.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Simulation exited with error code %ERRORLEVEL%.
)
pause
