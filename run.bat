@echo off
title AITraffic - Autonomous Traffic Light Management Platform
cd /d "%~dp0"

:: Launch Master Interactive Management Console
python main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Process exited with code %ERRORLEVEL%.
    pause
)
