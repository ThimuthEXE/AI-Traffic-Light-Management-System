@echo off
title Push AITraffic to GitHub
cd /d "%~dp0.."

echo =========================================================================
echo   Pushing AITraffic v2.0 and Backup Branch to GitHub
echo   Repository: https://github.com/ThimuthEXE/AI-Traffic-Light-Management-System
echo =========================================================================
echo.

echo [1/2] Pushing pre-modular-backup branch and tags to GitHub...
git push origin pre-modular-backup --tags

echo.
echo [2/2] Pushing main branch (v2.0.0 modular package) to GitHub...
git push origin main

echo.
if %ERRORLEVEL% EQU 0 (
    echo =========================================================================
    echo   [SUCCESS] GitHub repository is 100%% up to date!
    echo =========================================================================
) else (
    echo [NOTICE] Please complete the GitHub authentication prompt if displayed.
)
echo.
pause
