@echo off
title AI Traffic Light - Machine Learning Predictor
cd /d "%~dp0.."
echo Running Scikit-Learn Traffic Congestion Forecasts...
python "ai_engine\traffic_predictor\run_prediction_demo.py"
pause
