@echo off
title AI Traffic Light - YOLOv8 Computer Vision Monitor
echo Starting YOLOv8 Traffic Detection & PCU Monitor...
python "ai_engine\models\run_yolo_traffic_monitor.py" --source "data\sample_videos\demo_intersection.mp4"
pause
