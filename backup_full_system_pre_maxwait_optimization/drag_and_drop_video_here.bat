@echo off
title YOLOv8 AI Traffic Monitor - Custom Video Loader
cd /d "%~dp0"

if "%~1"=="" (
    echo ====================================================================
    echo No video file dropped. Opening Windows File Explorer to pick video...
    echo ====================================================================
    python -c "import tkinter as tk, os, subprocess; from tkinter import filedialog; root=tk.Tk(); root.withdraw(); f=filedialog.askopenfilename(title='Select Traffic Video', filetypes=[('Video Files', '*.mp4 *.avi *.mov *.mkv')]); root.destroy(); subprocess.run(['python', 'ai_engine/models/run_yolo_traffic_monitor.py', f]) if f else print('No file selected.')"
) else (
    echo ====================================================================
    echo Loading Video: %1
    echo ====================================================================
    python "ai_engine\models\run_yolo_traffic_monitor.py" %1
)
pause
