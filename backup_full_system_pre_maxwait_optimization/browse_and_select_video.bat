@echo off
title YOLOv8 AI Traffic Monitor - Video File Picker
cd /d "%~dp0"
echo Opening Windows File Browser...
python -c "import tkinter as tk, os, subprocess; from tkinter import filedialog; root=tk.Tk(); root.withdraw(); f=filedialog.askopenfilename(title='Select Traffic Video', filetypes=[('Video Files', '*.mp4 *.avi *.mov *.mkv')]); root.destroy(); subprocess.run(['python', 'ai_engine/models/run_yolo_traffic_monitor.py', f]) if f else print('No file selected.')"
pause
