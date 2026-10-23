"""
AITraffic Unified CLI & Interactive Management Console
"""

import sys
import os
import argparse
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BANNER = r"""
================================================================================
   🚦 AITraffic v2.0 — Autonomous Traffic Management & Telemetry Platform
================================================================================
"""

MENU = """
  [1] Run 4-Junction Grid Simulation (Synchronized 2x2 Network)
  [2] Run 2-Junction Corridor Simulation (Arterial Green Wave)
  [3] Run Single-Junction Simulation (4-Way Intersection)
  [4] Launch Web Dashboard & FastAPI Telemetry (Port 8000)
  [5] Launch YOLOv8 Video Traffic Monitor
  [6] Run Scientific Performance Benchmark Suite
  [7] Run Automated Regression & Verification Tests
  [0] Exit
================================================================================
"""

def run_script(rel_path: str):
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(full_path):
        print(f"[ERROR] Target script not found: {full_path}")
        return
    print(f"\n[AITRAFFIC] Launching: {rel_path} ...\n")
    subprocess.run([sys.executable, "-u", full_path], cwd=PROJECT_ROOT)

def run_dashboard():
    print("\n[AITRAFFIC] Starting FastAPI Telemetry Server on http://localhost:8000/dashboard/ ...\n")
    import webbrowser
    import threading
    import time

    def _open_delayed():
        time.sleep(2.5)
        webbrowser.open("http://localhost:8000/dashboard/")

    threading.Thread(target=_open_delayed, daemon=True).start()
    subprocess.run([sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"], cwd=PROJECT_ROOT)

def interactive_menu():
    while True:
        print(BANNER)
        print(MENU)
        choice = input("Select an option (0-7): ").strip()
        if choice == "1":
            run_script("3_four_junction/run_four_junction_simulation.py")
        elif choice == "2":
            run_script("2_two_junction/run_two_junction_simulation.py")
        elif choice == "3":
            run_script("1_single_junction/run_single_junction_simulation.py")
        elif choice == "4":
            run_dashboard()
        elif choice == "5":
            run_script("ai_engine/models/run_yolo_traffic_monitor.py")
        elif choice == "6":
            print("\nSelect Benchmark Target:")
            print("  [1] Single Junction Benchmark")
            print("  [2] 2-Junction Corridor Benchmark")
            print("  [3] 4-Junction Grid Benchmark")
            b_choice = input("Choice (1-3): ").strip()
            if b_choice == "1":
                run_script("1_single_junction/run_single_junction_benchmark.py")
            elif b_choice == "2":
                run_script("2_two_junction/run_two_junction_benchmark.py")
            elif b_choice == "3":
                run_script("3_four_junction/run_four_junction_benchmark.py")
        elif choice == "7":
            run_script("tests/test_dashboard.py")
        elif choice == "0":
            print("\nExiting AITraffic. Goodbye!\n")
            sys.exit(0)
        else:
            print("\n[WARNING] Invalid selection. Please choose an option between 0 and 7.\n")
        input("\nPress Enter to return to main menu...")
        os.system("cls" if os.name == "nt" else "clear")

def main():
    parser = argparse.ArgumentParser(description="AITraffic Management Platform CLI")
    parser.add_argument("command", nargs="?", choices=["sim", "dashboard", "vision", "test", "benchmark"], help="Command to run directly")
    parser.add_argument("target", nargs="?", help="Target option for command (e.g. 1, 2, 4 for sim/benchmark)")

    args = parser.parse_args()

    if not args.command:
        interactive_menu()
        return

    if args.command == "sim":
        target = args.target or "4"
        if target == "1":
            run_script("1_single_junction/run_single_junction_simulation.py")
        elif target == "2":
            run_script("2_two_junction/run_two_junction_simulation.py")
        elif target == "4":
            run_script("3_four_junction/run_four_junction_simulation.py")
        else:
            print(f"[ERROR] Unknown simulation target: {target}. Use 1, 2, or 4.")
    elif args.command == "dashboard":
        run_dashboard()
    elif args.command == "vision":
        run_script("ai_engine/models/run_yolo_traffic_monitor.py")
    elif args.command == "test":
        run_script("tests/test_dashboard.py")
    elif args.command == "benchmark":
        target = args.target or "4"
        if target == "1":
            run_script("1_single_junction/run_single_junction_benchmark.py")
        elif target == "2":
            run_script("2_two_junction/run_two_junction_benchmark.py")
        elif target == "4":
            run_script("3_four_junction/run_four_junction_benchmark.py")
        else:
            print(f"[ERROR] Unknown benchmark target: {target}. Use 1, 2, or 4.")

if __name__ == "__main__":
    main()
