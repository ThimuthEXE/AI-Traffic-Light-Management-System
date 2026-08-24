"""
Machine Learning Traffic Congestion Prediction - Interactive Scenario Inspector
Tests trained Scikit-Learn Random Forest models across diverse real-world situations:
- Morning Tidal Commuter Rush
- Evening Tidal Commuter Rush
- Midday & Weekend Shopping Surge
- Severe Weather Disruption
"""

import sys
import os
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_engine.traffic_predictor.predictor_service import TrafficPredictorService

def run_scenarios():
    service = TrafficPredictorService()

    scenarios = [
        {
            "name": "1. Weekday Morning Inbound Rush",
            "dt": datetime.datetime(2026, 8, 25, 8, 0),  # Tuesday 8:00 AM
            "weather": "Clear",
            "desc": "Heavy inbound commuter surge towards city center (North/East)."
        },
        {
            "name": "2. Weekday Evening Outbound Rush",
            "dt": datetime.datetime(2026, 8, 27, 17, 30),  # Thursday 5:30 PM
            "weather": "Clear",
            "desc": "Heavy outbound commuter surge returning home (South/West)."
        },
        {
            "name": "3. Morning Rush with Heavy Rain",
            "dt": datetime.datetime(2026, 8, 26, 8, 15),  # Wednesday 8:15 AM
            "weather": "Heavy Rain",
            "desc": "Extreme morning congestion compounded by severe weather delay."
        },
        {
            "name": "4. Saturday Afternoon Commercial Flow",
            "dt": datetime.datetime(2026, 8, 29, 14, 0),  # Saturday 2:00 PM
            "weather": "Clear",
            "desc": "Balanced recreational and shopping traffic across all directions."
        },
        {
            "name": "5. Late Night Off-Peak Flow",
            "dt": datetime.datetime(2026, 8, 24, 2, 30),  # Monday 2:30 AM
            "weather": "Clear",
            "desc": "Minimal nighttime vehicle activity."
        }
    ]

    print("\n" + "=" * 75)
    print("AI MACHINE LEARNING TRAFFIC CONGESTION & PROACTIVE TIMING FORECASTS")
    print("Course: KDU IT 3182 Essentials of AI | Model: Random Forest Multi-Output")
    print("=" * 75)

    for sc in scenarios:
        res = service.predict(target_dt=sc["dt"], weather=sc["weather"])
        
        print(f"\n[SCENARIO] {sc['name']}")
        print(f"   Context: {sc['desc']}")
        print(f"   Time: {res['timestamp']} | Weather: {res['weather']} | Rush Hour: {res['is_rush_hour']}")
        print(f"   Critical Tidal Approach: {res['critical_tidal_approach']}")
        
        print("   Forecasted Approach Flow Rates:")
        flows = res["predicted_flows_vpm"]
        states = res["predicted_congestion_state"]
        for d in ["North", "South", "East", "West"]:
            print(f"     * {d:6s}: {flows[d]:5.1f} veh/min  [{states[d]}]")
            
        timing = res["proactive_timing_recommendation"]
        print(f"   Proactive Signal Baseline:")
        print(f"     -> Phase NS (North/South): {timing['Phase_NS_Green']}s Green")
        print(f"     -> Phase EW (East/West):   {timing['Phase_EW_Green']}s Green")
        print("-" * 75)

if __name__ == "__main__":
    run_scenarios()
