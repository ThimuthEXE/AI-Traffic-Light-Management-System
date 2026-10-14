import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_engine.simulation.controllers import AIFuzzyController

def test_mpc_preallocation():
    ctrl = AIFuzzyController(min_green=8.0, max_green=38.0)
    print("Initialized AIFuzzyController successfully!")

    class MockV:
        def __init__(self, d, lane, wait):
            self.direction = d
            self.lane_idx = lane
            self.wait_time = wait
            self.has_cleared_intersection = False

    # Scenario 1: Uncongested
    vehs_light = [MockV('N', 0, 2.0), MockV('S', 0, 1.5)]
    intervals = {'N': 2.0, 'S': 2.0, 'E': 2.0, 'W': 2.0}
    dec_light = ctrl.get_next_phase_decision('NS', vehs_light, intervals)
    print("\nScenario 1 (Light traffic):")
    print("  Total Green:", dec_light['total_green_sec'])
    print("  MPC Proactive Cap:", dec_light['mpc_proactive_cap'])
    print("  Projected Max Wait:", dec_light['mpc_projected_max_wait'])
    assert dec_light['mpc_proactive_cap'] is False, "Expected no proactive cap for light traffic"

    # Scenario 2: Heavy North Surge with Cross-Axis East Queue Approaching 21s
    # North demands ~31s green, but East wait would breach 48s ceiling -> Method 1 caps green
    vehs_heavy_cross = [MockV('N', i % 2, 4.0) for i in range(14)]
    for i in range(8):
        vehs_heavy_cross.append(MockV('E', i % 2, 21.0 - i * 0.5))
    
    dec_heavy = ctrl.get_next_phase_decision('NS', vehs_heavy_cross, intervals)
    print("\nScenario 2 (Heavy East approach wait ~21s):")
    print("  Total Green:", dec_heavy['total_green_sec'])
    print("  MPC Proactive Cap:", dec_heavy['mpc_proactive_cap'])
    print("  Projected Max Wait:", dec_heavy['mpc_projected_max_wait'])
    print("  Bottleneck Approach:", dec_heavy['mpc_bottleneck_dir'])
    print("  Target Ceiling Sec:", dec_heavy['target_ceiling_sec'])
    assert dec_heavy['mpc_proactive_cap'] is True, "Expected MPC proactive capping to trigger"
    assert dec_heavy['total_green_sec'] <= 16.0, f"Expected capped green, got {dec_heavy['total_green_sec']}"
    assert dec_heavy['target_ceiling_sec'] == 28.0, "Expected target ceiling of 28.0s (48s - 20s)"

    print("\nSUCCESS: All Method 1 MPC Lookahead tests passed successfully!")

if __name__ == "__main__":
    test_mpc_preallocation()
