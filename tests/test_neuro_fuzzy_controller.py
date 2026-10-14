import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai_engine.simulation.controllers import AIFuzzyController

def test_neuro_fuzzy_controller():
    ctrl = AIFuzzyController(min_green=22.0, max_green=58.0)
    print("Initialized AIFuzzyController successfully!")

    class MockV:
        def __init__(self, d, lane, wait):
            self.direction = d
            self.lane_idx = lane
            self.wait_time = wait
            self.has_cleared_intersection = False

    # Scenario 1: Light traffic
    vehs_light = [MockV('N', 0, 2.0), MockV('S', 0, 1.5)]
    intervals = {'N': 4.0, 'S': 4.0, 'E': 4.0, 'W': 4.0}
    dec_light = ctrl.get_next_phase_decision('NS', vehs_light, intervals)
    print("\nScenario 1 (Light traffic):")
    print("  Total Green:", dec_light['total_green_sec'])
    print("  Fuzzy Green:", dec_light['fuzzy_green_sec'])
    print("  ML Green:", dec_light['ml_predicted_green_sec'])
    print("  Rule Activations:", dec_light['rule_activations'])
    assert dec_light['total_green_sec'] >= 22.0, "Expected minimum green >= 22.0s"
    assert dec_light['through_green_sec'] >= 16.0, "Expected through green >= 16.0s"
    assert dec_light['turn_green_sec'] >= 6.0, "Expected turn green >= 6.0s"
    assert 'micro' in dec_light['rule_activations'], "Expected micro fuzzy rule to activate"

    # Scenario 2: Heavy surge traffic
    vehs_heavy = [MockV('N', i % 2, 40.0) for i in range(16)]
    for i in range(4):
        vehs_heavy.append(MockV('S', i % 2, 20.0))
    dec_heavy = ctrl.get_next_phase_decision('NS', vehs_heavy, {'N': 0.6, 'S': 1.2, 'E': 3.0, 'W': 3.0})
    print("\nScenario 2 (Heavy surge):")
    print("  Policy:", dec_heavy['policy'])
    print("  Total Green:", dec_heavy['total_green_sec'])
    print("  Fuzzy Green:", dec_heavy['fuzzy_green_sec'])
    print("  ML Green:", dec_heavy['ml_predicted_green_sec'])
    print("  Rule Activations:", dec_heavy['rule_activations'])
    assert dec_heavy['total_green_sec'] <= 58.0, "Expected green capped at 58.0s"
    assert dec_heavy['fuzzy_green_sec'] > 30.0, "Expected fuzzy green > 30s for heavy surge"
    assert dec_heavy['ml_active'] is True, "Expected ML model to be active"

    print("\nSUCCESS: All Neuro-Fuzzy Controller tests passed successfully!")

if __name__ == "__main__":
    test_neuro_fuzzy_controller()
