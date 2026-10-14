"""
Mamdani Fuzzy Logic Traffic Signal Controller
High dynamic range green duration allocation:
- Empty / Sparse: 8.0s
- Light Queue (1-3 PCU): 10.0s - 13.0s
- Medium Queue (4-8 PCU): 16.0s - 22.0s
- Heavy Queue (9-15 PCU): 26.0s - 32.0s
- Severe Congestion / Starvation (16+ PCU): 35.0s - 45.0s
"""

class FuzzyMembership:
    @staticmethod
    def trimf(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        else:
            return (c - x) / (c - b) if c != b else 1.0

    @staticmethod
    def trapmf(x, a, b, c, d):
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        elif b <= x <= c:
            return 1.0
        else:
            return (d - x) / (d - c) if d != c else 1.0


class FuzzyTrafficController:
    def __init__(self, min_green=22.0, max_green=58.0):
        self.min_green = min_green
        self.max_green = max_green
        span = max_green - min_green
        self.OUTPUT_CONSEQUENTS = {
            'micro':    round(min_green, 1),
            'short':    round(min_green + span * 0.20, 1),
            'medium':   round(min_green + span * 0.45, 1),
            'long':     round(min_green + span * 0.72, 1),
            'extended': round(max_green, 1)
        }

    def _membership_queue(self, pcu):
        return {
            'very_low': FuzzyMembership.trapmf(pcu, 0, 0, 1.5, 3.0),
            'low': FuzzyMembership.trimf(pcu, 2.0, 4.5, 8.0),
            'medium': FuzzyMembership.trimf(pcu, 6.0, 10.0, 15.0),
            'high': FuzzyMembership.trimf(pcu, 12.0, 18.0, 24.0),
            'very_high': FuzzyMembership.trapmf(pcu, 20.0, 28.0, 60.0, 60.0)
        }

    def _membership_wait_time(self, wait_sec):
        return {
            'short': FuzzyMembership.trapmf(wait_sec, 0, 0, 15.0, 30.0),
            'moderate': FuzzyMembership.trimf(wait_sec, 25.0, 45.0, 65.0),
            'long': FuzzyMembership.trapmf(wait_sec, 55.0, 75.0, 120.0, 120.0)
        }

    def _membership_cross_traffic(self, cross_pcu):
        return {
            'low': FuzzyMembership.trapmf(cross_pcu, 0, 0, 3.0, 6.0),
            'moderate': FuzzyMembership.trimf(cross_pcu, 4.0, 8.0, 14.0),
            'high': FuzzyMembership.trapmf(cross_pcu, 10.0, 16.0, 50.0, 50.0)
        }

    def compute_green_duration(self, current_pcu: float, max_wait_time: float = 0.0, cross_pcu: float = 0.0) -> dict:
        if current_pcu <= 0.8:
            return {
                'green_duration': self.min_green,
                'raw_computed': self.min_green,
                'current_pcu': round(current_pcu, 2),
                'max_wait_time': round(max_wait_time, 1),
                'cross_pcu': round(cross_pcu, 2),
                'rule_activations': {'micro': 1.0}
            }

        q = self._membership_queue(current_pcu)
        w = self._membership_wait_time(max_wait_time)
        c = self._membership_cross_traffic(cross_pcu)

        rule_strengths = {
            'micro': 0.0,
            'short': 0.0,
            'medium': 0.0,
            'long': 0.0,
            'extended': 0.0
        }

        # Rule 1: Very low queue -> Micro green (8s)
        rule_strengths['micro'] = max(rule_strengths['micro'], q['very_low'])

        # Rule 2: Low queue -> Short green (12s)
        rule_strengths['short'] = max(rule_strengths['short'], q['low'])

        # Rule 3: Medium queue -> Medium green (20s)
        rule_strengths['medium'] = max(rule_strengths['medium'], q['medium'])

        # Rule 4: High queue -> Long green (30s)
        rule_strengths['long'] = max(rule_strengths['long'], q['high'])

        # Rule 5: Very High queue or Long wait -> Extended green (40s)
        r5 = max(q['very_high'], w['long'])
        rule_strengths['extended'] = max(rule_strengths['extended'], r5)

        # Cross traffic adjustments (if cross traffic is very heavy, slightly bias towards clearing faster)
        if c['high'] > 0 and (q['low'] > 0 or q['very_low'] > 0):
            rule_strengths['micro'] = max(rule_strengths['micro'], 0.8)

        # Centroid Defuzzification
        num = sum(strength * self.OUTPUT_CONSEQUENTS[cat] for cat, strength in rule_strengths.items() if strength > 0)
        den = sum(strength for strength in rule_strengths.values() if strength > 0)

        default_fallback = round(self.min_green + (self.max_green - self.min_green) * 0.35, 1)
        computed = (num / den) if den > 0 else default_fallback
        final_green = max(self.min_green, min(self.max_green, computed))

        return {
            'green_duration': round(final_green, 1),
            'raw_computed': round(computed, 1),
            'current_pcu': round(current_pcu, 2),
            'max_wait_time': round(max_wait_time, 1),
            'cross_pcu': round(cross_pcu, 2),
            'rule_activations': {k: round(v, 2) for k, v in rule_strengths.items() if v > 0}
        }
