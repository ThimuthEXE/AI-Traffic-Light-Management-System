"""
Traffic Signal State Machine with ML Dynamic Asymmetric Phasing
Features conflict-free Protected Phasing + ML Asymmetric Single-Approach Exclusive Green.
General Sir John Kotelawala Defence University (KDU) - IT 3182 Essentials of AI
"""

import math
import pygame


class SignalPhase:
    # Balanced East-West Movement Cycle
    EW_THROUGH_GREEN = "EW_THROUGH_GREEN"
    EW_THROUGH_YELLOW = "EW_THROUGH_YELLOW"
    EW_THROUGH_ALL_RED = "EW_THROUGH_ALL_RED"
    EW_LEFT_GREEN = "EW_LEFT_GREEN"
    EW_LEFT_YELLOW = "EW_LEFT_YELLOW"
    ALL_RED_1 = "ALL_RED_1"

    # Balanced North-South Movement Cycle
    NS_THROUGH_GREEN = "NS_THROUGH_GREEN"
    NS_THROUGH_YELLOW = "NS_THROUGH_YELLOW"
    NS_THROUGH_ALL_RED = "NS_THROUGH_ALL_RED"
    NS_LEFT_GREEN = "NS_LEFT_GREEN"
    NS_LEFT_YELLOW = "NS_LEFT_YELLOW"
    ALL_RED_2 = "ALL_RED_2"

    # Asymmetric Exclusive Green Phases (Full Green for Heavy Approach, Opposing Held at Red)
    N_EXCLUSIVE_GREEN = "N_EXCLUSIVE_GREEN"
    N_EXCLUSIVE_YELLOW = "N_EXCLUSIVE_YELLOW"
    
    S_EXCLUSIVE_GREEN = "S_EXCLUSIVE_GREEN"
    S_EXCLUSIVE_YELLOW = "S_EXCLUSIVE_YELLOW"

    E_EXCLUSIVE_GREEN = "E_EXCLUSIVE_GREEN"
    E_EXCLUSIVE_YELLOW = "E_EXCLUSIVE_YELLOW"

    W_EXCLUSIVE_GREEN = "W_EXCLUSIVE_GREEN"
    W_EXCLUSIVE_YELLOW = "W_EXCLUSIVE_YELLOW"


class TrafficSignalManager:
    def __init__(self, yellow_duration=2.0, all_red_duration=1.0):
        self.yellow_duration = yellow_duration
        self.all_red_duration = all_red_duration
        
        self.current_phase = SignalPhase.EW_THROUGH_GREEN
        self.allocated_total_green = 20.0
        self.allocated_through_green = 14.0
        self.allocated_left_green = 6.0
        self.time_in_state = 0.0
        
        self.active_policy = "BALANCED_PHASE"

        self.signal_positions = {
            'E': (530, 480),
            'W': (750, 240),
            'S': (510, 240),
            'N': (770, 480)
        }

    def get_signals_for_direction(self, direction: str) -> tuple:
        """Returns (signal_through, signal_turn) for direction ('N', 'S', 'E', 'W')."""
        d = direction.upper()

        # === 1. Asymmetric Exclusive Green Phases ===
        if self.current_phase == SignalPhase.N_EXCLUSIVE_GREEN:
            return ('GREEN', 'GREEN') if d == 'N' else ('RED', 'RED')
        elif self.current_phase == SignalPhase.N_EXCLUSIVE_YELLOW:
            return ('YELLOW', 'YELLOW') if d == 'N' else ('RED', 'RED')

        elif self.current_phase == SignalPhase.S_EXCLUSIVE_GREEN:
            return ('GREEN', 'GREEN') if d == 'S' else ('RED', 'RED')
        elif self.current_phase == SignalPhase.S_EXCLUSIVE_YELLOW:
            return ('YELLOW', 'YELLOW') if d == 'S' else ('RED', 'RED')

        elif self.current_phase == SignalPhase.E_EXCLUSIVE_GREEN:
            return ('GREEN', 'GREEN') if d == 'E' else ('RED', 'RED')
        elif self.current_phase == SignalPhase.E_EXCLUSIVE_YELLOW:
            return ('YELLOW', 'YELLOW') if d == 'E' else ('RED', 'RED')

        elif self.current_phase == SignalPhase.W_EXCLUSIVE_GREEN:
            return ('GREEN', 'GREEN') if d == 'W' else ('RED', 'RED')
        elif self.current_phase == SignalPhase.W_EXCLUSIVE_YELLOW:
            return ('YELLOW', 'YELLOW') if d == 'W' else ('RED', 'RED')

        # === 2. Balanced East-West Phasing ===
        elif self.current_phase == SignalPhase.EW_THROUGH_GREEN:
            return ('GREEN', 'RED') if d in ['E', 'W'] else ('RED', 'RED')
        elif self.current_phase == SignalPhase.EW_THROUGH_YELLOW:
            return ('YELLOW', 'RED') if d in ['E', 'W'] else ('RED', 'RED')
        elif self.current_phase == SignalPhase.EW_THROUGH_ALL_RED:
            return ('RED', 'RED')

        elif self.current_phase == SignalPhase.EW_LEFT_GREEN:
            return ('RED', 'GREEN') if d in ['E', 'W'] else ('RED', 'RED')
        elif self.current_phase == SignalPhase.EW_LEFT_YELLOW:
            return ('RED', 'YELLOW') if d in ['E', 'W'] else ('RED', 'RED')
        elif self.current_phase == SignalPhase.ALL_RED_1:
            return ('RED', 'RED')

        # === 3. Balanced North-South Phasing ===
        elif self.current_phase == SignalPhase.NS_THROUGH_GREEN:
            return ('GREEN', 'RED') if d in ['N', 'S'] else ('RED', 'RED')
        elif self.current_phase == SignalPhase.NS_THROUGH_YELLOW:
            return ('YELLOW', 'RED') if d in ['N', 'S'] else ('RED', 'RED')
        elif self.current_phase == SignalPhase.NS_THROUGH_ALL_RED:
            return ('RED', 'RED')

        elif self.current_phase == SignalPhase.NS_LEFT_GREEN:
            return ('RED', 'GREEN') if d in ['N', 'S'] else ('RED', 'RED')
        elif self.current_phase == SignalPhase.NS_LEFT_YELLOW:
            return ('RED', 'YELLOW') if d in ['N', 'S'] else ('RED', 'RED')
        elif self.current_phase == SignalPhase.ALL_RED_2:
            return ('RED', 'RED')

        return ('RED', 'RED')

    def get_signal_state(self, direction: str) -> str:
        st, sl = self.get_signals_for_direction(direction)
        if 'GREEN' in (st, sl): return 'GREEN'
        if 'YELLOW' in (st, sl): return 'YELLOW'
        return 'RED'

    @property
    def active_green_axis(self) -> str:
        if "EW" in self.current_phase or "E_" in self.current_phase or "W_" in self.current_phase:
            return 'EW'
        elif "NS" in self.current_phase or "N_" in self.current_phase or "S_" in self.current_phase:
            return 'NS'
        return 'NONE'

    @property
    def allocated_green(self) -> float:
        if "EXCLUSIVE_GREEN" in self.current_phase:
            return self.allocated_total_green
        elif "THROUGH_GREEN" in self.current_phase:
            return self.allocated_through_green
        elif "LEFT_GREEN" in self.current_phase:
            return self.allocated_left_green
        return self.allocated_through_green

    def _split_allocated_green(self, total_green: float,
                                through_green: float = None,
                                turn_green: float = None):
        """Set the through/turn green allocation.

        If through_green and turn_green are supplied (from CycleObserver),
        those exact values are used.  Otherwise falls back to fixed 68/32.
        """
        self.allocated_total_green   = total_green
        if through_green is not None and turn_green is not None:
            self.allocated_through_green = max(7.0,  round(through_green, 1))
            self.allocated_left_green    = max(4.5,  round(turn_green,    1))
        else:
            self.allocated_through_green = max(7.0, round(total_green * 0.68, 1))
            self.allocated_left_green    = max(4.5, round(total_green * 0.32, 1))

    def update(self, dt: float, next_decision_calc_fn=None) -> tuple:
        self.time_in_state += dt
        phase_switched = False
        completed_phase = None
        new_phase = None
        new_green = 0.0

        # === EXCLUSIVE ASYMMETRIC PHASES ===
        if "EXCLUSIVE_GREEN" in self.current_phase:
            if self.time_in_state >= self.allocated_total_green:
                completed_phase = self.current_phase
                if self.current_phase == SignalPhase.N_EXCLUSIVE_GREEN: self.current_phase = SignalPhase.N_EXCLUSIVE_YELLOW
                elif self.current_phase == SignalPhase.S_EXCLUSIVE_GREEN: self.current_phase = SignalPhase.S_EXCLUSIVE_YELLOW
                elif self.current_phase == SignalPhase.E_EXCLUSIVE_GREEN: self.current_phase = SignalPhase.E_EXCLUSIVE_YELLOW
                elif self.current_phase == SignalPhase.W_EXCLUSIVE_GREEN: self.current_phase = SignalPhase.W_EXCLUSIVE_YELLOW
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = self.current_phase

        elif "EXCLUSIVE_YELLOW" in self.current_phase:
            if self.time_in_state >= self.yellow_duration:
                completed_phase = self.current_phase
                # After exclusive yellow, switch to all-red before transitioning to cross-axis
                if "N_" in completed_phase or "S_" in completed_phase:
                    self.current_phase = SignalPhase.ALL_RED_2
                else:
                    self.current_phase = SignalPhase.ALL_RED_1
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = self.current_phase

        # === BALANCED EAST-WEST SEQUENCE ===
        elif self.current_phase == SignalPhase.EW_THROUGH_GREEN:
            if self.time_in_state >= self.allocated_through_green:
                completed_phase = SignalPhase.EW_THROUGH_GREEN
                self.current_phase = SignalPhase.EW_THROUGH_YELLOW
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = SignalPhase.EW_THROUGH_YELLOW

        elif self.current_phase == SignalPhase.EW_THROUGH_YELLOW:
            if self.time_in_state >= self.yellow_duration:
                completed_phase = SignalPhase.EW_THROUGH_YELLOW
                self.current_phase = SignalPhase.EW_THROUGH_ALL_RED
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = SignalPhase.EW_THROUGH_ALL_RED

        elif self.current_phase == SignalPhase.EW_THROUGH_ALL_RED:
            if self.time_in_state >= self.all_red_duration:
                completed_phase = SignalPhase.EW_THROUGH_ALL_RED
                self.current_phase = SignalPhase.EW_LEFT_GREEN
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = SignalPhase.EW_LEFT_GREEN

        elif self.current_phase == SignalPhase.EW_LEFT_GREEN:
            if self.time_in_state >= self.allocated_left_green:
                completed_phase = SignalPhase.EW_LEFT_GREEN
                self.current_phase = SignalPhase.EW_LEFT_YELLOW
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = SignalPhase.EW_LEFT_YELLOW

        elif self.current_phase == SignalPhase.EW_LEFT_YELLOW:
            if self.time_in_state >= self.yellow_duration:
                completed_phase = SignalPhase.EW_LEFT_YELLOW
                self.current_phase = SignalPhase.ALL_RED_1
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = SignalPhase.ALL_RED_1

        elif self.current_phase == SignalPhase.ALL_RED_1:
            if self.time_in_state >= self.all_red_duration:
                completed_phase = SignalPhase.ALL_RED_1
                self.time_in_state = 0.0

                if next_decision_calc_fn is not None:
                    decision = next_decision_calc_fn('NS')
                    policy      = decision.get("policy", "BALANCED_PHASE")
                    total_green = decision.get("total_green_sec",
                                  decision.get("green_duration", 22.0))
                    thru_g      = decision.get("through_green_sec", None)
                    turn_g      = decision.get("turn_green_sec",    None)
                    self.active_policy = policy

                    if policy == "N_EXCLUSIVE_GREEN":
                        self.current_phase = SignalPhase.N_EXCLUSIVE_GREEN
                        self._split_allocated_green(total_green, thru_g, turn_g)
                    elif policy == "S_EXCLUSIVE_GREEN":
                        self.current_phase = SignalPhase.S_EXCLUSIVE_GREEN
                        self._split_allocated_green(total_green, thru_g, turn_g)
                    else:
                        self.current_phase = SignalPhase.NS_THROUGH_GREEN
                        self._split_allocated_green(total_green, thru_g, turn_g)
                else:
                    self.current_phase = SignalPhase.NS_THROUGH_GREEN

                new_green = self.allocated_green
                phase_switched = True
                new_phase = self.current_phase

        # === BALANCED NORTH-SOUTH SEQUENCE ===
        elif self.current_phase == SignalPhase.NS_THROUGH_GREEN:
            if self.time_in_state >= self.allocated_through_green:
                completed_phase = SignalPhase.NS_THROUGH_GREEN
                self.current_phase = SignalPhase.NS_THROUGH_YELLOW
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = SignalPhase.NS_THROUGH_YELLOW

        elif self.current_phase == SignalPhase.NS_THROUGH_YELLOW:
            if self.time_in_state >= self.yellow_duration:
                completed_phase = SignalPhase.NS_THROUGH_YELLOW
                self.current_phase = SignalPhase.NS_THROUGH_ALL_RED
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = SignalPhase.NS_THROUGH_ALL_RED

        elif self.current_phase == SignalPhase.NS_THROUGH_ALL_RED:
            if self.time_in_state >= self.all_red_duration:
                completed_phase = SignalPhase.NS_THROUGH_ALL_RED
                self.current_phase = SignalPhase.NS_LEFT_GREEN
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = SignalPhase.NS_LEFT_GREEN

        elif self.current_phase == SignalPhase.NS_LEFT_GREEN:
            if self.time_in_state >= self.allocated_left_green:
                completed_phase = SignalPhase.NS_LEFT_GREEN
                self.current_phase = SignalPhase.NS_LEFT_YELLOW
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = SignalPhase.NS_LEFT_YELLOW

        elif self.current_phase == SignalPhase.NS_LEFT_YELLOW:
            if self.time_in_state >= self.yellow_duration:
                completed_phase = SignalPhase.NS_LEFT_YELLOW
                self.current_phase = SignalPhase.ALL_RED_2
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = SignalPhase.ALL_RED_2

        elif self.current_phase == SignalPhase.ALL_RED_2:
            if self.time_in_state >= self.all_red_duration:
                completed_phase = SignalPhase.ALL_RED_2
                self.time_in_state = 0.0

                if next_decision_calc_fn is not None:
                    decision = next_decision_calc_fn('EW')
                    policy      = decision.get("policy", "BALANCED_PHASE")
                    total_green = decision.get("total_green_sec",
                                  decision.get("green_duration", 22.0))
                    thru_g      = decision.get("through_green_sec", None)
                    turn_g      = decision.get("turn_green_sec",    None)
                    self.active_policy = policy

                    if policy == "E_EXCLUSIVE_GREEN":
                        self.current_phase = SignalPhase.E_EXCLUSIVE_GREEN
                        self._split_allocated_green(total_green, thru_g, turn_g)
                    elif policy == "W_EXCLUSIVE_GREEN":
                        self.current_phase = SignalPhase.W_EXCLUSIVE_GREEN
                        self._split_allocated_green(total_green, thru_g, turn_g)
                    else:
                        self.current_phase = SignalPhase.EW_THROUGH_GREEN
                        self._split_allocated_green(total_green, thru_g, turn_g)
                else:
                    self.current_phase = SignalPhase.EW_THROUGH_GREEN

                new_green = self.allocated_green
                phase_switched = True
                new_phase = self.current_phase

        return phase_switched, completed_phase, new_phase, new_green

    def force_emergency_axis(self, emergency_axis: str, emergency_green_time: float = 16.0):
        if emergency_axis == 'EW' and "EW" not in self.current_phase and "E_" not in self.current_phase and "W_" not in self.current_phase:
            self.current_phase = SignalPhase.EW_THROUGH_GREEN
            self._split_allocated_green(emergency_green_time)
            self.time_in_state = 0.0
        elif emergency_axis == 'NS' and "NS" not in self.current_phase and "N_" not in self.current_phase and "S_" not in self.current_phase:
            self.current_phase = SignalPhase.NS_THROUGH_GREEN
            self._split_allocated_green(emergency_green_time)
            self.time_in_state = 0.0

    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        for direction, (px, py) in self.signal_positions.items():
            sig_th, sig_lt = self.get_signals_for_direction(direction)

            box_w, box_h = 44, 48
            box_rect = pygame.Rect(px - box_w // 2, py - box_h // 2, box_w, box_h)
            pygame.draw.rect(surface, (20, 24, 30), box_rect, border_radius=6)
            pygame.draw.rect(surface, (70, 80, 95), box_rect, width=1, border_radius=6)

            turn_col = (231, 76, 60) if sig_lt == 'RED' else (241, 196, 15) if sig_lt == 'YELLOW' else (46, 204, 113)
            turn_pos = (px - 11, py - 4)
            pygame.draw.circle(surface, turn_col, turn_pos, 7)
            pygame.draw.circle(surface, (20, 20, 20), turn_pos, 7, 1)

            th_col = (231, 76, 60) if sig_th == 'RED' else (241, 196, 15) if sig_th == 'YELLOW' else (46, 204, 113)
            th_pos = (px + 11, py - 4)
            pygame.draw.circle(surface, th_col, th_pos, 7)
            pygame.draw.circle(surface, (20, 20, 20), th_pos, 7, 1)

            tiny_font = pygame.font.SysFont("Segoe UI", 9, bold=True)
            surface.blit(tiny_font.render("TURN", True, (160, 180, 200)), (px - 20, py + 8))
            surface.blit(tiny_font.render("THRU", True, (160, 180, 200)), (px + 4, py + 8))

            rem_sec = 0
            if "EXCLUSIVE_GREEN" in self.current_phase:
                rem_sec = max(0, int(math.ceil(self.allocated_total_green - self.time_in_state)))
            elif "GREEN" in self.current_phase:
                active_limit = self.allocated_through_green if "THROUGH" in self.current_phase else self.allocated_left_green
                rem_sec = max(0, int(math.ceil(active_limit - self.time_in_state)))
            elif "YELLOW" in self.current_phase:
                rem_sec = max(0, int(math.ceil(self.yellow_duration - self.time_in_state)))

            is_active = (sig_th == 'GREEN' or sig_lt == 'GREEN' or sig_th == 'YELLOW' or sig_lt == 'YELLOW')
            if is_active and rem_sec > 0:
                timer_txt = font.render(f"{rem_sec}s", True, (255, 255, 255))
                timer_bg = pygame.Rect(px - 14, py - 32, 28, 16)
                pygame.draw.rect(surface, (0, 0, 0), timer_bg, border_radius=3)
                pygame.draw.rect(surface, (241, 196, 15), timer_bg, width=1, border_radius=3)
                surface.blit(timer_txt, (px - timer_txt.get_width() // 2, py - 33))
