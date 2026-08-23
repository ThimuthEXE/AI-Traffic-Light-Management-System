"""
Traffic Signal State Machine & Visualizer
Provides predictable, driver-safe countdown timers (decrementing monotonically),
calculating new cycle durations strictly at phase transition boundaries.
"""

import math
import pygame

class SignalPhase:
    EW_GREEN = "EW_GREEN"
    EW_YELLOW = "EW_YELLOW"
    ALL_RED_1 = "ALL_RED_1"
    NS_GREEN = "NS_GREEN"
    NS_YELLOW = "NS_YELLOW"
    ALL_RED_2 = "ALL_RED_2"

class TrafficSignalManager:
    def __init__(self, yellow_duration=2.5, all_red_duration=1.0):
        self.yellow_duration = yellow_duration
        self.all_red_duration = all_red_duration
        
        self.current_phase = SignalPhase.EW_GREEN
        self.allocated_green = 16.0
        self.time_in_state = 0.0
        
        # 4 Signal heads on intersection corners
        self.signal_positions = {
            'E': (530, 470),
            'W': (750, 250),
            'S': (530, 250),
            'N': (750, 470)
        }

    def get_signal_state(self, direction: str) -> str:
        if self.current_phase == SignalPhase.EW_GREEN:
            return 'GREEN' if direction in ['E', 'W'] else 'RED'
        elif self.current_phase == SignalPhase.EW_YELLOW:
            return 'YELLOW' if direction in ['E', 'W'] else 'RED'
        elif self.current_phase in [SignalPhase.ALL_RED_1, SignalPhase.ALL_RED_2]:
            return 'RED'
        elif self.current_phase == SignalPhase.NS_GREEN:
            return 'GREEN' if direction in ['N', 'S'] else 'RED'
        elif self.current_phase == SignalPhase.NS_YELLOW:
            return 'YELLOW' if direction in ['N', 'S'] else 'RED'
        return 'RED'

    @property
    def active_green_axis(self) -> str:
        if self.current_phase in [SignalPhase.EW_GREEN, SignalPhase.EW_YELLOW]:
            return 'EW'
        elif self.current_phase in [SignalPhase.NS_GREEN, SignalPhase.NS_YELLOW]:
            return 'NS'
        return 'NONE'

    def update(self, dt: float, next_green_duration_calc_fn=None) -> tuple:
        """
        Advances the signal clock strictly without mid-cycle timer jumps.
        Returns: (phase_switched_bool, completed_phase_name, newly_started_phase_name, new_allocated_green)
        """
        self.time_in_state += dt
        phase_switched = False
        completed_phase = None
        new_phase = None
        new_green = 0.0

        if self.current_phase == SignalPhase.EW_GREEN:
            if self.time_in_state >= self.allocated_green:
                completed_phase = SignalPhase.EW_GREEN
                self.current_phase = SignalPhase.EW_YELLOW
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = SignalPhase.EW_YELLOW

        elif self.current_phase == SignalPhase.EW_YELLOW:
            if self.time_in_state >= self.yellow_duration:
                completed_phase = SignalPhase.EW_YELLOW
                self.current_phase = SignalPhase.ALL_RED_1
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = SignalPhase.ALL_RED_1

        elif self.current_phase == SignalPhase.ALL_RED_1:
            if self.time_in_state >= self.all_red_duration:
                completed_phase = SignalPhase.ALL_RED_1
                self.current_phase = SignalPhase.NS_GREEN
                self.time_in_state = 0.0
                
                # Predict next green duration strictly at phase start
                if next_green_duration_calc_fn is not None:
                    self.allocated_green = next_green_duration_calc_fn('NS')
                new_green = self.allocated_green
                phase_switched = True
                new_phase = SignalPhase.NS_GREEN

        elif self.current_phase == SignalPhase.NS_GREEN:
            if self.time_in_state >= self.allocated_green:
                completed_phase = SignalPhase.NS_GREEN
                self.current_phase = SignalPhase.NS_YELLOW
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = SignalPhase.NS_YELLOW

        elif self.current_phase == SignalPhase.NS_YELLOW:
            if self.time_in_state >= self.yellow_duration:
                completed_phase = SignalPhase.NS_YELLOW
                self.current_phase = SignalPhase.ALL_RED_2
                self.time_in_state = 0.0
                phase_switched = True
                new_phase = SignalPhase.ALL_RED_2

        elif self.current_phase == SignalPhase.ALL_RED_2:
            if self.time_in_state >= self.all_red_duration:
                completed_phase = SignalPhase.ALL_RED_2
                self.current_phase = SignalPhase.EW_GREEN
                self.time_in_state = 0.0
                
                # Predict next green duration strictly at phase start
                if next_green_duration_calc_fn is not None:
                    self.allocated_green = next_green_duration_calc_fn('EW')
                new_green = self.allocated_green
                phase_switched = True
                new_phase = SignalPhase.EW_GREEN

        return phase_switched, completed_phase, new_phase, new_green

    def force_emergency_axis(self, emergency_axis: str, emergency_green_time: float = 16.0):
        if emergency_axis == 'EW' and self.current_phase not in [SignalPhase.EW_GREEN, SignalPhase.EW_YELLOW]:
            self.current_phase = SignalPhase.EW_GREEN
            self.allocated_green = emergency_green_time
            self.time_in_state = 0.0
        elif emergency_axis == 'NS' and self.current_phase not in [SignalPhase.NS_GREEN, SignalPhase.NS_YELLOW]:
            self.current_phase = SignalPhase.NS_GREEN
            self.allocated_green = emergency_green_time
            self.time_in_state = 0.0

    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        """Draw signal heads with clean second-by-second countdown timers."""
        for direction, (px, py) in self.signal_positions.items():
            state = self.get_signal_state(direction)
            
            casing_rect = pygame.Rect(px - 14, py - 35, 28, 70)
            pygame.draw.rect(surface, (30, 30, 30), casing_rect, border_radius=6)
            pygame.draw.rect(surface, (70, 70, 70), casing_rect, width=2, border_radius=6)

            red_color = (255, 30, 30) if state == 'RED' else (60, 10, 10)
            pygame.draw.circle(surface, red_color, (px, py - 20), 8)

            yellow_color = (255, 200, 0) if state == 'YELLOW' else (60, 50, 5)
            pygame.draw.circle(surface, yellow_color, (px, py), 8)

            green_color = (30, 255, 30) if state == 'GREEN' else (10, 60, 10)
            pygame.draw.circle(surface, green_color, (px, py + 20), 8)

            # Continuous, monotonic countdown calculation
            if state == 'GREEN':
                rem_time = max(0.0, self.allocated_green - self.time_in_state)
            elif state == 'YELLOW':
                rem_time = max(0.0, self.yellow_duration - self.time_in_state)
            elif state == 'RED':
                # Time remaining until next green
                if self.current_phase in [SignalPhase.EW_GREEN, SignalPhase.NS_GREEN]:
                    rem_time = max(0.0, (self.allocated_green - self.time_in_state) + self.yellow_duration + self.all_red_duration)
                elif self.current_phase in [SignalPhase.EW_YELLOW, SignalPhase.NS_YELLOW]:
                    rem_time = max(0.0, (self.yellow_duration - self.time_in_state) + self.all_red_duration)
                else:
                    rem_time = max(0.0, self.all_red_duration - self.time_in_state)

            display_sec = int(math.ceil(rem_time))
            timer_txt = font.render(f"{display_sec:02d}s", True, (240, 240, 240))
            surface.blit(timer_txt, (px - 12, py + 38))
