"""
smoother.py
-----------
Command smoother / debouncer — prevents rapid flickering of
navigation commands when detections are unstable at zone borders.

Algorithm:
  - A command must be "voted" for at least N consecutive frames
    before it replaces the current command (hysteresis).
  - STOP and APPROACHING_FAST commands bypass the delay (safety first).
  - Provides a confidence score (0–1) for the current command.

Author: AI Navigation Project
"""

from collections import deque

# Frames a command must persist before becoming "official"
CONFIRM_FRAMES   = 4    # standard commands
URGENT_FRAMES    = 1    # STOP / fast-approach — instant

# Commands that bypass the debouncer
URGENT_COMMANDS  = {"STOP !", "APPROACHING FAST — STOP"}


class CommandSmoother:
    """
    Smooths navigation commands using a voting window.

    Usage
    -----
        smoother = CommandSmoother()
        stable_cmd = smoother.update(raw_command)
    """

    def __init__(self, window: int = CONFIRM_FRAMES):
        self.window          = window
        self._buffer         = deque(maxlen=window)
        self._current        = "Path Clear"
        self._candidate      = "Path Clear"
        self._candidate_count = 0

    def update(self, raw_command: str) -> str:
        """
        Feed the raw command for this frame and return the stable command.

        Parameters
        ----------
        raw_command : str — raw output from Navigator.decide()

        Returns
        -------
        str : smoothed stable command
        """
        self._buffer.append(raw_command)

        # Urgent commands bypass debounce
        if raw_command in URGENT_COMMANDS:
            self._current        = raw_command
            self._candidate      = raw_command
            self._candidate_count = 0
            return self._current

        # Count consecutive occurrences of the new command
        if raw_command == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate       = raw_command
            self._candidate_count = 1

        # Promote candidate to current if it has held long enough
        if self._candidate_count >= self.window:
            self._current = self._candidate

        return self._current

    @property
    def confidence(self) -> float:
        """
        Fraction of the recent window that agrees with current command.
        """
        if not self._buffer:
            return 1.0
        matches = sum(1 for c in self._buffer if c == self._current)
        return matches / len(self._buffer)

    def reset(self):
        self._buffer.clear()
        self._current        = "Path Clear"
        self._candidate      = "Path Clear"
        self._candidate_count = 0


class SpeedAwareVoice:
    """
    Wraps VoiceAssistant to modulate speech rate and urgency
    based on obstacle proximity.

    - Danger zones trigger faster speech rate + "WARNING" prefix
    - Cooldown is shortened for close obstacles
    """

    def __init__(self, voice_assistant,
                 normal_rate: int = 175,
                 urgent_rate: int = 220):
        self._voice       = voice_assistant
        self._normal_rate = normal_rate
        self._urgent_rate = urgent_rate
        self._last_urgent = False

    def speak(self, command: str, min_distance: float):
        """
        Speak command with rate adjusted to distance urgency.

        Parameters
        ----------
        command      : navigation command string
        min_distance : closest obstacle distance in metres
        """
        if self._voice is None or not self._voice.enabled:
            return

        is_urgent = min_distance < 1.5 or command == "STOP !"

        if is_urgent and not self._last_urgent:
            self._voice.set_rate(self._urgent_rate)
            self._last_urgent = True
        elif not is_urgent and self._last_urgent:
            self._voice.set_rate(self._normal_rate)
            self._last_urgent = False

        # Add urgency prefix for very close obstacles
        if min_distance < 0.8:
            spoken = f"WARNING. {command}"
        elif min_distance < 1.5:
            spoken = f"Caution. {command}"
        else:
            spoken = command

        self._voice.speak(spoken)
