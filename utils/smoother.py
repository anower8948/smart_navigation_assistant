"""
smoother.py
-----------
Command smoother tuned for 7-command 5-zone navigation.

Per-command confirm windows:
  STOP             → 1 frame  (instant, safety)
  Move Left/Right  → 2 frames (fast response)
  Slight Left/Right→ 2 frames (fast response)
  Move Forward     → 3 frames (a little more stable)
  Path Clear       → 3 frames (stable)

Uses majority vote over window so one noisy frame doesn't reset.

Author: AI Navigation Project
"""

from collections import deque
import time

from utils.navigator import (
    CMD_STOP, CMD_VERY_SLOW, CMD_SLOW,
    CMD_LEFT, CMD_SLIGHT_LEFT,
    CMD_RIGHT, CMD_SLIGHT_RIGHT,
    CMD_FORWARD, CMD_CLEAR
)

# Per-command confirm window sizes
# Speed commands confirm quickly — direction commands a touch slower
CONFIRM_WINDOWS = {
    CMD_STOP:          1,   # instant — safety
    CMD_VERY_SLOW:     1,   # instant — safety
    CMD_SLOW:          2,   # fast
    CMD_LEFT:          2,
    CMD_SLIGHT_LEFT:   2,
    CMD_RIGHT:         2,
    CMD_SLIGHT_RIGHT:  2,
    CMD_FORWARD:       3,
    CMD_CLEAR:         3,
}

DEFAULT_WINDOW  = 2
INSTANT_CMDS    = {CMD_STOP, CMD_VERY_SLOW}

# Voice cooldowns (seconds) by proximity
COOLDOWN_CRITICAL = 0.8
COOLDOWN_DANGER   = 1.5
COOLDOWN_NORMAL   = 2.8


class CommandSmoother:
    """
    Majority-vote smoother with per-command window sizes.
    Safety commands (STOP) are always instant.
    """

    def __init__(self):
        self._window   = DEFAULT_WINDOW
        self._buf      = deque(maxlen=6)   # large enough for all windows
        self._current  = CMD_CLEAR

    def update(self, raw: str) -> str:
        # Instant for safety
        if raw in INSTANT_CMDS:
            self._current = raw
            self._buf.clear()
            return self._current

        self._buf.append(raw)
        win = CONFIRM_WINDOWS.get(raw, DEFAULT_WINDOW)

        # Majority vote over the last `win` frames
        recent = list(self._buf)[-win:]
        if len(recent) >= win:
            counts = {}
            for c in recent:
                counts[c] = counts.get(c, 0) + 1
            winner, votes = max(counts.items(), key=lambda x: x[1])
            if votes >= (win + 1) // 2:   # simple majority
                self._current = winner

        return self._current

    @property
    def confidence(self) -> float:
        if not self._buf:
            return 1.0
        win = CONFIRM_WINDOWS.get(self._current, DEFAULT_WINDOW)
        recent = list(self._buf)[-win:]
        if not recent:
            return 1.0
        return sum(1 for c in recent if c == self._current) / len(recent)

    def reset(self):
        self._buf.clear()
        self._current = CMD_CLEAR


class SpeedAwareVoice:
    """
    Speaks navigation commands with adaptive rate and urgency prefix.
    Tuned for 7 commands — slight turns use softer language.
    """

    # What to actually say for each command
    SPEECH_TEXT = {
        CMD_STOP:          "Stop! Obstacle very close.",
        CMD_VERY_SLOW:     "Walk very slowly. Obstacle extremely close.",
        CMD_SLOW:          "Walk slowly. Obstacle nearby.",
        CMD_LEFT:          "Move Left.",
        CMD_SLIGHT_LEFT:   "Bear Left.",
        CMD_RIGHT:         "Move Right.",
        CMD_SLIGHT_RIGHT:  "Bear Right.",
        CMD_FORWARD:       "Move Forward.",
        CMD_CLEAR:         "Path clear. Move Forward.",
    }

    def __init__(self, voice_assistant,
                 normal_rate: int = 165,
                 urgent_rate: int = 205):
        self._va          = voice_assistant
        self._normal_rate = normal_rate
        self._urgent_rate = urgent_rate
        self._last_cmd    = ""
        self._last_t      = 0.0
        self._last_urgent = False

    def speak(self, command: str, min_distance: float):
        if self._va is None or not self._va.enabled:
            return

        now = time.time()

        # Adaptive cooldown by proximity
        if min_distance < 1.2:
            cooldown = COOLDOWN_CRITICAL
        elif min_distance < 2.5:
            cooldown = COOLDOWN_DANGER
        else:
            cooldown = COOLDOWN_NORMAL

        same = (command == self._last_cmd)
        if same and (now - self._last_t) < cooldown:
            return

        # Adjust TTS speed
        is_urgent = min_distance < 2.0 or command == CMD_STOP
        if is_urgent and not self._last_urgent:
            self._va.set_rate(self._urgent_rate)
            self._last_urgent = True
        elif not is_urgent and self._last_urgent:
            self._va.set_rate(self._normal_rate)
            self._last_urgent = False

        # Speak the natural-language version
        text = self.SPEECH_TEXT.get(command, command)

        # Add proximity prefix for close obstacles
        if min_distance < 1.2:
            text = f"WARNING! {text}"
        elif min_distance < 2.0:
            text = f"Caution. {text}"

        self._va.speak(text)
        self._last_cmd = command
        self._last_t   = now
