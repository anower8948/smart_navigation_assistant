"""
smoother.py
-----------
Command smoother for 5-direction + speed-tier navigation.

Per-command confirm windows:
  STOP / Walk Very Slowly → 1 frame (instant — safety)
  Walk Slowly             → 2 frames
  All direction commands  → 2 frames
  Path Clear              → 3 frames (most stable)

Author: AI Navigation Project
"""

from collections import deque
import time

from utils.navigator import (
    CMD_STOP, CMD_VERY_SLOW, CMD_SLOW,
    CMD_LEFT, CMD_LEFT_MID, CMD_FORWARD,
    CMD_RIGHT_MID, CMD_RIGHT, CMD_CLEAR
)

CONFIRM_WINDOWS = {
    CMD_STOP:       1,
    CMD_VERY_SLOW:  1,
    CMD_SLOW:       2,
    CMD_LEFT:       2,
    CMD_LEFT_MID:   2,
    CMD_FORWARD:    2,
    CMD_RIGHT_MID:  2,
    CMD_RIGHT:      2,
    CMD_CLEAR:      3,
}

DEFAULT_WINDOW = 2
INSTANT_CMDS   = {CMD_STOP, CMD_VERY_SLOW}

COOLDOWN_CRITICAL = 0.8
COOLDOWN_DANGER   = 1.5
COOLDOWN_NORMAL   = 2.8


class CommandSmoother:
    def __init__(self):
        self._buf     = deque(maxlen=6)
        self._current = CMD_CLEAR

    def update(self, raw: str) -> str:
        if raw in INSTANT_CMDS:
            self._current = raw
            self._buf.clear()
            return self._current

        self._buf.append(raw)
        win    = CONFIRM_WINDOWS.get(raw, DEFAULT_WINDOW)
        recent = list(self._buf)[-win:]

        if len(recent) >= win:
            counts = {}
            for c in recent:
                counts[c] = counts.get(c, 0) + 1
            winner, votes = max(counts.items(), key=lambda x: x[1])
            if votes >= (win + 1) // 2:
                self._current = winner

        return self._current

    @property
    def confidence(self) -> float:
        if not self._buf:
            return 1.0
        win    = CONFIRM_WINDOWS.get(self._current, DEFAULT_WINDOW)
        recent = list(self._buf)[-win:]
        return sum(1 for c in recent if c == self._current) / max(len(recent), 1)

    def reset(self):
        self._buf.clear()
        self._current = CMD_CLEAR


class SpeedAwareVoice:
    SPEECH_TEXT = {
        CMD_STOP:       "Stop! Obstacle very close.",
        CMD_VERY_SLOW:  "Walk very slowly. Obstacle extremely close.",
        CMD_SLOW:       "Walk slowly. Obstacle nearby.",
        CMD_LEFT:       "Move Left.",
        CMD_LEFT_MID:   "Move Left Middle.",
        CMD_FORWARD:    "Move Forward.",
        CMD_RIGHT_MID:  "Move Right Middle.",
        CMD_RIGHT:      "Move Right.",
        CMD_CLEAR:      "Path clear. Move Forward.",
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

        if min_distance < 1.2:
            cooldown = COOLDOWN_CRITICAL
        elif min_distance < 2.5:
            cooldown = COOLDOWN_DANGER
        else:
            cooldown = COOLDOWN_NORMAL

        if command == self._last_cmd and (now - self._last_t) < cooldown:
            return

        is_urgent = min_distance < 2.0 or command == CMD_STOP
        if is_urgent and not self._last_urgent:
            self._va.set_rate(self._urgent_rate)
            self._last_urgent = True
        elif not is_urgent and self._last_urgent:
            self._va.set_rate(self._normal_rate)
            self._last_urgent = False

        text = self.SPEECH_TEXT.get(command, command)
        if min_distance < 0.4:
            text = f"WARNING! {text}"
        elif min_distance < 2.0:
            text = f"Caution. {text}"

        self._va.speak(text)
        self._last_cmd = command
        self._last_t   = now
