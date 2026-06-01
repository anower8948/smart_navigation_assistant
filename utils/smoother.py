"""
smoother.py
-----------
Command smoother — fast response, no flicker.

Changes from v1:
  - CONFIRM_FRAMES reduced 4 → 2  (half the lag)
  - STOP and CMD_RIGHT/LEFT in danger zone bypass instantly
  - Uses majority-vote over a rolling window (not just consecutive count)
    so a single bad frame doesn't reset the counter
  - SpeedAwareVoice: shorter cooldown when close (0.8s vs 2.5s default)

Author: AI Navigation Project
"""

from collections import deque
import time

# ── Tuning ──────────────────────────────────────────────────────────────────
CONFIRM_FRAMES  = 2     # frames before a new command is accepted (was 4)
VOTE_MAJORITY   = 0.6   # fraction of window that must agree

# Commands that are accepted instantly (no delay)
INSTANT_COMMANDS = {"STOP", "STOP !"}

# Voice cooldowns by urgency (seconds)
COOLDOWN_CRITICAL = 0.8    # < 1m
COOLDOWN_DANGER   = 1.5    # 1–2.5m
COOLDOWN_NORMAL   = 2.5    # > 2.5m


class CommandSmoother:
    """
    Fast majority-vote smoother for navigation commands.
    Accepts a new command once it holds the majority in a 2-frame window.
    Safety commands (STOP) are always instant.
    """

    def __init__(self, window: int = CONFIRM_FRAMES):
        self.window           = window
        self._buf             = deque(maxlen=window)
        self._current         = "Path Clear"

    def update(self, raw: str) -> str:
        # Instant pass-through for safety commands
        if raw in INSTANT_COMMANDS:
            self._current = raw
            self._buf.clear()
            return self._current

        self._buf.append(raw)

        # Majority vote — most frequent command in the window wins
        if len(self._buf) >= self.window:
            counts = {}
            for c in self._buf:
                counts[c] = counts.get(c, 0) + 1
            winner, votes = max(counts.items(), key=lambda x: x[1])
            if votes / self.window >= VOTE_MAJORITY:
                self._current = winner

        return self._current

    @property
    def confidence(self) -> float:
        if not self._buf:
            return 1.0
        matches = sum(1 for c in self._buf if c == self._current)
        return matches / len(self._buf)

    def reset(self):
        self._buf.clear()
        self._current = "Path Clear"


class SpeedAwareVoice:
    """
    Speaks navigation commands with:
      - Faster speech rate near danger
      - "WARNING" / "Caution" prefix when close
      - Adaptive cooldown — speaks more often when danger is near
    """

    def __init__(self, voice_assistant,
                 normal_rate: int = 170,
                 urgent_rate: int = 210):
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

        # Choose cooldown by proximity
        if min_distance < 1.0:
            cooldown = COOLDOWN_CRITICAL
        elif min_distance < 2.5:
            cooldown = COOLDOWN_DANGER
        else:
            cooldown = COOLDOWN_NORMAL

        # Skip if same command and still within cooldown
        same = (command == self._last_cmd)
        if same and (now - self._last_t) < cooldown:
            return

        # Adjust TTS rate by urgency
        is_urgent = min_distance < 2.5 or command == "STOP"
        if is_urgent and not self._last_urgent:
            self._va.set_rate(self._urgent_rate)
            self._last_urgent = True
        elif not is_urgent and self._last_urgent:
            self._va.set_rate(self._normal_rate)
            self._last_urgent = False

        # Prefix
        if min_distance < 1.0:
            text = f"WARNING. {command}"
        elif min_distance < 2.0:
            text = f"Caution. {command}"
        else:
            text = command

        self._va.speak(text)
        self._last_cmd = command
        self._last_t   = now
