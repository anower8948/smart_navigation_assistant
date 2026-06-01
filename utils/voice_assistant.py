"""
voice_assistant.py
------------------
Text-to-speech guidance using pyttsx3.
Speaks navigation commands asynchronously so that it does not
block the video processing loop.

Fallback: if pyttsx3 fails (e.g. no audio device in a server/headless
environment), the module silently degrades — video still works.

Author: AI Navigation Project
"""

import threading
import time
import queue
import sys

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("[VoiceAssistant] pyttsx3 not installed — voice disabled.")


# Minimum seconds between repeated identical commands
REPEAT_COOLDOWN = 2.5   # seconds


class VoiceAssistant:
    """
    Non-blocking TTS wrapper.

    Internally runs a background daemon thread that reads from a queue
    and speaks each command. Commands that are identical to the previous
    command within REPEAT_COOLDOWN seconds are silently dropped to avoid
    speech spam.
    """

    def __init__(self, rate: int = 175, volume: float = 1.0,
                 enabled: bool = True):
        """
        Parameters
        ----------
        rate    : speech rate (words per minute)
        volume  : 0.0 – 1.0
        enabled : set False to disable voice globally
        """
        self.enabled        = enabled and PYTTSX3_AVAILABLE
        self._rate          = rate
        self._volume        = volume
        self._queue         = queue.Queue(maxsize=4)
        self._last_command  = None
        self._last_spoken_t = 0.0
        self._engine        = None
        self._thread        = None

        if self.enabled:
            self._init_engine()
            self._start_thread()

    # ──────────────────────────────────────────
    # Engine initialisation
    # ──────────────────────────────────────────
    def _init_engine(self):
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   self._rate)
            self._engine.setProperty("volume", self._volume)

            # Try to pick a pleasant voice
            voices = self._engine.getProperty("voices")
            for v in voices:
                if "english" in v.name.lower() or "en_" in v.id.lower():
                    self._engine.setProperty("voice", v.id)
                    break

            print(f"[VoiceAssistant] TTS engine ready  "
                  f"(rate={self._rate}, volume={self._volume})")
        except Exception as e:
            print(f"[VoiceAssistant] Engine init failed: {e} — voice disabled.")
            self.enabled = False

    # ──────────────────────────────────────────
    # Background speaker thread
    # ──────────────────────────────────────────
    def _start_thread(self):
        self._thread = threading.Thread(
            target=self._speaker_loop,
            daemon=True,
            name="VoiceSpeaker"
        )
        self._thread.start()

    def _speaker_loop(self):
        """Daemon thread: pop commands from queue and speak them."""
        while True:
            try:
                text = self._queue.get(timeout=0.5)
                if text is None:
                    break
                if self._engine:
                    self._engine.say(text)
                    self._engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VoiceAssistant] Speak error: {e}")

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────
    def speak(self, command: str):
        """
        Queue a navigation command for speech.
        Drops if identical to last command within cooldown period.

        Parameters
        ----------
        command : str — navigation instruction to speak
        """
        if not self.enabled:
            return

        now = time.time()
        same_as_last = (command == self._last_command)
        within_cooldown = (now - self._last_spoken_t) < REPEAT_COOLDOWN

        if same_as_last and within_cooldown:
            return   # Avoid repeating the same instruction

        # Drop oldest if queue is full (non-blocking)
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass

        self._queue.put(command)
        self._last_command  = command
        self._last_spoken_t = now

    def stop(self):
        """Gracefully shut down the speaker thread."""
        if self.enabled and self._thread and self._thread.is_alive():
            self._queue.put(None)   # sentinel
            self._thread.join(timeout=2.0)
            print("[VoiceAssistant] Stopped.")

    def set_rate(self, rate: int):
        """Change speech rate on the fly."""
        self._rate = rate
        if self._engine:
            self._engine.setProperty("rate", rate)

    def set_volume(self, volume: float):
        """Change volume on the fly (0.0 – 1.0)."""
        self._volume = max(0.0, min(1.0, volume))
        if self._engine:
            self._engine.setProperty("volume", self._volume)
