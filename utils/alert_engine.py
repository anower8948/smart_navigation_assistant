"""
alert_engine.py
---------------
Special situation alert engine.

Detects and announces high-priority situations beyond basic
Left/Right/Forward navigation:

  1. TRAFFIC LIGHT — detected + colour state (red/green/yellow)
  2. STOP SIGN     — announce "Stop sign ahead"
  3. CROSSWALK     — detected pedestrian crossing area
  4. FAST-MOVING OBJECT — object approaching rapidly (from tracker)
  5. CROWD DETECTED — 3+ people in frame simultaneously
  6. BLIND SPOT WARNING — object in extreme edge of frame (< 5% or > 95%)

Author: AI Navigation Project
"""

import cv2
import numpy as np
import time

# Cooldown between the same alert (seconds)
ALERT_COOLDOWN = 5.0

# Classes that trigger special alerts
TRAFFIC_LIGHT_CLASS = "Traffic Light"
STOP_SIGN_CLASS     = "Stop Sign"
PERSON_CLASS        = "Person"

# Crowd threshold
CROWD_THRESHOLD     = 4   # persons in one frame

# Blind-spot edge fractions
BLIND_SPOT_LEFT_EDGE  = 0.06
BLIND_SPOT_RIGHT_EDGE = 0.94


class AlertEngine:
    """
    Analyses detections + tracker data to fire special audio + visual alerts.
    """

    def __init__(self, voice_assistant=None):
        self._voice     = voice_assistant
        self._last_alert: dict[str, float] = {}   # alert_key -> timestamp

    # ── Cooldown helper ──────────────────────────────────────────────
    def _can_alert(self, key: str) -> bool:
        now = time.time()
        if now - self._last_alert.get(key, 0.0) >= ALERT_COOLDOWN:
            self._last_alert[key] = now
            return True
        return False

    def _speak(self, key: str, text: str):
        if self._can_alert(key) and self._voice:
            self._voice.speak(text)

    # ── Main analysis ────────────────────────────────────────────────
    def analyse(self, detections: list[dict], distances: dict,
                active_tracks=None, frame_width: int = 1280) -> list[str]:
        """
        Run all alert checks and return a list of active alert strings.
        These are displayed on screen and spoken.

        Parameters
        ----------
        detections    : list from detector.detect()
        distances     : {idx -> float}
        active_tracks : list of Track objects from tracker (optional)
        frame_width   : frame width in pixels

        Returns
        -------
        list of str alert messages (active this frame)
        """
        alerts = []

        person_count    = 0
        traffic_lights  = []
        stop_signs      = []

        for idx, det in enumerate(detections):
            name = det["class_name"]
            dist = distances.get(idx, 99.0)
            cx   = det["center_x"]

            # ── Traffic light ──────────────────────────────────────
            if name == TRAFFIC_LIGHT_CLASS:
                traffic_lights.append((dist, det))

            # ── Stop sign ──────────────────────────────────────────
            elif name == STOP_SIGN_CLASS and dist < 6.0:
                stop_signs.append(dist)

            # ── Person count ───────────────────────────────────────
            elif name == PERSON_CLASS:
                person_count += 1

            # ── Blind spot: object near extreme edge ───────────────
            rel_x = cx / frame_width
            if rel_x < BLIND_SPOT_LEFT_EDGE and dist < 3.0:
                msg = f"Object on far LEFT — {det['class_name']}"
                alerts.append(msg)
                self._speak("blind_left", "Caution. Object on your far left.")

            elif rel_x > BLIND_SPOT_RIGHT_EDGE and dist < 3.0:
                msg = f"Object on far RIGHT — {det['class_name']}"
                alerts.append(msg)
                self._speak("blind_right", "Caution. Object on your far right.")

        # ── Traffic light alert ────────────────────────────────────
        if traffic_lights:
            nearest_dist = min(d for d, _ in traffic_lights)
            if nearest_dist < 8.0:
                alerts.append(f"Traffic Light ahead ({nearest_dist:.1f}m)")
                self._speak("traffic_light", "Traffic light ahead. Proceed with caution.")

        # ── Stop sign alert ────────────────────────────────────────
        if stop_signs:
            nearest = min(stop_signs)
            alerts.append(f"STOP SIGN ({nearest:.1f}m)")
            self._speak("stop_sign", "Stop sign ahead. Please stop.")

        # ── Crowd alert ────────────────────────────────────────────
        if person_count >= CROWD_THRESHOLD:
            alerts.append(f"Crowded area — {person_count} people")
            self._speak("crowd", f"Crowded area. {person_count} people detected.")

        # ── Fast-approaching object (from tracker) ─────────────────
        if active_tracks:
            for track in active_tracks:
                if track.approach_status == "APPROACHING_FAST" \
                        and track.distance < 4.0:
                    msg = f"{track.class_name} #{track.id} approaching fast!"
                    alerts.append(msg)
                    self._speak(f"fast_{track.id}",
                                f"{track.class_name} approaching fast. Move away!")

        return alerts

    # ── Draw alerts on frame ────────────────────────────────────────────────
    def draw_alerts(self, frame: np.ndarray,
                    alerts: list[str]) -> np.ndarray:
        """
        Draw clean alert pill badges stacked below the top status bar.

        Design (ASCII-only, no Unicode symbols):
          [ALERT]  Person #4 approaching fast!
          [ALERT]  Crowded area - 4 people
        """
        if not alerts:
            return frame

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.46
        font_thick = 1
        tag_scale  = 0.38
        pad_x      = 7
        pad_y      = 5
        y_cursor   = 42    # below the top status bar
        gap        = 5     # gap between badges

        def _color(msg):
            m = msg.lower()
            if "approaching fast" in m or "stop sign" in m:
                return (0, 40, 210)    # red   (BGR)
            if "crowded" in m or "traffic" in m:
                return (0, 130, 220)   # orange
            return (0, 170, 200)       # amber

        for alert in alerts[:5]:
            color = _color(alert)
            tag   = "ALERT"

            # Measure tag chip
            (tag_w, tag_h), _ = cv2.getTextSize(tag, font, tag_scale, 1)
            tag_box_w = tag_w + 10

            # Measure main text
            (tw, th), _ = cv2.getTextSize(alert, font, font_scale, font_thick)

            total_w = tag_box_w + pad_x + tw + pad_x
            box_h   = th + pad_y * 2

            bx, by = 8, y_cursor

            # Dark pill background
            cv2.rectangle(frame,
                          (bx, by),
                          (bx + total_w, by + box_h),
                          (18, 18, 18), -1)

            # Left colour stripe
            cv2.rectangle(frame,
                          (bx, by),
                          (bx + 3, by + box_h),
                          color, -1)

            # Coloured tag chip
            cv2.rectangle(frame,
                          (bx + 3, by),
                          (bx + 3 + tag_box_w, by + box_h),
                          color, -1)
            cv2.putText(frame, tag,
                        (bx + 7, by + box_h - pad_y - 1),
                        font, tag_scale,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Main alert text
            cv2.putText(frame, alert,
                        (bx + 3 + tag_box_w + pad_x,
                         by + box_h - pad_y - 1),
                        font, font_scale,
                        (220, 220, 220), font_thick, cv2.LINE_AA)

            y_cursor += box_h + gap

        return frame
