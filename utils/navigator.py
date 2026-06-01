"""
navigator.py
------------
Navigation decision engine for the Smart Navigation Assistant.

Screen zones:
    ┌──────────┬──────────┬──────────┐
    │   LEFT   │  CENTER  │  RIGHT   │
    │  (0-33%) │ (33-67%) │ (67-100%)│
    └──────────┴──────────┴──────────┘

Decision priority:
    1. STOP  — any obstacle within DANGER_DISTANCE in CENTER zone
    2. MOVE LEFT  — obstacle in RIGHT zone
    3. MOVE RIGHT — obstacle in LEFT zone
    4. MOVE LEFT or MOVE RIGHT — obstacles in CENTER (pick open side)
    5. MOVE FORWARD — no obstacles detected

Author: AI Navigation Project
"""

import cv2
import numpy as np
from utils.distance_estimator import DANGER_DISTANCE

# ──────────────────────────────────────────────
# Navigation commands
# ──────────────────────────────────────────────
CMD_FORWARD     = "Move Forward"
CMD_LEFT        = "Move Left"
CMD_RIGHT       = "Move Right"
CMD_STOP        = "STOP !"
CMD_SLOW        = "Slow Down"
CMD_CLEAR       = "Path Clear"

# Zone boundary fractions of frame width
LEFT_BOUNDARY   = 0.33   # 0 → 33%  = LEFT zone
RIGHT_BOUNDARY  = 0.67   # 67% → 100% = RIGHT zone

# ──────────────────────────────────────────────
# Colour map for overlay
# ──────────────────────────────────────────────
CMD_COLORS = {
    CMD_FORWARD: (0,   200, 0),      # green
    CMD_LEFT:    (255, 165, 0),      # orange
    CMD_RIGHT:   (255, 165, 0),      # orange
    CMD_STOP:    (0,   0,   220),    # red
    CMD_SLOW:    (0,   165, 255),    # amber
    CMD_CLEAR:   (0,   220, 0),      # bright green
}

# Minimum confidence to consider a detection for navigation
MIN_NAV_CONFIDENCE = 0.40


class Navigator:
    """
    Analyses detections + distances to produce a navigation command
    and overlays HUD elements on the frame.
    """

    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self._last_command = CMD_CLEAR

    def update_frame_size(self, w: int, h: int):
        self.frame_width  = w
        self.frame_height = h

    # ──────────────────────────────────────────
    # Zone classification
    # ──────────────────────────────────────────
    def _zone(self, center_x: int) -> str:
        rel = center_x / self.frame_width
        if rel < LEFT_BOUNDARY:
            return "LEFT"
        elif rel < RIGHT_BOUNDARY:
            return "CENTER"
        else:
            return "RIGHT"

    # ──────────────────────────────────────────
    # Core decision function
    # ──────────────────────────────────────────
    def decide(self, detections: list[dict], distances: dict) -> str:
        """
        Compute navigation command from detections + distances.

        Parameters
        ----------
        detections : list of dicts from detector.detect()
        distances  : dict {index -> float (metres)} from distance_estimator

        Returns
        -------
        str : navigation command
        """
        if not detections:
            self._last_command = CMD_CLEAR
            return CMD_CLEAR

        left_obst   = []   # (distance, name)
        center_obst = []
        right_obst  = []

        for idx, det in enumerate(detections):
            if det["confidence"] < MIN_NAV_CONFIDENCE:
                continue

            zone = self._zone(det["center_x"])
            dist = distances.get(idx, 99.0)
            entry = (dist, det["class_name"])

            if zone == "LEFT":
                left_obst.append(entry)
            elif zone == "CENTER":
                center_obst.append(entry)
            else:
                right_obst.append(entry)

        # ── Priority 1: STOP if critically close in CENTER ────────────
        for dist, name in center_obst:
            if dist < 0.8:
                self._last_command = CMD_STOP
                return CMD_STOP

        # ── Priority 2: Danger in CENTER → choose open side ───────────
        if center_obst:
            min_center_dist = min(d for d, _ in center_obst)
            if min_center_dist <= DANGER_DISTANCE:
                # Try to route around the center obstacle
                left_clear  = not left_obst  or min(d for d, _ in left_obst)  > DANGER_DISTANCE
                right_clear = not right_obst or min(d for d, _ in right_obst) > DANGER_DISTANCE

                if left_clear and right_clear:
                    cmd = CMD_LEFT   # default: go left when both open
                elif left_clear:
                    cmd = CMD_LEFT
                elif right_clear:
                    cmd = CMD_RIGHT
                else:
                    cmd = CMD_STOP   # both sides blocked
                self._last_command = cmd
                return cmd

            # Center obstacle but still far — slow down
            if min_center_dist <= 3.0:
                self._last_command = CMD_SLOW
                return CMD_SLOW

        # ── Priority 3: Obstacle only on left → go right ──────────────
        if left_obst and not center_obst and not right_obst:
            self._last_command = CMD_RIGHT
            return CMD_RIGHT

        # ── Priority 4: Obstacle only on right → go left ──────────────
        if right_obst and not center_obst and not left_obst:
            self._last_command = CMD_LEFT
            return CMD_LEFT

        # ── Priority 5: Obstacles on sides but not center ─────────────
        if (left_obst or right_obst) and not center_obst:
            self._last_command = CMD_FORWARD
            return CMD_FORWARD

        # ── Default: clear path ────────────────────────────────────────
        self._last_command = CMD_CLEAR
        return CMD_CLEAR

    # ──────────────────────────────────────────
    # HUD overlay
    # ──────────────────────────────────────────
    def draw_hud(self, frame: np.ndarray, command: str,
                 detections: list[dict], distances: dict) -> np.ndarray:
        """
        Draw navigation HUD overlay on frame:
          - Zone dividers
          - Navigation command banner
          - Object list panel (name, confidence, distance)
          - Direction arrow

        Returns annotated frame.
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # ── Zone dividers ─────────────────────────────────────────────
        left_x  = int(w * LEFT_BOUNDARY)
        right_x = int(w * RIGHT_BOUNDARY)

        cv2.line(overlay, (left_x, 0),  (left_x, h),  (100, 100, 100), 1)
        cv2.line(overlay, (right_x, 0), (right_x, h), (100, 100, 100), 1)

        # Zone labels (top)
        for label, x in [("LEFT", 10), ("CENTER", left_x + 10), ("RIGHT", right_x + 10)]:
            cv2.putText(overlay, label, (x, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # ── Command banner (bottom centre) ───────────────────────────
        color  = CMD_COLORS.get(command, (255, 255, 255))
        banner = f"  {command}  "
        (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3)
        bx = (w - bw) // 2
        by = h - 30

        # Semi-transparent background
        cv2.rectangle(overlay,
                      (bx - 12, by - bh - 12),
                      (bx + bw + 12, by + 12),
                      (20, 20, 20), -1)
        cv2.putText(overlay, banner, (bx, by),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3)

        # ── Direction arrow ──────────────────────────────────────────
        self._draw_arrow(overlay, command, w, h)

        # ── Object info panel (top-right) ─────────────────────────────
        self._draw_object_panel(overlay, detections, distances, w)

        # Blend overlay with original for transparency
        alpha = 0.88
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def _draw_arrow(self, frame, command, w, h):
        """Draw a directional arrow icon near the command banner."""
        cx = w // 2
        cy = h - 110
        size = 28
        color = CMD_COLORS.get(command, (255, 255, 255))
        thick = 4

        if command == CMD_FORWARD or command == CMD_CLEAR:
            pts = np.array([[cx, cy - size],
                            [cx - size // 2, cy + size // 2],
                            [cx + size // 2, cy + size // 2]], np.int32)
            cv2.fillPoly(frame, [pts], color)

        elif command == CMD_LEFT:
            pts = np.array([[cx - size, cy],
                            [cx + size // 2, cy - size // 2],
                            [cx + size // 2, cy + size // 2]], np.int32)
            cv2.fillPoly(frame, [pts], color)

        elif command == CMD_RIGHT:
            pts = np.array([[cx + size, cy],
                            [cx - size // 2, cy - size // 2],
                            [cx - size // 2, cy + size // 2]], np.int32)
            cv2.fillPoly(frame, [pts], color)

        elif command == CMD_STOP:
            cv2.rectangle(frame,
                          (cx - size, cy - size),
                          (cx + size, cy + size),
                          color, -1)
            cv2.putText(frame, "!", (cx - 8, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        elif command == CMD_SLOW:
            # Double horizontal lines for "slow"
            for dy in [-8, 8]:
                cv2.line(frame,
                         (cx - size, cy + dy),
                         (cx + size, cy + dy),
                         color, thick)

    def _draw_object_panel(self, frame, detections, distances, frame_width):
        """Draw a small info panel listing detected objects."""
        if not detections:
            return

        panel_x = frame_width - 240
        panel_y = 35
        line_h  = 22

        cv2.rectangle(frame,
                      (panel_x - 8, panel_y - 20),
                      (frame_width - 5, panel_y + len(detections) * line_h + 5),
                      (20, 20, 20), -1)

        cv2.putText(frame, "DETECTED OBJECTS", (panel_x - 4, panel_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

        for idx, det in enumerate(detections):
            dist = distances.get(idx, 0.0)
            text = f"{det['class_name']}  {det['confidence']*100:.0f}%  {dist:.1f}m"
            y    = panel_y + (idx + 1) * line_h

            # Colour-code by distance
            if dist < 0.8:
                tc = (0, 0, 255)
            elif dist < 1.5:
                tc = (0, 100, 255)
            elif dist < 3.0:
                tc = (0, 165, 255)
            else:
                tc = (180, 255, 180)

            cv2.putText(frame, text, (panel_x - 4, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, tc, 1)
