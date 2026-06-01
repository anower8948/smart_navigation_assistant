"""
navigator.py
------------
Navigation decision engine — clear, fast, and stable.

Screen zones (tighter CENTER = more decisive Left/Right commands):
    ┌───────────┬──────────────┬───────────┐
    │   LEFT    │    CENTER    │   RIGHT   │
    │  (0-28%)  │  (28%-72%)  │ (72-100%) │
    └───────────┴──────────────┴───────────┘

Decision priority (highest → lowest):
    1. STOP       — center obstacle < 1.0m
    2. MOVE LEFT  — center obstacle, right side has more space
    3. MOVE RIGHT — center obstacle, left side has more space
    4. MOVE RIGHT — obstacle on LEFT only (within range)
    5. MOVE LEFT  — obstacle on RIGHT only (within range)
    6. MOVE FORWARD — obstacles exist but none blocking path
    7. PATH CLEAR   — no obstacles at all

Key improvements over v1:
    - Narrower CENTER zone (28–72%) → Left/Right fire more often
    - Only obstacles within MAX_NAV_DIST (5m) affect navigation
    - Clear side = truly empty OR farther than center obstacle by margin
    - Removed CMD_SLOW (was causing confusion, replaced by Move Fwd)
    - Bigger, cleaner command banner with large arrow
    - Object panel removed from HUD (shown in top bar by main.py)

Author: AI Navigation Project
"""

import cv2
import numpy as np

# ── Navigation commands ────────────────────────────────────────────────────
CMD_FORWARD = "Move Forward"
CMD_LEFT    = "Move Left"
CMD_RIGHT   = "Move Right"
CMD_STOP    = "STOP"
CMD_CLEAR   = "Path Clear"

# ── Zone boundaries (fraction of frame width) ─────────────────────────────
# Narrower CENTER = more decisive Left/Right output
LEFT_BOUNDARY  = 0.28   # 0  → 28%  = LEFT zone
RIGHT_BOUNDARY = 0.72   # 72% → 100% = RIGHT zone

# ── Distance thresholds (metres) ──────────────────────────────────────────
STOP_DIST    = 1.0    # STOP if center obstacle closer than this
MAX_NAV_DIST = 5.0    # ignore obstacles farther than this
MARGIN       = 0.5    # side must be this much farther than center to be "clear"

# ── Minimum confidence for navigation decisions ────────────────────────────
MIN_CONF = 0.42

# ── Command colours (BGR) ─────────────────────────────────────────────────
CMD_COLORS = {
    CMD_FORWARD: (0,   210, 0),
    CMD_LEFT:    (0,   165, 255),
    CMD_RIGHT:   (0,   165, 255),
    CMD_STOP:    (0,   0,   240),
    CMD_CLEAR:   (0,   230, 100),
}


class Navigator:
    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        self.fw = frame_width
        self.fh = frame_height
        self._last = CMD_CLEAR

    def update_frame_size(self, w: int, h: int):
        self.fw = w
        self.fh = h

    # ── Zone helper ────────────────────────────────────────────────────────
    def _zone(self, cx: int) -> str:
        r = cx / self.fw
        if r < LEFT_BOUNDARY:
            return "L"
        elif r < RIGHT_BOUNDARY:
            return "C"
        else:
            return "R"

    # ── Core decision ──────────────────────────────────────────────────────
    def decide(self, detections: list[dict], distances: dict) -> str:
        """
        Return a navigation command based on obstacle zones and distances.
        Only considers obstacles within MAX_NAV_DIST metres.
        """
        left_dists   = []
        center_dists = []
        right_dists  = []

        for idx, det in enumerate(detections):
            if det["confidence"] < MIN_CONF:
                continue
            dist = distances.get(idx, 99.0)
            if dist > MAX_NAV_DIST:
                continue          # too far away — ignore
            zone = self._zone(det["center_x"])
            if zone == "L":
                left_dists.append(dist)
            elif zone == "C":
                center_dists.append(dist)
            else:
                right_dists.append(dist)

        # Closest obstacle in each zone (99 = empty)
        cl = min(left_dists,   default=99.0)
        cc = min(center_dists, default=99.0)
        cr = min(right_dists,  default=99.0)

        # ── 1. STOP ────────────────────────────────────────────────────────
        if cc < STOP_DIST:
            return self._set(CMD_STOP)

        # ── 2. CENTER blocked → steer to open side ─────────────────────────
        if center_dists:
            # A side is "clear" if it has no obstacle OR its closest is
            # significantly farther than the centre obstacle
            left_clear  = (cl == 99.0) or (cl > cc + MARGIN)
            right_clear = (cr == 99.0) or (cr > cc + MARGIN)

            if left_clear and right_clear:
                # Both open — pick side with MORE space
                cmd = CMD_LEFT if cl >= cr else CMD_RIGHT
            elif left_clear:
                cmd = CMD_LEFT
            elif right_clear:
                cmd = CMD_RIGHT
            else:
                cmd = CMD_STOP   # all zones blocked
            return self._set(cmd)

        # ── 3. Side-only obstacles ─────────────────────────────────────────
        if left_dists and not right_dists:
            return self._set(CMD_RIGHT)
        if right_dists and not left_dists:
            return self._set(CMD_LEFT)
        if left_dists and right_dists:
            # Both sides occupied, center clear → go toward more space
            cmd = CMD_RIGHT if cl < cr else CMD_LEFT
            return self._set(cmd)

        # ── 4. No nearby obstacles ─────────────────────────────────────────
        if not detections:
            return self._set(CMD_CLEAR)
        return self._set(CMD_FORWARD)

    def _set(self, cmd: str) -> str:
        self._last = cmd
        return cmd

    # ── HUD overlay ────────────────────────────────────────────────────────
    def draw_hud(self, frame: np.ndarray, command: str,
                 detections: list[dict], distances: dict) -> np.ndarray:
        """
        Draw a clean, readable navigation HUD:
          - Subtle zone dividers with coloured highlights
          - Large command banner at bottom
          - Big directional arrow above banner
          - Danger distance bar on left edge
        """
        h, w = frame.shape[:2]
        color = CMD_COLORS.get(command, (255, 255, 255))

        left_x  = int(w * LEFT_BOUNDARY)
        right_x = int(w * RIGHT_BOUNDARY)

        # ── Zone dividers ──────────────────────────────────────────────────
        # Highlight the safe zone with a faint tint
        overlay = frame.copy()
        # Dim the blocked zones slightly
        if command == CMD_LEFT:
            cv2.rectangle(overlay, (right_x, 35), (w, h-80),
                          (0, 0, 60), -1)
        elif command == CMD_RIGHT:
            cv2.rectangle(overlay, (0, 35), (left_x, h-80),
                          (0, 0, 60), -1)
        elif command == CMD_STOP:
            cv2.rectangle(overlay, (left_x, 35), (right_x, h-80),
                          (0, 0, 80), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Zone lines
        cv2.line(frame, (left_x,  35), (left_x,  h-80), (80, 80, 80), 1)
        cv2.line(frame, (right_x, 35), (right_x, h-80), (80, 80, 80), 1)

        # Zone labels — only LEFT and RIGHT, not CENTER (less clutter)
        cv2.putText(frame, "L", (left_x // 2 - 6, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        cv2.putText(frame, "R", (right_x + (w - right_x) // 2 - 6, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

        # ── Bottom command banner ──────────────────────────────────────────
        banner_h = 72
        banner_y = h - banner_h

        # Solid colour background bar
        cv2.rectangle(frame, (0, banner_y), (w, h), (12, 12, 12), -1)
        # Coloured left accent stripe
        cv2.rectangle(frame, (0, banner_y), (6, h), color, -1)

        # Command text — centred, large
        font_scale = 1.6
        thickness  = 3
        (tw, th), _ = cv2.getTextSize(
            command, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        tx = (w - tw) // 2
        ty = banner_y + (banner_h + th) // 2

        cv2.putText(frame, command, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness,
                    cv2.LINE_AA)

        # ── Large directional arrow ────────────────────────────────────────
        self._draw_arrow(frame, command, w, banner_y, color)

        # ── Closest obstacle distance badge ───────────────────────────────
        if distances:
            min_d = min(distances.values())
            self._draw_dist_badge(frame, min_d, w, banner_y)

        return frame

    def _draw_arrow(self, frame, command, w, banner_y, color):
        """Draw a large, clear arrow above the command banner."""
        cx   = w // 2
        ay   = banner_y - 20   # arrow tip y
        size = 40              # larger than before

        if command in (CMD_FORWARD, CMD_CLEAR):
            # Up arrow
            pts = np.array([
                [cx,        ay - size],
                [cx - size, ay + size // 2],
                [cx - size // 3, ay + size // 2],
                [cx - size // 3, ay + size],
                [cx + size // 3, ay + size],
                [cx + size // 3, ay + size // 2],
                [cx + size, ay + size // 2],
            ], np.int32)
            cv2.fillPoly(frame, [pts], color)

        elif command == CMD_LEFT:
            # Left arrow
            pts = np.array([
                [cx - size,      ay],
                [cx - size // 2, ay - size],
                [cx - size // 2, ay - size // 3],
                [cx + size,      ay - size // 3],
                [cx + size,      ay + size // 3],
                [cx - size // 2, ay + size // 3],
                [cx - size // 2, ay + size],
            ], np.int32)
            cv2.fillPoly(frame, [pts], color)

        elif command == CMD_RIGHT:
            # Right arrow (mirror)
            pts = np.array([
                [cx + size,      ay],
                [cx + size // 2, ay - size],
                [cx + size // 2, ay - size // 3],
                [cx - size,      ay - size // 3],
                [cx - size,      ay + size // 3],
                [cx + size // 2, ay + size // 3],
                [cx + size // 2, ay + size],
            ], np.int32)
            cv2.fillPoly(frame, [pts], color)

        elif command == CMD_STOP:
            # Bold red STOP octagon-ish shape
            cv2.rectangle(frame,
                           (cx - size, ay - size),
                           (cx + size, ay + size),
                           color, -1)
            cv2.putText(frame, "!", (cx - 10, ay + size // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4,
                        (255, 255, 255), 3, cv2.LINE_AA)

    def _draw_dist_badge(self, frame, min_dist: float, w: int, banner_y: int):
        """Show the closest obstacle distance as a badge on the right."""
        label = f"Nearest: {min_dist:.1f}m"
        if min_dist < 1.0:
            bc = (0, 0, 220)
        elif min_dist < 2.5:
            bc = (0, 100, 255)
        elif min_dist < 4.0:
            bc = (0, 200, 255)
        else:
            bc = (0, 200, 100)

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        rx = w - tw - 20
        ry = banner_y - 14

        cv2.rectangle(frame, (rx - 6, ry - th - 4),
                       (rx + tw + 6, ry + 6), (20, 20, 20), -1)
        cv2.putText(frame, label, (rx, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bc, 2, cv2.LINE_AA)
