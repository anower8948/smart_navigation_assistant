"""
navigator.py
------------
5-Direction Precision Navigation Engine
========================================

Direction commands (left → right):
    Move Left  |  Move Left Middle  |  Move Forward  |  Move Right Middle  |  Move Right

    These map directly to the 5 screen zones:
    ┌──────────┬────────────────┬──────────────┬─────────────────┬──────────┐
    │   FAR    │      MID       │    CENTER    │       MID       │   FAR    │
    │   LEFT   │     LEFT       │   FORWARD    │      RIGHT      │  RIGHT   │
    │  0–20%   │   20–40%       │   40–60%     │    60–80%       │ 80–100%  │
    │ Move Left│ Move Left Mid  │ Move Forward │ Move Right Mid  │Move Right│
    └──────────┴────────────────┴──────────────┴─────────────────┴──────────┘

Speed tiers (checked FIRST, highest priority):
    < 0.40m   → STOP
    0.40–0.69m → Walk Very Slowly
    0.70–1.99m → Walk Slowly
    >= 2.0m   → direction logic

Direction logic:
    - Each zone gets a danger score = sum(1/dist) for obstacles inside it
    - The SAFEST zone (lowest danger) becomes the recommended direction
    - Ties broken by preferring the zone closest to current path (center)

Author: AI Navigation Project
"""

import cv2
import numpy as np

# ── Direction Commands (left → right) ─────────────────────────────────────
CMD_STOP        = "STOP"
CMD_VERY_SLOW   = "Walk Very Slowly"
CMD_SLOW        = "Walk Slowly"
CMD_LEFT        = "Move Left"
CMD_LEFT_MID    = "Move Left Middle"
CMD_FORWARD     = "Move Forward"
CMD_RIGHT_MID   = "Move Right Middle"
CMD_RIGHT       = "Move Right"
CMD_CLEAR       = "Path Clear"

ALL_COMMANDS = [
    CMD_STOP, CMD_VERY_SLOW, CMD_SLOW,
    CMD_LEFT, CMD_LEFT_MID, CMD_FORWARD,
    CMD_RIGHT_MID, CMD_RIGHT, CMD_CLEAR
]

# ── 5 Zones (equal 20% slices) ────────────────────────────────────────────
# Each zone maps 1-to-1 to a direction command
ZONE_BOUNDS = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
Z_FL, Z_ML, Z_C, Z_MR, Z_FR = 0, 1, 2, 3, 4

# Zone → recommended direction when that zone is the safest
ZONE_TO_CMD = {
    Z_FL: CMD_LEFT,
    Z_ML: CMD_LEFT_MID,
    Z_C:  CMD_FORWARD,
    Z_MR: CMD_RIGHT_MID,
    Z_FR: CMD_RIGHT,
}

# ── Distance / Speed Thresholds ───────────────────────────────────────────
STOP_DIST      = 0.40   # < 0.40m  → STOP (uses global closest)
VERY_SLOW_DIST = 0.69   # 0.40–0.69m → Walk Very Slowly (center only)
SLOW_DIST      = 1.99   # 0.70–1.99m → Walk Slowly     (center only)
FREE_WALK_DIST = 2.0    # >= 2.0m  → full direction logic

MAX_NAV_DIST   = 6.0    # ignore obstacles beyond this

# ── Minimum confidence ────────────────────────────────────────────────────
MIN_CONF = 0.40

# ── Command colours (BGR) ─────────────────────────────────────────────────
CMD_COLORS = {
    CMD_STOP:       (0,   0,   240),   # red
    CMD_VERY_SLOW:  (0,   50,  220),   # deep red-orange
    CMD_SLOW:       (0,   120, 255),   # orange
    CMD_LEFT:       (0,   120, 255),   # orange
    CMD_LEFT_MID:   (0,   190, 255),   # amber
    CMD_FORWARD:    (0,   210, 0),     # green
    CMD_RIGHT_MID:  (0,   190, 255),   # amber
    CMD_RIGHT:      (0,   120, 255),   # orange
    CMD_CLEAR:      (0,   230, 100),   # bright green
}

# ── Arrow angles (degrees, 0=up, CW positive) ─────────────────────────────
CMD_ANGLES = {
    CMD_STOP:       None,
    CMD_VERY_SLOW:  None,
    CMD_SLOW:       None,
    CMD_LEFT:       -90,    # ← full left
    CMD_LEFT_MID:   -45,    # ↖ diagonal left
    CMD_FORWARD:    0,      # ↑ straight ahead
    CMD_RIGHT_MID:  +45,    # ↗ diagonal right
    CMD_RIGHT:      +90,    # → full right
    CMD_CLEAR:      0,      # ↑ straight ahead
}


class Navigator:
    """
    5-direction navigation engine.
    Picks the safest of 5 walking directions based on zone danger scores.
    """

    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        self.fw   = frame_width
        self.fh   = frame_height
        self._last = CMD_CLEAR

    def update_frame_size(self, w: int, h: int):
        self.fw = w
        self.fh = h

    def _zone(self, cx: int) -> int:
        """Map pixel x → zone index 0–4."""
        r = cx / self.fw
        for i in range(5):
            if r < ZONE_BOUNDS[i + 1]:
                return i
        return Z_FR

    @staticmethod
    def _danger(dists: list) -> float:
        """Danger score = sum(1/dist). Empty zone = 0."""
        return sum(1.0 / max(d, 0.1) for d in dists) if dists else 0.0

    # ── Core decision ──────────────────────────────────────────────────────
    def decide(self, detections: list[dict], distances: dict) -> str:

        # Collect obstacle distances per zone
        zone_dists: list[list] = [[] for _ in range(5)]
        for idx, det in enumerate(detections):
            if det["confidence"] < MIN_CONF:
                continue
            dist = distances.get(idx, 99.0)
            if dist > MAX_NAV_DIST:
                continue
            z = self._zone(det["center_x"])
            zone_dists[z].append(dist)

        # Closest obstacle in each zone
        zone_min = [min(zd, default=99.0) for zd in zone_dists]
        global_min = min(zone_min)
        cc_min     = zone_min[Z_C]

        # ── Speed tiers (priority over direction) ─────────────────────────
        if global_min < STOP_DIST:            # < 0.40m anywhere
            return self._set(CMD_STOP)
        if cc_min <= VERY_SLOW_DIST:          # 0.40–0.69m center
            return self._set(CMD_VERY_SLOW)
        if cc_min <= SLOW_DIST:               # 0.70–1.99m center
            return self._set(CMD_SLOW)

        # ── Direction logic (>= 2.0m) ─────────────────────────────────────
        # No obstacles at all → clear
        if global_min == 99.0:
            return self._set(CMD_CLEAR)

        # Danger score per zone
        ds = [self._danger(zone_dists[z]) for z in range(5)]

        # Find the safest zone (lowest danger score)
        # Tie-break: prefer zones closer to center (index 2)
        # Priority order for equal scores: C > ML > MR > FL > FR
        center_pref = [2, 1, 3, 0, 4]   # preference order (center-first)

        best_zone  = None
        best_score = float("inf")
        for z in center_pref:
            if ds[z] < best_score:
                best_score = ds[z]
                best_zone  = z

        return self._set(ZONE_TO_CMD[best_zone])

    def _set(self, cmd: str) -> str:
        self._last = cmd
        return cmd

    # ══════════════════════════════════════════════════════════════════════
    #  HUD Rendering
    # ══════════════════════════════════════════════════════════════════════

    def draw_hud(self, frame: np.ndarray, command: str,
                 detections: list[dict], distances: dict) -> np.ndarray:
        h, w  = frame.shape[:2]
        color = CMD_COLORS.get(command, (255, 255, 255))
        zx    = [int(w * b) for b in ZONE_BOUNDS]   # 6 x-boundaries

        # Danger scores for rendering
        zone_dists: list[list] = [[] for _ in range(5)]
        for idx, det in enumerate(detections):
            if det["confidence"] < MIN_CONF:
                continue
            dist = distances.get(idx, 99.0)
            if dist > MAX_NAV_DIST:
                continue
            zone_dists[self._zone(det["center_x"])].append(dist)
        ds = [self._danger(zone_dists[z]) for z in range(5)]

        TOP = 35
        BOT = h - 82

        # ── Zone danger tints ──────────────────────────────────────────────
        overlay = frame.copy()
        for z in range(5):
            if ds[z] > 0:
                intensity = min(ds[z] / 1.5, 1.0)
                cv2.rectangle(overlay,
                              (zx[z], TOP), (zx[z+1], BOT),
                              (0, 0, int(60 * intensity)), -1)

        # ── Safe path highlight (recommended zone = green) ─────────────────
        safe_z = self._cmd_to_zone(command)
        if safe_z is not None:
            cv2.rectangle(overlay,
                          (zx[safe_z], TOP), (zx[safe_z+1], BOT),
                          (0, 50, 0), -1)

        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

        # ── Zone divider lines ─────────────────────────────────────────────
        for x in zx[1:-1]:
            cv2.line(frame, (x, TOP), (x, BOT), (55, 55, 55), 1)

        # ── Zone direction labels ──────────────────────────────────────────
        zone_labels = ["◄ Left", "◄ Mid L", "Forward", "Mid R ►", "Right ►"]
        for z, lbl in enumerate(zone_labels):
            lx = (zx[z] + zx[z+1]) // 2
            ly = TOP + 18
            # colour: green if recommended, red if dangerous, grey otherwise
            if z == safe_z:
                lc = (0, 230, 80)
            elif ds[z] > 1.0:
                lc = (0, 60, 220)
            elif ds[z] > 0.25:
                lc = (0, 160, 255)
            else:
                lc = (90, 90, 90)

            # Danger score badge
            score_lbl = f"{ds[z]:.1f}" if ds[z] > 0 else ""
            (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(frame, lbl,
                        (lx - tw//2, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, lc, 1, cv2.LINE_AA)
            if score_lbl:
                cv2.putText(frame, score_lbl,
                            (lx - 8, ly + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            (120, 120, 120), 1, cv2.LINE_AA)

        # ── Danger bar (thin strip above banner) ──────────────────────────
        self._draw_danger_bar(frame, ds, zx, BOT + 2, w)

        # ── Bottom command banner ──────────────────────────────────────────
        banner_h = 80
        banner_y = h - banner_h
        cv2.rectangle(frame, (0, banner_y), (w, h), (10, 10, 10), -1)
        cv2.rectangle(frame, (0, banner_y), (8, h), color, -1)
        cv2.line(frame, (0, banner_y), (w, banner_y), color, 2)

        # Command text centred
        fs, thick = 1.45, 3
        (tw, th), _ = cv2.getTextSize(command, cv2.FONT_HERSHEY_DUPLEX, fs, thick)
        tx = (w - tw) // 2
        ty = banner_y + (banner_h + th) // 2 - 2
        cv2.putText(frame, command, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, fs, color, thick, cv2.LINE_AA)

        # ── Direction arrow ────────────────────────────────────────────────
        self._draw_arrow(frame, command, w, banner_y, color)

        # ── Nearest obstacle badge ─────────────────────────────────────────
        if distances:
            self._draw_dist_badge(frame, min(distances.values()), w, banner_y)

        return frame

    def _cmd_to_zone(self, command: str):
        """Return zone index for a direction command, or None."""
        return {v: k for k, v in ZONE_TO_CMD.items()}.get(command, None)

    def _draw_arrow(self, frame, command, w, banner_y, color):
        cx  = w // 2
        ay  = banner_y - 20
        sz  = 36

        if command == CMD_STOP:
            cv2.rectangle(frame, (cx-sz, ay-sz), (cx+sz, ay+sz), color, -1)
            cv2.putText(frame, "!", (cx-10, ay+sz//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5,
                        (255,255,255), 3, cv2.LINE_AA)
            return

        if command == CMD_VERY_SLOW:
            for off in [-sz//2, sz//4]:
                pts = np.array([[cx, ay+off],
                                [cx-sz, ay+off-sz//2],
                                [cx+sz, ay+off-sz//2]], np.int32)
                cv2.polylines(frame, [pts], False, color, 4, cv2.LINE_AA)
            cv2.putText(frame, "!!", (cx-16, ay+sz),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255,255,255), 2, cv2.LINE_AA)
            return

        if command == CMD_SLOW:
            pts = np.array([[cx, ay],
                            [cx-sz, ay-sz//2],
                            [cx+sz, ay-sz//2]], np.int32)
            cv2.polylines(frame, [pts], False, color, 4, cv2.LINE_AA)
            cv2.putText(frame, "!", (cx-8, ay+sz//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255,255,255), 2, cv2.LINE_AA)
            return

        # Rotated chevron arrow for all direction commands
        angle = CMD_ANGLES.get(command, 0)
        arrow_pts = np.array([
            [0,      -sz],
            [-sz//2,  sz//3],
            [-sz//5,  sz//3],
            [-sz//5,  sz],
            [ sz//5,  sz],
            [ sz//5,  sz//3],
            [ sz//2,  sz//3],
        ], dtype=np.float32)

        if angle != 0:
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            arrow_pts = (rot @ arrow_pts.T).T

        arrow_pts = (arrow_pts + [cx, ay]).astype(np.int32)
        cv2.fillPoly(frame, [arrow_pts], color)

    def _draw_dist_badge(self, frame, min_dist, w, banner_y):
        label = f"Nearest  {min_dist:.1f} m"
        if min_dist < 0.4:
            bc = (0, 0, 240)
        elif min_dist < 0.7:
            bc = (0, 50, 220)
        elif min_dist < 2.0:
            bc = (0, 120, 255)
        elif min_dist < 4.0:
            bc = (0, 190, 255)
        else:
            bc = (60, 210, 60)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        rx = w - tw - 18
        ry = banner_y - 12
        cv2.rectangle(frame, (rx-6, ry-th-4), (rx+tw+6, ry+6), (18,18,18), -1)
        cv2.putText(frame, label, (rx, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, bc, 2, cv2.LINE_AA)

    def _draw_danger_bar(self, frame, ds, zx, bar_y, w):
        bar_h = 6
        for z in range(5):
            x1, x2 = zx[z], zx[z+1]
            d = ds[z]
            if d < 0.25:
                c = (0, 180, 0)
            elif d < 1.0:
                c = (0, 200, 255)
            else:
                c = (0, 0, 220)
            cv2.rectangle(frame, (x1, bar_y), (x2, bar_y+bar_h), c, -1)
        cv2.rectangle(frame, (0, bar_y), (w, bar_y+bar_h), (40,40,40), 1)
