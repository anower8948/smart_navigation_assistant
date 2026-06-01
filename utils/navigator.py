"""
navigator.py
------------
5-Zone Precision Navigation Engine
===================================
Designed for real street walking — gives gentle diagonal guidance
instead of harsh hard-left/hard-right commands.

5 Zones (equal 20% slices of frame width):
    ┌────────┬──────────┬──────────┬──────────┬────────┐
    │  FAR   │   MID    │  CENTER  │   MID    │  FAR   │
    │  LEFT  │   LEFT   │          │  RIGHT   │ RIGHT  │
    │ 0-20%  │ 20-40%   │ 40-60%   │ 60-80%   │80-100% │
    └────────┴──────────┴──────────┴──────────┴────────┘

7 Commands (priority order):
    1. STOP             — obstacle < STOP_DIST in CENTER
    2. Move Left        — CENTER/MID-LEFT blocked, far-left is open
    3. Slight Left      — CENTER partially blocked, mid-left is cleaner
    4. Move Right       — CENTER/MID-RIGHT blocked, far-right is open
    5. Slight Right     — CENTER partially blocked, mid-right is cleaner
    6. Move Forward     — obstacles exist but path is walkable
    7. Path Clear       — no obstacles within range

Decision algorithm:
    - Each zone gets a "danger score" = sum of (1/distance) for all
      obstacles in that zone (closer obstacle = higher danger)
    - CENTER danger is the primary trigger
    - LEFT vs RIGHT decision compares weighted danger of the two halves
    - Diagonal (Slight) vs Full turn depends on how blocked CENTER is
      and how clear the mid zones are

Author: AI Navigation Project
"""

import cv2
import numpy as np

# ── 7 Navigation Commands ──────────────────────────────────────────────────
CMD_STOP         = "STOP"
CMD_VERY_SLOW    = "Walk Very Slowly"
CMD_SLOW         = "Walk Slowly"
CMD_LEFT         = "Move Left"
CMD_SLIGHT_LEFT  = "Slight Left"
CMD_RIGHT        = "Move Right"
CMD_SLIGHT_RIGHT = "Slight Right"
CMD_FORWARD      = "Move Forward"
CMD_CLEAR        = "Path Clear"

# Export list for other modules
ALL_COMMANDS = [
    CMD_STOP, CMD_VERY_SLOW, CMD_SLOW,
    CMD_LEFT, CMD_SLIGHT_LEFT,
    CMD_RIGHT, CMD_SLIGHT_RIGHT,
    CMD_FORWARD, CMD_CLEAR
]

# ── Zone boundaries (5 equal zones, 20% each) ─────────────────────────────
# Zone indices: 0=FAR_LEFT  1=MID_LEFT  2=CENTER  3=MID_RIGHT  4=FAR_RIGHT
ZONE_BOUNDS = [0.0, 0.20, 0.40, 0.60, 0.80, 1.0]
ZONE_NAMES  = ["FAR_LEFT", "MID_LEFT", "CENTER", "MID_RIGHT", "FAR_RIGHT"]
Z_FL, Z_ML, Z_C, Z_MR, Z_FR = 0, 1, 2, 3, 4

# ── Distance thresholds (metres) ──────────────────────────────────────────
#  Speed tiers (user-defined rules):
#    < 0.40m          →  STOP
#    0.40 – 0.69m     →  Walk Very Slowly
#    0.70 – 1.99m     →  Walk Slowly
#    >= 2.0m          →  direction logic (free to walk)
STOP_DIST      = 0.40   # hard stop threshold
VERY_SLOW_DIST = 0.69   # walk very slowly upper bound
SLOW_DIST      = 1.99   # walk slowly upper bound
FREE_WALK_DIST = 2.0    # >= this → full direction navigation

#  Direction / zone thresholds
SLIGHT_DIST    = 3.0    # mild centre blockage → slight turn
FULL_TURN_DIST = 5.0    # strong centre blockage → full turn
MAX_NAV_DIST   = 6.0    # ignore obstacles beyond this distance

# ── Danger score thresholds ────────────────────────────────────────────────
# danger_score = sum(1/dist) for all obstacles in zone
# Higher = more dangerous
DANGER_SLIGHT   = 0.25   # mild blockage → slight turn
DANGER_FULL     = 0.60   # significant blockage → full turn

# ── Minimum detection confidence for navigation ────────────────────────────
MIN_CONF = 0.40

# ── Command colours (BGR) ─────────────────────────────────────────────────
CMD_COLORS = {
    CMD_STOP:         (0,   0,   240),   # red
    CMD_VERY_SLOW:    (0,   60,  220),   # deep red-orange
    CMD_SLOW:         (0,   140, 255),   # orange
    CMD_LEFT:         (0,   140, 255),   # deep orange
    CMD_SLIGHT_LEFT:  (0,   200, 255),   # amber
    CMD_RIGHT:        (0,   140, 255),   # deep orange
    CMD_SLIGHT_RIGHT: (0,   200, 255),   # amber
    CMD_FORWARD:      (0,   210, 0),     # green
    CMD_CLEAR:        (0,   230, 100),   # bright green
}

# ── Arrow directions for HUD (angle in degrees, 0=up, CW positive) ────────
CMD_ANGLES = {
    CMD_STOP:         None,    # special stop icon
    CMD_VERY_SLOW:    None,    # special slow icon
    CMD_SLOW:         None,    # special slow icon
    CMD_LEFT:         -90,     # full left
    CMD_SLIGHT_LEFT:  -45,     # diagonal up-left
    CMD_RIGHT:        +90,     # full right
    CMD_SLIGHT_RIGHT: +45,     # diagonal up-right
    CMD_FORWARD:      0,       # straight up
    CMD_CLEAR:        0,       # straight up
}


class Navigator:
    """
    5-zone precision navigation engine.
    Produces 7 graduated commands for smooth real-world walking guidance.
    """

    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        self.fw   = frame_width
        self.fh   = frame_height
        self._last = CMD_CLEAR

    def update_frame_size(self, w: int, h: int):
        self.fw = w
        self.fh = h

    # ── Zone classification ────────────────────────────────────────────────
    def _zone(self, cx: int) -> int:
        """Return zone index (0–4) for a given center_x pixel."""
        r = cx / self.fw
        for i, bound in enumerate(ZONE_BOUNDS[1:]):
            if r < bound:
                return i
        return Z_FR

    # ── Danger score for a zone ────────────────────────────────────────────
    @staticmethod
    def _danger(dists: list) -> float:
        """
        Danger score = sum(1/d) for each obstacle distance d.
        Closer obstacles contribute much more than far ones.
        Returns 0.0 if no obstacles.
        """
        if not dists:
            return 0.0
        return sum(1.0 / max(d, 0.1) for d in dists)

    # ── Core decision ──────────────────────────────────────────────────────
    def decide(self, detections: list[dict], distances: dict) -> str:
        """
        Analyse 5 zones and return the most appropriate navigation command.
        """
        # Collect distances per zone
        zone_dists: list[list] = [[] for _ in range(5)]

        for idx, det in enumerate(detections):
            if det["confidence"] < MIN_CONF:
                continue
            dist = distances.get(idx, 99.0)
            if dist > MAX_NAV_DIST:
                continue
            z = self._zone(det["center_x"])
            zone_dists[z].append(dist)

        # Danger scores per zone
        ds = [self._danger(zone_dists[z]) for z in range(5)]

        # Closest distance across ALL zones (global)
        all_dists_flat = [d for zd in zone_dists for d in zd]
        global_min = min(all_dists_flat, default=99.0)

        # Closest distance in CENTER zone only
        cc_min = min(zone_dists[Z_C], default=99.0)

        # ══════════════════════════════════════════════════════════════
        #  SPEED / SAFETY TIER  (based on closest CENTER obstacle)
        # ══════════════════════════════════════════════════════════════

        # ── Tier 1: STOP — < 0.40m anywhere ──────────────────────────
        if global_min < STOP_DIST:           # < 0.40m
            return self._set(CMD_STOP)

        # ── Tier 2: Walk Very Slowly — center 0.40–0.69m ─────────────
        if cc_min <= VERY_SLOW_DIST:         # 0.40–0.69m
            return self._set(CMD_VERY_SLOW)

        # ── Tier 3: Walk Slowly — center 0.70–1.99m ──────────────────
        if cc_min <= SLOW_DIST:              # 0.70–1.99m
            return self._set(CMD_SLOW)

        # ══════════════════════════════════════════════════════════════
        #  DIRECTION LOGIC  (center obstacle >= 2.0m → navigate freely)
        # ══════════════════════════════════════════════════════════════

        # ── Tier 4: Center blocked ≥ 2m → steer to open side ─────────
        if ds[Z_C] > 0:
            # Weighted danger of left half vs right half
            left_danger  = ds[Z_FL] * 0.5 + ds[Z_ML] * 1.0
            right_danger = ds[Z_MR] * 1.0 + ds[Z_FR] * 0.5

            # Full turn vs slight nudge based on danger intensity
            full_turn   = (ds[Z_C] >= DANGER_FULL) or (cc_min < FULL_TURN_DIST)
            slight_turn = (ds[Z_C] >= DANGER_SLIGHT) and not full_turn

            if left_danger <= right_danger:
                if slight_turn and ds[Z_ML] < DANGER_SLIGHT:
                    return self._set(CMD_SLIGHT_LEFT)
                return self._set(CMD_LEFT)
            else:
                if slight_turn and ds[Z_MR] < DANGER_SLIGHT:
                    return self._set(CMD_SLIGHT_RIGHT)
                return self._set(CMD_RIGHT)

        # ── Tier 5: Side-only obstacles (center clear) ────────────────
        # Mid-left busy → nudge right
        if ds[Z_ML] >= DANGER_SLIGHT and ds[Z_MR] < DANGER_SLIGHT:
            return self._set(CMD_SLIGHT_RIGHT)
        # Mid-right busy → nudge left
        if ds[Z_MR] >= DANGER_SLIGHT and ds[Z_ML] < DANGER_SLIGHT:
            return self._set(CMD_SLIGHT_LEFT)
        # Far-left busy → nudge right
        if ds[Z_FL] >= DANGER_SLIGHT and ds[Z_FR] < DANGER_SLIGHT:
            return self._set(CMD_SLIGHT_RIGHT)
        # Far-right busy → nudge left
        if ds[Z_FR] >= DANGER_SLIGHT and ds[Z_FL] < DANGER_SLIGHT:
            return self._set(CMD_SLIGHT_LEFT)

        # ── Tier 6: Obstacles exist but all far ───────────────────────
        if any(zone_dists):
            return self._set(CMD_FORWARD)

        # ── Tier 7: Truly clear ───────────────────────────────────────
        return self._set(CMD_CLEAR)

    def _set(self, cmd: str) -> str:
        self._last = cmd
        return cmd

    # ══════════════════════════════════════════════════════════════════════
    #  HUD Rendering
    # ══════════════════════════════════════════════════════════════════════

    def draw_hud(self, frame: np.ndarray, command: str,
                 detections: list[dict], distances: dict) -> np.ndarray:
        """
        Render the full navigation HUD:
          - 5-zone dividers with per-zone danger tint
          - Directional path highlight (shows WHERE to walk)
          - Large command banner + diagonal arrow
          - Nearest obstacle badge
        """
        h, w  = frame.shape[:2]
        color = CMD_COLORS.get(command, (255, 255, 255))

        # Zone x-coordinates
        zx = [int(w * b) for b in ZONE_BOUNDS]  # 6 boundary x positions

        # Danger scores for current frame
        zone_dists: list[list] = [[] for _ in range(5)]
        for idx, det in enumerate(detections):
            if det["confidence"] < MIN_CONF:
                continue
            dist = distances.get(idx, 99.0)
            if dist > MAX_NAV_DIST:
                continue
            z = self._zone(det["center_x"])
            zone_dists[z].append(dist)
        ds = [self._danger(zone_dists[z]) for z in range(5)]

        overlay = frame.copy()

        # ── Per-zone danger tint ───────────────────────────────────────────
        TOP = 35
        BOT = h - 78

        for z in range(5):
            x1, x2 = zx[z], zx[z + 1]
            d = ds[z]
            if d == 0:
                continue
            # Intensity: clamp danger to 0–1 range for colour
            intensity = min(d / 1.5, 1.0)
            # Red channel increases with danger
            r = int(60 * intensity)
            cv2.rectangle(overlay, (x1, TOP), (x2, BOT),
                          (0, 0, r), -1)

        # ── Path highlight — show the recommended walk corridor ────────────
        walk_x1, walk_x2 = self._walk_corridor(command, zx, w)
        cv2.rectangle(overlay, (walk_x1, TOP), (walk_x2, BOT),
                      (0, 60, 0), -1)   # green tint = safe path

        # Blend overlay
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

        # ── Zone divider lines ─────────────────────────────────────────────
        for x in zx[1:-1]:   # inner 4 lines only
            cv2.line(frame, (x, TOP), (x, BOT), (60, 60, 60), 1)

        # ── Zone labels (small, non-intrusive) ────────────────────────────
        zone_labels = ["FL", "ML", "C", "MR", "FR"]
        for z, label in enumerate(zone_labels):
            lx = (zx[z] + zx[z + 1]) // 2 - 8
            # Colour-code label by danger
            lc = (80, 80, 80) if ds[z] < DANGER_SLIGHT else (
                 (0, 180, 255) if ds[z] < DANGER_FULL else (0, 80, 255))
            cv2.putText(frame, label, (lx, TOP + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, lc, 1, cv2.LINE_AA)

        # ── Bottom command banner ──────────────────────────────────────────
        banner_h = 78
        banner_y = h - banner_h

        cv2.rectangle(frame, (0, banner_y), (w, h), (10, 10, 10), -1)
        # Thick colour accent on left
        cv2.rectangle(frame, (0, banner_y), (8, h), color, -1)
        # Thin colour line on top of banner
        cv2.line(frame, (0, banner_y), (w, banner_y), color, 2)

        # Command text
        font_scale = 1.5
        thick = 3
        (tw, th), _ = cv2.getTextSize(
            command, cv2.FONT_HERSHEY_DUPLEX, font_scale, thick)
        tx = (w - tw) // 2
        ty = banner_y + (banner_h + th) // 2 - 2
        cv2.putText(frame, command, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thick,
                    cv2.LINE_AA)

        # ── Directional arrow ──────────────────────────────────────────────
        self._draw_arrow(frame, command, w, banner_y, color)

        # ── Nearest obstacle badge ─────────────────────────────────────────
        if distances:
            min_d = min(distances.values())
            self._draw_dist_badge(frame, min_d, w, banner_y)

        # ── Zone danger bar (thin horizontal bar above banner) ─────────────
        self._draw_danger_bar(frame, ds, zx, banner_y, w)

        return frame

    def _walk_corridor(self, command: str,
                       zx: list, w: int) -> tuple:
        """
        Return (x1, x2) of the recommended walk corridor to highlight green.
        """
        if command == CMD_CLEAR or command == CMD_FORWARD:
            return zx[1], zx[4]           # full middle (ML to MR)
        elif command == CMD_SLIGHT_LEFT:
            return zx[0], zx[3]           # shift left (FL to MR)
        elif command == CMD_LEFT:
            return zx[0], zx[2]           # far left half
        elif command == CMD_SLIGHT_RIGHT:
            return zx[2], zx[5]           # shift right (C to FR)  -- fixed
        elif command == CMD_RIGHT:
            return zx[3], zx[5]           # far right half
        else:
            return zx[2], zx[3]           # just center (STOP — nowhere)

    def _draw_arrow(self, frame, command, w, banner_y, color):
        """
        Draw a rotated arrow above the banner.
        Diagonal arrows for Slight Left/Right, orthogonal for others.
        """
        cx  = w // 2
        ay  = banner_y - 18
        sz  = 38

        angle = CMD_ANGLES.get(command, 0)

        if command == CMD_STOP:
            # Bold red filled square with !
            cv2.rectangle(frame, (cx-sz, ay-sz), (cx+sz, ay+sz), color, -1)
            cv2.putText(frame, "!", (cx-10, ay + sz//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5,
                        (255, 255, 255), 3, cv2.LINE_AA)
            return

        if command == CMD_VERY_SLOW:
            # Double downward chevron (slow down hard)
            for offset in [-sz//2, sz//4]:
                pts = np.array([
                    [cx,       ay + offset],
                    [cx - sz,  ay + offset - sz//2],
                    [cx + sz,  ay + offset - sz//2],
                ], np.int32)
                cv2.polylines(frame, [pts], False, color, 4, cv2.LINE_AA)
            cv2.putText(frame, "!!", (cx - 16, ay + sz),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            return

        if command == CMD_SLOW:
            # Single downward chevron
            pts = np.array([
                [cx,       ay],
                [cx - sz,  ay - sz//2],
                [cx + sz,  ay - sz//2],
            ], np.int32)
            cv2.polylines(frame, [pts], False, color, 4, cv2.LINE_AA)
            cv2.putText(frame, "!", (cx - 8, ay + sz//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            return

        # Build a chevron arrow and rotate it
        # Arrow points: tip up, then rotate
        arrow_pts = np.array([
            [0,     -sz],        # tip
            [-sz//2, sz//3],     # left base wing
            [-sz//5, sz//3],     # left inner
            [-sz//5, sz],        # left tail
            [ sz//5, sz],        # right tail
            [ sz//5, sz//3],     # right inner
            [ sz//2, sz//3],     # right base wing
        ], dtype=np.float32)

        # Rotate by angle (degrees)
        if angle != 0:
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            arrow_pts = (rot @ arrow_pts.T).T

        arrow_pts = (arrow_pts + [cx, ay]).astype(np.int32)
        cv2.fillPoly(frame, [arrow_pts], color)

    def _draw_dist_badge(self, frame, min_dist: float,
                          w: int, banner_y: int):
        """Nearest obstacle distance badge — bottom right."""
        label = f"Nearest  {min_dist:.1f} m"
        if min_dist < 1.2:
            bc = (0, 0, 230)
        elif min_dist < 2.5:
            bc = (0, 80, 255)
        elif min_dist < 4.0:
            bc = (0, 190, 255)
        else:
            bc = (60, 210, 60)

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        rx = w - tw - 18
        ry = banner_y - 12
        cv2.rectangle(frame, (rx-6, ry-th-4), (rx+tw+6, ry+6),
                      (18, 18, 18), -1)
        cv2.putText(frame, label, (rx, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, bc, 2, cv2.LINE_AA)

    def _draw_danger_bar(self, frame, ds: list,
                          zx: list, banner_y: int, w: int):
        """
        Draw a thin horizontal danger bar just above the command banner.
        Each zone segment is coloured by its danger score.
        Green=safe, yellow=caution, red=danger.
        """
        bar_h  = 6
        bar_y  = banner_y - bar_h - 2

        for z in range(5):
            x1, x2 = zx[z], zx[z + 1]
            d = ds[z]
            if d < DANGER_SLIGHT:
                seg_color = (0, 180, 0)       # green
            elif d < DANGER_FULL:
                seg_color = (0, 200, 255)     # yellow/amber
            else:
                seg_color = (0, 0, 220)       # red

            cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h),
                          seg_color, -1)

        # Bar border
        cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h),
                      (40, 40, 40), 1)
