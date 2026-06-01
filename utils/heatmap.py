"""
heatmap.py
----------
Real-time danger zone heatmap overlay.

How it works:
  - Each detected obstacle "deposits" heat proportional to:
      * How close it is (closer = hotter)
      * Its size in pixels (bigger bbox = more heat)
  - Heat accumulates over time with exponential decay
  - The heatmap is rendered as a semi-transparent colour overlay
    (blue → green → yellow → red = safe → moderate → danger → critical)

Also provides a mini radar display (bird's-eye view) showing
obstacle positions relative to the user.

Author: AI Navigation Project
"""

import cv2
import numpy as np

# Decay factor applied per frame (1.0 = no decay, 0.0 = instant reset)
HEAT_DECAY      = 0.82

# Gaussian blur radius for smooth heat spread
HEAT_BLUR       = 51    # must be odd

# Max heat value (clips above this for colour mapping)
HEAT_MAX        = 255.0

# Heatmap blend alpha (0 = invisible, 1 = fully opaque)
HEAT_ALPHA      = 0.38

# Radar panel size in pixels
RADAR_SIZE      = 160
RADAR_PADDING   = 12

# Max real-world distance shown on radar (metres)
RADAR_MAX_DIST  = 8.0


class DangerHeatmap:
    """
    Maintains and renders a persistent danger heatmap overlay.
    """

    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        self.fw = frame_width
        self.fh = frame_height
        # Float32 accumulation buffer
        self._heat = np.zeros((frame_height, frame_width), dtype=np.float32)

    def update_size(self, w: int, h: int):
        if w != self.fw or h != self.fh:
            self.fw = w
            self.fh = h
            self._heat = np.zeros((h, w), dtype=np.float32)

    # ── Heat update ───────────────────────────────────────────────────
    def update(self, detections: list[dict], distances: dict):
        """
        Decay existing heat and deposit new heat from current detections.
        """
        # Decay
        self._heat *= HEAT_DECAY

        for idx, det in enumerate(detections):
            dist   = distances.get(idx, 10.0)
            cx, cy = det["center_x"], det["center_y"]
            w, h   = det["width"],    det["height"]

            # Heat intensity: inverse square of distance, capped
            intensity = min(HEAT_MAX, (HEAT_MAX / max(dist, 0.5) ** 1.5))

            # Gaussian blob size proportional to bounding box
            sigma_x = max(int(w * 0.6), 20)
            sigma_y = max(int(h * 0.6), 20)

            # Create a small Gaussian patch and add to the buffer
            patch_w = sigma_x * 4 + 1
            patch_h = sigma_y * 4 + 1
            patch   = self._gaussian_patch(patch_w, patch_h,
                                           sigma_x, sigma_y) * intensity

            # Compute region to paste
            x1 = cx - patch_w // 2
            y1 = cy - patch_h // 2
            x2 = x1 + patch_w
            y2 = y1 + patch_h

            # Clip to frame bounds
            fx1 = max(x1, 0);  fy1 = max(y1, 0)
            fx2 = min(x2, self.fw); fy2 = min(y2, self.fh)
            px1 = fx1 - x1;   py1 = fy1 - y1
            px2 = px1 + (fx2 - fx1)
            py2 = py1 + (fy2 - fy1)

            if fx2 > fx1 and fy2 > fy1:
                self._heat[fy1:fy2, fx1:fx2] += patch[py1:py2, px1:px2]

        # Clip max
        np.clip(self._heat, 0, HEAT_MAX, out=self._heat)

    @staticmethod
    def _gaussian_patch(w: int, h: int,
                         sigma_x: float, sigma_y: float) -> np.ndarray:
        """Return a normalised 2D Gaussian patch of size (h, w)."""
        cx, cy = w // 2, h // 2
        xs = np.arange(w, dtype=np.float32) - cx
        ys = np.arange(h, dtype=np.float32) - cy
        xx, yy = np.meshgrid(xs, ys)
        patch = np.exp(-(xx**2 / (2*sigma_x**2) + yy**2 / (2*sigma_y**2)))
        return patch

    # ── Render overlay ────────────────────────────────────────────────
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Blend the heatmap as a colour overlay onto the frame.
        Returns the annotated frame (in-place blend).
        """
        # Normalise to 0-255 uint8
        norm = (self._heat / HEAT_MAX * 255).astype(np.uint8)

        # Gaussian blur for smooth gradients
        blurred = cv2.GaussianBlur(norm, (HEAT_BLUR, HEAT_BLUR), 0)

        # Apply COLORMAP_JET: blue=cool/safe → red=hot/danger
        coloured = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

        # Only show heat where it's significant (> threshold)
        mask = blurred > 15
        coloured[~mask] = 0

        # Blend with original frame
        cv2.addWeighted(coloured, HEAT_ALPHA,
                        frame,    1 - HEAT_ALPHA, 0, frame)
        return frame

    def reset(self):
        """Clear the heatmap buffer."""
        self._heat[:] = 0.0


# ─────────────────────────────────────────────────────────────
# Mini Radar / Bird's-Eye View
# ─────────────────────────────────────────────────────────────

class MiniRadar:
    """
    Draws a bird's-eye radar panel showing obstacle positions
    relative to the user (bottom-centre of radar = user position).
    """

    def __init__(self, size: int = RADAR_SIZE,
                 max_dist: float = RADAR_MAX_DIST):
        self.size     = size
        self.max_dist = max_dist
        self.half     = size // 2

    def draw(self, frame: np.ndarray, detections: list[dict],
             distances: dict, frame_width: int,
             frame_height: int) -> np.ndarray:
        """
        Draw the radar panel in the bottom-left corner of the frame.
        """
        import cv2

        pad = RADAR_PADDING
        rx  = pad
        ry  = frame_height - self.size - pad

        # Background
        radar_bg = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        radar_bg[:] = (15, 15, 15)

        # Concentric range rings
        for ring in [0.25, 0.5, 0.75, 1.0]:
            r = int(ring * self.size // 2)
            cv2.circle(radar_bg, (self.half, self.size - 10),
                       r, (40, 80, 40), 1)

        # Cross-hairs
        cv2.line(radar_bg, (self.half, 0), (self.half, self.size),
                 (30, 60, 30), 1)
        cv2.line(radar_bg, (0, self.size - 10), (self.size, self.size - 10),
                 (30, 60, 30), 1)

        # User position (triangle at bottom centre)
        ux, uy = self.half, self.size - 12
        pts = np.array([[ux, uy-8], [ux-5, uy+4], [ux+5, uy+4]], np.int32)
        cv2.fillPoly(radar_bg, [pts], (0, 255, 100))

        # Plot each detection
        for idx, det in enumerate(detections):
            dist = distances.get(idx, 10.0)
            if dist > self.max_dist:
                continue

            # Horizontal offset: map frame center_x to radar X
            rel_x  = (det["center_x"] / frame_width) - 0.5  # -0.5 to +0.5
            radar_x = int(self.half + rel_x * self.size * 0.9)

            # Vertical offset: distance → radar Y (closer = lower)
            rel_d  = dist / self.max_dist          # 0=very close, 1=far
            radar_y = int((self.size - 12) - rel_d * (self.size - 20))

            # Colour by distance
            if dist < 1.5:
                dot_color = (0, 0, 255)
                r = 6
            elif dist < 3.0:
                dot_color = (0, 100, 255)
                r = 5
            elif dist < 5.0:
                dot_color = (0, 200, 255)
                r = 4
            else:
                dot_color = (0, 255, 150)
                r = 3

            radar_x = np.clip(radar_x, r, self.size - r)
            radar_y = np.clip(radar_y, r, self.size - r)
            cv2.circle(radar_bg, (radar_x, radar_y), r, dot_color, -1)

            # Class initial
            initial = det["class_name"][0].upper()
            cv2.putText(radar_bg, initial,
                        (radar_x + r + 1, radar_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, dot_color, 1)

        # Border
        cv2.rectangle(radar_bg, (0, 0),
                       (self.size-1, self.size-1), (60, 120, 60), 1)

        # Title
        cv2.putText(radar_bg, "RADAR",
                    (self.half - 18, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 100), 1)

        # Paste onto frame
        ry2 = ry + self.size
        rx2 = rx + self.size
        if ry >= 0 and ry2 <= frame_height and rx2 <= frame_width:
            frame[ry:ry2, rx:rx2] = radar_bg

        return frame
