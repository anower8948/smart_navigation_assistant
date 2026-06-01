"""
tracker.py
----------
Multi-object centroid tracker with trajectory history.

Features:
  - Assigns persistent IDs to detected objects across frames
  - Maintains a position history (trail) per object
  - Computes velocity vector (dx, dy pixels/frame)
  - Predicts whether an object is APPROACHING the camera
    (bounding box growing = object moving closer)
  - Classifies approach speed: FAST / SLOW / STABLE / RECEDING

Algorithm: IoU + centroid distance Hungarian-style greedy matching.
No external tracking library needed — pure NumPy.

Author: AI Navigation Project
"""

import numpy as np
from collections import defaultdict, deque

# Max frames an object can be missing before its ID is dropped
MAX_MISSING_FRAMES = 10

# History length for trail drawing and velocity smoothing
TRAIL_LENGTH = 20

# Pixels/frame velocity threshold to classify approach
APPROACH_VELOCITY_THRESHOLD = 1.5   # px/frame (area growth rate)

# IoU threshold for matching detections to existing tracks
IOU_THRESHOLD = 0.25

# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def _iou(boxA, boxB):
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)


def _centroid_dist(c1, c2):
    return np.hypot(c1[0]-c2[0], c1[1]-c2[1])


# ─────────────────────────────────────────────────────────────
# Track object
# ─────────────────────────────────────────────────────────────

class Track:
    _next_id = 1

    def __init__(self, detection: dict, distance: float):
        self.id            = Track._next_id
        Track._next_id    += 1
        self.class_name    = detection["class_name"]
        self.bbox          = detection["bbox"]
        self.center        = (detection["center_x"], detection["center_y"])
        self.area          = detection["width"] * detection["height"]
        self.distance      = distance
        self.missing       = 0
        self.age           = 1   # frames since creation

        # History queues
        self.center_history: deque = deque(maxlen=TRAIL_LENGTH)
        self.area_history:   deque = deque(maxlen=TRAIL_LENGTH)
        self.center_history.append(self.center)
        self.area_history.append(self.area)

    def update(self, detection: dict, distance: float):
        self.bbox       = detection["bbox"]
        self.center     = (detection["center_x"], detection["center_y"])
        self.area       = detection["width"] * detection["height"]
        self.distance   = distance
        self.missing    = 0
        self.age       += 1
        self.center_history.append(self.center)
        self.area_history.append(self.area)

    # ── Velocity (pixels/frame) ────────────────────────────────
    @property
    def velocity(self) -> tuple:
        """Mean velocity (vx, vy) over recent history."""
        hist = list(self.center_history)
        if len(hist) < 2:
            return (0.0, 0.0)
        dxs = [hist[i+1][0]-hist[i][0] for i in range(len(hist)-1)]
        dys = [hist[i+1][1]-hist[i][1] for i in range(len(hist)-1)]
        return (float(np.mean(dxs)), float(np.mean(dys)))

    # ── Area growth rate → approaching / receding ─────────────
    @property
    def area_growth_rate(self) -> float:
        """Positive = growing (approaching), negative = shrinking."""
        hist = list(self.area_history)
        if len(hist) < 3:
            return 0.0
        # Linear regression slope
        x = np.arange(len(hist), dtype=float)
        y = np.array(hist, dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    @property
    def approach_status(self) -> str:
        """APPROACHING_FAST | APPROACHING | STABLE | RECEDING"""
        rate = self.area_growth_rate
        if rate > APPROACH_VELOCITY_THRESHOLD * 3:
            return "APPROACHING_FAST"
        elif rate > APPROACH_VELOCITY_THRESHOLD:
            return "APPROACHING"
        elif rate < -APPROACH_VELOCITY_THRESHOLD:
            return "RECEDING"
        else:
            return "STABLE"

    @property
    def is_approaching(self) -> bool:
        return "APPROACHING" in self.approach_status


# ─────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────

class ObjectTracker:
    """
    Maintains persistent tracks across frames.
    Call update() each frame with the current detections + distances.
    Returns a list of active Track objects.
    """

    def __init__(self):
        self.tracks: list[Track] = []

    def update(self, detections: list[dict],
               distances: dict) -> list[Track]:
        """
        Match detections to existing tracks and return updated track list.

        Parameters
        ----------
        detections : list from detector.detect()
        distances  : dict {detection_index -> float}

        Returns
        -------
        list of active Track objects
        """
        # ── Step 1: match detections to existing tracks ────────────────
        unmatched_det_indices = list(range(len(detections)))
        matched_track_ids     = set()

        for track in self.tracks:
            if not unmatched_det_indices:
                break

            # Score each unmatched detection against this track
            best_score = -1.0
            best_det_i = -1

            for det_i in unmatched_det_indices:
                det = detections[det_i]
                # Only match same class
                if det["class_name"] != track.class_name:
                    iou_w = 0.5   # allow cross-class match at lower score
                else:
                    iou_w = 1.0

                iou   = _iou(track.bbox, det["bbox"])
                cdist = _centroid_dist(track.center,
                                       (det["center_x"], det["center_y"]))
                # Normalise centroid distance (lower = better)
                cdist_score = max(0.0, 1.0 - cdist / 300.0)
                score = iou_w * (iou * 0.6 + cdist_score * 0.4)

                if score > best_score and iou >= IOU_THRESHOLD:
                    best_score = score
                    best_det_i = det_i

            if best_det_i >= 0:
                dist = distances.get(best_det_i, 99.0)
                track.update(detections[best_det_i], dist)
                matched_track_ids.add(track.id)
                unmatched_det_indices.remove(best_det_i)

        # ── Step 2: age unmatched existing tracks ─────────────────────
        for track in self.tracks:
            if track.id not in matched_track_ids:
                track.missing += 1

        # ── Step 3: create new tracks for unmatched detections ─────────
        for det_i in unmatched_det_indices:
            dist = distances.get(det_i, 99.0)
            self.tracks.append(Track(detections[det_i], dist))

        # ── Step 4: drop stale tracks ──────────────────────────────────
        self.tracks = [t for t in self.tracks
                       if t.missing <= MAX_MISSING_FRAMES]

        # Return only currently visible tracks
        return [t for t in self.tracks if t.missing == 0]

    def draw_trails(self, frame: np.ndarray,
                    active_tracks: list[Track]) -> np.ndarray:
        """
        Draw motion trails and approach arrows for each active track.
        """
        import cv2

        for track in active_tracks:
            hist = list(track.center_history)
            if len(hist) < 2:
                continue

            # Trail colour: red if approaching, cyan otherwise
            if track.approach_status == "APPROACHING_FAST":
                trail_color = (0, 0, 255)
            elif track.is_approaching:
                trail_color = (0, 100, 255)
            else:
                trail_color = (200, 200, 50)

            # Draw fading trail
            for i in range(1, len(hist)):
                alpha = int(255 * i / len(hist))
                thickness = max(1, int(3 * i / len(hist)))
                cv2.line(frame, hist[i-1], hist[i], trail_color, thickness)

            # ── Track ID chip ──────────────────────────────────────────
            cx, cy = track.center

            # Approach badge text: show only when notably approaching
            status = track.approach_status
            if status == "APPROACHING_FAST":
                badge = f"#{track.id}  FAST"
            elif status == "APPROACHING":
                badge = f"#{track.id}  near"
            else:
                badge = f"#{track.id}"

            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.38
            font_thick = 1
            pad_x, pad_y = 5, 3

            (tw, th), _ = cv2.getTextSize(badge, font, font_scale, font_thick)

            bx = cx + 8
            by = cy - 8
            # clamp inside frame
            bx = max(bx, 2)
            by = max(by, th + pad_y + 2)

            # Dark semi-transparent chip
            cv2.rectangle(frame,
                          (bx, by - th - pad_y),
                          (bx + tw + pad_x * 2, by + pad_y),
                          (20, 20, 20), -1)
            # Coloured left border
            cv2.rectangle(frame,
                          (bx, by - th - pad_y),
                          (bx + 2, by + pad_y),
                          trail_color, -1)
            cv2.putText(frame, badge,
                        (bx + pad_x, by),
                        font, font_scale, (210, 210, 210), font_thick, cv2.LINE_AA)

            # ── Approach velocity arrow ────────────────────────────────
            if track.is_approaching:
                vx, vy = track.velocity
                mag = np.hypot(vx, vy)
                if mag > 0:
                    scale = 28.0 / mag
                    ex = int(cx + vx * scale)
                    ey = int(cy + vy * scale)
                    cv2.arrowedLine(frame, (cx, cy), (ex, ey),
                                    trail_color, 2, tipLength=0.35)

        return frame
