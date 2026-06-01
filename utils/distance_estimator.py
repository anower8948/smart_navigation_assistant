"""
distance_estimator.py
---------------------
Approximates the real-world distance of detected objects using
bounding-box height and a reference focal-length calibration approach.

Formula:
    distance (m) = (real_height * focal_length) / pixel_height

Where focal_length is derived from:
    focal_length = (pixel_height_ref * distance_ref) / real_height_ref

Because we don't have exact camera calibration, we use well-known
average real-world heights for each class and a fixed focal-length
estimated for a standard 1080p camera at ~60° FoV.

Author: AI Navigation Project
"""

import numpy as np

# ──────────────────────────────────────────────
# Real-world approximate heights (meters)
# Used for focal-length based distance estimation
# ──────────────────────────────────────────────
OBJECT_REAL_HEIGHTS = {
    "Person":        1.70,   # average adult
    "Bicycle":       1.00,
    "Car":           1.50,
    "Motorcycle":    1.10,
    "Bus":           3.20,
    "Truck":         2.80,
    "Traffic Light": 0.80,
    "Stop Sign":     0.75,
    "Bench":         0.90,
    "Chair":         0.90,
    "Couch":         0.85,
    "Potted Plant":  0.50,
    "Dining Table":  0.75,
    "TV":            0.60,
    "Laptop":        0.35,
    "Refrigerator":  1.80,
    "Dog":           0.55,
    "Cat":           0.30,
    "Bird":          0.25,
}

# Fallback height for unknown objects
DEFAULT_REAL_HEIGHT = 1.0   # metres

# ──────────────────────────────────────────────
# Camera focal length (pixels) — estimated for
# a typical webcam / phone camera at 1080p
# (This can be calibrated per camera for higher accuracy)
# ──────────────────────────────────────────────
FOCAL_LENGTH_PIXELS = 700.0  # approximate

# Distance clamp limits (metres)
MIN_DISTANCE = 0.2
MAX_DISTANCE = 30.0

# "Danger zone" threshold: object is considered very close
DANGER_DISTANCE = 1.5   # metres


class DistanceEstimator:
    """
    Estimates distance from camera to detected object using
    perspective projection and known object heights.
    """

    def __init__(self, focal_length: float = FOCAL_LENGTH_PIXELS,
                 frame_height: int = 720):
        """
        Parameters
        ----------
        focal_length : float
            Camera focal length in pixels (default ≈700 for typical webcam).
        frame_height : int
            Frame height in pixels, used to scale focal length dynamically.
        """
        self.focal_length = focal_length
        self.frame_height = frame_height

        print(f"[DistanceEstimator] Focal length: {self.focal_length:.0f}px | "
              f"Frame height reference: {self.frame_height}px")

    def update_frame_size(self, frame_height: int):
        """
        Update focal length when frame resolution changes.
        Scales focal length proportionally.
        """
        if frame_height != self.frame_height and frame_height > 0:
            scale = frame_height / self.frame_height
            self.focal_length = FOCAL_LENGTH_PIXELS * scale
            self.frame_height = frame_height

    def estimate(self, detection: dict) -> float:
        """
        Estimate distance (metres) for a single detection.

        Parameters
        ----------
        detection : dict from detector.detect()

        Returns
        -------
        float : estimated distance in metres
        """
        class_name   = detection.get("class_name", "Unknown")
        pixel_height = detection.get("height", 1)

        if pixel_height <= 0:
            return MAX_DISTANCE

        real_height = OBJECT_REAL_HEIGHTS.get(class_name, DEFAULT_REAL_HEIGHT)

        # Core pinhole camera formula
        distance = (real_height * self.focal_length) / pixel_height

        # Clamp to plausible range
        distance = max(MIN_DISTANCE, min(MAX_DISTANCE, distance))
        return round(distance, 2)

    def estimate_all(self, detections: list[dict]) -> dict:
        """
        Estimate distance for each detection.

        Returns
        -------
        dict : {detection_index -> distance_metres}
        """
        distances = {}
        for idx, det in enumerate(detections):
            distances[idx] = self.estimate(det)
        return distances

    @staticmethod
    def is_danger(distance: float) -> bool:
        """Returns True if object is within the danger threshold."""
        return distance <= DANGER_DISTANCE

    @staticmethod
    def danger_level(distance: float) -> str:
        """
        Returns a human-readable danger label.
        'CRITICAL' < 0.8m | 'DANGER' < 1.5m | 'CAUTION' < 3m | 'SAFE' otherwise
        """
        if distance < 0.8:
            return "CRITICAL"
        elif distance < DANGER_DISTANCE:
            return "DANGER"
        elif distance < 3.0:
            return "CAUTION"
        else:
            return "SAFE"
