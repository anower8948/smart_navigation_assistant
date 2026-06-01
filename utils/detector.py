"""
detector.py
-----------
Handles YOLOv8 model loading and object detection.
Uses Ultralytics YOLOv8 Nano (yolov8n.pt) pretrained on COCO dataset.

COCO classes detected (relevant to navigation):
  0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck,
  13: bench, 56: chair, 57: couch, 60: dining table, 63: laptop,
  64: mouse, 66: keyboard, 67: cell phone, 72: tv, 73: laptop

Author: AI Navigation Project
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch

# ──────────────────────────────────────────────
# Navigation-relevant COCO class IDs
# ──────────────────────────────────────────────
OBSTACLE_CLASSES = {
    0:  "Person",
    1:  "Bicycle",
    2:  "Car",
    3:  "Motorcycle",
    5:  "Bus",
    7:  "Truck",
    9:  "Traffic Light",
    11: "Stop Sign",
    13: "Bench",
    14: "Bird",
    15: "Cat",
    16: "Dog",
    56: "Chair",
    57: "Couch",
    58: "Potted Plant",
    60: "Dining Table",
    62: "TV",
    63: "Laptop",
    66: "Keyboard",
    72: "Refrigerator",
    73: "Book",
    74: "Clock",
    75: "Vase",
    76: "Scissors",
    79: "Toothbrush",
}

# Colour palette for bounding boxes (BGR)
CLASS_COLORS = {
    "Person":        (0,   200, 255),
    "Car":           (0,   100, 255),
    "Bus":           (0,   50,  200),
    "Truck":         (50,  50,  200),
    "Motorcycle":    (0,   165, 255),
    "Bicycle":       (0,   255, 200),
    "Bench":         (180, 120, 50),
    "Chair":         (180, 80,  80),
    "Traffic Light": (0,   255, 0),
    "Stop Sign":     (0,   0,   255),
}
DEFAULT_COLOR = (200, 200, 200)


class ObjectDetector:
    """
    Wraps YOLOv8 for obstacle detection.

    Parameters
    ----------
    model_path : str
        Path to yolov8n.pt weights file.
    confidence : float
        Minimum confidence threshold (0–1).
    device : str
        'cuda' for GPU, 'cpu' for CPU.
    """

    def __init__(self, model_path: str = "models/yolov8n.pt",
                 confidence: float = 0.40,
                 device: str = None):

        # Auto-select device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.confidence = confidence

        print(f"[Detector] Loading model from: {model_path}")
        print(f"[Detector] Device: {self.device.upper()}")

        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"[Detector] Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"[Detector] Failed to load model: {e}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a single BGR frame.

        Returns
        -------
        list of dict with keys:
            class_id, class_name, confidence,
            bbox (x1, y1, x2, y2), center_x, center_y, width, height
        """
        if frame is None or frame.size == 0:
            return []

        try:
            results = self.model(
                frame,
                conf=self.confidence,
                verbose=False,
                device=self.device
            )[0]
        except Exception as e:
            print(f"[Detector] Inference error: {e}")
            return []

        detections = []
        if results.boxes is None:
            return detections

        for box in results.boxes:
            class_id = int(box.cls[0])

            # Only keep navigation-relevant obstacles
            if class_id not in OBSTACLE_CLASSES:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width  = x2 - x1
            height = y2 - y1
            cx = x1 + width  // 2
            cy = y1 + height // 2

            detections.append({
                "class_id":   class_id,
                "class_name": OBSTACLE_CLASSES[class_id],
                "confidence": conf,
                "bbox":       (x1, y1, x2, y2),
                "center_x":  cx,
                "center_y":  cy,
                "width":     width,
                "height":    height,
            })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: list[dict],
                        distances: dict = None) -> np.ndarray:
        """
        Draw bounding boxes and clean label badges on frame.

        Label design:
          ┌─────────────────────────┐
          │  Person  87%  · 2.3 m  │  ← single-line pill, dark bg + colour accent
          └─────────────────────────┘
        """
        annotated = frame.copy()

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            name  = det["class_name"]
            conf  = det["confidence"]
            color = CLASS_COLORS.get(name, DEFAULT_COLOR)
            dist  = distances.get(idx, None) if distances else None

            # ── Bounding box (thin, clean) ─────────────────────────────
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Corner accent marks (small L-shapes at corners)
            corner = 10
            thick  = 3
            for cx_, cy_, dx, dy in [
                (x1, y1,  1,  1), (x2, y1, -1,  1),
                (x1, y2,  1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(annotated, (cx_, cy_), (cx_ + dx*corner, cy_), color, thick)
                cv2.line(annotated, (cx_, cy_), (cx_, cy_ + dy*corner), color, thick)

            # ── Single-line label pill ─────────────────────────────────
            if dist is not None:
                label = f"{name}  {conf*100:.0f}%   {dist:.1f} m"
            else:
                label = f"{name}  {conf*100:.0f}%"

            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.48
            font_thick = 1
            pad_x, pad_y = 7, 5

            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

            # Position: just above top-left corner of bbox
            lx = x1
            ly = max(y1 - pad_y - 2, th + pad_y + 2)

            # Dark pill background
            cv2.rectangle(annotated,
                          (lx, ly - th - pad_y),
                          (lx + tw + pad_x * 2, ly + pad_y - 2),
                          (18, 18, 18), -1)

            # Left colour accent stripe
            cv2.rectangle(annotated,
                          (lx, ly - th - pad_y),
                          (lx + 3, ly + pad_y - 2),
                          color, -1)

            # Label text (white, crisp)
            cv2.putText(annotated, label,
                        (lx + pad_x, ly),
                        font, font_scale, (230, 230, 230), font_thick, cv2.LINE_AA)

            # ── Small centre dot ──────────────────────────────────────
            cv2.circle(annotated,
                       (det["center_x"], det["center_y"]),
                       4, color, -1)

        return annotated
