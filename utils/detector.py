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
        Draw bounding boxes, labels, confidence and distance on frame.

        Parameters
        ----------
        frame      : BGR image (numpy array)
        detections : list from detect()
        distances  : dict mapping detection index -> estimated distance (m)

        Returns
        -------
        Annotated BGR frame
        """
        annotated = frame.copy()

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            name  = det["class_name"]
            conf  = det["confidence"]
            color = CLASS_COLORS.get(name, DEFAULT_COLOR)
            dist  = distances.get(idx, None) if distances else None

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label line 1: class + confidence
            label1 = f"{name}  {conf * 100:.0f}%"
            # Label line 2: distance
            label2 = f"Dist: {dist:.1f}m" if dist is not None else ""

            label_y = max(y1 - 10, 20)

            # Background pill for readability
            (tw1, th1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated,
                          (x1, label_y - th1 - 4),
                          (x1 + tw1 + 6, label_y + 4),
                          color, -1)
            cv2.putText(annotated, label1,
                        (x1 + 3, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            if label2:
                ly2 = label_y + th1 + 8
                (tw2, th2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
                cv2.rectangle(annotated,
                              (x1, ly2 - th2 - 2),
                              (x1 + tw2 + 6, ly2 + 4),
                              (30, 30, 30), -1)
                cv2.putText(annotated, label2,
                            (x1 + 3, ly2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

            # Center dot
            cv2.circle(annotated,
                       (det["center_x"], det["center_y"]),
                       5, color, -1)

        return annotated
