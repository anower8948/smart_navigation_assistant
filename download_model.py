"""
download_model.py
-----------------
Helper script to download yolov8n.pt into the models/ directory.
Run this once before main.py if models/yolov8n.pt is missing.

Usage:
    python download_model.py
"""

import os
import urllib.request

MODEL_URL  = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
MODEL_PATH = "models/yolov8n.pt"


def download():
    os.makedirs("models", exist_ok=True)

    if os.path.isfile(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1_000_000
        print(f"[Download] Model already exists: {MODEL_PATH} ({size_mb:.1f} MB)")
        return

    print(f"[Download] Downloading YOLOv8 Nano weights from:\n  {MODEL_URL}")
    print("[Download] This is ~6 MB — please wait ...")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"\r  [{bar}] {pct:.0f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=progress)
        print(f"\n[Download] Saved to: {MODEL_PATH}")
    except Exception as e:
        print(f"\n[Download] Error: {e}")
        print("[Download] Alternatively, run:  from ultralytics import YOLO; YOLO('yolov8n.pt')")
        print("           Ultralytics will auto-download the model on first use.")


if __name__ == "__main__":
    download()
