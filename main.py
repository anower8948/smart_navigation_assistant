"""
main.py
-------
AI-Based Smart Navigation Assistant for Visually Impaired People
================================================================
Entry point — orchestrates the full pipeline:
    1. Open video file (or webcam)
    2. Run YOLOv8 object detection on every frame
    3. Estimate distances to detected obstacles
    4. Decide navigation command (Left / Right / Forward / STOP)
    5. Speak the command via TTS
    6. Render annotated video on screen and save to output/result.mp4

Usage
-----
    python main.py                          # default sample video
    python main.py --input videos/my.mp4   # custom video file
    python main.py --webcam                 # webcam mode (device 0)
    python main.py --help                   # all options

Requirements
------------
    pip install -r requirements.txt

Optimised for NVIDIA RTX 2060 GPU (CUDA 11.x+).

Author: AI Navigation Project
"""

import argparse
import os
import sys
import time
import cv2
import torch
import numpy as np

# ── Local module imports ──────────────────────────────────────────────────────
from utils.detector          import ObjectDetector
from utils.distance_estimator import DistanceEstimator
from utils.navigator          import Navigator, CMD_STOP, CMD_CLEAR
from utils.voice_assistant    import VoiceAssistant

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_VIDEO   = "videos/sample_video.mp4"
DEFAULT_MODEL   = "models/yolov8n.pt"
OUTPUT_VIDEO    = "output/result.mp4"
WINDOW_TITLE    = "Smart Navigation Assistant | Press Q to quit"

# Display scale (resize for screen if needed)
DISPLAY_WIDTH   = 1280
DISPLAY_HEIGHT  = 720


# ═════════════════════════════════════════════════════════════════════════════
#  Argument parser
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Smart Navigation Assistant for Visually Impaired People"
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_VIDEO,
        help=f"Path to input video file (default: {DEFAULT_VIDEO})"
    )
    parser.add_argument(
        "--webcam", action="store_true",
        help="Use webcam instead of video file"
    )
    parser.add_argument(
        "--cam-id", type=int, default=0,
        help="Webcam device ID (default: 0)"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Path to YOLOv8 weights (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_VIDEO,
        help=f"Output video path (default: {OUTPUT_VIDEO})"
    )
    parser.add_argument(
        "--conf", type=float, default=0.40,
        help="Detection confidence threshold 0–1 (default: 0.40)"
    )
    parser.add_argument(
        "--no-voice", action="store_true",
        help="Disable text-to-speech voice guidance"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run headlessly without showing a window (useful on servers)"
    )
    parser.add_argument(
        "--save-output", action="store_true", default=True,
        help="Save annotated video to output file (default: True)"
    )
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
#  Utility: draw a heads-up FPS counter
# ═════════════════════════════════════════════════════════════════════════════
def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
    return frame


def draw_title_bar(frame: np.ndarray) -> np.ndarray:
    """Draw a top title banner."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 32), (15, 15, 15), -1)
    cv2.putText(frame,
                "AI Smart Navigation Assistant for Visually Impaired  |  YOLOv8n + COCO",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 230, 255), 1)
    return frame


# ═════════════════════════════════════════════════════════════════════════════
#  Open video source
# ═════════════════════════════════════════════════════════════════════════════
def open_source(args) -> cv2.VideoCapture:
    if args.webcam:
        print(f"[Main] Opening webcam (device {args.cam_id}) ...")
        cap = cv2.VideoCapture(args.cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    else:
        video_path = args.input
        if not os.path.isfile(video_path):
            print(f"[Main] ERROR: Video file not found: {video_path}")
            print("[Main] Tip: Place a video in the 'videos/' folder "
                  "or pass --input <path>")
            sys.exit(1)
        print(f"[Main] Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[Main] ERROR: Could not open video source.")
        sys.exit(1)

    return cap


# ═════════════════════════════════════════════════════════════════════════════
#  Setup output writer
# ═════════════════════════════════════════════════════════════════════════════
def setup_writer(cap: cv2.VideoCapture, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"[Main] Output will be saved to: {output_path}  "
          f"({width}x{height} @ {fps:.1f} fps)")
    return writer


# ═════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═════════════════════════════════════════════════════════════════════════════
def run(args):
    print("\n" + "═" * 60)
    print("  AI Smart Navigation Assistant — Starting")
    print("═" * 60)

    # ── GPU check ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Main] GPU detected: {gpu_name}")
    else:
        print("[Main] No GPU detected — running on CPU (slower).")

    # ── Initialise modules ─────────────────────────────────────────────
    detector   = ObjectDetector(model_path=args.model, confidence=args.conf)
    estimator  = DistanceEstimator()
    navigator  = Navigator()
    voice      = VoiceAssistant(enabled=not args.no_voice)

    # ── Open source ────────────────────────────────────────────────────
    cap    = open_source(args)
    writer = None
    if args.save_output:
        writer = setup_writer(cap, args.output)

    # ── Main loop ──────────────────────────────────────────────────────
    frame_count  = 0
    fps_display  = 0.0
    t_start_loop = time.time()
    prev_cmd     = ""

    print("\n[Main] Processing started. Press 'Q' in window to quit.\n")

    try:
        while True:
            t_frame = time.time()
            ret, frame = cap.read()

            if not ret:
                print("[Main] End of video stream.")
                break

            # ── Resize for consistent processing ──────────────────────
            h_orig, w_orig = frame.shape[:2]
            if w_orig != DISPLAY_WIDTH or h_orig != DISPLAY_HEIGHT:
                frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            frame_h, frame_w = frame.shape[:2]

            # Update module sizes
            navigator.update_frame_size(frame_w, frame_h)
            estimator.update_frame_size(frame_h)

            # ── Detection ──────────────────────────────────────────────
            detections = detector.detect(frame)

            # ── Distance estimation ────────────────────────────────────
            distances = estimator.estimate_all(detections)

            # ── Navigation decision ────────────────────────────────────
            command = navigator.decide(detections, distances)

            # ── Voice guidance (only on command change) ────────────────
            if command != prev_cmd:
                voice.speak(command)
                prev_cmd = command

            # ── Draw detections ────────────────────────────────────────
            frame = detector.draw_detections(frame, detections, distances)

            # ── Draw HUD (zones, command banner, object panel) ─────────
            frame = navigator.draw_hud(frame, command, detections, distances)

            # ── FPS counter ────────────────────────────────────────────
            elapsed = time.time() - t_frame
            fps_display = 1.0 / elapsed if elapsed > 0 else 0.0
            frame = draw_fps(frame, fps_display)
            frame = draw_title_bar(frame)

            # ── Write output ───────────────────────────────────────────
            if writer:
                writer.write(frame)

            # ── Display ────────────────────────────────────────────────
            if not args.no_display:
                cv2.imshow(WINDOW_TITLE, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q") or key == 27:
                    print("[Main] User quit.")
                    break

            frame_count += 1

            # Progress log every 50 frames
            if frame_count % 50 == 0:
                total_elapsed = time.time() - t_start_loop
                avg_fps = frame_count / total_elapsed
                print(f"[Main] Frame {frame_count:5d} | "
                      f"FPS {avg_fps:.1f} | "
                      f"Objects: {len(detections):2d} | "
                      f"Command: {command}")

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")

    finally:
        # ── Cleanup ────────────────────────────────────────────────────
        cap.release()
        if writer:
            writer.release()
            print(f"\n[Main] Output saved: {args.output}")
        cv2.destroyAllWindows()
        voice.stop()

        total_time = time.time() - t_start_loop
        avg_fps    = frame_count / total_time if total_time > 0 else 0
        print(f"\n[Main] Processed {frame_count} frames in {total_time:.1f}s "
              f"(avg {avg_fps:.1f} FPS)")
        print("[Main] Done.")


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()
    run(args)
