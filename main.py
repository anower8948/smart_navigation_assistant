"""
main.py
-------
AI-Based Smart Navigation Assistant for Visually Impaired People
================================================================
Advanced pipeline v2:

  1. Open video file or webcam
  2. YOLOv8n object detection (GPU-accelerated)
  3. Distance estimation (pinhole camera model)
  4. Multi-object tracking with trajectory history (ObjectTracker)
  5. Navigation decision with LEFT/CENTER/RIGHT zone logic
  6. Command smoothing / debouncing (CommandSmoother)
  7. Speed-aware voice guidance (SpeedAwareVoice + pyttsx3)
  8. Danger heatmap overlay (DangerHeatmap)
  9. Mini radar / bird's-eye view panel (MiniRadar)
 10. Special situation alerts — traffic light, stop sign,
     crowd detection, fast-approaching objects, blind spots (AlertEngine)
 11. Motion trail overlay per tracked object
 12. Saves annotated output to output/result.mp4

Usage
-----
    python main.py                         # default sample video
    python main.py --input videos/my.mp4  # custom video
    python main.py --webcam               # live webcam
    python main.py --no-voice             # silent mode
    python main.py --no-heatmap           # disable heatmap
    python main.py --no-radar             # disable radar panel
    python main.py --help                 # all options

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
from utils.detector           import ObjectDetector
from utils.distance_estimator import DistanceEstimator
from utils.navigator          import (Navigator, CMD_STOP, CMD_CLEAR,
                                      CMD_LEFT, CMD_LEFT_MID, CMD_FORWARD,
                                      CMD_RIGHT_MID, CMD_RIGHT,
                                      CMD_VERY_SLOW, CMD_SLOW)
from utils.voice_assistant    import VoiceAssistant
from utils.tracker            import ObjectTracker
from utils.heatmap            import DangerHeatmap, MiniRadar
from utils.smoother           import CommandSmoother, SpeedAwareVoice
from utils.alert_engine       import AlertEngine

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_VIDEO   = "videos/sample_video.mp4"
DEFAULT_MODEL   = "models/yolov8n.pt"
OUTPUT_VIDEO    = "output/result.mp4"
WINDOW_TITLE    = "Smart Navigation Assistant v2  |  Q = quit  |  H = heatmap  |  R = radar"

DISPLAY_WIDTH   = 1280
DISPLAY_HEIGHT  = 720


# ═════════════════════════════════════════════════════════════════════════════
#  Argument parser
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="AI Smart Navigation Assistant v2 — Advanced Edition"
    )
    p.add_argument("--input",      type=str,   default=DEFAULT_VIDEO)
    p.add_argument("--webcam",     action="store_true")
    p.add_argument("--cam-id",     type=int,   default=0)
    p.add_argument("--model",      type=str,   default=DEFAULT_MODEL)
    p.add_argument("--output",     type=str,   default=OUTPUT_VIDEO)
    p.add_argument("--conf",       type=float, default=0.40)
    p.add_argument("--no-voice",   action="store_true")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--no-heatmap", action="store_true",
                   help="Disable danger heatmap overlay")
    p.add_argument("--no-radar",   action="store_true",
                   help="Disable mini radar panel")
    p.add_argument("--no-trails",  action="store_true",
                   help="Disable motion trail overlay")
    p.add_argument("--save-output", action="store_true", default=True)
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
#  HUD helpers
# ═════════════════════════════════════════════════════════════════════════════
def draw_topbar(frame: np.ndarray, fps: float,
                frame_count: int, n_objects: int,
                command: str, conf: float) -> np.ndarray:
    """Top status bar with FPS, object count, command confidence."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 32), (10, 10, 10), -1)

    left  = f"Smart Navigation Assistant v2  |  GPU: {'YES' if torch.cuda.is_available() else 'CPU'}"
    right = f"FPS: {fps:.1f}  |  Objects: {n_objects}  |  Cmd Confidence: {conf*100:.0f}%  |  Frame: {frame_count}"

    cv2.putText(frame, left,  (8,  22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 255), 1)
    (rw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, right, (w - rw - 8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 200, 160), 1)
    return frame


def draw_approach_warning(frame: np.ndarray,
                           active_tracks) -> np.ndarray:
    """Flash a bold warning banner when any object is fast-approaching."""
    fast = [t for t in active_tracks
            if t.approach_status == "APPROACHING_FAST" and t.distance < 3.0]
    if not fast:
        return frame

    h, w = frame.shape[:2]
    msg  = f"! OBJECT APPROACHING FAST: {fast[0].class_name.upper()} !"
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
    bx = (w - tw) // 2

    # Flashing effect — blink every 15 frames using time
    if int(time.time() * 4) % 2 == 0:
        cv2.rectangle(frame,
                      (bx - 10, h // 2 - th - 14),
                      (bx + tw + 10, h // 2 + 10),
                      (0, 0, 200), -1)
        cv2.putText(frame, msg, (bx, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
    return frame


# ═════════════════════════════════════════════════════════════════════════════
#  Video source
# ═════════════════════════════════════════════════════════════════════════════
def open_source(args) -> cv2.VideoCapture:
    if args.webcam:
        print(f"[Main] Opening webcam (device {args.cam_id}) ...")
        cap = cv2.VideoCapture(args.cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    else:
        if not os.path.isfile(args.input):
            print(f"[Main] ERROR: Video file not found: {args.input}")
            sys.exit(1)
        print(f"[Main] Opening video: {args.input}")
        cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print("[Main] ERROR: Could not open video source.")
        sys.exit(1)
    return cap


def setup_writer(cap: cv2.VideoCapture, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"[Main] Output → {output_path}  ({width}x{height} @ {fps:.1f} fps)")
    return writer


# ═════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═════════════════════════════════════════════════════════════════════════════
def run(args):
    print("\n" + "═" * 65)
    print("  AI Smart Navigation Assistant v2 — Advanced Edition")
    print("═" * 65)

    if torch.cuda.is_available():
        print(f"[Main] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[Main] No GPU — running on CPU.")

    # ── Init modules ──────────────────────────────────────────────────
    detector  = ObjectDetector(model_path=args.model, confidence=args.conf)
    estimator = DistanceEstimator()
    navigator = Navigator()
    voice     = VoiceAssistant(enabled=not args.no_voice)
    tracker   = ObjectTracker()
    heatmap   = DangerHeatmap()
    radar     = MiniRadar()
    smoother  = CommandSmoother()
    speed_voice = SpeedAwareVoice(voice)
    alert_eng = AlertEngine(voice_assistant=voice)

    # Feature toggles (can also be toggled with keyboard at runtime)
    show_heatmap = not args.no_heatmap
    show_radar   = not args.no_radar
    show_trails  = not args.no_trails

    # ── Open source + writer ──────────────────────────────────────────
    cap    = open_source(args)
    writer = setup_writer(cap, args.output) if args.save_output else None

    frame_count = 0
    fps_display = 0.0
    t_loop      = time.time()
    prev_stable = ""

    print("\n[Main] Running. Keys: Q=quit | H=toggle heatmap | "
          "R=toggle radar | T=toggle trails\n")

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[Main] End of stream.")
                break

            # Resize
            frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            fh, fw = frame.shape[:2]

            # Update module sizes
            navigator.update_frame_size(fw, fh)
            estimator.update_frame_size(fh)
            heatmap.update_size(fw, fh)

            # ── Detection ─────────────────────────────────────────────
            detections = detector.detect(frame)

            # ── Distance estimation ───────────────────────────────────
            distances = estimator.estimate_all(detections)

            # ── Tracking ──────────────────────────────────────────────
            active_tracks = tracker.update(detections, distances)

            # ── Navigation decision ───────────────────────────────────
            raw_cmd    = navigator.decide(detections, distances)
            stable_cmd = smoother.update(raw_cmd)
            cmd_conf   = smoother.confidence

            # ── Voice guidance ────────────────────────────────────────
            min_dist = min(distances.values(), default=99.0)
            # Speak on every command change OR if distance dropped sharply
            if stable_cmd != prev_stable:
                speed_voice.speak(stable_cmd, min_dist)
                prev_stable = stable_cmd
            elif stable_cmd in (CMD_STOP, CMD_VERY_SLOW, CMD_SLOW,
                                CMD_LEFT, CMD_LEFT_MID,
                                CMD_RIGHT_MID, CMD_RIGHT) \
                    and min_dist < 2.0:
                speed_voice.speak(stable_cmd, min_dist)  # repeat for proximity cmds

            # ── Special alerts ────────────────────────────────────────
            alerts = alert_eng.analyse(detections, distances,
                                       active_tracks, fw)

            # ── Render: heatmap (bottom layer) ────────────────────────
            if show_heatmap:
                heatmap.update(detections, distances)
                frame = heatmap.draw_overlay(frame)

            # ── Render: bounding boxes ────────────────────────────────
            frame = detector.draw_detections(frame, detections, distances)

            # ── Render: motion trails ─────────────────────────────────
            if show_trails:
                frame = tracker.draw_trails(frame, active_tracks)

            # ── Render: navigator HUD ─────────────────────────────────
            frame = navigator.draw_hud(frame, stable_cmd,
                                        detections, distances)

            # ── Render: approach warning ──────────────────────────────
            frame = draw_approach_warning(frame, active_tracks)

            # ── Render: alert banners ─────────────────────────────────
            frame = alert_eng.draw_alerts(frame, alerts)

            # ── Render: radar ─────────────────────────────────────────
            if show_radar:
                frame = radar.draw(frame, detections, distances, fw, fh)

            # ── Render: top status bar ────────────────────────────────
            fps_display = 1.0 / max(time.time() - t0, 1e-6)
            frame = draw_topbar(frame, fps_display, frame_count,
                                 len(detections), stable_cmd, cmd_conf)

            # ── Write output ──────────────────────────────────────────
            if writer:
                writer.write(frame)

            # ── Display ───────────────────────────────────────────────
            if not args.no_display:
                cv2.imshow(WINDOW_TITLE, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[Main] User quit.")
                    break
                elif key in (ord("h"), ord("H")):
                    show_heatmap = not show_heatmap
                    print(f"[Main] Heatmap: {'ON' if show_heatmap else 'OFF'}")
                elif key in (ord("r"), ord("R")):
                    show_radar = not show_radar
                    print(f"[Main] Radar: {'ON' if show_radar else 'OFF'}")
                elif key in (ord("t"), ord("T")):
                    show_trails = not show_trails
                    print(f"[Main] Trails: {'ON' if show_trails else 'OFF'}")

            frame_count += 1

            if frame_count % 60 == 0:
                avg = frame_count / (time.time() - t_loop)
                print(f"[Main] Frame {frame_count:5d} | "
                      f"FPS {avg:.1f} | Objects {len(detections)} | "
                      f"Tracks {len(active_tracks)} | Cmd: {stable_cmd}")

    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"\n[Main] Saved: {args.output}")
        cv2.destroyAllWindows()
        voice.stop()

        total = time.time() - t_loop
        print(f"[Main] {frame_count} frames in {total:.1f}s "
              f"(avg {frame_count/max(total,1):.1f} FPS)")
        print("[Main] Done.")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()
    run(args)
