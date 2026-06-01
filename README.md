# AI-Based Smart Navigation Assistant for Visually Impaired People
### v3 — 5-Zone Precision Navigation · YOLOv8n · Real-Time Tracking · Smart Voice

---

## Overview

A real-time AI navigation assistant that helps visually impaired people walk safely in outdoor environments. The system detects obstacles using **YOLOv8 Nano**, estimates distances, tracks each object across frames, and delivers precise spoken directional guidance — all running locally on a consumer GPU or CPU with no internet required.

---

## What's New in v3

| Change | Detail |
|--------|--------|
| **5-Zone Navigation** | Screen split into 5 equal zones — each maps to one direction command |
| **5 Direction Commands** | Move Left · Move Left Middle · Move Forward · Move Right Middle · Move Right |
| **Diagonal Guidance** | New "Middle" commands for smooth diagonal correction on crowded streets |
| **Danger Scoring** | Each zone gets a danger score = sum(1/distance); safest zone wins |
| **Speed-Tier Priority** | STOP / Walk Very Slowly / Walk Slowly always override direction logic |
| **Clean Label Badges** | Single-line dark pill: `Person  87%   2.3 m` with colour accent stripe |
| **Clean Track ID Chips** | Compact `#4`, `#4  near`, `#4  FAST` chips — no raw floating text |
| **Fixed Alert Style** | Replaced broken `???` Unicode symbols with proper `[ALERT]` tag chips |

---

## Navigation System

### 5-Zone Screen Division

```
┌──────────┬────────────────┬──────────────┬─────────────────┬──────────┐
│  FAR     │   MID LEFT     │   CENTRE     │   MID RIGHT     │  FAR     │
│  LEFT    │                │              │                 │  RIGHT   │
│  0–20%   │   20–40%       │   40–60%     │   60–80%        │ 80–100%  │
│          │                │              │                 │          │
│Move Left │Move Left Middle│ Move Forward │Move Right Middle│Move Right│
└──────────┴────────────────┴──────────────┴─────────────────┴──────────┘
```

### Speed-Tier Priority System

Speed tiers are checked **first** — they always override direction logic:

| Priority | Condition | Command |
|----------|-----------|---------|
| 1 (highest) | Any obstacle **< 0.40 m** (anywhere) | **STOP** |
| 2 | Centre obstacle **0.40 – 0.69 m** | **Walk Very Slowly** |
| 3 | Centre obstacle **0.70 – 1.99 m** | **Walk Slowly** |
| 4 | All obstacles **≥ 2.0 m** | Direction logic (5 options) |
| 5 (default) | No obstacles detected | **Path Clear** |

### Direction Algorithm (when ≥ 2.0 m)

1. Each obstacle is assigned to its zone based on centre pixel X
2. Danger score per zone = `sum(1 / distance)` for all obstacles inside it
3. Safest zone = lowest danger score (tie-break: prefer centre → mid → far)
4. Output = direction command mapped to that zone

### Command Smoothing

| Command | Confirm Window | Reason |
|---------|---------------|--------|
| STOP / Walk Very Slowly | 1 frame (instant) | Safety critical |
| Walk Slowly / All directions | 2 frames | Balanced stability |
| Path Clear | 3 frames | Requires most stability |

---

## Features

### Core Navigation
- **YOLOv8 Nano** — detects 20 obstacle-relevant COCO classes at 60–80+ FPS on GPU
- **Pinhole Distance Estimation** — estimates metric distance from bounding box height
- **5-Zone Navigation Engine** — danger-score-based safest path selection
- **Speed-Aware Voice Guidance** — rate increases near danger; cooldown prevents repetition
- **Non-Blocking TTS** — pyttsx3 runs in background thread, never drops frames

### Tracking & Awareness
- **Multi-Object Tracking** — persistent IDs via IoU + centroid matching, survives brief occlusions
- **Velocity Estimation** — tracks direction and speed of each object
- **Approach Classification** — `APPROACHING_FAST` / `APPROACHING` / `STABLE` / `RECEDING`
- **Motion Trails** — fading trajectory per object (red = fast approach, orange = approaching)

### Overlays & HUD
- **5-Zone HUD** — zone dividers, danger tint per zone, green safe-zone highlight
- **Direction Arrow** — rotated chevron: −90° ← full left, −45° ↖ mid-left, 0° ↑ forward, +45° ↗ mid-right, +90° → full right
- **Danger Bar Strip** — thin green/amber/red bar per zone above the command banner
- **Nearest Obstacle Badge** — distance in metres shown bottom-right
- **Command Banner** — large text + colour accent stripe at the bottom of the frame
- **Danger Heatmap** — Gaussian overlay accumulating risk zones over time (toggle: `H`)
- **Mini Radar Panel** — 160×160 bird's-eye obstacle map bottom-left (toggle: `R`)

### Alert Engine
Spoken + visual alerts for special situations:

| Alert | Trigger |
|-------|---------|
| `[ALERT]  Traffic Light ahead` | Traffic light detected within 8 m |
| `[ALERT]  STOP SIGN` | Stop sign detected within 6 m |
| `[ALERT]  Crowded area — N people` | 4+ persons in frame simultaneously |
| `[ALERT]  Person #4 approaching fast!` | Object area growing rapidly within 4 m |
| `[ALERT]  Object on far LEFT/RIGHT` | Object at frame edge within 3 m |

Alert badges use dark pill style with `[ALERT]` tag chip — no Unicode symbols.

### Label Design
- **Bounding boxes** — thin 2px line + small L-corner accent marks
- **Object labels** — single-line dark pill: `Car  91%   4.1 m` with coloured left stripe
- **Track ID chips** — compact: `#4` (stable), `#4  near` (approaching), `#4  FAST` (fast approach)

---

## Detected Classes (20 COCO Categories)

Person, Bicycle, Car, Motorcycle, Bus, Truck, Traffic Light, Stop Sign, Bench, Bird, Cat, Dog, Chair, Couch, Potted Plant, Dining Table, TV, Laptop, Keyboard, Refrigerator

---

## Project Structure

```
smart_navigation_assistant/
│
├── main.py                      ← Full pipeline entry point
├── requirements.txt             ← All dependencies
├── download_model.py            ← Pre-download yolov8n.pt
├── README.md
│
├── models/
│   └── yolov8n.pt               ← Auto-downloaded on first run
│
├── videos/
│   └── sample_video.mp4         ← Place your test video here
│
├── output/
│   └── result.mp4               ← Annotated output (auto-generated)
│
└── utils/
    ├── __init__.py
    ├── detector.py              ← YOLOv8n wrapper, 20 obstacle classes, clean label badges
    ├── distance_estimator.py    ← Pinhole camera distance estimation
    ├── navigator.py             ← 5-zone navigation engine + HUD rendering
    ├── smoother.py              ← Command debouncer + SpeedAwareVoice
    ├── tracker.py               ← Multi-object tracker + trails + approach detection
    ├── heatmap.py               ← Danger heatmap overlay + mini radar panel
    ├── alert_engine.py          ← Special alerts with clean [ALERT] badge style
    └── voice_assistant.py       ← Non-blocking pyttsx3 TTS wrapper
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/anower8948/smart_navigation_assistant.git
cd smart_navigation_assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install PyTorch with CUDA

**RTX 3070 / 3080 / 3090 (CUDA 12.x):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**RTX 2060 / 2070 / 2080 (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU only:**
```bash
pip install torch torchvision
```

### 4. Install all dependencies
```bash
pip install -r requirements.txt
```

The YOLOv8n model (`yolov8n.pt`) downloads automatically on first run. To pre-download:
```bash
python download_model.py
```

---

## Running

### Default — uses `videos/sample_video.mp4`
```bash
python main.py
```

### Custom video file
```bash
python main.py --input path/to/video.mp4
```

### Live webcam
```bash
python main.py --webcam
python main.py --webcam --cam-id 1   # second camera
```

### Silent / headless mode
```bash
python main.py --no-voice --no-display
```

### All options
```
  --input         Input video path          (default: videos/sample_video.mp4)
  --webcam        Use webcam instead of file
  --cam-id        Webcam device ID          (default: 0)
  --model         YOLOv8 weights path       (default: models/yolov8n.pt)
  --output        Output video path         (default: output/result.mp4)
  --conf          Detection confidence      (default: 0.40)
  --no-voice      Disable TTS voice output
  --no-display    Headless mode (no window)
  --no-heatmap    Disable danger heatmap overlay
  --no-radar      Disable mini radar panel
  --no-trails     Disable motion trail overlay
```

### Runtime keyboard shortcuts
| Key | Action |
|-----|--------|
| `H` | Toggle danger heatmap on / off |
| `R` | Toggle mini radar panel on / off |
| `T` | Toggle motion trails on / off |
| `Q` / `Esc` | Quit |

---

## How Distance Estimation Works

```
Distance (m) = (Real Object Height × Focal Length px) / Bounding Box Height px
```

Reference heights used per class:

| Class | Height |
|-------|--------|
| Person | 1.70 m |
| Car / Truck | 1.50 m |
| Bus | 3.00 m |
| Motorcycle | 1.10 m |
| Bicycle | 1.00 m |
| Bench / Chair | 0.80 m |
| Default | 1.00 m |

---

## Performance

| Hardware | Approx FPS |
|----------|-----------|
| NVIDIA RTX 3070 (CUDA 12.x) | 60–80+ FPS |
| NVIDIA RTX 2060 (CUDA 11.8) | 30–55 FPS |
| Intel Core i7 (CPU only) | 8–15 FPS |
| Google Colab T4 GPU | 20–40 FPS |

---

## Recommended Test Videos

| Source | Link |
|--------|------|
| Pixabay — NYC Manhattan street | https://pixabay.com/videos/new-york-city-manhattan-people-cars-1044/ |
| Pixabay — Street people + cars | https://pixabay.com/videos/street-people-city-cars-traffic-22516/ |
| Pixabay — Broadway crossing | https://pixabay.com/videos/broadway-street-new-york-crossing-10836/ |
| Videezy — Street walking POV | https://www.videezy.com/free-video/street-walking |
| KITTI Dataset (driving) | http://www.cvlibs.net/datasets/kitti/ |
| BDD100K Dataset | https://bdd-data.berkeley.edu/ |

---

## Future Improvements

- [ ] Full camera calibration (OpenCV chessboard) for precise distance
- [ ] Stereo / depth camera support (Intel RealSense, OAK-D)
- [ ] Monocular depth network (Depth Anything v2 / MiDaS) fallback
- [ ] Pedestrian trajectory prediction (Social LSTM)
- [ ] Semantic segmentation — detect walkable pavement vs road
- [ ] TensorRT quantisation for edge deployment (Jetson Nano / Raspberry Pi)
- [ ] Multi-language voice output (Korean, Bengali, etc.)
- [ ] Haptic feedback wristband via BLE
- [ ] Smart glasses / AR headset integration
- [ ] Formal user study with visually impaired participants

---

## References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [COCO Dataset](https://cocodataset.org/)
- [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/)
- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
- [pyttsx3 Docs](https://pyttsx3.readthedocs.io/)
- [OpenCV Docs](https://docs.opencv.org/)
- [PyTorch](https://pytorch.org/)

---

*Woosong University · AI & Big Data · Smart Navigation Assistant v3*
