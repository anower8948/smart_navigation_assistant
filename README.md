# AI-Based Smart Navigation Assistant for Visually Impaired People
### Advanced Edition v2 — YOLOv8 + Tracking + Heatmap + Radar + Smart Voice

---

## Overview

A real-time AI navigation assistant that helps visually impaired people navigate safely using computer vision. It detects obstacles, estimates distances, tracks object movement, and provides spoken navigation guidance — all running locally on a GPU or CPU.

### What's New in v2
| Feature | Description |
|---|---|
| **Multi-Object Tracking** | Persistent IDs, trajectory trails, approach prediction |
| **Danger Heatmap** | Real-time colour overlay showing risk zones (blue→red) |
| **Mini Radar** | Bird's-eye view panel showing obstacle positions |
| **Command Smoother** | Debouncer prevents flickering commands at zone borders |
| **Speed-Aware Voice** | Speech rate increases near danger; "WARNING." prefix added |
| **Alert Engine** | Special alerts: traffic lights, stop signs, crowds, blind spots |
| **Fast-Approach Warning** | Flashing banner when object moves toward camera rapidly |
| **Live Key Toggles** | H=heatmap, R=radar, T=trails — toggle without restarting |

---

## Project Structure

```
smart_navigation_assistant/
│
├── main.py                      ← Full pipeline entry point (v2)
├── requirements.txt
├── README.md
├── download_model.py            ← Helper to pre-download yolov8n.pt
│
├── models/
│   └── yolov8n.pt               ← Auto-downloaded on first run
│
├── videos/
│   └── sample_video.mp4         ← Place your test video here
│
├── output/
│   └── result.mp4               ← Annotated output (generated)
│
└── utils/
    ├── __init__.py
    ├── detector.py              ← YOLOv8n wrapper, 25 COCO classes
    ├── distance_estimator.py    ← Pinhole camera distance estimation
    ├── navigator.py             ← LEFT/CENTER/RIGHT zone navigation
    ├── voice_assistant.py       ← Non-blocking pyttsx3 TTS
    ├── tracker.py               ← Multi-object centroid tracker + trails
    ├── heatmap.py               ← Danger heatmap + mini radar panel
    ├── smoother.py              ← Command debouncer + speed-aware voice
    └── alert_engine.py          ← Traffic light, crowd, blind-spot alerts
```

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/anower8948/smart_navigation_assistant.git
cd smart_navigation_assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

### 3. Install PyTorch with CUDA (RTX 2060 — CUDA 11.8)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
> CPU only: `pip install torch torchvision`

### 4. Install all dependencies
```bash
pip install -r requirements.txt
```

---

## Running

### Default (uses `videos/sample_video.mp4`)
```bash
python main.py
```

### Custom video
```bash
python main.py --input path/to/video.mp4
```

### Webcam
```bash
python main.py --webcam
```

### Silent / headless mode
```bash
python main.py --no-voice --no-display
```

### All options
```bash
python main.py --help
```
```
  --input        Input video path (default: videos/sample_video.mp4)
  --webcam       Use webcam
  --cam-id       Webcam device ID (default: 0)
  --model        YOLOv8 weights path (default: models/yolov8n.pt)
  --output       Output video path (default: output/result.mp4)
  --conf         Detection confidence threshold (default: 0.40)
  --no-voice     Disable TTS voice
  --no-display   Headless mode
  --no-heatmap   Disable danger heatmap overlay
  --no-radar     Disable mini radar panel
  --no-trails    Disable motion trail overlay
```

### Runtime keyboard shortcuts (while window is open)
| Key | Action |
|-----|--------|
| `H` | Toggle danger heatmap on/off |
| `R` | Toggle mini radar on/off |
| `T` | Toggle motion trails on/off |
| `Q` / `Esc` | Quit |

---

## How It Works

### Detection
YOLOv8 Nano runs on each frame detecting 25 navigation-relevant COCO classes.

### Screen Zones
```
┌──────────────┬──────────────┬──────────────┐
│     LEFT     │    CENTER    │    RIGHT     │
│   0% – 33%   │  33% – 67%  │  67% – 100%  │
└──────────────┴──────────────┴──────────────┘
```

### Navigation Logic
| Situation | Command |
|-----------|---------|
| Center obstacle < 0.8m | **STOP !** |
| Center obstacle within 4m | **Move Left** or **Move Right** (picks open side) |
| Left obstacle only | **Move Right** |
| Right obstacle only | **Move Left** |
| Both sides: steers to wider gap | **Move Left / Move Right** |
| Clear path | **Move Forward** |

### Distance Estimation
```
distance (m) = (real_height × focal_length_px) / bbox_height_px
```
Known average heights used per class (Person=1.7m, Car=1.5m, Bus=3.2m…).

### Tracking
Each detected object gets a persistent ID. The tracker:
- Matches detections frame-to-frame using IoU + centroid distance
- Computes velocity vector (direction + speed)
- Classifies approach: `APPROACHING_FAST` / `APPROACHING` / `STABLE` / `RECEDING`
- Draws fading colour trails (red=approaching, cyan=moving away)

### Danger Heatmap
A Gaussian blob is deposited at each obstacle location every frame, weighted by distance (closer = hotter). Heat decays over time. Rendered as a COLORMAP_JET overlay (blue=safe → red=danger).

### Mini Radar
A 160×160 bird's-eye panel shows all obstacles as coloured dots:
- Red dot = < 1.5m away
- Orange dot = 1.5–3m
- Cyan dot = 3–5m
- Green dot = > 5m

### Command Smoother
A command must appear for 4 consecutive frames before it replaces the current one. STOP bypasses this (instant). Prevents flickering when objects are at zone borders.

### Alert Engine
Special situations announced by voice and shown on screen:
- Traffic light detected ahead
- Stop sign ahead
- Crowded area (3+ people)
- Fast-approaching object
- Blind-spot objects (at extreme frame edges)

---

## Performance

| Hardware | Approx FPS |
|----------|-----------|
| NVIDIA RTX 2060 (CUDA) | 30–55 FPS |
| Intel Core i7 (CPU) | 8–15 FPS |
| Google Colab T4 GPU | 20–40 FPS |

---

## Recommended Test Videos

| Source | Link |
|--------|------|
| Pixabay — NYC Manhattan street | https://pixabay.com/videos/new-york-city-manhattan-people-cars-1044/ |
| Pixabay — Street people + cars | https://pixabay.com/videos/street-people-city-cars-traffic-22516/ |
| Pixabay — Broadway crossing | https://pixabay.com/videos/broadway-street-new-york-crossing-10836/ |
| Videezy — Street walking | https://www.videezy.com/free-video/street-walking |
| KITTI Dataset | http://www.cvlibs.net/datasets/kitti/ |
| BDD100K Dataset | https://bdd-data.berkeley.edu/ |

---

## Future Improvements
- [ ] Real camera calibration for precise distance
- [ ] Depth camera support (Intel RealSense / ZED)
- [ ] Audio spatialization (left/right ear = left/right obstacle)
- [ ] GPS + map overlay
- [ ] Haptic feedback via BLE wearable
- [ ] Mobile deployment (TFLite / CoreML)
- [ ] Multi-language voice guidance
- [ ] Smart glasses integration

---

## References
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [COCO Dataset](https://cocodataset.org/)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [BDD100K](https://bdd-data.berkeley.edu/)
- [pyttsx3](https://pyttsx3.readthedocs.io/)
- [OpenCV](https://docs.opencv.org/)

---

*AI & Big Data — University Project | Smart Navigation Assistant v2*
