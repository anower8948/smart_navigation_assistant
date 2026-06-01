# AI-Based Smart Navigation Assistant for Visually Impaired People
### Using Computer Vision (YOLOv8 + Distance Estimation + Voice Guidance)

---

## Overview

This project is a **real-time AI navigation assistant** designed to help visually impaired people navigate their environment safely.

It takes a **video file or live webcam feed** as input, detects nearby obstacles using **YOLOv8 Nano**, estimates their distances, and provides **spoken navigation commands** ("Move Left", "Move Right", "STOP", etc.) in real time.

### Key Features
- Object detection via **YOLOv8n** (pretrained COCO — no custom training needed)
- **Distance estimation** from bounding box size + perspective calibration
- **Left / Center / Right** zone-based navigation logic
- **Voice guidance** via `pyttsx3` (offline, no internet required)
- Annotated video output with bounding boxes, confidence scores, and distances
- Saves processed video to `output/result.mp4`
- Supports **NVIDIA GPU** (RTX 2060 / CUDA) for real-time performance

---

## Project Structure

```
smart_navigation_assistant/
│
├── main.py                    ← Entry point — run the full pipeline
├── requirements.txt           ← Python dependencies
├── README.md                  ← This file
│
├── models/
│   └── yolov8n.pt             ← YOLOv8 Nano weights (auto-downloaded)
│
├── videos/
│   └── sample_video.mp4       ← Place your test video here
│
├── output/
│   └── result.mp4             ← Annotated output video (generated)
│
└── utils/
    ├── __init__.py
    ├── detector.py            ← YOLOv8 object detection wrapper
    ├── distance_estimator.py  ← Approximate distance from bounding box
    ├── navigator.py           ← Navigation decision logic + HUD overlay
    └── voice_assistant.py     ← Non-blocking pyttsx3 TTS wrapper
```

---

## Installation

### 1. Clone / Download the project

```bash
# If downloaded as ZIP:
unzip smart_navigation_assistant.zip
cd smart_navigation_assistant
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install PyTorch with CUDA (RTX 2060 — CUDA 11.8)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> For CPU-only (slower):
> ```bash
> pip install torch torchvision
> ```

### 4. Install all other dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Default run (uses `videos/sample_video.mp4`)

```bash
python main.py
```

### Custom video file

```bash
python main.py --input path/to/your_video.mp4
```

### Webcam mode

```bash
python main.py --webcam
# or specify device ID:
python main.py --webcam --cam-id 0
```

### Headless mode (server / no display)

```bash
python main.py --no-display
```

### Disable voice

```bash
python main.py --no-voice
```

### All options

```bash
python main.py --help
```

```
optional arguments:
  --input      Path to input video file (default: videos/sample_video.mp4)
  --webcam     Use webcam instead of video file
  --cam-id     Webcam device ID (default: 0)
  --model      Path to YOLOv8 weights (default: models/yolov8n.pt)
  --output     Output video path (default: output/result.mp4)
  --conf       Detection confidence threshold 0-1 (default: 0.40)
  --no-voice   Disable voice guidance
  --no-display Run without GUI window
```

---

## How It Works

### 1. Detection
YOLOv8 Nano runs on every frame and detects navigation-relevant objects:
`Person`, `Car`, `Bus`, `Truck`, `Motorcycle`, `Bicycle`, `Bench`, `Chair`, `Traffic Light`, `Stop Sign`, etc.

### 2. Screen Zones

```
┌──────────────┬──────────────┬──────────────┐
│     LEFT     │    CENTER    │    RIGHT     │
│   0% – 33%   │  33% – 67%  │  67% – 100%  │
└──────────────┴──────────────┴──────────────┘
```

### 3. Distance Estimation

Using the pinhole camera model:

```
distance (m) = (real_object_height × focal_length_px) / bounding_box_height_px
```

Known average real-world heights are used per class (e.g. Person = 1.7m, Car = 1.5m).

### 4. Navigation Logic

| Situation | Command |
|-----------|---------|
| Obstacle in CENTER, dist < 0.8m | **STOP !** |
| Obstacle in CENTER, dist < 1.5m | **Move Left / Move Right** |
| Obstacle in CENTER, dist < 3.0m | **Slow Down** |
| Obstacle only on LEFT | **Move Right** |
| Obstacle only on RIGHT | **Move Left** |
| No obstacles | **Move Forward** |

### 5. Voice Guidance
`pyttsx3` speaks commands on change with a 2.5-second cooldown to avoid repetition.

---

## Sample Test Video Suggestions

| Source | Description |
|--------|-------------|
| [Pexels – Street Walking POV](https://www.pexels.com/search/videos/walking%20street/) | First-person pedestrian footage |
| [KITTI Vision Dataset](http://www.cvlibs.net/datasets/kitti/) | Driving / urban scenes |
| [BDD100K Dataset](https://bdd-data.berkeley.edu/) | Diverse driving scenes |
| [COCO Dataset](https://cocodataset.org/#home) | Static object images |
| Your phone camera | Walk around campus / outside |

> Simply rename your video to `sample_video.mp4` and place it in `videos/`

---

## Output

The annotated result is saved to `output/result.mp4` and includes:
- Bounding boxes with class name + confidence %
- Estimated distance for each object
- Zone dividers (LEFT | CENTER | RIGHT)
- Navigation command banner at the bottom
- Direction arrow
- Detected object panel (top-right)
- FPS counter

---

## Screenshots

> *(Add screenshots of your demo here for the university presentation)*
>
> Suggested capture: `ffmpeg -i output/result.mp4 -vf fps=0.5 screenshots/frame_%04d.png`

---

## Performance

| Hardware | Approximate FPS |
|----------|----------------|
| NVIDIA RTX 2060 (CUDA) | 35–60 FPS |
| Intel Core i7 (CPU only) | 8–15 FPS |
| Google Colab (T4 GPU) | 20–40 FPS |

---

## Future Improvements

- [ ] Real camera calibration for more accurate distance estimation
- [ ] Depth camera support (Intel RealSense / ZED)
- [ ] Custom-trained model for indoor navigation
- [ ] GPS and map integration
- [ ] Haptic feedback via USB/BLE wearable
- [ ] Mobile deployment (Android/iOS via TFLite or CoreML)
- [ ] Multi-language voice guidance
- [ ] Integration with smart glasses (e.g. VUZIX)
- [ ] Audio spatialization (left ear = left obstacle, right ear = right obstacle)

---

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [COCO Dataset](https://cocodataset.org/#home)
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
- [pyttsx3 Documentation](https://pyttsx3.readthedocs.io/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## License

MIT License — Free for academic and educational use.

---

*Project developed for university demonstration — AI & Big Data program.*
