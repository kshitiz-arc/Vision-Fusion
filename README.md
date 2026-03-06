<div align="center">

# 🔭 VisionFusion AI

### Unified Computer Vision Perception Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=for-the-badge&logo=opencv)](https://opencv.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=for-the-badge)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](LICENSE)

*Multi-Modal Vision Pipeline for Intelligent Scene Understanding*

</div>

---

## Abstract

**VisionFusion AI** is a modular, research-grade computer vision system that unifies classical OpenCV algorithms with modern deep learning inference into a single real-time perception pipeline. The system integrates face detection, motion analysis, edge extraction, contour classification, YOLOv8 object detection, multi-object tracking, and optional CNN scene classification — all configurable through a single YAML file and toggleable live via keyboard shortcuts.

Designed for use in robotics perception, autonomous systems, intelligent surveillance, and visual AI research.

---

## Architecture

```
Input (Webcam / Video / Image)
        │
        ▼
┌─────────────────────────────────────────────────┐
│              Perception Pipeline                │
│                                                 │
│  Preprocessing  →  Edge Detection               │
│       │                  │                      │
│       ▼                  ▼                      │
│  Face Detection     Contour Analysis            │
│       │                  │                      │
│       ▼                  ▼                      │
│  Motion Detection   Object Detection (YOLOv8)  │
│       │                  │                      │
│       └──────┬───────────┘                      │
│              ▼                                  │
│       Multi-Object Tracker                      │
│              │                                  │
│              ▼                                  │
│    CNN Classifier (optional)                    │
│              │                                  │
│              ▼                                  │
│       Visualization + HUD                       │
└─────────────────────────────────────────────────┘
        │
        ▼
   Display / Export
```

---

## Project Structure

```
Vision-Fusion/
│
├── main.py                    # Unified entry-point
├── requirements.txt
├── default.yaml               # Master configuration
│
├── configs/                   # Experiment-specific config overrides
│
├── modules/
│   ├── preprocessing.py       # Resize, denoise, CLAHE, normalization
│   ├── edge_detection.py      # Canny, Sobel, Laplacian, Scharr
│   ├── face_detection.py      # Haar / DNN (SSD ResNet-10) / MediaPipe
│   ├── motion_detection.py    # MOG2, KNN, Frame-Differencing
│   ├── contour_analysis.py    # Shape extraction and geometric classification
│   ├── object_detection.py    # YOLOv3/SSD via OpenCV DNN (legacy)
│   ├── yolov8_detector.py     # YOLOv8 via Ultralytics (recommended)
│   ├── tracking.py            # Centroid tracker + OpenCV CSRT/KCF
│   ├── visualization.py       # HUD, overlays, color schemes
│   └── cnn_classifier.py      # ResNet / MobileNet / EfficientNet
│
├── pipelines/
│   ├── perception_pipeline.py # Orchestrated multi-stage pipeline
│   └── training_pipeline.py   # CNN training harness
│
├── evaluation/
│   ├── metrics.py             # Accuracy, Precision, Recall, F1, mAP
│   └── evaluate.py            # Evaluation runner + report generation
│
├── data/
│   ├── dataset_loaders.py     # CIFAR-10, COCO, Custom, VideoFrame loaders
│   └── coco_classes.txt       # 80-class COCO label list
│
├── models/
│   └── pretrained/            # ← Place downloaded model weights here
│
├── experiments/
│   └── notebooks/
│       └── ablation_study.py  # Per-module latency benchmarking
│
└── utils/
    ├── logger.py              # Color-coded structured logging
    ├── config_loader.py       # YAML config loading and merging
    └── timer.py               # FPS counter and stage latency profiler
```

---

## Installation

### Prerequisites

- Python **3.10 or higher**
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/kshitiz-arc/Vision-Fusion.git
cd Vision-Fusion
```

### Step 2 — Create a virtual environment *(recommended)*

```bash
python -m venv venv

# Activate on Mac / Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## ⚠️ Required: Download YOLOv3 Weights

The object detection module requires the **YOLOv3 pre-trained weights file** (~236 MB) which is too large to ship with the repository. You must download it manually before running the pipeline.

**Download link:**
```
https://data.pjreddie.com/files/yolov3.weights
```

**Where to place it:**

After downloading, move the file to exactly this path inside the project:

```
Vision-Fusion/
└── models/
    └── pretrained/
        └── yolov3.weights      ← place the file here
```

**One-line download (Linux / Mac / Git Bash on Windows):**

```bash
wget -P models/pretrained/ https://data.pjreddie.com/files/yolov3.weights
```

**On Windows (PowerShell):**

```powershell
Invoke-WebRequest -Uri "https://data.pjreddie.com/files/yolov3.weights" `
                  -OutFile "models\pretrained\yolov3.weights"
```

> **Note:** If you prefer to use **YOLOv8** instead (no manual download required — weights auto-fetch on first run), set `method: "yolov8"` in `default.yaml` under `object_detection`. Install the Ultralytics package first:
> ```bash
> pip install ultralytics
> ```

---

## Face Detector Weights *(optional but recommended)*

For the DNN-based face detector, also download these two files into `models/pretrained/`:

```bash
# Face detection model (SSD ResNet-10)
wget -P models/pretrained/ https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel

# Config file
wget -P models/pretrained/ https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
```

If these are absent, the system automatically falls back to the built-in Haar cascade face detector.

---

## Running the System

### Live webcam

```bash
python main.py --mode webcam
```

### Video file

```bash
python main.py --mode video --source path/to/video.mp4
```

### Single image

```bash
python main.py --mode image --source path/to/image.jpg
```

### Disable specific modules

```bash
# Run with only face and edge detection
python main.py --mode webcam --disable objects motion contour cnn track
```

### Custom configuration

```bash
python main.py --mode webcam --config configs/my_config.yaml
```

---

## Keyboard Controls

When running in webcam or video mode, use these keys to toggle modules live:

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `E` | Toggle edge detection |
| `F` | Toggle face detection |
| `M` | Toggle motion detection |
| `O` | Toggle object detection |
| `C` | Toggle contour analysis |
| `N` | Toggle CNN classifier |
| `T` | Toggle object tracking |

---

## Configuration Reference

All behaviour is controlled through `default.yaml`. Key sections:

```yaml
input:
  mode: webcam          # webcam | video | image
  width: 1280
  height: 720
  flip: 1               # 1 = mirror (default) | 0 = vertical | null = off

object_detection:
  enabled: true
  method: yolo          # yolo (YOLOv3) | yolov8 | ssd
  model_path: "models/pretrained/yolov3.weights"   # ← downloaded file goes here
  config_path: "models/pretrained/yolov3.cfg"
  confidence_threshold: 0.5

face_detection:
  enabled: true
  method: dnn           # haar | dnn | mediapipe
  confidence_threshold: 0.7
  blur_faces: false

tracking:
  enabled: true
  algorithm: csrt       # csrt | kcf | mosse | centroid
```

---

## Training the CNN Classifier

To train a classification head on CIFAR-10:

```bash
python training_pipeline.py \
    --config default.yaml \
    --dataset cifar10 \
    --arch resnet18 \
    --epochs 50
```

Checkpoints are saved to `models/checkpoints/`.

---

## Evaluation

```bash
python evaluate.py \
    --config default.yaml \
    --weights models/checkpoints/resnet18_best.pth \
    --dataset cifar10 \
    --output evaluation/results
```

Outputs a full classification report and a confusion matrix heatmap.

---

## Performance Benchmarks

Tested on Intel Core i7, CPU-only, 1280×720 input:

| Configuration | FPS |
|---------------|-----|
| Edge detection only | ~95 |
| Edge + Face (Haar) | ~60 |
| Edge + Face + Motion | ~35 |
| Full pipeline (no CNN) | ~25 |
| Full pipeline + CNN | ~12 |

---

## Dependencies

Key packages from `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `opencv-python` | Core computer vision |
| `opencv-contrib-python` | Additional trackers |
| `torch` / `torchvision` | CNN inference and training |
| `ultralytics` | YOLOv8 object detection |
| `mediapipe` | Face / hand / pose detection |
| `PyYAML` | Configuration loading |
| `scikit-learn` | Evaluation metrics |
| `matplotlib` / `seaborn` | Plotting and confusion matrix |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## License

This project is released under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgements
YOLOv8 is provided by [Ultralytics](https://ultralytics.com). Pre-trained face detection models are from the OpenCV DNN model zoo.
