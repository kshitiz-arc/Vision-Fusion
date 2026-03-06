# VisionFusion AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/OpenCV-4.8%2B-green?style=for-the-badge&logo=opencv" />
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-orange?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Research%20Grade-blueviolet?style=for-the-badge" />
</p>

---

## Overview

**VisionFusion AI** is a modular, realtime computer vision engine that blends
classical OpenCV algorithms with modern deep learning models. It offers a
configurable pipeline that can process live webcam, video, or image inputs and
handle tasks such as edge detection, face/motion/object detection, contour
analysis, CNN-based classification, and multi-object tracking — all in a single
unified framework.

The system is designed for:

* Robotics perception
* Autonomous systems and UAVs
* Intelligent surveillance
* Visual AI research and prototyping
* Edge deployment on CPU/GPU hardware

It ships with builtin support for YOLOv8 (downloaded automatically) and can
fall back to traditional OpenCV DNN detectors and trackers.

---

## Features

* **Multi-stage pipeline** with independent toggles for each module
* **YOLOv8 object detection + tracking** (auto-download weights)
* Legacy support for YOLOv3 / SSD via OpenCV DNN
* Face detection (Haar, DNN, MediaPipe), motion analysis, contour shapes
* Optional CNN classifier (ResNet/MobileNet/EfficientNet) for scene labels
* Centroid & IoU trackers or YOLOv8s built-in IDs
* Configurable via YAML + command-line overrides
* Keyboard controls for live toggling (see below)
* Extensive logging and latency profiling

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/visionfusion-ai.git
cd visionfusion-ai
python -m venv .venv      # optional but recommended
source .venv/Scripts/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. (Optional) download pretrained weights

YOLOv8 models are auto-fetched on first run. Other weights (YOLOv3, face
SSD) are available in `models/pretrained/` and can be downloaded with the
provided script or manually:

```bash
# one-off downloads
mkdir -p models/pretrained
wget -P models/pretrained/ https://pjreddie.com/media/files/yolov3.weights
wget -P models/pretrained/ https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
```

### 3. Run the pipeline

```bash
python main.py --mode webcam
```

The default configuration (`configs/default.yaml`) uses YOLOv8 with tracking
enabled. Use `--config` to specify a custom YAML file.

### 4. CLI options

```
usage: main.py [-h] [--mode {webcam,video,image,reid}]
               [--source SOURCE] [--config CONFIG] [--no-display]
               [--max-frames MAX_FRAMES]
               [--disable {edge,face,motion,objects,contour,cnn,track} [disable ...]]
               [--enable {edge,face,motion,objects,contour,cnn,track} [enable ...]]
```

### 5. Keyboard controls (live webcam/video)

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `e` | Toggle edge detection |
| `f` | Toggle face detection |
| `m` | Toggle motion detection |
| `o` | Toggle object detection (YOLOv8/YOLOv3/SSD) |
| `c` | Toggle contour analysis |
| `n` | Toggle CNN classifier |
| `t` | Toggle tracking (centroid tracker if YOLOv8 IDs not active) |

---

## Configuration

All options live in `configs/default.yaml`. You can override values by
providing a different config file or using the `--enable`/`--disable` CLI
flags. Key sections include:

```yaml
system:
  log_level: "INFO"        # DEBUG for verbose output
...
object_detection:
  enabled: true
  method: "yolov8"        # alternatives: yolov3, ssd
  yolov8_model: "yolov8n.pt"
  track: true               # use YOLOv8 built-in tracking
  confidence_threshold: 0.5
  nms_threshold: 0.4
tracking:
  enabled: true
  algorithm: "csrt"        # centroid or OpenCV single-target tracker
```

You can chain overrides with `ConfigLoader.load_with_overrides()` if you
programmatically build configurations in experiments.

---

## Object Detection / Tracking

The object detection module now supports three methods:

* `yolov8` – uses the Ultralytics package; weights auto-download. Set
  `track: true` to run `model.track()` and receive `track_id`s.
* `yolo` – OpenCV DNN YOLOv3/v4 (requires `model_path` and `config_path`).
* `ssd`  – SSD MobileNet via OpenCV DNN.

When YOLOv8 tracking is active the pipeline no longer runs the centroid
tracker; IDs from the model are drawn directly. You can still toggle the
whole module with `o` or disable tracking from config for centroid-based
association.

---

## Development & Extensibility

* Add new modules under `modules/` and register them in
  `perception_pipeline.py`.
* Config loader and logger utilities make it easy to maintain structured
  settings and consistent output.
* Run `python -m pytest` to execute unit tests (if added).

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for
details.

---

## Acknowledgements

Built from a collection of academic and opensource CV scripts; YOLOv8 was
provided by Ultralytics and OpenCVs DNN module powers legacy detectors.

Enjoy building with VisionFusion AI!  
