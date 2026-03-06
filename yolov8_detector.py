"""
VisionFusion AI — YOLOv8 Object Detector
==========================================
Replaces the old YOLOv3/OpenCV-DNN object detection with a clean
YOLOv8 (Ultralytics) detector. Tracking and Re-ID logic fully removed.

Drop this file into:  visionfusion-ai/modules/yolov8_detector.py

Then in perception_pipeline.py replace:
    from modules.object_detection import ObjectDetector
with:
    from modules.yolov8_detector  import YOLOv8Detector as ObjectDetector

The public API — detect(frame) and draw(frame, detections) — is identical
to the original ObjectDetector so nothing else needs to change.

Standalone usage
----------------
    python modules/yolov8_detector.py              # webcam
    python modules/yolov8_detector.py video.mp4    # video file
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Attempt to import ultralytics; give a helpful error if missing
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO as _YOLO
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared Detection dataclass (same as modules/object_detection.py)
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """Single detected object."""
    bbox:       Tuple[int, int, int, int]   # (x1, y1, x2, y2) absolute pixels
    class_id:   int
    class_name: str
    confidence: float


# ---------------------------------------------------------------------------
# Per-class reproducible colors
# ---------------------------------------------------------------------------
_COLOR_PALETTE = np.random.default_rng(0).integers(50, 255, (100, 3), dtype=np.uint8)

def _class_color(class_id: int) -> Tuple[int, int, int]:
    c = _COLOR_PALETTE[class_id % 100]
    return (int(c[0]), int(c[1]), int(c[2]))


# ---------------------------------------------------------------------------
# YOLOv8Detector
# ---------------------------------------------------------------------------
class YOLOv8Detector:
    """
    YOLOv8-based object detector (Ultralytics).

    Tracking and Re-ID are completely removed — this does pure per-frame
    detection only, matching the interface expected by PerceptionPipeline.

    Parameters
    ----------
    cfg : dict
        The ``object_detection`` section from configs/default.yaml.
        Relevant keys:
            model_path          : YOLOv8 model name or .pt path
                                  e.g. "yolov8n.pt" | "yolov8s.pt" | "yolov8m.pt"
            confidence_threshold: minimum detection confidence  (default 0.4)
            class_filter        : list of COCO class ids to keep, or null for all
                                  e.g. [0, 2]  →  persons + cars only

    Example
    -------
    >>> det  = YOLOv8Detector(cfg["object_detection"])
    >>> dets = det.detect(frame)
    >>> out  = det.draw(frame, dets)
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg       = cfg
        self.conf_thr  = float(cfg.get("confidence_threshold", 0.4))
        self.cls_filter: Optional[List[int]] = cfg.get("class_filter", None)

        # Model: prefer "yolov8_model" key, fall back to "model_path"
        model_name = (cfg.get("yolov8_model")
                      or cfg.get("model_path", "yolov8n.pt"))

        if not _ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed.\n"
                "Run:  pip install ultralytics"
            )

        print(f"[YOLOv8Detector] Loading model '{model_name}' "
              f"(downloads ~6 MB on first run)…")
        self._model = _YOLO(model_name)
        print("[YOLOv8Detector] Model ready.")

    # ------------------------------------------------------------------
    # Public API  (same as ObjectDetector)
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 detection on *frame*.

        Parameters
        ----------
        frame : np.ndarray  BGR image from camera / video.

        Returns
        -------
        List[Detection]  — bboxes in (x1, y1, x2, y2) pixel coordinates.
        """
        results = self._model(
            frame,
            verbose=False,
            conf=self.conf_thr,
            classes=self.cls_filter,   # None → detect all 80 COCO classes
        )[0]

        detections: List[Detection] = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes   = results.boxes.xyxy.cpu().numpy()       # (N, 4) float
        confs   = results.boxes.conf.cpu().numpy()        # (N,)
        classes = results.boxes.cls.cpu().numpy().astype(int)  # (N,)

        for box, conf, cls_id in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            name = self._model.names.get(cls_id, str(cls_id))
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=cls_id,
                class_name=name,
                confidence=float(conf),
            ))

        return detections

    def draw(self, frame: np.ndarray,
             detections: List[Detection],
             thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes + labels on a copy of *frame*.

        Parameters
        ----------
        frame      : np.ndarray       BGR image.
        detections : List[Detection]  Output of :meth:`detect`.
        thickness  : int              Box line thickness.

        Returns
        -------
        np.ndarray  Annotated BGR image.
        """
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = _class_color(det.class_id)

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            # Label background + text
            label = f"{det.class_name}  {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(out,
                          (x1, y1 - th - 10),
                          (x1 + tw + 6, y1),
                          color, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return out

    @property
    def class_names(self) -> dict:
        """Return the model's class-id → name mapping."""
        return self._model.names


# ===========================================================================
# Standalone runner  (python modules/yolov8_detector.py [source])
# ===========================================================================

def _draw_hud(img: np.ndarray, fps: float,
              active_filter: str, det_count: int,
              show_help: bool) -> None:
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (w - 255, 5), (w - 5, 115), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, f"FPS:      {fps:5.1f}",           (w-245, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)
    cv2.putText(img, f"Filter:   {active_filter}",       (w-245, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
    cv2.putText(img, f"Objects:  {det_count}",           (w-245, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0),  1)
    if not show_help:
        cv2.putText(img, "H = help", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1)


def _draw_help(img: np.ndarray) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (320, 185), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    lines = [
        ("  CONTROLS",                (0, 210, 255), 2),
        ("",                           None,          1),
        ("  F   Cycle class filter",   (210, 210, 210), 1),
        ("  M   Toggle mirror",        (210, 210, 210), 1),
        ("  H   Toggle this help",     (210, 210, 210), 1),
        ("  Q   Quit",                 (210, 210, 210), 1),
    ]
    for i, (text, color, thickness) in enumerate(lines):
        if color:
            cv2.putText(img, text, (15, 38 + i * 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness)


# Class filter cycling (same as original script)
_CLASS_FILTERS = [None, [0], [2], [0, 2]]
_CLASS_LABELS  = ["All", "Person", "Car", "Person+Car"]


def main() -> None:
    source: int | str = 0
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        source = int(arg) if arg.isdigit() else arg

    # Minimal config so YOLOv8Detector can be constructed standalone
    cfg = {
        "yolov8_model":          "yolov8n.pt",
        "confidence_threshold":  0.4,
        "class_filter":          None,
    }
    detector = YOLOv8Detector(cfg)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source '{source}'")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mirror           = True
    show_help        = True
    filter_idx       = 0
    fps_smoothed     = 0.0

    print("YOLOv8 detector running.")
    print("Controls: F=filter  M=mirror  H=help  Q=quit\n")

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("Stream ended.")
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        # Apply current class filter
        detector.cls_filter = _CLASS_FILTERS[filter_idx]
        detections = detector.detect(frame)
        frame      = detector.draw(frame, detections)

        elapsed      = time.perf_counter() - t0
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * (1.0 / elapsed if elapsed > 0 else 0)

        _draw_hud(frame, fps_smoothed,
                  _CLASS_LABELS[filter_idx], len(detections), show_help)
        if show_help:
            _draw_help(frame)

        cv2.imshow("VisionFusion — YOLOv8 Detector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            print("Quit.")
            break
        elif key == ord('m'):
            mirror = not mirror
            print(f"Mirror: {'ON' if mirror else 'OFF'}")
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('f'):
            filter_idx = (filter_idx + 1) % len(_CLASS_FILTERS)
            print(f"Class filter: {_CLASS_LABELS[filter_idx]}")

    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")


if __name__ == "__main__":
    main()
