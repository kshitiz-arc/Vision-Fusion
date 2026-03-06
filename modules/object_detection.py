"""
VisionFusion AI — Object Detection Module
==========================================
Multi-backend object detector supporting YOLOv3/v4, SSD MobileNet,
and a lightweight Haar-based fallback — all through a unified interface.

Detection backends
------------------
yolo   : YOLOv3 / YOLOv4 via OpenCV DNN.
ssd    : SSD MobileNet v2 COCO via OpenCV DNN.
haar   : Haar cascade (car, full-body, etc.) — for offline use without weights.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

# COCO 80-class label list (abbreviated — full list loaded from file when available)
COCO_CLASSES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet",
    "tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
    "oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush",
]


@dataclass
class Detection:
    """Structured result for a single detected object.

    ``track_id`` is optional and only populated when using a detector that
    provides its own tracking (e.g. YOLOv8 `.track()`).
    """
    bbox:       Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    class_id:   int
    class_name: str
    confidence: float
    track_id:   Optional[int] = None


class ObjectDetector:
    """
    Unified object detection module.

    Parameters
    ----------
    cfg : dict
        The ``object_detection`` section from the global config.

    Example
    -------
    >>> detector = ObjectDetector(cfg["object_detection"])
    >>> dets     = detector.detect(frame)
    >>> out      = detector.draw(frame, dets)
    """

    # Per-class reproducible colors based on class_id hash
    _COLOR_PALETTE = np.random.default_rng(0).integers(
        50, 255, size=(100, 3), dtype=np.uint8
    )

    def __init__(self, cfg: dict) -> None:
        self.cfg        = cfg
        self.method     = cfg.get("method", "yolo").lower()
        self.conf_thr   = float(cfg.get("confidence_threshold", 0.5))
        self.nms_thr    = float(cfg.get("nms_threshold", 0.4))
        self.input_size = tuple(cfg.get("input_size", [416, 416]))
        self.classes    = self._load_classes(cfg.get("classes_path", ""))
        # optional yolov8-specific parameters
        self.track      = bool(cfg.get("track", False))
        self.model_path = cfg.get("yolov8_model", "yolov8n.pt")

        # network loader may differ depending on method
        if self.method == "yolov8":
            try:
                from ultralytics import YOLO
            except ImportError:
                logger.error("ultralytics package required for yolov8 method")
                self.model = None
            else:
                # YOLO constructor will download weights if necessary
                self.model = YOLO(self.model_path)
                logger.info("Loaded YOLOv8 model %s", self.model_path)
            self.net = None
        else:
            self.net = self._load_network()
            self.model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run object detection on *frame*.

        Returns
        -------
        List[Detection]  Detected objects with bboxes in absolute pixel coords.
        """
        if self.method == "yolov8":
            if self.model is None:
                return []
            return self._detect_yolov8(frame)
        if self.net is None:
            return []
        if self.method == "yolo":
            return self._detect_yolo(frame)
        elif self.method == "ssd":
            return self._detect_ssd(frame)
        return []

    def draw(self, frame: np.ndarray, detections: List[Detection],
             thickness: int = 2) -> np.ndarray:
        """Draw annotated bounding boxes on a copy of *frame*."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = tuple(int(c) for c in self._COLOR_PALETTE[det.class_id % 100])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Network loaders
    # ------------------------------------------------------------------

    def _load_network(self):
        model  = self.cfg.get("model_path", "")
        config = self.cfg.get("config_path", "")
        if not (Path(model).exists() and Path(config).exists()):
            logger.warning(
                "Object detection weights not found at '%s'. "
                "Download YOLOv3 weights and cfg, or run "
                "scripts/download_models.sh", model
            )
            return None
        net = cv2.dnn.readNet(model, config)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("Loaded object detection network from %s", model)
        return net

    @staticmethod
    def _load_classes(path: str) -> List[str]:
        p = Path(path)
        if p.exists():
            with p.open() as fh:
                return [line.strip() for line in fh if line.strip()]
        return COCO_CLASSES

    # ------------------------------------------------------------------
    # YOLO inference
    # ------------------------------------------------------------------

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, self.input_size, swapRB=True, crop=False
        )
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()

        # OpenCV 4.x returns a flat array; older versions return array-of-arrays
        unconnected = self.net.getUnconnectedOutLayers()
        if isinstance(unconnected, np.ndarray) and unconnected.ndim == 2:
            output_names = [layer_names[i[0] - 1] for i in unconnected]
        else:
            output_names = [layer_names[i - 1] for i in unconnected.flatten()]

        try:
            outputs = self.net.forward(output_names)
        except Exception as e:
            logger.error("YOLO forward pass failed: %s", e)
            return []

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for det in output:
                scores     = det[5:]
                class_id   = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < self.conf_thr:
                    continue
                cx, cy, bw, bh = (det[:4] * [w, h, w, h]).astype(int)
                x1 = max(0, cx - bw // 2)
                y1 = max(0, cy - bh // 2)
                x2 = min(w, cx + bw // 2)
                y2 = min(h, cy + bh // 2)
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(confidence)
                class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_thr, self.nms_thr
        )
        results = []
        flat_idx = indices.flatten() if hasattr(indices, "flatten") else indices
        for i in flat_idx:
            x, y, bw, bh = boxes[i]
            cid  = class_ids[i]
            name = self.classes[cid] if cid < len(self.classes) else str(cid)
            results.append(Detection(
                bbox=(x, y, x + bw, y + bh),
                class_id=cid,
                class_name=name,
                confidence=confidences[i],
            ))
        return results

    # ------------------------------------------------------------------
    # YOLOv8 inference

    def _detect_yolov8(self, frame: np.ndarray) -> List[Detection]:
        """Use the Ultralytics YOLO API. If ``self.track`` is True, run
        :meth:`YOLO.track` to also obtain object IDs; otherwise use a plain
        prediction.
        """
        if self.track:
            results = self.model.track(
                frame,
                persist=True,
                verbose=False,
                conf=self.conf_thr,
                tracker=self.cfg.get("yolov8_tracker", "bytetrack.yaml"),
            )
        else:
            results = self.model(frame)
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        confidences = res.boxes.conf.cpu().numpy() if res.boxes is not None else []
        classes = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else []
        ids = None
        if self.track and hasattr(res.boxes, "id") and res.boxes.id is not None:
            ids = res.boxes.id.cpu().numpy().astype(int)

        output = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cid = int(classes[i])
            name = self.classes[cid] if cid < len(self.classes) else str(cid)
            track_id = int(ids[i]) if ids is not None else None
            output.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=cid,
                class_name=name,
                confidence=float(confidences[i]),
                track_id=track_id,
            ))
        if ids is not None:
            logger.debug("yolov8 detected %d objects with ids %s", len(ids), ids.tolist())
        return output

    # ------------------------------------------------------------------
    # SSD inference
    # ------------------------------------------------------------------

    def _detect_ssd(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843,
            (300, 300), 127.5
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        results    = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.conf_thr:
                continue
            cid  = int(detections[0, 0, i, 1])
            box  = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            name = self.classes[cid] if cid < len(self.classes) else str(cid)
            results.append(Detection(
                bbox=(x1, y1, x2, y2),
                class_id=cid,
                class_name=name,
                confidence=conf,
            ))
        return results
