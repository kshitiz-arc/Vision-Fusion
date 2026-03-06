"""
VisionFusion AI — Face Detection Module
=========================================
Multi-backend face detector supporting Haar cascades, OpenCV DNN (SSD ResNet-10),
and MediaPipe Face Detection.

Backend comparison
------------------
haar      : Fast, lightweight. Prone to false positives in non-frontal views.
dnn       : SSD ResNet-10 via OpenCV DNN. Good balance of speed and accuracy.
mediapipe : BlazeFace model. Robust to scale/rotation. Requires ``mediapipe`` package.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FaceDetection:
    """Structured result for a single detected face."""
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) in pixel coords
    confidence: float = 1.0
    landmarks: List[Tuple[int, int]] = field(default_factory=list)


class FaceDetector:
    """
    Unified face detection interface.

    Parameters
    ----------
    cfg : dict
        The ``face_detection`` section from the global config.

    Example
    -------
    >>> detector = FaceDetector(cfg["face_detection"])
    >>> detections = detector.detect(frame)
    >>> annotated  = detector.draw(frame, detections)
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg        = cfg
        self.method     = cfg.get("method", "dnn").lower()
        self.conf_thr   = float(cfg.get("confidence_threshold", 0.7))
        self.blur_faces = bool(cfg.get("blur_faces", False))
        self._detector  = self._init_detector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect all faces in *frame*.

        Returns
        -------
        List[FaceDetection]
        """
        if self.method == "haar":
            return self._detect_haar(frame)
        elif self.method == "dnn":
            return self._detect_dnn(frame)
        elif self.method == "mediapipe":
            return self._detect_mediapipe(frame)
        return []

    def draw(self, frame: np.ndarray,
             detections: List[FaceDetection],
             color: Tuple[int, int, int] = (0, 255, 0),
             thickness: int = 2) -> np.ndarray:
        """
        Annotate *frame* with bounding boxes and optional face blur.

        Returns a copy of the frame with annotations.
        """
        out = frame.copy()
        for det in detections:
            x, y, w, h = det.bbox
            if self.blur_faces:
                face_roi = out[y:y + h, x:x + w]
                if face_roi.size > 0:
                    out[y:y + h, x:x + w] = cv2.GaussianBlur(face_roi, (91, 91), 0)
            else:
                cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
                label = f"Face {det.confidence:.2f}"
                cv2.putText(out, label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Detector initializers
    # ------------------------------------------------------------------

    def _init_detector(self):
        if self.method == "haar":
            cascade_path = self.cfg.get("haar_cascade",
                                        "haarcascade_frontalface_default.xml")
            # cv2.data.haarcascades provides bundled models
            full_path = str(Path(cv2.data.haarcascades) / cascade_path)
            detector  = cv2.CascadeClassifier(full_path)
            if detector.empty():
                logger.error("Failed to load Haar cascade: %s", full_path)
            return detector

        elif self.method == "dnn":
            model_path  = self.cfg.get("dnn_model", "")
            config_path = self.cfg.get("dnn_config", "")
            if Path(model_path).exists() and Path(config_path).exists():
                return cv2.dnn.readNetFromCaffe(config_path, model_path)
            logger.warning("DNN face model not found — falling back to Haar.")
            self.method = "haar"
            return self._init_detector()

        elif self.method == "mediapipe":
            try:
                import mediapipe as mp
                mp_fd     = mp.solutions.face_detection
                return mp_fd.FaceDetection(min_detection_confidence=self.conf_thr)
            except ImportError:
                logger.warning("mediapipe not installed — falling back to Haar.")
                self.method = "haar"
                return self._init_detector()

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _detect_haar(self, frame: np.ndarray) -> List[FaceDetection]:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        rects = self._detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )
        results = []
        for (x, y, w, h) in (rects if len(rects) else []):
            results.append(FaceDetection(bbox=(int(x), int(y), int(w), int(h))))
        return results

    def _detect_dnn(self, frame: np.ndarray) -> List[FaceDetection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self._detector.setInput(blob)
        detections = self._detector.forward()
        results    = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.conf_thr:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            results.append(FaceDetection(
                bbox=(x1, y1, x2 - x1, y2 - y1),
                confidence=conf,
            ))
        return results

    def _detect_mediapipe(self, frame: np.ndarray) -> List[FaceDetection]:
        import mediapipe as mp
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result  = self._detector.process(rgb)
        results = []
        if not result.detections:
            return results
        h, w = frame.shape[:2]
        for detection in result.detections:
            bb   = detection.location_data.relative_bounding_box
            x    = int(bb.xmin * w)
            y    = int(bb.ymin * h)
            bw   = int(bb.width * w)
            bh   = int(bb.height * h)
            conf = detection.score[0] if detection.score else 1.0
            results.append(FaceDetection(
                bbox=(x, y, bw, bh),
                confidence=float(conf),
            ))
        return results
