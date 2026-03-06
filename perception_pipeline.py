"""
VisionFusion AI — Perception Pipeline
=======================================
Orchestrates all vision modules into a single streaming inference pipeline.

Pipeline stages
---------------
1. Capture       — fetch frame from camera / video / image
2. Preprocessing — resize, denoise, color conversion
3. Edge Detection — Canny / Sobel / etc.
4. Face Detection — Haar / DNN / MediaPipe
5. Motion Detection — MOG2 / KNN / frame-diff
6. Object Detection — YOLOv3 / SSD
7. Contour Analysis — shape extraction and classification
8. CNN Classification — ResNet / MobileNet per-ROI
9. Multi-Object Tracking — centroid / IoU association
10. Visualization — composite display frame
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from modules import (
    Preprocessor, EdgeDetector, FaceDetector, MotionDetector,
    ContourAnalyzer, ObjectDetector, CentroidTracker, Visualizer,
    CNNClassifier,
    FaceDetection, MotionRegion, ContourShape, Detection, Track,
    ClassificationResult,
)
# YOLOv8 detector — used when config method == "yolov8"
try:
    from modules.yolov8_detector import YOLOv8Detector
    _YOLOV8_AVAILABLE = True
except ImportError:
    _YOLOV8_AVAILABLE = False
from utils.logger import get_logger
from utils.timer  import FPSCounter, StageTimer

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """All outputs produced by one pass of the perception pipeline."""
    frame:            np.ndarray                       # annotated BGR frame
    raw_frame:        np.ndarray                       # unmodified input frame
    fps:              float                            = 0.0
    edge_map:         Optional[np.ndarray]             = None
    face_detections:  List[FaceDetection]              = field(default_factory=list)
    motion_regions:   List[MotionRegion]               = field(default_factory=list)
    object_detections: List[Detection]                 = field(default_factory=list)
    contour_shapes:   List[ContourShape]               = field(default_factory=list)
    tracks:           List[Track]                      = field(default_factory=list)
    cnn_result:       Optional[ClassificationResult]   = None
    latencies_ms:     Dict[str, float]                 = field(default_factory=dict)


class PerceptionPipeline:
    """
    Full multi-stage computer vision perception pipeline.

    Parameters
    ----------
    cfg : dict
        Root VisionFusion configuration dictionary (loaded from YAML).

    Example
    -------
    >>> pipeline = PerceptionPipeline(cfg)
    >>> pipeline.start("video", source="sample.mp4")
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        logger.info("Initialising VisionFusion AI Perception Pipeline …")

        # — Module instantiation -------------------------------------------
        self.preprocessor   = Preprocessor(cfg.get("preprocessing", {}))
        self.edge_detector  = EdgeDetector(cfg.get("edge_detection", {}))
        self.face_detector  = FaceDetector(cfg.get("face_detection", {}))
        self.motion_detector = MotionDetector(cfg.get("motion_detection", {}))
        # Auto-select YOLOv8 or legacy OpenCV-DNN object detector
        _od_cfg = cfg.get("object_detection", {})
        if _od_cfg.get("method", "yolo") == "yolov8" and _YOLOV8_AVAILABLE:
            self.object_detector = YOLOv8Detector(_od_cfg)
            logger.info("Object detector: YOLOv8 (Ultralytics)")
        else:
            self.object_detector = ObjectDetector(_od_cfg)
            logger.info("Object detector: YOLOv3 / OpenCV DNN")
        self.contour_analyzer = ContourAnalyzer(cfg.get("contour_analysis", {}))
        self.tracker        = CentroidTracker(
            max_disappeared=cfg.get("tracking", {}).get("max_disappeared", 30),
            max_distance=cfg.get("tracking", {}).get("max_distance", 50),
        )
        self.cnn_classifier = CNNClassifier(cfg.get("cnn_classifier", {}))
        self.visualizer     = Visualizer(cfg.get("visualization", {}))

        # — Module enable flags -------------------------------------------
        self._enabled = {
            "edge":    cfg.get("edge_detection",   {}).get("enabled", True),
            "face":    cfg.get("face_detection",   {}).get("enabled", True),
            "motion":  cfg.get("motion_detection", {}).get("enabled", True),
            "objects": cfg.get("object_detection", {}).get("enabled", True),
            "contour": cfg.get("contour_analysis", {}).get("enabled", True),
            "cnn":     cfg.get("cnn_classifier",   {}).get("enabled", False),
            "track":   cfg.get("tracking",         {}).get("enabled", True),
        }

        # — Diagnostics -----------------------------------------------
        self._fps     = FPSCounter(window=30)
        self._timer   = StageTimer()

        logger.info("Pipeline ready. Active modules: %s",
                    [k for k, v in self._enabled.items() if v])

    # ------------------------------------------------------------------
    # Streaming entry-points
    # ------------------------------------------------------------------

    def start(self, mode: str = "webcam",
              source: int | str = 0,
              display: bool = True,
              max_frames: Optional[int] = None) -> None:
        """
        Start the pipeline on a live video stream or file.

        Parameters
        ----------
        mode       : "webcam" | "video" | "image"
        source     : camera index (int) or file path (str)
        display    : whether to show cv2.imshow window
        max_frames : optional frame budget (useful for benchmarking)
        """
        if mode == "image":
            self._run_image(str(source))
            return

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error("Cannot open video source: %s", source)
            return

        # Apply requested resolution
        inp = self.cfg.get("input", {})
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  inp.get("width",  1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, inp.get("height",  720))

        logger.info("Streaming from '%s' (mode=%s) …  Press Q to quit.", source, mode)
        frame_count = 0
        # Flip mode: 1=horizontal mirror (default for webcam), 0=vertical, -1=both, None=off
        flip_mode = inp.get("flip", 1)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of stream.")
                    break

                # Mirror the webcam so it feels natural (left/right correct)
                if mode == "webcam" and flip_mode is not None:
                    frame = cv2.flip(frame, int(flip_mode))

                result = self.process_frame(frame)
                self._fps.tick()
                frame_count += 1

                if display:
                    window_title = "VisionFusion AI"
                    cv2.imshow(window_title, result.frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                    self._handle_keypress(key)

                if max_frames and frame_count >= max_frames:
                    break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            logger.info(self._timer.summary())

    def process_frame(self, raw_frame: np.ndarray) -> PipelineResult:
        """
        Execute all pipeline stages on a single frame.

        Parameters
        ----------
        raw_frame : np.ndarray  Raw BGR frame from capture.

        Returns
        -------
        PipelineResult
        """
        result = PipelineResult(frame=raw_frame.copy(), raw_frame=raw_frame)

        # ── 1. Preprocessing ────────────────────────────────────────────
        with self._timer("preprocessing"):
            frame = self.preprocessor.process(raw_frame)

        working = frame.copy()

        # ── 2. Edge Detection ───────────────────────────────────────────
        edge_map = None
        if self._enabled["edge"]:
            with self._timer("edge_detection"):
                edge_map = self.edge_detector.detect(frame)
            result.edge_map = edge_map

        # ── 3. Face Detection ───────────────────────────────────────────
        if self._enabled["face"]:
            with self._timer("face_detection"):
                faces = self.face_detector.detect(frame)
                working = self.face_detector.draw(working, faces)
            result.face_detections = faces

        # ── 4. Motion Detection ─────────────────────────────────────────
        if self._enabled["motion"]:
            with self._timer("motion_detection"):
                motion_regions = self.motion_detector.detect(frame)
                working = self.motion_detector.draw(working, motion_regions)
            result.motion_regions = motion_regions

        # ── 5. Object Detection ─────────────────────────────────────────
        if self._enabled["objects"]:
            with self._timer("object_detection"):
                detections = self.object_detector.detect(frame)
                working = self.object_detector.draw(working, detections)
            result.object_detections = detections

        # ── 6. Contour Analysis ─────────────────────────────────────────
        if self._enabled["contour"]:
            with self._timer("contour_analysis"):
                shapes = self.contour_analyzer.analyze(frame)
                working = self.contour_analyzer.draw(working, shapes)
            result.contour_shapes = shapes

        # ── 7. CNN Classification ───────────────────────────────────────
        if self._enabled["cnn"]:
            with self._timer("cnn_classifier"):
                cnn_res = self.cnn_classifier.classify(frame)
                if cnn_res:
                    self.visualizer.put_text_with_bg(
                        working,
                        f"Scene: {cnn_res.class_name} ({cnn_res.confidence:.2f})",
                        (10, working.shape[0] - 20),
                    )
            result.cnn_result = cnn_res

        # ── 8. Multi-Object Tracking ────────────────────────────────────
        # Build a unified bbox list from whatever detectors are active.
        # Face bboxes are (x,y,w,h) — convert to (x1,y1,x2,y2) for tracker.
        track_bboxes: list = []
        if self._enabled["track"]:
            if result.object_detections:
                track_bboxes.extend([det.bbox for det in result.object_detections])
            if result.face_detections:
                for fd in result.face_detections:
                    x, y, w, h = fd.bbox
                    track_bboxes.append((x, y, x + w, y + h))

        if self._enabled["track"] and track_bboxes:
            with self._timer("tracking"):
                tracks = self.tracker.update(track_bboxes)
                working = CentroidTracker.draw(working, tracks)
            result.tracks = tracks

        # ── 9. Visualization ────────────────────────────────────────────
        with self._timer("visualization"):
            active = [k for k, v in self._enabled.items() if v]
            result.frame = self.visualizer.compose(
                working,
                fps=self._fps.fps,
                active_modules=active,
                edge_map=edge_map,
                overlay_edges=self.cfg.get("visualization", {}).get("overlay_edges", True),
            )

        result.fps = self._fps.fps
        return result

    # ------------------------------------------------------------------
    # Image mode
    # ------------------------------------------------------------------

    def _run_image(self, path: str) -> None:
        frame = cv2.imread(path)
        if frame is None:
            logger.error("Cannot read image: %s", path)
            return
        result = self.process_frame(frame)
        logger.info("Image processed. Showing result — press any key to close.")
        cv2.imshow("VisionFusion AI — Image", result.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Runtime controls
    # ------------------------------------------------------------------

    def _handle_keypress(self, key: int) -> None:
        """Toggle modules at runtime via keyboard shortcuts."""
        key_map = {
            ord("e"): "edge",
            ord("f"): "face",
            ord("m"): "motion",
            ord("o"): "objects",
            ord("c"): "contour",
            ord("n"): "cnn",
            ord("t"): "track",
        }
        if key in key_map:
            mod = key_map[key]
            self._enabled[mod] = not self._enabled[mod]
            state = "ON" if self._enabled[mod] else "OFF"
            logger.info("Module '%s' toggled %s", mod, state)

    def toggle_module(self, name: str, enabled: bool) -> None:
        """Programmatically enable / disable a pipeline module."""
        if name in self._enabled:
            self._enabled[name] = enabled

    @property
    def latency_summary(self) -> str:
        return self._timer.summary()
