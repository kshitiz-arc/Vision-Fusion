"""
VisionFusion AI — Motion Detection Module
===========================================
Background-subtraction and frame-differencing based motion analysis.

Algorithms
----------
mog2       : Gaussian Mixture Model (OpenCV MOG2) — handles illumination changes well.
knn        : K-Nearest Neighbours background subtractor.
frame_diff : Simple 3-frame temporal differencing — ultra-lightweight.
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MotionRegion:
    """A single detected motion region."""
    bbox: Tuple[int, int, int, int]    # (x, y, w, h)
    area: float
    centroid: Tuple[int, int]


class MotionDetector:
    """
    Configurable motion detection module.

    Parameters
    ----------
    cfg : dict
        The ``motion_detection`` section from the global config.

    Example
    -------
    >>> detector = MotionDetector(cfg["motion_detection"])
    >>> regions  = detector.detect(frame)
    """

    def __init__(self, cfg: dict) -> None:
        self.method          = cfg.get("method", "mog2").lower()
        self.min_area        = int(cfg.get("min_area", 800))
        self.history         = int(cfg.get("history", 500))
        self.var_threshold   = int(cfg.get("var_threshold", 16))
        self.detect_shadows  = bool(cfg.get("detect_shadows", True))
        self.blur_kernel     = int(cfg.get("blur_kernel", 21)) | 1   # ensure odd

        self._bg_subtractor  = self._init_subtractor()
        self._frame_buffer: deque = deque(maxlen=3)   # for frame_diff

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[MotionRegion]:
        """
        Detect motion regions in *frame*.

        Parameters
        ----------
        frame : np.ndarray  BGR image.

        Returns
        -------
        List[MotionRegion]
        """
        if self.method in ("mog2", "knn"):
            return self._detect_bg_subtraction(frame)
        else:
            return self._detect_frame_diff(frame)

    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return the raw foreground mask (uint8, single channel)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        if self.method in ("mog2", "knn"):
            mask = self._bg_subtractor.apply(blurred)
        else:
            mask = self._frame_diff_mask(blurred)
        _, binary = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)
        kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary    = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary    = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def draw(self, frame: np.ndarray, regions: List[MotionRegion],
             color: Tuple[int, int, int] = (0, 128, 255),
             thickness: int = 2) -> np.ndarray:
        """Draw bounding rectangles for each motion region."""
        out = frame.copy()
        for region in regions:
            x, y, w, h = region.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
            cx, cy = region.centroid
            cv2.circle(out, (cx, cy), 4, color, -1)
            cv2.putText(out, f"Motion {region.area:.0f}px",
                        (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
        return out

    def reset(self) -> None:
        """Re-initialise the background model."""
        self._bg_subtractor = self._init_subtractor()
        self._frame_buffer.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_subtractor(self):
        if self.method == "mog2":
            return cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows,
            )
        elif self.method == "knn":
            return cv2.createBackgroundSubtractorKNN(
                history=self.history,
                dist2Threshold=400.0,
                detectShadows=self.detect_shadows,
            )
        logger.warning("method '%s' unknown — using MOG2.", self.method)
        return cv2.createBackgroundSubtractorMOG2()

    def _detect_bg_subtraction(self, frame: np.ndarray) -> List[MotionRegion]:
        mask = self.get_foreground_mask(frame)
        return self._contours_to_regions(mask)

    def _detect_frame_diff(self, frame: np.ndarray) -> List[MotionRegion]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frame_buffer.append(gray)
        if len(self._frame_buffer) < 3:
            return []
        mask = self._frame_diff_mask(gray)
        return self._contours_to_regions(mask)

    def _frame_diff_mask(self, gray: np.ndarray) -> np.ndarray:
        if len(self._frame_buffer) < 2:
            return np.zeros_like(gray)
        buf = list(self._frame_buffer)
        diff1 = cv2.absdiff(buf[-1], buf[-2])
        diff2 = cv2.absdiff(buf[-1], buf[0]) if len(buf) >= 3 else diff1
        combined = cv2.bitwise_and(diff1, diff2)
        _, binary = cv2.threshold(combined, 25, 255, cv2.THRESH_BINARY)
        return binary

    def _contours_to_regions(self, mask: np.ndarray) -> List[MotionRegion]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] else x + w // 2
            cy = int(M["m01"] / M["m00"]) if M["m00"] else y + h // 2
            regions.append(MotionRegion(
                bbox=(x, y, w, h), area=area, centroid=(cx, cy)
            ))
        return regions
