"""
VisionFusion AI — Contour Analysis Module
==========================================
Shape extraction, classification, and geometric analysis of contours.

Provides
--------
* Multi-threshold contour extraction
* Shape classification (triangle / quadrilateral / circle / polygon)
* Geometric descriptors: area, perimeter, circularity, aspect ratio, solidity
* Convex hull and minimum enclosing circle overlays
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContourShape:
    """Geometric description of a single extracted contour."""
    contour:      np.ndarray
    bbox:         Tuple[int, int, int, int]   # (x, y, w, h)
    area:         float
    perimeter:    float
    shape_label:  str
    circularity:  float
    aspect_ratio: float
    solidity:     float
    vertices:     int


class ContourAnalyzer:
    """
    Contour extraction and shape analysis module.

    Parameters
    ----------
    cfg : dict
        The ``contour_analysis`` section from the global config.

    Example
    -------
    >>> analyzer = ContourAnalyzer(cfg["contour_analysis"])
    >>> shapes   = analyzer.analyze(frame)
    >>> out      = analyzer.draw(frame, shapes)
    """

    SHAPE_COLORS = {
        "triangle":     (255, 100, 100),
        "rectangle":    (100, 255, 100),
        "pentagon":     (100, 100, 255),
        "hexagon":      (255, 255,   0),
        "circle":       (  0, 255, 255),
        "polygon":      (200, 200, 200),
    }

    def __init__(self, cfg: dict) -> None:
        self.min_area           = int(cfg.get("min_area", 500))
        self.approx_epsilon     = float(cfg.get("approx_epsilon", 0.02))
        self.draw_bounding_rect = bool(cfg.get("draw_bounding_rect", True))
        self.draw_convex_hull   = bool(cfg.get("draw_convex_hull", False))
        self.show_shape_labels  = bool(cfg.get("show_shape_labels", True))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, frame: np.ndarray) -> List[ContourShape]:
        """
        Extract and describe all significant contours in *frame*.

        Parameters
        ----------
        frame : np.ndarray  BGR image.

        Returns
        -------
        List[ContourShape]
        """
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            shapes.append(self._describe(cnt, area))
        return shapes

    def draw(self, frame: np.ndarray, shapes: List[ContourShape]) -> np.ndarray:
        """Annotate *frame* with contours, bboxes, and shape labels."""
        out = frame.copy()
        for shape in shapes:
            color = self.SHAPE_COLORS.get(shape.shape_label, (180, 180, 180))
            cv2.drawContours(out, [shape.contour], -1, color, 2)

            if self.draw_bounding_rect:
                x, y, w, h = shape.bbox
                cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)

            if self.draw_convex_hull:
                hull = cv2.convexHull(shape.contour)
                cv2.drawContours(out, [hull], -1, (255, 255, 255), 1)

            if self.show_shape_labels:
                M  = cv2.moments(shape.contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"{shape.shape_label} ({shape.area:.0f})"
                    cv2.putText(out, label, (cx - 40, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                color, 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _describe(self, contour: np.ndarray, area: float) -> ContourShape:
        perimeter = cv2.arcLength(contour, True)
        approx    = cv2.approxPolyDP(contour,
                                     self.approx_epsilon * perimeter, True)
        vertices  = len(approx)

        circularity  = (4 * np.pi * area / (perimeter ** 2)
                        if perimeter > 0 else 0.0)
        hull_area    = cv2.contourArea(cv2.convexHull(contour))
        solidity     = area / hull_area if hull_area > 0 else 0.0
        x, y, w, h   = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0.0

        shape_label = self._classify(vertices, circularity)
        return ContourShape(
            contour=contour,
            bbox=(x, y, w, h),
            area=area,
            perimeter=perimeter,
            shape_label=shape_label,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            vertices=vertices,
        )

    @staticmethod
    def _classify(vertices: int, circularity: float) -> str:
        if circularity > 0.85:
            return "circle"
        if vertices == 3:
            return "triangle"
        if vertices == 4:
            return "rectangle"
        if vertices == 5:
            return "pentagon"
        if vertices == 6:
            return "hexagon"
        return "polygon"
