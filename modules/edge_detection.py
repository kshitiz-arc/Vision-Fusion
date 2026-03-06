"""
VisionFusion AI — Edge Detection Module
=========================================
Multi-algorithm edge and feature extraction with configurable overlays.

Supported algorithms
---------------------
* Canny         – gradient-based, gold standard for clean edges
* Sobel         – directional first-order derivative
* Laplacian     – second-order derivative (isotropic)
* Scharr        – improved Sobel with better rotational symmetry
* Structured Forests (SE) – learned edge detector (requires opencv-contrib)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class EdgeDetector:
    """
    Configurable edge detection module.

    Parameters
    ----------
    cfg : dict
        The ``edge_detection`` section from the global config.

    Example
    -------
    >>> detector = EdgeDetector(cfg["edge_detection"])
    >>> edges    = detector.detect(frame)
    >>> overlaid = detector.overlay(frame, edges)
    """

    METHODS = ("canny", "sobel", "laplacian", "scharr")

    def __init__(self, cfg: dict) -> None:
        self.method     = cfg.get("method", "canny").lower()
        self.t1         = int(cfg.get("canny_threshold1", 50))
        self.t2         = int(cfg.get("canny_threshold2", 150))
        self.aperture   = int(cfg.get("canny_aperture", 3))
        self.alpha      = float(cfg.get("overlay_alpha", 0.4))

        if self.method not in self.METHODS:
            logger.warning("Unknown edge method '%s'. Falling back to 'canny'.", self.method)
            self.method = "canny"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run edge detection on *frame* and return a grayscale edge map.

        Parameters
        ----------
        frame : np.ndarray
            BGR input image.

        Returns
        -------
        np.ndarray
            Single-channel uint8 edge map (0 = no edge, 255 = edge).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.method == "canny":
            return self._canny(gray)
        elif self.method == "sobel":
            return self._sobel(gray)
        elif self.method == "laplacian":
            return self._laplacian(gray)
        elif self.method == "scharr":
            return self._scharr(gray)
        return self._canny(gray)

    def overlay(self, frame: np.ndarray, edges: np.ndarray,
                color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """
        Blend a colored edge map onto the original frame.

        Parameters
        ----------
        frame  : np.ndarray  BGR image.
        edges  : np.ndarray  Grayscale edge map from :meth:`detect`.
        color  : tuple       BGR color for edge highlights.

        Returns
        -------
        np.ndarray  Composite BGR image.
        """
        colored_edges      = np.zeros_like(frame)
        colored_edges[edges > 0] = color
        return cv2.addWeighted(frame, 1.0, colored_edges, self.alpha, 0)

    def detect_and_overlay(self, frame: np.ndarray,
                           color: Tuple[int, int, int] = (0, 255, 255)
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience: run detection and overlay in one call."""
        edges = self.detect(frame)
        return self.overlay(frame, edges, color), edges

    # ------------------------------------------------------------------
    # Algorithm implementations
    # ------------------------------------------------------------------

    def _canny(self, gray: np.ndarray) -> np.ndarray:
        """Canny edge detector with optional Gaussian pre-blur."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, self.t1, self.t2, apertureSize=self.aperture)

    @staticmethod
    def _sobel(gray: np.ndarray) -> np.ndarray:
        """Sobel gradient magnitude (x + y combined)."""
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sx ** 2 + sy ** 2)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def _laplacian(gray: np.ndarray) -> np.ndarray:
        """Laplacian second-order edge detector."""
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        lap     = cv2.Laplacian(blurred, cv2.CV_64F)
        return cv2.convertScaleAbs(lap)

    @staticmethod
    def _scharr(gray: np.ndarray) -> np.ndarray:
        """Scharr operator — improved rotational symmetry over Sobel."""
        sx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        sy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(sx ** 2 + sy ** 2)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
