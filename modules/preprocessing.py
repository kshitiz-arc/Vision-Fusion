"""
VisionFusion AI — Preprocessing Module
========================================
Standardised image pre-processing pipeline stage.

Responsibilities
----------------
* Resizing / aspect-ratio-preserving letterboxing
* Color-space conversion (BGR ↔ RGB / GRAY / HSV / LAB)
* Noise suppression (Gaussian, Bilateral, Median)
* Normalization for CNN inference
* Histogram equalization & CLAHE
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Configurable preprocessing stage for the VisionFusion pipeline.

    Parameters
    ----------
    cfg : dict
        Preprocessing sub-section from the global config YAML.

    Example
    -------
    >>> prep = Preprocessor(cfg["preprocessing"])
    >>> frame_out = prep.process(frame)
    """

    COLOR_SPACE_MAP = {
        "BGR":  None,
        "RGB":  cv2.COLOR_BGR2RGB,
        "GRAY": cv2.COLOR_BGR2GRAY,
        "HSV":  cv2.COLOR_BGR2HSV,
        "LAB":  cv2.COLOR_BGR2LAB,
    }

    DENOISE_METHODS = {
        "gaussian":  "_gaussian_denoise",
        "bilateral": "_bilateral_denoise",
        "median":    "_median_denoise",
    }

    def __init__(self, cfg: dict) -> None:
        self.cfg            = cfg
        self.target_size: Tuple[int, int] | None = (
            tuple(cfg.get("resize", [])) or None  # type: ignore[assignment]
        )
        self.normalize      = cfg.get("normalize", True)
        self.denoise        = cfg.get("denoise", False)
        self.denoise_method = cfg.get("denoise_method", "gaussian")
        self.denoise_kernel = int(cfg.get("denoise_kernel", 5))
        self.color_space    = cfg.get("color_space", "BGR")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply the full preprocessing chain to a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR image from the capture source.

        Returns
        -------
        np.ndarray
            Pre-processed frame ready for downstream modules.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Received empty or None frame.")

        if self.target_size:
            frame = self.resize(frame, self.target_size)

        if self.denoise:
            frame = self._apply_denoise(frame)

        if self.color_space != "BGR":
            frame = self.convert_color(frame, self.color_space)

        return frame

    def to_cnn_tensor(self, frame: np.ndarray,
                      target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Prepare a frame as a float32 tensor normalized for CNN inference.

        Returns float32 array of shape (1, 3, H, W) suitable for PyTorch.
        """
        blob = cv2.resize(frame, target_size)
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32)
        # ImageNet mean/std normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        blob = (blob / 255.0 - mean) / std
        return blob.transpose(2, 0, 1)[np.newaxis]   # (1, C, H, W)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def resize(frame: np.ndarray,
               size: Tuple[int, int],
               letterbox: bool = False) -> np.ndarray:
        """
        Resize frame to *size* (W, H).

        If *letterbox* is True, preserve aspect ratio and pad with zeros.
        """
        if not letterbox:
            return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)

        h, w = frame.shape[:2]
        target_w, target_h = size
        scale  = min(target_w / w, target_h / h)
        new_w  = int(w * scale)
        new_h  = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas  = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
        return canvas

    @staticmethod
    def convert_color(frame: np.ndarray, target: str) -> np.ndarray:
        """Convert from BGR to *target* color space."""
        code = Preprocessor.COLOR_SPACE_MAP.get(target.upper())
        if code is None:
            return frame
        return cv2.cvtColor(frame, code)

    @staticmethod
    def apply_clahe(frame: np.ndarray,
                    clip_limit: float = 2.0,
                    tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).

        Enhances local contrast while suppressing noise amplification.
        Works on the L channel of LAB color space to preserve hue.
        """
        lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        l_eq  = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Private denoising helpers
    # ------------------------------------------------------------------

    def _apply_denoise(self, frame: np.ndarray) -> np.ndarray:
        method_name = self.DENOISE_METHODS.get(self.denoise_method, "_gaussian_denoise")
        return getattr(self, method_name)(frame)

    def _gaussian_denoise(self, frame: np.ndarray) -> np.ndarray:
        k = self.denoise_kernel | 1   # ensure odd
        return cv2.GaussianBlur(frame, (k, k), 0)

    def _bilateral_denoise(self, frame: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(frame, self.denoise_kernel, 75, 75)

    def _median_denoise(self, frame: np.ndarray) -> np.ndarray:
        k = self.denoise_kernel | 1
        return cv2.medianBlur(frame, k)
