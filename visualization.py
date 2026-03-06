"""
VisionFusion AI — Visualization Module
========================================
Composites detection overlays, HUD panels, and diagnostic information
onto video frames for real-time display and export.

Features
--------
* Flexible info-panel HUD (FPS, mode, active modules)
* Unified detection overlay (edges, faces, motion, objects, tracks)
* Split-view / combined-view panel modes
* Color-scheme management
"""

from __future__ import annotations

import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class Visualizer:
    """
    Composites all perception outputs onto a single display frame.

    Parameters
    ----------
    cfg : dict
        The ``visualization`` section from the global config.

    Example
    -------
    >>> vis = Visualizer(cfg["visualization"])
    >>> display_frame = vis.compose(frame, results)
    """

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL = 0.5
    FONT_MED   = 0.65

    SCHEMES = {
        "vivid": {
            "face":   (0, 255,  80),
            "edge":   (0, 255, 255),
            "motion": (0, 128, 255),
            "object": (255,  80,   0),
            "track":  (255, 255,   0),
            "hud":    (220, 220, 220),
            "panel":  (20,  20,  20),
        },
        "pastel": {
            "face":   (180, 255, 180),
            "edge":   (180, 255, 255),
            "motion": (180, 200, 255),
            "object": (255, 200, 180),
            "track":  (255, 255, 180),
            "hud":    (200, 200, 200),
            "panel":  (40,  40,  40),
        },
        "mono": {
            "face":   (200, 200, 200),
            "edge":   (255, 255, 255),
            "motion": (160, 160, 160),
            "object": (220, 220, 220),
            "track":  (255, 255, 255),
            "hud":    (200, 200, 200),
            "panel":  (30,  30,  30),
        },
    }

    def __init__(self, cfg: dict) -> None:
        self.show_fps       = bool(cfg.get("show_fps", True))
        self.show_timestamp = bool(cfg.get("show_timestamp", True))
        self.show_labels    = bool(cfg.get("show_labels", True))
        self.show_confidence = bool(cfg.get("show_confidence", True))
        self.thickness      = int(cfg.get("bbox_thickness", 2))
        self.font_scale     = float(cfg.get("font_scale", 0.6))
        self.panel_mode     = cfg.get("panel_mode", "combined")
        scheme_name         = cfg.get("color_scheme", "vivid")
        self.colors         = self.SCHEMES.get(scheme_name, self.SCHEMES["vivid"])

    # ------------------------------------------------------------------
    # Primary composition entry-point
    # ------------------------------------------------------------------

    def compose(self,
                frame: np.ndarray,
                fps: float = 0.0,
                active_modules: Optional[List[str]] = None,
                edge_map: Optional[np.ndarray] = None,
                overlay_edges: bool = True,
                extra_info: Optional[Dict[str, str]] = None) -> np.ndarray:
        """
        Compose the final display frame.

        Parameters
        ----------
        frame          : np.ndarray  Annotated BGR frame from the pipeline.
        fps            : float       Current FPS estimate.
        active_modules : list[str]   Names of active modules for the HUD.
        edge_map       : np.ndarray  Optional edge overlay.
        overlay_edges  : bool        Whether to alpha-blend edges.
        extra_info     : dict        Additional key-value text for the HUD.

        Returns
        -------
        np.ndarray  Display-ready BGR frame.
        """
        out = frame.copy()

        if overlay_edges and edge_map is not None:
            out = self._blend_edges(out, edge_map)

        self._draw_hud(out, fps, active_modules or [], extra_info or {})
        return out

    # ------------------------------------------------------------------
    # HUD panel
    # ------------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray, fps: float,
                  modules: List[str],
                  extra: Dict[str, str]) -> None:
        """Draw a semi-transparent HUD in the top-left corner."""
        lines = []
        if self.show_fps:
            lines.append(f"FPS: {fps:5.1f}")
        if self.show_timestamp:
            lines.append(datetime.now().strftime("%H:%M:%S"))
        for mod in modules:
            lines.append(f"[ON] {mod}")
        for k, v in extra.items():
            lines.append(f"{k}: {v}")

        if not lines:
            return

        padding   = 6
        line_h    = 20
        panel_h   = len(lines) * line_h + 2 * padding
        panel_w   = 200
        overlay   = frame.copy()
        panel_col = self.colors["panel"]
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), panel_col, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, text in enumerate(lines):
            y = padding + (i + 1) * line_h - 4
            cv2.putText(frame, text, (padding, y), self.FONT,
                        self.FONT_SMALL, self.colors["hud"], 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Edge overlay
    # ------------------------------------------------------------------

    def _blend_edges(self, frame: np.ndarray,
                     edges: np.ndarray) -> np.ndarray:
        color_edges = np.zeros_like(frame)
        color_edges[edges > 0] = self.colors["edge"]
        return cv2.addWeighted(frame, 1.0, color_edges, 0.35, 0)

    # ------------------------------------------------------------------
    # Static annotation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def put_text_with_bg(frame: np.ndarray, text: str,
                         pos: Tuple[int, int],
                         font_scale: float = 0.55,
                         text_color: Tuple[int, int, int] = (255, 255, 255),
                         bg_color: Tuple[int, int, int] = (0, 0, 0)) -> None:
        """Draw *text* at *pos* with a solid background rectangle."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, 1)
        x, y = pos
        cv2.rectangle(frame, (x - 2, y - th - 4),
                      (x + tw + 2, y + 2), bg_color, -1)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, text_color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_fps(frame: np.ndarray, fps: float,
                 color: Tuple[int, int, int] = (0, 255, 0)) -> None:
        """Overlay FPS counter in the top-right corner."""
        h, w = frame.shape[:2]
        label = f"FPS: {fps:.1f}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(frame, label, (w - tw - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    @staticmethod
    def create_side_panel(left: np.ndarray,
                          right: np.ndarray) -> np.ndarray:
        """Horizontally stack two frames of possibly different sizes."""
        h  = max(left.shape[0], right.shape[0])
        l  = cv2.copyMakeBorder(left,  0, h - left.shape[0],  0, 0,
                                 cv2.BORDER_CONSTANT)
        r  = cv2.copyMakeBorder(right, 0, h - right.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT)
        return np.hstack([l, r])
