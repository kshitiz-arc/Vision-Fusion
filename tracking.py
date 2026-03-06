"""
VisionFusion AI — Multi-Object Tracking Module
================================================
Centroid-based IoU tracker + optional OpenCV tracker per-target.

Architecture
------------
* ``CentroidTracker``  — lightweight, no external dependencies.
  Associates new detections to existing tracks via Euclidean distance.
* ``IOUTracker``       — IOU-based Hungarian assignment, frame-to-frame.
* ``OpenCVTracker``    — wraps cv2 CSRT / KCF for single-target video tracking.
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Track:
    """State of a single tracked object."""
    track_id:    int
    bbox:        Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    centroid:    Tuple[int, int]
    age:         int = 0                      # frames since creation
    disappeared: int = 0                      # consecutive missed frames
    history:     List[Tuple[int, int]] = field(default_factory=list)


class CentroidTracker:
    """
    Centroid-based multi-object tracker.

    Parameters
    ----------
    max_disappeared : int
        Frames allowed without a matching detection before a track is removed.
    max_distance    : int
        Maximum pixel distance for centroid-to-detection association.

    Example
    -------
    >>> tracker = CentroidTracker(max_disappeared=30)
    >>> tracks  = tracker.update(detections)
    """

    def __init__(self, max_disappeared: int = 30,
                 max_distance: int = 50) -> None:
        self.next_id        = 0
        self.max_disappeared = max_disappeared
        self.max_distance   = max_distance
        self.tracks: OrderedDict[int, Track] = OrderedDict()

    # ------------------------------------------------------------------

    def update(self, bboxes: List[Tuple[int, int, int, int]]) -> List[Track]:
        """
        Update tracks with a new set of bounding boxes.

        Parameters
        ----------
        bboxes : list of (x1, y1, x2, y2)

        Returns
        -------
        List[Track]  All currently active tracks.
        """
        if not bboxes:
            for track in self.tracks.values():
                track.disappeared += 1
            self.tracks = OrderedDict(
                {tid: t for tid, t in self.tracks.items()
                 if t.disappeared <= self.max_disappeared}
            )
            return list(self.tracks.values())

        new_centroids = [
            ((x1 + x2) // 2, (y1 + y2) // 2)
            for (x1, y1, x2, y2) in bboxes
        ]

        if not self.tracks:
            for i, (cent, bbox) in enumerate(zip(new_centroids, bboxes)):
                self._register(cent, bbox)
            return list(self.tracks.values())

        track_ids      = list(self.tracks.keys())
        track_cents    = [t.centroid for t in self.tracks.values()]
        dist_matrix    = np.linalg.norm(
            np.array(track_cents)[:, None] - np.array(new_centroids)[None], axis=2
        )  # (n_tracks, n_dets)

        row_ind = dist_matrix.min(axis=1).argsort()
        col_ind = dist_matrix.argmin(axis=1)[row_ind]

        used_rows, used_cols = set(), set()
        for row, col in zip(row_ind, col_ind):
            if row in used_rows or col in used_cols:
                continue
            if dist_matrix[row, col] > self.max_distance:
                continue
            tid   = track_ids[row]
            track = self.tracks[tid]
            track.centroid    = new_centroids[col]
            track.bbox        = bboxes[col]
            track.disappeared = 0
            track.age        += 1
            track.history.append(new_centroids[col])
            used_rows.add(row)
            used_cols.add(col)

        for row in range(len(track_ids)):
            if row not in used_rows:
                self.tracks[track_ids[row]].disappeared += 1

        for col in range(len(new_centroids)):
            if col not in used_cols:
                self._register(new_centroids[col], bboxes[col])

        self.tracks = OrderedDict(
            {tid: t for tid, t in self.tracks.items()
             if t.disappeared <= self.max_disappeared}
        )
        return list(self.tracks.values())

    def _register(self, centroid: Tuple[int, int],
                  bbox: Tuple[int, int, int, int]) -> None:
        self.tracks[self.next_id] = Track(
            track_id=self.next_id,
            bbox=bbox,
            centroid=centroid,
            history=[centroid],
        )
        self.next_id += 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 0

    # ------------------------------------------------------------------

    @staticmethod
    def draw(frame: np.ndarray, tracks: List[Track],
             draw_trails: bool = True) -> np.ndarray:
        """Annotate frame with track IDs, bboxes, and motion trails."""
        out    = frame.copy()
        colors = np.random.default_rng(42).integers(80, 255, (500, 3))

        for track in tracks:
            tid   = track.track_id
            color = tuple(int(c) for c in colors[tid % 500])
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"ID {tid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 1, cv2.LINE_AA)

            if draw_trails and len(track.history) > 1:
                for j in range(1, len(track.history)):
                    cv2.line(out, track.history[j - 1],
                             track.history[j], color, 2)

        return out


class SingleObjectTracker:
    """
    Wraps an OpenCV tracker algorithm for single-target tracking.

    Parameters
    ----------
    algorithm : str
        One of: csrt | kcf | medianflow | mosse
    """

    @staticmethod
    def _get_algorithms() -> dict:
        algos = {}
        # CSRT — may be in cv2 or cv2.legacy depending on version
        if hasattr(cv2, "TrackerCSRT_create"):
            algos["csrt"] = cv2.TrackerCSRT_create
        elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            algos["csrt"] = cv2.legacy.TrackerCSRT_create
        # KCF
        if hasattr(cv2, "TrackerKCF_create"):
            algos["kcf"] = cv2.TrackerKCF_create
        elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            algos["kcf"] = cv2.legacy.TrackerKCF_create
        # MOSSE / MedianFlow (legacy only)
        if hasattr(cv2, "legacy"):
            if hasattr(cv2.legacy, "TrackerMOSSE_create"):
                algos["mosse"] = cv2.legacy.TrackerMOSSE_create
            if hasattr(cv2.legacy, "TrackerMedianFlow_create"):
                algos["medianflow"] = cv2.legacy.TrackerMedianFlow_create
        return algos

    ALGORITHMS: dict = {}   # populated lazily

    def __init__(self, algorithm: str = "csrt") -> None:
        self.algorithm = algorithm.lower()
        self._tracker  = None
        self.initialized = False
        if not SingleObjectTracker.ALGORITHMS:
            SingleObjectTracker.ALGORITHMS = self._get_algorithms()

    def init(self, frame: np.ndarray,
             bbox: Tuple[int, int, int, int]) -> None:
        """Initialise tracker on *frame* with initial *bbox* (x, y, w, h)."""
        algorithms = self._get_algorithms()
        default = next(iter(algorithms.values())) if algorithms else None
        factory = algorithms.get(self.algorithm, default)
        if factory is None:
            logger.warning("No suitable OpenCV tracker found. SingleObjectTracker disabled.")
            return
        self._tracker    = factory()
        self._tracker.init(frame, bbox)
        self.initialized = True

    def update(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Update and return new bbox or None if tracking failed."""
        if not self.initialized or self._tracker is None:
            return None
        ok, box = self._tracker.update(frame)
        if not ok:
            self.initialized = False
            return None
        x, y, w, h = [int(v) for v in box]
        return (x, y, w, h)
