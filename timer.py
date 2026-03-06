"""
VisionFusion AI — Performance Timer
=====================================
Lightweight FPS counter and per-stage latency profiler.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque


class FPSCounter:
    """Rolling-window frames-per-second estimator.

    Parameters
    ----------
    window : int
        Number of recent frame intervals to average over.
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: Deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        """Register the current instant as a completed frame."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """Estimated FPS over the rolling window."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


class StageTimer:
    """Context-manager timer for profiling individual pipeline stages.

    Example
    -------
    >>> timer = StageTimer()
    >>> with timer("preprocessing"):
    ...     run_preprocessing()
    >>> print(timer.summary())
    """

    def __init__(self) -> None:
        self._records: dict[str, list[float]] = {}
        self._current_stage: str | None = None
        self._start: float = 0.0

    def __call__(self, stage: str) -> "StageTimer":
        self._current_stage = stage
        return self

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        stage = self._current_stage or "unknown"
        self._records.setdefault(stage, []).append(elapsed_ms)

    def summary(self) -> str:
        """Return a formatted summary of mean latencies per stage."""
        lines = ["Stage Latency Summary (ms):"]
        for stage, times in self._records.items():
            mean = sum(times) / len(times)
            lines.append(f"  {stage:<25} {mean:7.2f} ms  (n={len(times)})")
        return "\n".join(lines)

    def reset(self) -> None:
        self._records.clear()
