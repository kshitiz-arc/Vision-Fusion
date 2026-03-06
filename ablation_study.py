"""
VisionFusion AI — Ablation Study Script
=========================================
Systematically benchmarks individual pipeline modules to quantify their
contribution to overall system accuracy and latency.

Run as a script:
    python experiments/notebooks/ablation_study.py

Or import from a Jupyter notebook:
    from experiments.notebooks.ablation_study import run_ablation
    results = run_ablation(cfg, video_source="data/sample.mp4")
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import ConfigLoader
from utils.logger        import get_logger
from pipelines.perception_pipeline import PerceptionPipeline

logger = get_logger("ablation")

# Modules to ablate, one at a time
ALL_MODULES = ["edge", "face", "motion", "objects", "contour", "track"]


def benchmark_configuration(cfg: dict,
                              active_modules: List[str],
                              source: int | str = 0,
                              n_frames: int = 300) -> Dict:
    """
    Run the pipeline with a specific subset of modules and measure performance.

    Parameters
    ----------
    cfg            : base config dict
    active_modules : list of module names to enable
    source         : video source
    n_frames       : number of frames to benchmark over

    Returns
    -------
    dict with mean_fps, mean_latency_ms, std_latency_ms
    """
    pipeline = PerceptionPipeline(cfg)

    # Disable all, then enable only the specified set
    for mod in ALL_MODULES:
        pipeline.toggle_module(mod, mod in active_modules)
    pipeline.toggle_module("cnn", False)   # CNN excluded from latency bench

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", source)
        return {}

    latencies: List[float] = []
    frame_count = 0

    while frame_count < n_frames:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        t0     = time.perf_counter()
        _      = pipeline.process_frame(frame)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)
        frame_count += 1

    cap.release()

    mean_lat = float(np.mean(latencies))
    std_lat  = float(np.std(latencies))
    mean_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "modules":          active_modules,
        "mean_fps":         mean_fps,
        "mean_latency_ms":  mean_lat,
        "std_latency_ms":   std_lat,
        "n_frames":         frame_count,
    }


def run_ablation(cfg: dict,
                 video_source: int | str = 0,
                 n_frames: int = 200) -> List[Dict]:
    """
    Run the full ablation study.

    Tests:
    * Baseline (no modules)
    * Each module in isolation
    * All modules combined
    """
    results = []

    # 1. Baseline — preprocessing only
    logger.info("Running baseline (preprocessing only)…")
    r = benchmark_configuration(cfg, [], video_source, n_frames)
    r["label"] = "Baseline (preprocessor only)"
    results.append(r)

    # 2. Each module in isolation
    for mod in ALL_MODULES:
        logger.info("Ablating module: %s …", mod)
        r = benchmark_configuration(cfg, [mod], video_source, n_frames)
        r["label"] = f"Module: {mod}"
        results.append(r)

    # 3. All modules combined
    logger.info("Running full pipeline…")
    r = benchmark_configuration(cfg, ALL_MODULES, video_source, n_frames)
    r["label"] = "Full pipeline"
    results.append(r)

    _print_table(results)
    return results


def _print_table(results: List[Dict]) -> None:
    """Print a formatted ablation results table."""
    print("\n" + "=" * 72)
    print("  VisionFusion AI — Ablation Study Results")
    print("=" * 72)
    print(f"  {'Configuration':<30} {'FPS':>8} {'Mean (ms)':>12} {'Std (ms)':>10}")
    print("  " + "-" * 66)
    for r in results:
        label = r.get("label", str(r.get("modules")))
        fps   = r.get("mean_fps",        0)
        mean  = r.get("mean_latency_ms", 0)
        std   = r.get("std_latency_ms",  0)
        print(f"  {label:<30} {fps:>8.1f} {mean:>12.2f} {std:>10.2f}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
    cfg      = ConfigLoader.load(str(cfg_path))
    run_ablation(cfg, video_source=0, n_frames=100)
