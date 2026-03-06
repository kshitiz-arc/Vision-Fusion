#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        VisionFusion AI — Unified Computer Vision Perception Engine           ║
║                                                                              ║
║   Multi-Modal Vision Pipeline for Intelligent Scene Understanding            ║
║   Classical CV  ×  Deep Learning  ×  Real-Time Video Processing              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
-----
    python main.py --mode webcam
    python main.py --mode video  --source path/to/video.mp4
    python main.py --mode image  --source path/to/image.jpg
    python main.py --mode webcam --config configs/my_experiment.yaml
    python main.py --mode webcam --disable edge motion

Keyboard Controls (live mode)
------------------------------
    Q / ESC  — quit
    e        — toggle edge detection
    f        — toggle face detection
    m        — toggle motion detection
    o        — toggle object detection
    c        — toggle contour analysis
    n        — toggle CNN classifier
    t        — toggle tracking
"""

from __future__ import annotations

import cv2
import sys
import time
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# --- re-ID demo constants --------------------------------------------------
YOLO_MODEL   = "yolov8n.pt"
CONFIDENCE   = 0.4
TRAIL_LEN    = 40
REID_TIMEOUT = 10.0
REID_GAP     = 1.5

CLASS_FILTERS = [None, 0, 2]
CLASS_NAMES   = ["All", "Person", "Car"]

import argparse
from pathlib import Path

# Ensure project root is importable from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import ConfigLoader
from utils.logger        import get_logger
from pipelines.perception_pipeline import PerceptionPipeline

logger = get_logger("visionfusion.main")

BANNER = r"""
██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗      ███████╗██╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗
██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║      ██╔════╝██║   ██║██╔════╝██║██╔═══██╗████╗  ██║
██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║█████╗█████╗  ██║   ██║███████╗██║██║   ██║██╔██╗ ██║
╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║╚════╝██╔══╝  ██║   ██║╚════██║██║██║   ██║██║╚██╗██║
 ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║      ██║     ╚██████╔╝███████║██║╚██████╔╝██║ ╚████║
  ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝      ╚═╝      ╚═════╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
  Unified Computer Vision Perception Engine  |  VisionFusion-AI v1.0
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionFusion AI — Multi-Modal Computer Vision Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["webcam", "video", "image", "reid"],
        default="webcam",
        help="Input mode (default: webcam)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Camera index (int) or path to video/image file",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless (no cv2.imshow window)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (useful for benchmarking)",
    )
    parser.add_argument(
        "--disable",
        nargs="+",
        choices=["edge", "face", "motion", "objects", "contour", "cnn", "track"],
        default=[],
        help="Disable specific pipeline modules",
    )
    parser.add_argument(
        "--enable",
        nargs="+",
        choices=["edge", "face", "motion", "objects", "contour", "cnn", "track"],
        default=[],
        help="Force-enable specific pipeline modules",
    )
    return parser.parse_args()


def resolve_source(mode: str, source_arg: str | None) -> int | str:
    """Determine the capture source from mode and CLI argument."""
    if mode == "webcam":
        if source_arg is None:
            return 0
        try:
            return int(source_arg)
        except ValueError:
            return source_arg
    if source_arg is None:
        logger.error("--source is required for mode '%s'.", mode)
        sys.exit(1)
    return source_arg


def main() -> None:
    print(BANNER)
    args = parse_args()

    # ── Load configuration ─────────────────────────────────────────────
    cfg = ConfigLoader.load(args.config)

    # propagate log level from config to all existing loggers
    import logging
    log_level = cfg.get("system", {}).get("log_level", "INFO").upper()
    for lname in list(logging.root.manager.loggerDict.keys()):
        logging.getLogger(lname).setLevel(log_level)

    # Apply mode override from CLI
    cfg["input"]["mode"] = args.mode

    # ── Build pipeline ─────────────────────────────────────────────────
    pipeline = PerceptionPipeline(cfg)

    # Apply module enable / disable flags from CLI
    for mod in args.disable:
        pipeline.toggle_module(mod, False)
        logger.info("Module '%s' disabled via CLI.", mod)
    for mod in args.enable:
        pipeline.toggle_module(mod, True)
        logger.info("Module '%s' enabled via CLI.", mod)

    # ── Resolve source ─────────────────────────────────────────────────
    # handle special modes that bypass the pipeline
    if args.mode == "reid":
        run_reid_demo(source=args.source)
        return

    source = resolve_source(args.mode, args.source)

    # ── Run ────────────────────────────────────────────────────────────
    logger.info("Starting VisionFusion AI  [mode=%s  source=%s]",
                args.mode, source)
    pipeline.start(
        mode=args.mode,
        source=source,
        display=not args.no_display,
        max_frames=args.max_frames,
    )

    logger.info("VisionFusion AI session ended.")
    logger.info(pipeline.latency_summary)



# ------------------------------------------------------------------
# re-ID demo utilities (merged from testing.py)
# ------------------------------------------------------------------

def id_to_color(track_id):
    rng = np.random.default_rng(int(track_id) * 7 + 13)
    return tuple(int(c) for c in rng.integers(80, 255, 3))


def draw_help(img):
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (340, 200), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    lines = [
        ("  CONTROLS",               (0, 210, 255), 2),
        ("",                          None,          1),
        ("  F   Cycle class filter",  (210, 210, 210), 1),
        ("  M   Toggle mirror",       (210, 210, 210), 1),
        ("  H   Toggle this help",    (210, 210, 210), 1),
        ("  Q   Quit",                (210, 210, 210), 1),
    ]
    for i, (text, color, thickness) in enumerate(lines):
        if color:
            cv2.putText(img, text, (15, 38 + i * 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness)


def draw_hud(img, fps, active_filter, track_count, reid_count, show_help):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (w - 250, 5), (w - 5, 120), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, f"FPS:     {fps:5.1f}",              (w-240, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,   255, 100), 1)
    cv2.putText(img, f"Filter:  {active_filter}",          (w-240, 55),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,   200, 255), 1)
    cv2.putText(img, f"Active:  {track_count} track(s)",   (w-240, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200,   0), 1)
    cv2.putText(img, f"Memory:  {reid_count} object(s)",   (w-240, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 200), 1)
    if not show_help:
        cv2.putText(img, "H = help", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1)


def draw_track(img, track_id, x1, y1, x2, y2, label, is_reid, history):
    color  = id_to_color(track_id)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if is_reid:
        cv2.rectangle(img, (x1-3, y1-3), (x2+3, y2+3), (255, 255, 255), 2)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    tag = f"ID:{track_id}  {label}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, tag, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    if is_reid:
        cv2.putText(img, "RE-ID", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    history.append((cx, cy))
    if len(history) > TRAIL_LEN:
        history.pop(0)
    for i in range(1, len(history)):
        alpha = i / len(history)
        tc = tuple(int(c * alpha) for c in color)
        cv2.line(img, history[i - 1], history[i], tc, 2)


def run_reid_demo(source=None) -> None:
    """Standalone re-ID tracker demo copied from testing.py."""
    print("Loading YOLOv8 model... (downloads ~6MB on first run)")
    model = YOLO(YOLO_MODEL)
    print("Model ready.\n")

    cap = cv2.VideoCapture(0 if source is None else source)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mirror           = True
    show_help        = True
    class_filter_idx = 0

    track_history = defaultdict(list)
    reid_memory   = {}
    seen_ids      = set()
    fps_smoothed  = 0.0

    print("Tracker running.")
    print("Controls: F=filter  M=mirror  H=help  Q=quit\n")

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from camera.")
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        class_filter = CLASS_FILTERS[class_filter_idx]
        results = model.track(
            frame,
            persist = True,
            verbose = False,
            classes = [class_filter] if class_filter is not None else None,
            conf    = CONFIDENCE,
            tracker = "bytetrack.yaml",
        )[0]

        now        = time.perf_counter()
        active_ids = set()

        expired = [tid for tid, info in reid_memory.items()
                   if now - info["last_seen"] > REID_TIMEOUT]
        for tid in expired:
            del reid_memory[tid]

        if results.boxes is not None and results.boxes.id is not None:
            boxes   = results.boxes.xyxy.cpu().numpy()
            ids     = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls_id in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[cls_id]

                active_ids.add(tid)
                seen_ids.add(tid)

                last_seen = reid_memory.get(tid, {}).get("last_seen", now)
                is_reid   = (tid in seen_ids) and \
                            (tid in reid_memory) and \
                            (now - last_seen > REID_GAP)

                reid_memory[tid] = {"last_seen": now, "label": label}

                draw_track(frame, tid, x1, y1, x2, y2, label,
                           is_reid, track_history[tid])

        for tid in seen_ids - active_ids:
            if tid not in reid_memory:
                reid_memory[tid] = {
                    "last_seen": now,
                    "label":     "unknown"
                }

        draw_hud(frame, fps_smoothed,
                 CLASS_NAMES[class_filter_idx],
                 len(active_ids), len(reid_memory), show_help)
        if show_help:
            draw_help(frame)

        cv2.imshow("Re-ID Tracker", frame)

        elapsed      = time.perf_counter() - t0
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * (1.0 / elapsed if elapsed > 0 else 0)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting.")
            break
        elif key == ord('m'):
            mirror = not mirror
            print(f"Mirror: {'ON' if mirror else 'OFF'}")
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('f'):
            class_filter_idx = (class_filter_idx + 1) % len(CLASS_FILTERS)
            print(f"Class filter: {CLASS_NAMES[class_filter_idx]}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Tracked {len(seen_ids)} unique object(s) total.")


if __name__ == "__main__":
    main()
