"""
VisionFusion AI — Evaluation Runner
======================================
Evaluates the trained CNN classifier against a test split and
prints a full metrics report.

Usage
-----
    python evaluation/evaluate.py \\
        --config configs/default.yaml \\
        --weights models/checkpoints/resnet18_best.pth \\
        --dataset cifar10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import ClassificationMetrics
from utils.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionFusion AI — Evaluation")
    p.add_argument("--config",  default="configs/default.yaml")
    p.add_argument("--weights", default=None)
    p.add_argument("--dataset", default="cifar10",
                   choices=["cifar10", "custom"])
    p.add_argument("--output",  default="evaluation/results")
    return p.parse_args()


def run_evaluation(cfg: dict, weights_path: str | None,
                   dataset_name: str, output_dir: str) -> None:
    try:
        import torch
        import torchvision.transforms as T
        import torchvision.datasets   as D
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error("PyTorch not installed — cannot run evaluation.")
        return

    from pipelines.training_pipeline import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch   = cfg["cnn_classifier"]["architecture"]
    n_cls  = cfg["cnn_classifier"]["num_classes"]

    model = build_model(arch, n_cls).to(device)
    if weights_path and Path(weights_path).exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        logger.info("Loaded weights from %s", weights_path)
    model.eval()

    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    root = cfg["datasets"].get(dataset_name, {}).get("root", f"data/{dataset_name}")
    ds   = D.CIFAR10(root=root, train=False, transform=val_tf, download=True)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            out    = model(inputs)
            preds  = out.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    class_names = cfg["cnn_classifier"].get(
        "class_names",
        ["airplane","automobile","bird","cat","deer",
         "dog","frog","horse","ship","truck"],
    )
    metrics = ClassificationMetrics(y_true, y_pred, class_names)

    report = metrics.report()
    print(report)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "report.txt").write_text(report)
    metrics.plot_confusion_matrix(str(out_path / "confusion_matrix.png"))
    logger.info("Evaluation artefacts saved to %s", out_path)


def main() -> None:
    args = parse_args()
    cfg  = ConfigLoader.load(args.config)
    run_evaluation(cfg, args.weights, args.dataset, args.output)


if __name__ == "__main__":
    main()
