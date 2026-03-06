"""
VisionFusion AI — Evaluation Metrics
========================================
Standard classification and detection metrics with reporting utilities.

Metrics provided
----------------
Classification : accuracy, precision, recall, F1, confusion matrix
Detection      : mAP (mean Average Precision), per-class AP, IoU

Usage
-----
    from evaluation.metrics import ClassificationMetrics, DetectionMetrics
    cm  = ClassificationMetrics(y_true, y_pred, class_names)
    print(cm.report())
    cm.plot_confusion_matrix("evaluation/results/confusion.png")
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Classification Metrics
# ============================================================================

class ClassificationMetrics:
    """
    Compute and visualise classification evaluation metrics.

    Parameters
    ----------
    y_true       : array-like of int  Ground-truth class indices.
    y_pred       : array-like of int  Predicted class indices.
    class_names  : list[str]          Human-readable class labels.
    """

    def __init__(self, y_true: List[int], y_pred: List[int],
                 class_names: Optional[List[str]] = None) -> None:
        self.y_true      = np.asarray(y_true)
        self.y_pred      = np.asarray(y_pred)
        self.n_classes   = max(self.y_true.max(), self.y_pred.max()) + 1
        self.class_names = class_names or [str(i) for i in range(self.n_classes)]
        self._cm         = self._compute_confusion_matrix()

    # ------------------------------------------------------------------

    def accuracy(self) -> float:
        return float(np.mean(self.y_true == self.y_pred))

    def precision(self, average: str = "macro") -> float:
        return self._sklearn_metric("precision", average)

    def recall(self, average: str = "macro") -> float:
        return self._sklearn_metric("recall", average)

    def f1_score(self, average: str = "macro") -> float:
        return self._sklearn_metric("f1", average)

    def per_class_precision(self) -> np.ndarray:
        tp = np.diag(self._cm)
        fp = self._cm.sum(axis=0) - tp
        return np.where((tp + fp) > 0, tp / (tp + fp), 0.0)

    def per_class_recall(self) -> np.ndarray:
        tp = np.diag(self._cm)
        fn = self._cm.sum(axis=1) - tp
        return np.where((tp + fn) > 0, tp / (tp + fn), 0.0)

    def per_class_f1(self) -> np.ndarray:
        p = self.per_class_precision()
        r = self.per_class_recall()
        return np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)

    def confusion_matrix(self) -> np.ndarray:
        return self._cm

    def report(self) -> str:
        """Return a formatted metrics report string."""
        lines = [
            "=" * 60,
            " VisionFusion AI — Classification Evaluation Report",
            "=" * 60,
            f"  Accuracy  : {self.accuracy():.4f}",
            f"  Precision : {self.precision():.4f}  (macro)",
            f"  Recall    : {self.recall():.4f}  (macro)",
            f"  F1 Score  : {self.f1_score():.4f}  (macro)",
            "",
            f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}",
            "  " + "-" * 54,
        ]
        for i, name in enumerate(self.class_names):
            p = self.per_class_precision()[i]
            r = self.per_class_recall()[i]
            f = self.per_class_f1()[i]
            lines.append(f"  {name:<20} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def plot_confusion_matrix(self, save_path: str | None = None,
                               figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot and optionally save the confusion matrix heatmap."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib / seaborn not installed — cannot plot.")
            return

        cm_norm = self._cm.astype(float) / (self._cm.sum(axis=1, keepdims=True) + 1e-8)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title("VisionFusion AI — Normalised Confusion Matrix")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info("Confusion matrix saved to %s", save_path)
        plt.show()

    # ------------------------------------------------------------------

    def _compute_confusion_matrix(self) -> np.ndarray:
        cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for t, p in zip(self.y_true, self.y_pred):
            cm[t, p] += 1
        return cm

    def _sklearn_metric(self, metric: str, average: str) -> float:
        try:
            from sklearn import metrics as skm
            fn = {"precision": skm.precision_score,
                  "recall":    skm.recall_score,
                  "f1":        skm.f1_score}[metric]
            return float(fn(self.y_true, self.y_pred,
                            average=average, zero_division=0))
        except ImportError:
            # Manual macro average fallback
            per_class = {"precision": self.per_class_precision,
                         "recall":    self.per_class_recall,
                         "f1":        self.per_class_f1}[metric]()
            return float(per_class.mean())


# ============================================================================
# Detection Metrics
# ============================================================================

class DetectionMetrics:
    """
    Compute mean Average Precision (mAP) for object detection results.

    Parameters
    ----------
    iou_threshold : float  IoU threshold for a prediction to be a true positive.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self._predictions:  List[Dict] = []   # {class_id, confidence, bbox}
        self._ground_truths: List[Dict] = []  # {class_id, bbox}

    def add_image(self, predictions: List[Dict],
                  ground_truths: List[Dict]) -> None:
        """
        Accumulate predictions and ground-truths for one image.

        Each entry: ``{"class_id": int, "confidence": float, "bbox": (x1,y1,x2,y2)}``
        """
        self._predictions.extend(predictions)
        self._ground_truths.extend(ground_truths)

    def compute_map(self) -> Tuple[float, Dict[int, float]]:
        """
        Compute mAP over all accumulated images.

        Returns
        -------
        (mAP, per_class_AP)
        """
        class_ids = set(gt["class_id"] for gt in self._ground_truths)
        per_class: Dict[int, float] = {}

        for cid in class_ids:
            preds = sorted(
                [p for p in self._predictions if p["class_id"] == cid],
                key=lambda x: -x["confidence"],
            )
            gts  = [g for g in self._ground_truths if g["class_id"] == cid]
            per_class[cid] = self._average_precision(preds, gts)

        map_val = float(np.mean(list(per_class.values()))) if per_class else 0.0
        return map_val, per_class

    # ------------------------------------------------------------------

    def _average_precision(self, preds: List[Dict],
                           gts: List[Dict]) -> float:
        tp_list, fp_list = [], []
        matched = set()

        for pred in preds:
            best_iou, best_idx = 0.0, -1
            for j, gt in enumerate(gts):
                iou = self._iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, j
            if best_iou >= self.iou_threshold and best_idx not in matched:
                tp_list.append(1)
                fp_list.append(0)
                matched.add(best_idx)
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)
        n_gt   = max(len(gts), 1)
        prec   = tp_cum / (tp_cum + fp_cum + 1e-9)
        rec    = tp_cum / n_gt

        # 11-point interpolation
        ap = 0.0
        for thr in np.linspace(0, 1, 11):
            p = prec[rec >= thr].max() if (rec >= thr).any() else 0.0
            ap += p / 11
        return ap

    @staticmethod
    def _iou(box_a: Tuple, box_b: Tuple) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)
        return inter / union if union > 0 else 0.0
