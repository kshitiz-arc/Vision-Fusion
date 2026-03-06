"""
VisionFusion AI — CNN Classifier Module
=========================================
PyTorch-based CNN classifier for scene-level and ROI-level recognition.

Architectures supported
-----------------------
resnet18, resnet50, mobilenet_v2, efficientnet_b0

Hybrid design
-------------
OpenCV provides candidate ROI bounding boxes (from object detection / contours).
This module classifies each ROI using a fine-tuned CNN, enabling
classical-CV + deep-learning co-processing without running a full DNN detector.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationResult:
    """Single classification result for one image or ROI."""
    class_id:   int
    class_name: str
    confidence: float
    top5:       List[Tuple[str, float]]   # (name, prob) for top-5 predictions


class CNNClassifier:
    """
    Configurable CNN image classifier backed by PyTorch.

    Parameters
    ----------
    cfg : dict
        The ``cnn_classifier`` section from the global config.

    Example
    -------
    >>> clf = CNNClassifier(cfg["cnn_classifier"])
    >>> result = clf.classify(frame)
    >>> print(result.class_name, result.confidence)
    """

    # ImageNet normalization constants
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, cfg: dict) -> None:
        self.cfg          = cfg
        self.arch         = cfg.get("architecture", "resnet18")
        self.num_classes  = int(cfg.get("num_classes", 10))
        self.input_size   = tuple(cfg.get("input_size", [224, 224]))
        self.conf_thr     = float(cfg.get("confidence_threshold", 0.6))
        self.model_path   = cfg.get("model_path", "")
        self.dataset_name = cfg.get("dataset", "cifar10")

        self.class_names  = self._default_class_names()
        self.model        = None
        self.device       = None
        self._torch_available = self._try_import_torch()

        if self._torch_available:
            self._build_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, frame: np.ndarray) -> Optional[ClassificationResult]:
        """
        Classify *frame* (or any numpy BGR image / ROI).

        Returns
        -------
        ClassificationResult or None if model unavailable.
        """
        if not self._torch_available or self.model is None:
            logger.debug("CNN classifier unavailable; skipping inference.")
            return None

        import torch
        import torch.nn.functional as F

        tensor = self._preprocess(frame)
        with torch.no_grad():
            logits = self.model(tensor.to(self.device))
            probs  = F.softmax(logits, dim=1)[0].cpu().numpy()

        top5_idx   = probs.argsort()[-5:][::-1]
        top5       = [(self.class_names[i], float(probs[i])) for i in top5_idx]
        best_id    = int(top5_idx[0])
        best_conf  = float(probs[best_id])
        best_name  = self.class_names[best_id]

        if best_conf < self.conf_thr:
            return None

        return ClassificationResult(
            class_id=best_id,
            class_name=best_name,
            confidence=best_conf,
            top5=top5,
        )

    def classify_rois(self, frame: np.ndarray,
                      bboxes: List[Tuple[int, int, int, int]]
                      ) -> List[Optional[ClassificationResult]]:
        """
        Classify multiple ROIs from *frame* given (x1, y1, x2, y2) *bboxes*.
        """
        results = []
        for (x1, y1, x2, y2) in bboxes:
            roi = frame[max(0, y1):y2, max(0, x1):x2]
            if roi.size == 0:
                results.append(None)
            else:
                results.append(self.classify(roi))
        return results

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        import torch
        import torchvision.models as models

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        arch_map = {
            "resnet18":       models.resnet18,
            "resnet50":       models.resnet50,
            "mobilenet_v2":   models.mobilenet_v2,
            "efficientnet_b0": models.efficientnet_b0,
        }
        builder = arch_map.get(self.arch, models.resnet18)
        model   = builder(weights=None)

        # Adjust final layer for num_classes
        self._adapt_head(model)

        model_file = Path(self.model_path)
        if model_file.exists():
            import torch
            state = torch.load(str(model_file),
                               map_location=self.device)
            model.load_state_dict(state, strict=False)
            logger.info("Loaded CNN weights from %s", model_file)
        else:
            logger.warning(
                "CNN model weights not found at '%s'. "
                "Running with random weights — predictions will be meaningless. "
                "Train via: python pipelines/training_pipeline.py",
                self.model_path,
            )

        model.eval()
        self.model = model.to(self.device)

    def _adapt_head(self, model) -> None:
        import torch.nn as nn
        n = self.num_classes
        if hasattr(model, "fc"):                       # ResNet
            model.fc = nn.Linear(model.fc.in_features, n)
        elif hasattr(model, "classifier"):
            last = model.classifier[-1]
            if hasattr(last, "in_features"):
                model.classifier[-1] = nn.Linear(last.in_features, n)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray):
        import torch
        import cv2

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb   = cv2.resize(rgb, self.input_size).astype(np.float32) / 255.0
        rgb   = (rgb - self._MEAN) / self._STD
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.float()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_import_torch() -> bool:
        try:
            import torch          # noqa: F401
            import torchvision    # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "PyTorch / torchvision not installed. "
                "CNN classification disabled. "
                "Install with: pip install torch torchvision"
            )
            return False

    def _default_class_names(self) -> List[str]:
        if self.dataset_name == "cifar10":
            return ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]
        return [str(i) for i in range(self.num_classes)]
