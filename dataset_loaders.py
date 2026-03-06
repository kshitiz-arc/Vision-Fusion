"""
VisionFusion AI — Dataset Loaders
====================================
Unified dataset access layer for CIFAR-10, COCO, and custom image folders.

Provides
--------
* ``CIFAR10Loader``     — auto-download, PyTorch DataLoader
* ``COCOLoader``        — detection dataset with COCO-format annotations
* ``CustomLoader``      — ImageFolder-style directory → DataLoader
* ``VideoFrameLoader``  — frame iterator over any OpenCV-readable video
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# CIFAR-10
# ============================================================================

class CIFAR10Loader:
    """
    Wraps torchvision CIFAR-10 dataset with sensible augmentation defaults.

    Parameters
    ----------
    root      : path to cache directory (will be created automatically)
    batch_size: mini-batch size
    num_workers: DataLoader worker threads
    """

    CLASS_NAMES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    def __init__(self, root: str = "data/cifar10",
                 batch_size: int = 32,
                 num_workers: int = 2) -> None:
        self.root        = root
        self.batch_size  = batch_size
        self.num_workers = num_workers
        Path(root).mkdir(parents=True, exist_ok=True)

    def get_loaders(self):
        """Return (train_loader, val_loader) tuple."""
        try:
            import torch
            import torchvision.transforms as T
            import torchvision.datasets   as D
            from torch.utils.data import DataLoader
        except ImportError:
            logger.error("torchvision required: pip install torchvision")
            return None, None

        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ])
        val_tf = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ])

        train_ds = D.CIFAR10(self.root, train=True,  transform=train_tf, download=True)
        val_ds   = D.CIFAR10(self.root, train=False, transform=val_tf,   download=True)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True,  num_workers=self.num_workers,
                                  pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size,
                                  shuffle=False, num_workers=self.num_workers,
                                  pin_memory=True)
        logger.info("CIFAR-10 loaded: %d train, %d val samples.",
                    len(train_ds), len(val_ds))
        return train_loader, val_loader


# ============================================================================
# COCO
# ============================================================================

class COCOLoader:
    """
    Minimal COCO detection dataset loader (requires pycocotools).

    Parameters
    ----------
    root            : path to COCO image directory
    annotation_file : path to COCO JSON annotation file
    """

    def __init__(self, root: str, annotation_file: str) -> None:
        self.root            = Path(root)
        self.annotation_file = annotation_file
        self._coco           = None

    def load(self):
        """Load COCO annotations and return the pycocotools COCO object."""
        try:
            from pycocotools.coco import COCO
            self._coco = COCO(self.annotation_file)
            logger.info("COCO dataset loaded: %d images, %d annotations.",
                        len(self._coco.imgs), len(self._coco.anns))
            return self._coco
        except ImportError:
            logger.error("pycocotools required: pip install pycocotools")
            return None

    def image_iter(self) -> Generator[Tuple[np.ndarray, List[dict]], None, None]:
        """Yield (image_bgr, annotations) pairs."""
        if self._coco is None:
            self.load()
        if self._coco is None:
            return
        for img_id, img_info in self._coco.imgs.items():
            img_path = self.root / img_info["file_name"]
            frame    = cv2.imread(str(img_path))
            if frame is None:
                continue
            ann_ids = self._coco.getAnnIds(imgIds=img_id)
            anns    = self._coco.loadAnns(ann_ids)
            yield frame, anns


# ============================================================================
# Custom ImageFolder
# ============================================================================

class CustomLoader:
    """
    Loads a custom classification dataset laid out as:

        root/
            class_a/ img1.jpg img2.jpg …
            class_b/ img1.jpg …

    Parameters
    ----------
    root       : dataset root directory
    batch_size : mini-batch size
    img_size   : (H, W) to resize all images to
    train_split: fraction of data for training
    """

    def __init__(self, root: str, batch_size: int = 32,
                 img_size: Tuple[int, int] = (224, 224),
                 train_split: float = 0.8) -> None:
        self.root        = root
        self.batch_size  = batch_size
        self.img_size    = img_size
        self.train_split = train_split

    def get_loaders(self):
        """Return (train_loader, val_loader, class_names) tuple."""
        try:
            import torch
            import torchvision.transforms as T
            import torchvision.datasets   as D
            from torch.utils.data import DataLoader, random_split
        except ImportError:
            logger.error("torchvision required: pip install torchvision")
            return None, None, []

        tf = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
        ds = D.ImageFolder(self.root, transform=tf)
        n  = len(ds)
        n_train = int(n * self.train_split)
        n_val   = n - n_train
        train_ds, val_ds = random_split(ds, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size,
                                  shuffle=False, num_workers=2)
        logger.info("Custom dataset loaded from '%s': %d classes, %d images.",
                    self.root, len(ds.classes), n)
        return train_loader, val_loader, ds.classes


# ============================================================================
# Video Frame Iterator
# ============================================================================

class VideoFrameLoader:
    """
    Iterate over frames of any OpenCV-readable video file or camera index.

    Example
    -------
    >>> for frame, idx in VideoFrameLoader("video.mp4"):
    ...     process(frame)
    """

    def __init__(self, source: int | str = 0,
                 max_frames: Optional[int] = None,
                 skip: int = 1) -> None:
        self.source     = source
        self.max_frames = max_frames
        self.skip       = max(1, skip)

    def __iter__(self) -> Generator[Tuple[np.ndarray, int], None, None]:
        cap   = cv2.VideoCapture(self.source)
        idx   = 0
        count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % self.skip == 0:
                    yield frame, idx
                    count += 1
                    if self.max_frames and count >= self.max_frames:
                        break
                idx += 1
        finally:
            cap.release()
