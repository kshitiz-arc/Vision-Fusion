"""
VisionFusion AI — Training Pipeline
=====================================
End-to-end training harness for the CNN classification head.

Supports
--------
* CIFAR-10  (auto-download via torchvision)
* Custom datasets (ImageFolder-compatible directory)
* Warmup + cosine LR scheduling
* TensorBoard logging
* Best-model checkpointing
* Early stopping

Usage
-----
    python pipelines/training_pipeline.py \\
        --config configs/default.yaml \\
        --dataset cifar10 \\
        --epochs 50 \\
        --arch resnet18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger      import get_logger
from utils.config_loader import ConfigLoader

logger = get_logger("training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionFusion AI — CNN Training Pipeline"
    )
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "custom"])
    parser.add_argument("--arch",    default="resnet18",
                        choices=["resnet18", "resnet50",
                                 "mobilenet_v2", "efficientnet_b0"])
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--batch",   type=int, default=None)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--output",  default=None)
    return parser.parse_args()


def build_dataloaders(cfg: dict, dataset_name: str):
    """Build PyTorch DataLoader objects for train / val / test splits."""
    import torch
    import torchvision.transforms as T
    import torchvision.datasets  as D
    from torch.utils.data import DataLoader, random_split

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])
    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])

    root = cfg["datasets"].get(dataset_name, {}).get("root", f"data/{dataset_name}")
    Path(root).mkdir(parents=True, exist_ok=True)

    if dataset_name == "cifar10":
        train_ds = D.CIFAR10(root=root, train=True,  transform=train_tf, download=True)
        val_ds   = D.CIFAR10(root=root, train=False, transform=val_tf,   download=True)
    else:
        full_ds = D.ImageFolder(root=root, transform=train_tf)
        n_train = int(0.8 * len(full_ds))
        n_val   = len(full_ds) - n_train
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        val_ds.dataset.transform = val_tf

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader


def build_model(arch: str, num_classes: int):
    """Instantiate and configure a torchvision model."""
    import torchvision.models as M
    import torch.nn as nn

    factory_map = {
        "resnet18":       M.resnet18,
        "resnet50":       M.resnet50,
        "mobilenet_v2":   M.mobilenet_v2,
        "efficientnet_b0": M.efficientnet_b0,
    }
    model = factory_map[arch](weights=None)

    # Adapt classification head
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        last = model.classifier[-1]
        if hasattr(last, "in_features"):
            model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model


def train(cfg: dict, arch: str, dataset_name: str,
          output_dir: str | None = None) -> None:
    """Main training loop."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch torchvision")
        return

    t_cfg    = cfg["training"]
    epochs   = t_cfg["epochs"]
    lr       = t_cfg["learning_rate"]
    wd       = t_cfg["weight_decay"]
    warmup   = t_cfg.get("warmup_epochs", 5)
    patience = t_cfg.get("early_stopping_patience", 10)
    ckpt_dir = Path(output_dir or t_cfg.get("checkpoint_dir", "models/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    num_classes  = cfg["cnn_classifier"]["num_classes"]
    train_loader, val_loader = build_dataloaders(cfg, dataset_name)
    model        = build_model(arch, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup)
    scheduler        = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                                     milestones=[warmup])

    best_acc, no_improve = 0.0, 0

    for epoch in range(1, epochs + 1):
        # ── Training pass ──────────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds    = outputs.max(1)
            correct    += preds.eq(labels).sum().item()
            total      += labels.size(0)

        train_acc  = correct / total
        train_loss /= total

        # ── Validation pass ────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                val_loss   += loss.item() * inputs.size(0)
                _, preds    = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total   += labels.size(0)

        val_acc  = val_correct / val_total
        val_loss /= val_total
        scheduler.step()

        logger.info(
            "Epoch [%03d/%03d]  train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f  lr=%.6f",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
            optimizer.param_groups[0]["lr"],
        )

        # ── Checkpointing ──────────────────────────────────────────
        if val_acc > best_acc:
            best_acc   = val_acc
            no_improve = 0
            ckpt_path  = ckpt_dir / f"{arch}_best.pth"
            torch.save(model.state_dict(), str(ckpt_path))
            logger.info("  ✓ New best model saved → %s (acc=%.4f)", ckpt_path, best_acc)
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info("Early stopping triggered after %d epochs without improvement.", epoch)
            break

    logger.info("Training complete. Best validation accuracy: %.4f", best_acc)


def main() -> None:
    args = parse_args()
    cfg  = ConfigLoader.load(args.config)

    # CLI overrides
    if args.epochs: cfg["training"]["epochs"] = args.epochs
    if args.batch:  cfg["training"]["batch_size"] = args.batch
    if args.lr:     cfg["training"]["learning_rate"] = args.lr
    if args.arch:   cfg["cnn_classifier"]["architecture"] = args.arch

    train(cfg, arch=args.arch, dataset_name=args.dataset,
          output_dir=args.output)


if __name__ == "__main__":
    main()
