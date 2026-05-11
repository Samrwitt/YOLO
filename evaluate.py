#!/usr/bin/env python3
"""
Load a saved LeNet checkpoint and report Fashion-MNIST test accuracy.

Example:
  python evaluate.py --checkpoint runs/lenet_1234567890/best.pt

For YOLO detection metrics (mAP), use `evaluate_yolo.py` instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import LeNet5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained LeNet checkpoint.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("./data"))
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def build_test_loader(data_root: Path, batch_size: int) -> DataLoader:
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    test_set = datasets.FashionMNIST(data_root, train=False, download=True, transform=tfm)
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = LeNet5(num_classes=10).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = build_test_loader(args.data_root, args.batch_size)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    print(f"Checkpoint epoch (at save): {ckpt.get('epoch', 'n/a')}")
    print(f"Test loss: {total_loss / total:.4f}")
    print(f"Test accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    main()
