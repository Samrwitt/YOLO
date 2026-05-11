#!/usr/bin/env python3
"""
Train LeNet-5 on Fashion-MNIST (classification baseline for the assignment).

For YOLO (object detection), use `train_yolo.py` — detection uses different labels, losses,
and metrics than image-level classification.

Typical flow:
  python train.py --epochs 15

Checkpoints and metrics are written under ./runs/ for plots / report tables.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import LeNet5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LeNet-5 on Fashion-MNIST for CNN coursework.")
    p.add_argument("--data-root", type=Path, default=Path("./data"), help="Directory for downloaded datasets.")
    p.add_argument("--run-dir", type=Path, default=Path("./runs"), help="Where to store checkpoints and metrics.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001, help="Adam learning rate.")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(args: argparse.Namespace):
    """
    Return (train_loader, test_loader, num_classes).

    Fashion-MNIST matches LeNet's original 28×28 grayscale setting but is harder than MNIST,
    which supports a clearer discussion of capacity and generalization in the write-up.
    """
    args.data_root.mkdir(parents=True, exist_ok=True)

    # Normalize to zero-mean unit-ish variance — stabilizes optimization.
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST channel stats
        ]
    )
    train_set = datasets.FashionMNIST(args.data_root, train=True, download=True, transform=tfm)
    test_set = datasets.FashionMNIST(args.data_root, train=False, download=True, transform=tfm)
    num_classes = 10

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    return train_loader, test_loader, num_classes


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return (average loss, accuracy) on the given loader."""
    model.eval()
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
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, num_classes = build_loaders(args)

    model = LeNet5(num_classes=num_classes).to(device)

    # Adam is a reasonable default for small student projects (no LR schedule complexity).
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    run_name = f"lenet_{int(time.time())}"
    run_path = args.run_dir / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "test_loss": te_loss, "test_acc": te_acc}
        history.append(row)
        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | test loss {te_loss:.4f} acc {te_acc:.4f}")

        if te_acc > best_acc:
            best_acc = te_acc
            ckpt = {"model": model.state_dict(), "epoch": epoch, "test_acc": te_acc, "args": vars(args)}
            torch.save(ckpt, run_path / "best.pt")

    with (run_path / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")

    print(f"Done. Best test accuracy: {best_acc:.4f}. Artifacts in {run_path}")


if __name__ == "__main__":
    main()
