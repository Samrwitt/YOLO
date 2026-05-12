#!/usr/bin/env python3
"""
Train our **PyTorch YOLOv1-style** detector (`TinyYoloV1`) on PASCAL VOC 2007.

This is the detection counterpart to `train.py` (LeNet on Fashion-MNIST): same framework
(PyTorch), different task (bounding boxes + classes), different labels and loss.

Example:
  python train_yolo.py --data-root ./data --epochs 40 --batch-size 16

The first run downloads VOC2007 into `--data-root` (can be several hundred MB).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import VocYoloGridDataset, voc_yolo_collate
from datasets.voc_yolo import VOC_CLASSES
from models.yolo_loss import YoloV1Loss
from models.yolo_v1_tiny import TinyYoloV1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TinyYOLOv1 (PyTorch) on VOC2007.")
    p.add_argument("--data-root", type=Path, default=Path("./data"), help="Root for VOC download (see torchvision VOC).")
    p.add_argument("--run-dir", type=Path, default=Path("./runs/yolo"), help="Checkpoints + metrics.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=224, help="Must match backbone stride to grid (224 -> 7x7).")
    p.add_argument("--grid", type=int, default=7, help="S in SxS YOLO grid.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model: torch.nn.Module, criterion: YoloV1Loss, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for imgs, gt, obj, cls in loader:
        imgs = imgs.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        obj = obj.to(device, non_blocking=True)
        cls = cls.to(device, non_blocking=True)
        pred = model(imgs)
        loss, _ = criterion(pred, gt, obj, cls)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = VocYoloGridDataset(args.data_root, "train", img_size=args.img_size, grid_size=args.grid)
    val_ds = VocYoloGridDataset(args.data_root, "val", img_size=args.img_size, grid_size=args.grid)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=voc_yolo_collate,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=voc_yolo_collate,
        pin_memory=pin,
    )

    num_classes = len(VOC_CLASSES)

    model = TinyYoloV1(num_classes=num_classes, grid_size=args.grid, in_ch=3).to(device)
    criterion = YoloV1Loss(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_name = f"yolo_v1_tiny_{int(time.time())}"
    run_path = args.run_dir / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for imgs, gt, obj, cls in pbar:
            imgs = imgs.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            obj = obj.to(device, non_blocking=True)
            cls = cls.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(imgs)
            loss, parts = criterion(pred, gt, obj, cls)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=f"{running/steps:.3f}", coord=float(parts["coord"]), cls=float(parts["cls"]))

        train_loss = running / max(steps, 1)
        val_loss = validate(model, criterion, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "args": vars(args)}
            torch.save(ckpt, run_path / "best.pt")

    with (run_path / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")

    print(f"Done. Best val loss {best_val:.4f}. Checkpoints in {run_path}")


if __name__ == "__main__":
    main()
