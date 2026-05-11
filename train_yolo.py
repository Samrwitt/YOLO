#!/usr/bin/env python3
"""
Train a YOLO detection model (Ultralytics YOLOv8 family).

Why a separate script from `train.py`?
- LeNet solves *classification* (one label per image) on Fashion-MNIST.
- YOLO solves *object detection* (bounding boxes + class ids). The loss, data labels,
  and metrics (mAP) are different, so training goes through Ultralytics' battle-tested loop.

Before running:
  pip install -r requirements.txt

Example (small built-in demo dataset, good for laptops):
  python train_yolo.py --data coco128.yaml --model yolov8n.pt --epochs 30

Use your own dataset by passing a YOLO-format `data.yaml` (paths, class names, train/val).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO (Ultralytics) for detection coursework.")
    # `coco128.yaml` ships with Ultralytics — 128 COCO images for quick experiments.
    p.add_argument(
        "--data",
        type=str,
        default="coco128.yaml",
        help="YOLO data YAML (built-in name like coco128.yaml, or path to your yaml).",
    )
    # Start from pretrained nano weights; faster convergence than training from scratch.
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Checkpoint or architecture YAML, e.g. yolov8n.pt, yolov8s.pt, yolov8n.yaml.",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640, help="Square training image size.")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", type=Path, default=Path("runs/detect"), help="Ultralytics project dir.")
    p.add_argument("--name", type=str, default="yolo_train", help="Run name under project.")
    p.add_argument("--device", type=str, default="", help="cuda, cpu, or 0,1,... (empty = auto).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Import here so `python train.py` (LeNet-only) does not require ultralytics.
    from ultralytics import YOLO

    # Load model graph + weights (or architecture yaml if training from scratch).
    model = YOLO(args.model)

    # `train()` handles augmentations, optimizer, loss (box + cls + dfl), validation mAP.
    # `exist_ok` avoids crashing if you re-run the same experiment name in development.
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        device=args.device or None,
    )

    # After training, `model.trainer` holds paths Ultralytics used for this run.
    trainer = getattr(model, "trainer", None)
    save_dir = Path(trainer.save_dir) if trainer is not None else args.project / args.name
    print(f"Training finished. Artifacts directory: {save_dir}")
    print(f"Best weights (typical path): {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
