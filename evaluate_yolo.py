#!/usr/bin/env python3
"""
Validate a trained YOLO checkpoint (mAP, precision, recall on the val split from data.yaml).

Example:
  python evaluate_yolo.py --weights runs/detect/yolo_train/weights/best.pt --data coco128.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate YOLO weights on a YOLO-format dataset.")
    p.add_argument("--weights", type=Path, required=True, help="Path to best.pt or last.pt.")
    p.add_argument("--data", type=str, default="coco128.yaml", help="Same data YAML used for training.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="", help="cuda, cpu, or device id (empty = auto).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    # Load trained weights into the YOLO wrapper (architecture matches training).
    model = YOLO(str(args.weights))

    # `val` runs NMS + metric computation; returns metrics object with maps50, maps50-95, etc.
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
    )

    # Print a compact summary for lab reports (Ultralytics also logs richer tables).
    box = metrics.box
    print("Validation summary:")
    print(f"  mAP50-95: {float(box.map):.4f}")
    print(f"  mAP50:    {float(box.map50):.4f}")
    print(f"  mean P:   {float(box.mp):.4f}")
    print(f"  mean R:   {float(box.mr):.4f}")


if __name__ == "__main__":
    main()
