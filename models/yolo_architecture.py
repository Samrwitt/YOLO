"""
YOLO (You Only Look Once) — context for this repository's **TinyYOLOv1** implementation.

Where the code lives:
- `yolo_v1_tiny.py` — convolutional backbone + 1×1 detection head, S×S grid, one box per cell.
- `yolo_loss.py` — YOLOv1-style multi-part loss (coordinates, objectness, classification).
- `../datasets/voc_yolo.py` — PASCAL VOC 2007 loading + assignment of GT boxes to grid cells.
- `../train_yolo.py` / `../evaluate_yolo.py` — training and mAP@0.5 evaluation.

Motivation vs region-based detectors:
- Two-stage pipelines (proposals, then classifiers) can be accurate but heavier; YOLO frames
  detection as dense prediction from one CNN pass.

Relation to LeNet (`lenet.py`):
- LeNet outputs a **single** class distribution for the whole image.
- YOLO outputs **many** localized predictions (per grid cell), enabling multiple objects per image.

Modern YOLO variants (v3–v8) add multi-scale heads, better anchors / anchor-free designs, and
improved training tricks; this project keeps the original **single-scale grid** idea so the
entire model fits in a small student codebase.
"""
