"""
YOLO (You Only Look Once) — single-stage object detection.

Motivation (high level):
- Two-stage detectors (e.g. R-CNN family) first propose regions, then classify each region.
  That pipeline is accurate but slower and more complex to train end-to-end at scale.
- YOLO reframes detection as *dense prediction*: one CNN forward pass predicts many boxes
  and class probabilities over a grid / feature pyramid, trading some localization nuance
  for speed and simplicity.

Predecessors vs YOLO:
- Classical CNNs like LeNet map an image to *one* label vector (classification). They do not
  output spatial sets of boxes; extending them naively to many objects requires extra
  structure (region proposals, anchors, etc.).
- YOLO adds heads that predict box offsets and per-class scores at multiple scales (in
  modern variants), with a backbone (CNN) shared across the image.

Innovations across YOLO generations (for your presentation):
- YOLOv1: single network, S×S grid, limited small-object behavior.
- Later versions: better backbones, anchor / anchor-free designs, feature pyramid networks,
  improved loss (CIoU/DIoU), data augmentation (mosaic), and training efficiency.

This repository trains a *modern* YOLO implementation via the Ultralytics library
(see `train_yolo.py`). The YAML/weights define the concrete depth-width schedule; your
report should cite the exact variant (e.g. YOLOv8n) and discuss compute vs accuracy.
"""

# Intentionally no heavy imports here — this module documents architecture for coursework.
