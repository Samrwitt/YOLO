Deep Learning for Computer Vision

This project compares **two CNN architectures** implemented in **PyTorch**:

- **LeNet-5 (1998)** — image classification on **Fashion-MNIST**
- **YOLO (2016 / YOLOv1-style)** — single-stage object detection on **PASCAL VOC 2007**

## Setup

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Part 1 — LeNet-5 (classification)

Open `notebooks/lenet_train_eval.ipynb` and **Run All**.

Training and evaluation logic lives entirely in that notebook. Outputs:

- Checkpoint: `runs/lenet_<timestamp>/best.pt`
- Metrics: `runs/lenet_<timestamp>/metrics.jsonl`

## Part 2 — YOLO (2016 / YOLOv1-style) (detection)

Open `notebooks/yolo_train_eval.ipynb` and **Run All** (VOC2007 downloads on first successful training).

Outputs:

- Checkpoint: `runs/yolo/yolo_v1_tiny_<timestamp>/best.pt`
- Metrics: `runs/yolo/yolo_v1_tiny_<timestamp>/metrics.jsonl`

## Plots

Each notebook ends with a **Plot training curves** section that reads `metrics.jsonl`, saves PNGs under `runs/figures/`, and displays plots inline.

## deliverables

- `docs/lenet5.md` — LeNet-5 motivation, predecessors, innovations, tradeoffs
- `docs/yolo_v1.md` — YOLO (2016 / v1-style) motivation, predecessors vs R-CNN, innovations, tradeoffs

---

### Notes about “YOLO 2016” scope

The original YOLOv1 predicts **\(S \times S\)** grid outputs with **\(B\)** boxes per cell.  
This coursework implementation keeps the original *single-stage grid formulation* but uses **one box per cell** (B=1) to keep target assignment and loss readable while preserving the key idea: **detect in one forward pass** (no region proposals).

---
BY Lamrot Ibsa, Ruhama Yohannes and Samrawit Kahsay
