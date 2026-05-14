# Assignment 3 — Findings Report  
## Deep Learning for Computer Vision (LeNet-5 & YOLO-style Detection)

**Project:** PyTorch implementations with training/evaluation in Jupyter notebooks.  
**Recorded runs:** `runs/lenet_1778588535`, `runs/yolo/yolo_v1_tiny_1778587728` (metrics in each folder’s `metrics.jsonl`).

---

## 1. Objective

Compare two CNN paradigms required by the assignment:

- **LeNet-5 (1998):** image-level **classification** on **Fashion-MNIST**.
- **YOLO (2016 / YOLOv1-style):** **single-stage object detection** on **PASCAL VOC 2007**, implemented as a compact **TinyYOLOv1** grid model for coursework.

---

## 2. Experimental setup

| Component | LeNet-5 | YOLO-style (Tiny) |
|-----------|---------|-------------------|
| **Dataset** | Fashion-MNIST (10 classes, 28×28 grayscale) | PASCAL VOC 2007 train/val |
| **Task** | One label per image | Bounding boxes + 20 object classes |
| **Model** | `LeNet5` in `models/lenet.py` | `TinyYoloV1` + `YoloV1Loss` |
| **Training logged** | Train/test loss & accuracy per epoch | Train/val loss per epoch |
| **Primary metric** | Test **top-1 accuracy** | **mAP@0.5** on VOC val (plus validation loss for model selection) |

Hyperparameters (epochs, batch size, learning rate) are defined in:

- `notebooks/lenet_train_eval.ipynb`
- `notebooks/yolo_train_eval.ipynb`

---

## 3. Results — LeNet-5 (Fashion-MNIST)

### Summary

| Metric | Value |
|--------|------:|
| **Best test accuracy** | **90.60%** |
| **Epoch of best checkpoint** | **13** (of 15 trained) |
| Final epoch (15) train / test accuracy | 92.55% / 89.88% |

### Interpretation

- Learning is **stable**: train accuracy rises from ~75% (epoch 1) to ~92.5% (epoch 15).
- **Peak generalization** occurs at **epoch 13**; later epochs show **slightly lower** test accuracy while train accuracy still increases — a small **train–test gap** consistent with **mild overfitting** or metric noise.
- For reporting, prefer the **saved best checkpoint** (highest test accuracy) rather than the final epoch.

---

## 4. Results — Tiny YOLOv1-style (VOC 2007)

### Validation loss (model selection)

| Metric | Value |
|--------|------:|
| **Lowest validation loss** | **3.338** |
| **Epoch of best checkpoint** | **16** (of 40 trained) |
| Training epochs completed | 40 |

After ~epoch 16, **training loss** continues to decrease while **validation loss** increases — clear **overfitting** to the training set. The **best model** for detection quality should be taken from **epoch 16**, not the last epoch.

### Detection metric (VOC val, IoU = 0.5)

| Metric | Value |
|--------|------:|
| **Mean AP@0.5** (20 classes) | **0.0035** (~0.35%) |

Representative per-class AP values from the evaluation run were **very low** for most classes (often near 0); among the higher values were **horse**, **train**, and **diningtable** (still well below practical deployment levels).

### Interpretation

- **mAP@0.5** reflects **localization + classification** jointly; it is **much harder** to push than Fashion-MNIST accuracy.
- The implementation uses a **deliberately small** detector (single-scale grid, **one box per cell** in this codebase) and a **limited training budget** relative to published YOLO systems — **low mAP is expected** in a teaching setting.
- **Validation loss** and **mAP** are different: the network can minimize loss components while still producing poor precision/recall on crowded VOC scenes.

---

## 5. Cross-architecture comparison

| Aspect | LeNet-5 | YOLO-style |
|--------|---------|------------|
| **Output** | Single class distribution | Dense grid of boxes + scores |
| **Inductive bias** | Translation tolerance via conv/pool | Spatial grid + shared conv backbone |
| **Observed headline metric** | **~90.6%** test accuracy | **~0.35%** mean AP@0.5 (this run) |
| **Role in the course narrative** | Early CNN success on simple inputs | Real-time detection idea; tradeoffs for speed vs accuracy |

---

## 6. Strengths, limitations, and tradeoffs

**Strengths**

- End-to-end **PyTorch** pipelines with **reproducible metrics** (`metrics.jsonl`, checkpoints under `runs/`).
- Clear separation of **classification** vs **detection** objectives and metrics.

**Limitations**

- LeNet: modest capacity; Fashion-MNIST is simpler than large-scale natural-image benchmarks.
- YOLO coursework model: simplified vs original YOLOv1 (e.g., **B=1**); single scale; **40 epochs** may be insufficient for competitive VOC mAP without augmentation and schedules.

**Computational tradeoffs**

- LeNet is **lightweight** (fast iterations).
- Detection requires **more compute per image** (resolution, VOC loading, dense loss); improving mAP typically requires **more parameters, epochs, and engineering**.

---

## 7. Conclusions

1. **LeNet-5** demonstrates that classical convolutional structure achieves **strong Fashion-MNIST accuracy (~90.6%)**, with best performance at **epoch 13**, illustrating generalization and the value of **checkpoint selection** by validation/test performance.

2. **Tiny YOLOv1-style** training on VOC shows **meaningful learning in loss** (best **val loss ~3.34 at epoch 16**) but **low mAP** under the simplified architecture and training regime — consistent with the **difficulty of detection** and the **gap** between teaching-scale models and production detectors.

3. Together, the experiments support the course themes: **motivation for CNN architectures**, **differences between classification and detection**, and **computational tradeoffs** when scaling vision models.

---

## 8. Artifacts (for markers / slides)

- Notebooks: `notebooks/lenet_train_eval.ipynb`, `notebooks/yolo_train_eval.ipynb`
- Metrics: `runs/*/metrics.jsonl`
- Background notes: `docs/lenet5.md`, `docs/yolo_v1.md`
- **Slides:** `docs/findings_presentation.pptx` (generated from repo script)

---

*Figures: training curves can be reproduced from `metrics.jsonl` via the plot cells in each notebook; PNGs are written under `runs/figures/` when those cells are executed.*
