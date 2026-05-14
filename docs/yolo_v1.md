# YOLO (2016 / YOLOv1) — Background, Motivation, and Tradeoffs

## Motivation: “You Only Look Once”

Before YOLO, many high-accuracy detectors were **two-stage**:

- generate candidate regions (proposals),
- classify each region and refine its bounding box.

This pipeline can be accurate but often has **higher latency** because the model (or parts of it) are applied many times per image. YOLO (Redmon et al., 2016) reframed detection as a **single neural network** that predicts **bounding boxes and classes in one forward pass**, enabling real-time detection.

## Key predecessors and what YOLO changed

### Sliding-window detectors and DPM

- **Deformable Part Models (DPM, 2008)**: used handcrafted HOG features and part-based scoring to detect objects in a sliding-window style. Detection was still a post-processing pipeline with separate feature extraction, scoring, and non-max suppression.
- **OverFeat (2014)**: introduced CNNs for dense sliding-window detection, producing classification and localization scores from many image positions. It was an important step toward using convolutional networks for detection, but it still scored windows rather than predicting a unified global grid.
- **Other dense prediction methods**: early CNN-based detectors computed score maps over image positions and scales, but they did not fully unify classification and bounding-box regression into a single direct output tensor.

### R-CNN family (region-based detectors)

- **R-CNN (2014)**: generate region proposals (Selective Search), run a CNN on each region, classify + regress boxes. Accurate but slow due to per-proposal CNN computation.
- **Fast R-CNN (2015)**: compute CNN features once per image, then pool features for each proposal (ROI pooling). Much faster than R-CNN, still proposal-driven.
- **Faster R-CNN (2015)**: replaces external proposals with a learned Region Proposal Network (RPN). Strong accuracy, but still fundamentally two-stage (propose → classify).

### SSD and single-shot predecessors

- **SSD (2016)**: predicts boxes from multiple feature maps at different scales, showing that single-shot detection can match two-stage accuracy if multi-scale output is used.
- **YOLO difference**: from the start, YOLO predicts all detections in one forward pass using a single global feature map, not region proposals or a sliding-window stage.

**YOLO difference**: no proposal stage. YOLO predicts a dense set of candidate boxes directly from the image, making detection “single shot”.

### Unified detection vs earlier pipelines

YOLO’s key change is to treat detection as a unified regression problem over an image-level grid, while many predecessors still separated proposal/scores or relied on sliding-window scoring.

## Core YOLOv1 idea (architecture-level)

YOLOv1 divides the input image into an **\(S \times S\)** grid. Each cell predicts:

- bounding box parameters (center + size),
- an **objectness/confidence** score,
- and class probabilities.

This turns detection into a **single dense prediction problem** similar in spirit to semantic segmentation heads, but with box geometry outputs.

## Architectural innovations introduced

- **Single-stage detection**: one forward pass produces all detections.
- **Unified loss** for localization + objectness + classification.
- **Global reasoning**: predicting detections from full-image features helps reduce some background false positives compared to purely local window classification (though it introduces other errors).

## Strengths

- **Speed**: efficient inference; good for real-time applications.
- **Simple pipeline**: no external proposal generator, fewer moving pieces.
- **End-to-end training**: jointly learns features for detection.

## Weaknesses (especially YOLOv1)

- **Localization precision**: early YOLO struggled with small objects and precise box placement.
- **Grid constraint**: if multiple objects’ centers fall into the same grid cell, the model can miss one (capacity bottleneck per cell).
- **Single-scale prediction**: YOLOv1 uses a single grid scale; modern detectors use multi-scale heads to improve small-object recall.

## Quantitative failure analysis for this repo’s YOLO implementation

- **Best checkpoint**: epoch 16 with validation loss **3.338**.
- **mAP@0.5** on VOC 2007 validation: **0.0035** (~0.35%).
- **Per-class AP**: most classes are near zero; higher values appear only for classes like `horse`, `train`, and `diningtable`, still far below practical detection quality.

### Observed failure modes

- **Poor small-object recall**: the 7×7 grid is too coarse for many VOC objects, so small objects are often missed entirely.
- **Multiple objects per cell**: because this implementation uses one box per cell, scenes with two objects whose centers fall in the same grid cell produce false negatives.
- **Low confidence on valid detections**: predicted boxes can have too-low objectness scores, meaning correct boxes are filtered out during post-processing.
- **Loss vs metric mismatch**: the network may continue to reduce training loss while mAP remains very low, showing that joint regression/classification loss does not guarantee good detection precision/recall on complex scenes.

These quantitative failure modes explain why later YOLO variants add multi-scale heads, more anchors/boxes per cell, and stronger localization refinements.

## Computational tradeoffs

- **Lower latency** than two-stage detectors (fewer per-image operations).
- **Potential accuracy tradeoff** (especially on small objects) due to coarse grids and limited per-cell capacity.

## How this repo’s implementation maps to YOLOv1 (and what’s simplified)

This project keeps the original YOLOv1 “single shot grid” formulation:

- outputs are shaped like `(5 + C, S, S)` (box + objectness + classes per cell),
- training assigns each ground-truth object to the cell containing its center,
- evaluation reports **mAP@0.5** on VOC2007 val.

To keep the code coursework-sized and readable, it uses **one box per cell (B=1)**. The original YOLOv1 uses \(B>1\) and a “responsibility” rule to choose which predicted box best matches a ground-truth object. The simplification preserves the core concept while reducing complexity.

## What you should emphasize in your presentation

- **Motivation**: real-time detection by removing the proposal stage.
- **Innovation**: dense prediction + unified loss in one CNN.
- **Tradeoff**: speed vs small-object localization/recall; why later YOLO versions add multi-scale heads and other improvements.

## Where this appears in the repo

- Model + decode head: `models/yolo_v1_tiny.py`; loss: `models/yolo_loss.py`.
- VOC loading / grid targets: `datasets/voc_yolo.py`.
- Train / evaluate (mAP@0.5) / plots: `notebooks/yolo_train_eval.ipynb`.
- Written results template: `docs/assignment_findings.md`.
