#!/usr/bin/env python3
"""
Evaluate our **PyTorch TinyYOLOv1** checkpoint on PASCAL VOC 2007 val.

Reports **mean AP at IoU=0.5** (VOC-style single threshold) using all ground-truth boxes.
This is a lightweight teaching metric: a full COCO-style evaluation would add multiple
IoU thresholds and area ranges.

Example:
  python evaluate_yolo.py --checkpoint runs/yolo/yolo_v1_tiny_.../best.pt --data-root ./data
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, nms
from tqdm import tqdm

from datasets import VocYoloEvalDataset, voc_eval_collate
from datasets.voc_yolo import VOC_CLASSES
from models.yolo_v1_tiny import TinyYoloV1, boxes_cxcywh_to_xyxy, decode_predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TinyYOLOv1 on VOC2007 val (mAP@0.5).")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("./data"))
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--grid", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--conf-thres", type=float, default=0.15, help="Confidence gate before NMS.")
    p.add_argument("--nms-thres", type=float, default=0.45)
    p.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold for AP matching.")
    return p.parse_args()


def decode_image_detections(
    pred: torch.Tensor,
    conf_thres: float,
    nms_thres: float,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    pred: (1, 5+C, S, S) for a single image.
    Returns filtered tensors (K,), (K,4), (K,) scores, xyxy boxes in [0,1], class ids.
    """
    cx, cy, w, h, conf, tcls = decode_predictions(pred)
    cls_prob, cls_id = tcls.softmax(dim=-1).max(dim=-1)
    score = conf * cls_prob  # joint score for ranking

    cx = cx[0].reshape(-1)
    cy = cy[0].reshape(-1)
    w = w[0].reshape(-1)
    h = h[0].reshape(-1)
    score = score[0].reshape(-1)
    cls_id = cls_id[0].reshape(-1)

    keep = score > conf_thres
    if keep.sum() == 0:
        return (
            torch.zeros(0, device=pred.device),
            torch.zeros(0, 4, device=pred.device),
            torch.zeros(0, dtype=torch.long, device=pred.device),
        )

    cx, cy, w, h, score, cls_id = cx[keep], cy[keep], w[keep], h[keep], score[keep], cls_id[keep]
    boxes = boxes_cxcywh_to_xyxy(cx, cy, w, h)

    # Per-class NMS (standard practice avoids suppressing different classes in the same spot).
    kept_scores: List[torch.Tensor] = []
    kept_boxes: List[torch.Tensor] = []
    kept_cls: List[torch.Tensor] = []
    for c in range(num_classes):
        m = cls_id == c
        if not m.any():
            continue
        b = boxes[m]
        s = score[m]
        idx = nms(b, s, nms_thres)
        kept_boxes.append(b[idx])
        kept_scores.append(s[idx])
        kept_cls.append(torch.full((idx.numel(),), c, device=pred.device, dtype=torch.long))

    if not kept_scores:
        return (
            torch.zeros(0, device=pred.device),
            torch.zeros(0, 4, device=pred.device),
            torch.zeros(0, dtype=torch.long, device=pred.device),
        )

    scores_out = torch.cat(kept_scores, dim=0)
    boxes_out = torch.cat(kept_boxes, dim=0)
    cls_out = torch.cat(kept_cls, dim=0)
    return scores_out, boxes_out, cls_out


def compute_ap(
    detections: List[Tuple[int, float, torch.Tensor]],
    gt_by_image_class: Dict[int, torch.Tensor],
    iou_thres: float,
) -> float:
    """
    detections: list of (image_id, score, box_xyxy) for one class, any order.
    gt_by_image_class: image_id -> (Ni,4) GT boxes of that class on that image (xyxy [0,1]).
    """
    if not gt_by_image_class:
        return 0.0

    num_gt = int(sum(g.shape[0] for g in gt_by_image_class.values()))
    if num_gt == 0:
        return 0.0

    if not detections:
        return 0.0

    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    tp = torch.zeros(len(detections))
    fp = torch.zeros(len(detections))

    matched: Dict[int, List[bool]] = defaultdict(list)
    for gi, g in gt_by_image_class.items():
        matched[gi] = [False] * g.shape[0]

    for i, (img_id, score, box) in enumerate(detections):
        gts = gt_by_image_class.get(img_id)
        if gts is None or gts.numel() == 0:
            fp[i] = 1
            continue
        ious = box_iou(box.unsqueeze(0), gts)[0]
        best_iou, best_j = float(ious.max().item()), int(ious.argmax().item())
        if best_iou >= iou_thres and not matched[img_id][best_j]:
            tp[i] = 1
            matched[img_id][best_j] = True
        else:
            fp[i] = 1

    tp_c = torch.cumsum(tp, dim=0)
    fp_c = torch.cumsum(fp, dim=0)
    rec = tp_c / max(num_gt, 1)
    prec = tp_c / torch.clamp(tp_c + fp_c, min=1e-8)

    # VOC-style AP: area under precision-recall curve (monotonic envelope).
    mrec = torch.cat([torch.tensor([0.0]), rec, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), prec, torch.tensor([0.0])])
    for j in range(mpre.numel() - 1, 0, -1):
        mpre[j - 1] = torch.maximum(mpre[j - 1], mpre[j])
    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    if idx.numel() == 0:
        return 0.0
    ap = ((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum().item()
    return float(ap)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(VOC_CLASSES)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TinyYoloV1(num_classes=num_classes, grid_size=args.grid, in_ch=3).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = VocYoloEvalDataset(args.data_root, "val", img_size=args.img_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=voc_eval_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # Per-class detection lists and GT buckets.
    dets_per_class: DefaultDict[int, List[Tuple[int, float, torch.Tensor]]] = defaultdict(list)
    gt_per_class: DefaultDict[int, Dict[int, torch.Tensor]] = defaultdict(dict)

    # First pass: build GT structure per (class, image).
    for img_idx in range(len(ds)):
        _, boxes, labels = ds[img_idx]
        if boxes.numel() == 0:
            continue
        for j in range(boxes.shape[0]):
            c = int(labels[j].item())
            b = boxes[j].unsqueeze(0)
            if img_idx not in gt_per_class[c]:
                gt_per_class[c][img_idx] = b
            else:
                gt_per_class[c][img_idx] = torch.cat([gt_per_class[c][img_idx], b], dim=0)

    # Second pass: run network and collect detections.
    img_id = 0
    for imgs, _, _ in tqdm(loader, desc="inference"):
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)
        for b in range(preds.shape[0]):
            s, bx, cid = decode_image_detections(
                preds[b : b + 1], args.conf_thres, args.nms_thres, num_classes
            )
            for k in range(s.numel()):
                dets_per_class[int(cid[k].item())].append((img_id + b, float(s[k].item()), bx[k].detach().cpu()))
        img_id += imgs.shape[0]

    aps: List[float] = []
    for c in range(num_classes):
        ap = compute_ap(dets_per_class[c], gt_per_class[c], args.iou_thres)
        aps.append(ap)
        print(f"AP50 {VOC_CLASSES[c]:12s}: {ap:.4f}")

    mean_ap = sum(aps) / max(len(aps), 1)
    print(f"Mean AP@0.5 (VOC val, {num_classes} classes): {mean_ap:.4f}")


if __name__ == "__main__":
    main()
