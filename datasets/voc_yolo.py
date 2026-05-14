"""
PASCAL VOC 2007 → tensors our tiny YOLO can actually train on.

torchvision gives us annotation dicts per image. We resize to a square, scale boxes with the
same resize, then for training we **snap** each object to the grid cell that contains the
center of its box. If two objects fight for the same cell (happens), we keep the bigger one
— that's a simplification, not how full YOLO handles crowded scenes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import functional as TF

# VOC's 20 class names — order matters because we turn names into indices 0..19
VOC_CLASSES: Tuple[str, ...] = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(VOC_CLASSES)}


def _flatten_voc_objects(obj_field: Any) -> List[Dict[str, Any]]:
    """The XML parser sometimes gives one dict, sometimes a list — normalize to a list."""
    if obj_field is None:
        return []
    if isinstance(obj_field, dict):
        return [obj_field]
    if isinstance(obj_field, list):
        return obj_field
    return []


def _parse_annotation_to_boxes(annotation: Dict[str, Any], orig_w: int, orig_h: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pull out every non-difficult object as xyxy in **original image pixels**,
    plus integer labels. Empty images come back as zero-row tensors.
    """
    objs: List[Dict[str, Any]] = []
    ann = annotation.get("annotation", annotation)
    if "object" in ann:
        objs = _flatten_voc_objects(ann["object"])

    xyxy_list: List[List[float]] = []
    labels: List[int] = []
    for obj in objs:
        if str(obj.get("difficult", "0")) == "1":
            continue
        name = obj["name"]
        if isinstance(name, (list, tuple)):
            name = name[0]
        name = str(name).strip()
        if name not in CLASS_TO_IDX:
            continue
        bb = obj["bndbox"]
        xmin = float(bb["xmin"])
        ymin = float(bb["ymin"])
        xmax = float(bb["xmax"])
        ymax = float(bb["ymax"])
        xmin = max(0.0, min(xmin, float(orig_w - 1)))
        xmax = max(0.0, min(xmax, float(orig_w)))
        ymin = max(0.0, min(ymin, float(orig_h - 1)))
        ymax = max(0.0, min(ymax, float(orig_h)))
        if xmax <= xmin + 1 or ymax <= ymin + 1:
            continue
        xyxy_list.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_IDX[name])

    if not xyxy_list:
        return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)

    return torch.tensor(xyxy_list, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def xyxy_to_cxcywh(xyxy: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = xyxy.unbind(dim=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1).clamp(min=1.0)
    h = (y2 - y1).clamp(min=1.0)
    return torch.stack([cx, cy, w, h], dim=-1)


def build_yolo_targets(
    boxes_cxcywh: torch.Tensor,
    labels: torch.Tensor,
    grid_size: int,
    img_size: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Figure out which grid cell "owns" each object (the one its center falls into), then
    stash cxcywh in 0–1 coordinates so the loss lines up with `decode_predictions`.

    Returns gt_box / obj_mask / cls_target, each shaped for an S×S grid.
    """
    S = grid_size
    gt_box = torch.zeros(S, S, 4)
    obj_mask = torch.zeros(S, S)
    cls_target = torch.zeros(S, S, dtype=torch.long)

    if boxes_cxcywh.numel() == 0:
        return gt_box, obj_mask, cls_target

    # Scale box coords so they live on the resized square (still in pixel units here)
    b = boxes_cxcywh.clone()
    b[..., [0, 2]] /= img_size
    b[..., [1, 3]] /= img_size
    b[..., 0] = b[..., 0].clamp(0.0, 1.0)
    b[..., 1] = b[..., 1].clamp(0.0, 1.0)
    b[..., 2] = b[..., 2].clamp(1e-4, 1.0)
    b[..., 3] = b[..., 3].clamp(1e-4, 1.0)

    best_area: Dict[Tuple[int, int], float] = {}
    for i in range(b.shape[0]):
        cx, cy, w, h = b[i].tolist()
        cx_i = min(int(cx * S), S - 1)
        cy_i = min(int(cy * S), S - 1)
        area = w * h
        key = (cy_i, cx_i)
        # Two objects in the same cell: keep whichever has larger area (crude but simple)
        if key not in best_area or area > best_area[key]:
            best_area[key] = area
            gt_box[cy_i, cx_i, :] = b[i]
            cls_target[cy_i, cx_i] = int(labels[i].item())
            obj_mask[cy_i, cx_i] = 1.0

    return gt_box, obj_mask, cls_target


class VocYoloGridDataset(Dataset):
    """VOC train/val split, resized to `img_size`, with per-cell targets for training."""

    def __init__(self, root: Path | str, image_set: str, img_size: int = 224, grid_size: int = 7) -> None:
        self.root = Path(root)
        self.img_size = img_size
        self.S = grid_size
        # No torchvision Compose — we need to scale boxes and pixels together by hand
        self.voc = datasets.VOCDetection(self.root.as_posix(), year="2007", image_set=image_set, download=True)

    def __len__(self) -> int:
        return len(self.voc)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pil, target = self.voc[idx]
        pil = pil.convert("RGB")
        w0, h0 = pil.size

        boxes_xyxy, labels = _parse_annotation_to_boxes(target, w0, h0)
        sx = self.img_size / float(w0)
        sy = self.img_size / float(h0)
        if boxes_xyxy.numel() > 0:
            boxes_xyxy[:, [0, 2]] *= sx
            boxes_xyxy[:, [1, 3]] *= sy

        img = TF.to_tensor(pil)
        img = TF.resize(img, [self.img_size, self.img_size])

        boxes_cxcywh = xyxy_to_cxcywh(boxes_xyxy) if boxes_xyxy.numel() > 0 else torch.zeros(0, 4)
        gt_box, obj_mask, cls_target = build_yolo_targets(boxes_cxcywh, labels, self.S, float(self.img_size))

        # Same mean/std people use with ImageNet weights — works fine for from-scratch too
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std

        return img, gt_box, obj_mask, cls_target


def voc_yolo_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack batch dimension for images and per-cell tensors."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    gt = torch.stack([b[1] for b in batch], dim=0)
    obj = torch.stack([b[2] for b in batch], dim=0)
    cls = torch.stack([b[3] for b in batch], dim=0)
    return imgs, gt, obj, cls


class VocYoloEvalDataset(Dataset):
    """
    Same images as the grid dataset, but we return **every** GT box for mAP.

    Training collapses to one object per cell; evaluation shouldn't, or metrics lie.
    """

    def __init__(self, root: Path | str, image_set: str, img_size: int = 224) -> None:
        self.root = Path(root)
        self.img_size = img_size
        self.voc = datasets.VOCDetection(self.root.as_posix(), year="2007", image_set=image_set, download=True)

    def __len__(self) -> int:
        return len(self.voc)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pil, target = self.voc[idx]
        pil = pil.convert("RGB")
        w0, h0 = pil.size
        boxes_xyxy, labels = _parse_annotation_to_boxes(target, w0, h0)
        sx = self.img_size / float(w0)
        sy = self.img_size / float(h0)
        if boxes_xyxy.numel() > 0:
            boxes_xyxy[:, [0, 2]] *= sx
            boxes_xyxy[:, [1, 3]] *= sy
            boxes_xyxy /= float(self.img_size)

        img = TF.to_tensor(pil)
        img = TF.resize(img, [self.img_size, self.img_size])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        return img, boxes_xyxy, labels


def voc_eval_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    imgs = torch.stack([b[0] for b in batch], dim=0)
    boxes = [b[1] for b in batch]
    labels = [b[2] for b in batch]
    return imgs, boxes, labels
