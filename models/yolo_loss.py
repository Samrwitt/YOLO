"""
YOLOv1-style combined loss (coordinate + objectness + classification).

This mirrors the original paper's intent:
- Penalize box errors only on cells that contain an object's center (responsible cells).
- Down-weight background cells for objectness so the network is not drowned by negatives.
- Train class probabilities only where an object is present.

Objectness uses `binary_cross_entropy_with_logits` on the raw score (stable). Box and class
terms use decoded boxes / class logits from `decode_predictions`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .yolo_v1_tiny import decode_predictions


class YoloV1Loss(nn.Module):
    def __init__(self, num_classes: int, lambda_coord: float = 5.0, lambda_noobj: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(
        self,
        pred: torch.Tensor,
        gt_box: torch.Tensor,
        obj_mask: torch.Tensor,
        cls_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        pred: (N, 5+C, S, S) raw head outputs.
        gt_box: (N, S, S, 4) normalized cx, cy, w, h (zeros where no object).
        obj_mask: (N, S, S) float/bool 1 where a ground-truth center falls in the cell.
        cls_target: (N, S, S) long class index in [0, C-1] (ignored where obj_mask==0).
        """
        cx, cy, w, h, conf, tcls = decode_predictions(pred)
        obj = obj_mask.float()
        noobj = 1.0 - obj

        pred_box = torch.stack([cx, cy, w, h], dim=-1)
        coord_err = F.smooth_l1_loss(pred_box, gt_box, reduction="none").sum(dim=-1)
        loss_coord = (obj * coord_err).sum() / obj.sum().clamp(min=1.0)

        # Raw objectness logits live in channel index 4 before sigmoid.
        tobj_raw = pred.permute(0, 2, 3, 1).contiguous()[..., 4]
        loss_obj = (obj * F.binary_cross_entropy_with_logits(tobj_raw, torch.ones_like(tobj_raw), reduction="none")).sum()
        loss_obj = loss_obj / obj.sum().clamp(min=1.0)
        loss_noobj = (
            noobj * F.binary_cross_entropy_with_logits(tobj_raw, torch.zeros_like(tobj_raw), reduction="none")
        ).sum() / noobj.sum().clamp(min=1.0)

        logits = tcls.reshape(-1, self.num_classes)
        mask_flat = obj.view(-1) > 0.5
        if mask_flat.any():
            loss_cls = F.cross_entropy(logits[mask_flat], cls_target.reshape(-1)[mask_flat], reduction="mean")
        else:
            loss_cls = torch.zeros((), device=pred.device, dtype=pred.dtype)

        loss = self.lambda_coord * loss_coord + loss_obj + self.lambda_noobj * loss_noobj + loss_cls
        parts = {
            "loss": loss.detach(),
            "coord": (self.lambda_coord * loss_coord).detach(),
            "obj": loss_obj.detach(),
            "noobj": (self.lambda_noobj * loss_noobj).detach(),
            "cls": loss_cls.detach(),
        }
        return loss, parts
