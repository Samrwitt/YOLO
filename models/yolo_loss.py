"""
Loss for our tiny YOLOv1-style head — three pieces that get added together.

1) Box term: only on cells where we actually placed a ground-truth object (the "responsible"
   cell for that object). Smooth L1 on decoded cx, cy, w, h vs the target.
2) Objectness: should the network think there's an object here? BCE-with-logits on the raw
   score — 1 on object cells, 0 elsewhere, but empty cells are scaled down by lambda_noobj
   so they don't overwhelm the loss.
3) Classes: plain cross-entropy on the logits, again only on cells that have an object.

We use `binary_cross_entropy_with_logits` on objectness so we never apply sigmoid twice.
Box/class terms use `decode_predictions` so we're comparing apples to apples in image space.
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
        pred — raw network output (N, 5+C, S, S).
        gt_box — (N, S, S, 4) with cx,cy,w,h in [0,1]; zeros where there's nothing to predict.
        obj_mask — 1 where a GT center landed in that cell, else 0.
        cls_target — class index per cell; values in empty cells are ignored.
        """
        cx, cy, w, h, conf, tcls = decode_predictions(pred)
        obj = obj_mask.float()
        noobj = 1.0 - obj

        pred_box = torch.stack([cx, cy, w, h], dim=-1)
        coord_err = F.smooth_l1_loss(pred_box, gt_box, reduction="none").sum(dim=-1)
        loss_coord = (obj * coord_err).sum() / obj.sum().clamp(min=1.0)

        # Channel 4 is objectness *before* sigmoid — feed that straight into BCEWithLogits
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
