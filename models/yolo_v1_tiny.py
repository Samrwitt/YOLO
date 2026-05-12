"""
Tiny YOLOv1-style detector in pure PyTorch (coursework-sized).

YOLOv1 (Redmon et al., 2016) divides the image into an S×S grid; each cell predicts
B bounding boxes plus class scores. This file uses **one box per cell** (B=1) to keep
the loss and target assignment readable, while preserving the core idea: **dense
prediction in one forward pass** (no region-proposal stage like R-CNN).

Compared to LeNet (`lenet.py`):
- LeNet maps the whole image to a **single** class vector.
- YOLO maps a spatial grid to **many** (box + class) vectors at once, suited to
  multiple objects and localization.

The backbone is a small Darknet-like stack (conv + maxpool) ending at S×S; a 1×1 head
emits (5 + num_classes) channels per cell: offsets for center within the cell, size,
objectness, and class logits.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyYoloV1(nn.Module):
    """
    Backbone: repeated conv → maxpool until spatial size is `grid`×`grid`.

    Head: 1×1 convolution mapping C channels to `5 + num_classes` outputs per cell.
    Raw outputs are interpreted by `decode_predictions` (sigmoid / exp + grid).
    """

    def __init__(self, num_classes: int = 20, grid_size: int = 7, in_ch: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.S = grid_size
        self.out_ch = 5 + num_classes

        def conv_block(cin: int, cout: int, k: int = 3, s: int = 1, p: int | None = None) -> nn.Sequential:
            if p is None:
                p = k // 2
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True),
            )

        # Input is fixed square (default 224). Stride to grid: 224 / 7 = 32 (four pools of 2).
        self.net = nn.Sequential(
            conv_block(in_ch, 16, k=7, s=2, p=3),
            nn.MaxPool2d(2),
            conv_block(16, 32),
            nn.MaxPool2d(2),
            conv_block(32, 64),
            nn.MaxPool2d(2),
            conv_block(64, 128),
            nn.MaxPool2d(2),
            conv_block(128, 256),
            conv_block(256, 256),
            nn.Conv2d(256, self.out_ch, kernel_size=1, stride=1, padding=0),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns tensor of shape (N, 5 + C, S, S) — raw logits / pre-activations for head.
        """
        out = self.net(x)
        if out.shape[2] != self.S or out.shape[3] != self.S:
            raise ValueError(
                f"Expected spatial output {self.S}x{self.S}, got {out.shape[2]}x{out.shape[3]}. "
                "Resize the input image so H and W match the architecture stride (224 for S=7)."
            )
        return out


def _make_grid(S: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    # Cell indices: gy along height (rows), gx along width (cols), matching (i,j) in image space.
    gy, gx = torch.meshgrid(
        torch.arange(S, device=device, dtype=dtype),
        torch.arange(S, device=device, dtype=dtype),
        indexing="ij",
    )
    return gx, gy


def decode_predictions(pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    pred: (N, 5+C, S, S) -> per-cell decoded quantities in **normalized image coordinates** [0,1].

    Decoding (YOLOv1 spirit):
    - Center (cx, cy) = (sigmoid(tx) + gx) / S, (sigmoid(ty) + gy) / S with gx,gy cell indices.
    - Width/height as fractions of image: exp(tw)/S, exp(th)/S (stabilized with clamp).
    - Objectness: sigmoid(tobj).
    - Classes: raw logits for softmax/CE in the loss.
    """
    N, _, S, _ = pred.shape
    pred = pred.permute(0, 2, 3, 1).contiguous()  # (N, S, S, 5+C)
    tx, ty, tw, th, tobj = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3], pred[..., 4]
    tcls = pred[..., 5:]

    gx, gy = _make_grid(S, pred.device, pred.dtype)
    gx = gx.view(1, S, S).expand(N, -1, -1)
    gy = gy.view(1, S, S).expand(N, -1, -1)

    cx = (torch.sigmoid(tx) + gx) / S
    cy = (torch.sigmoid(ty) + gy) / S
    w = torch.exp(torch.clamp(tw, max=8.0)) / S
    h = torch.exp(torch.clamp(th, max=8.0)) / S
    conf = torch.sigmoid(tobj)
    return cx, cy, w, h, conf, tcls


def boxes_cxcywh_to_xyxy(cx: torch.Tensor, cy: torch.Tensor, w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """All in normalized [0,1] coords; returns (..., 4) as xmin, ymin, xmax, ymax."""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)
