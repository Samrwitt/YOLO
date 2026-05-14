"""
Tiny YOLOv1-style detector — all PyTorch, sized so you can read it in one sitting.

Original YOLO (Redmon et al., 2016) chops the image into an S×S grid and has each cell spit
out bounding boxes + class scores in a single forward pass. We use **one box per cell** so
the training code stays short; you still get the "dense detection" idea without drowning in
anchor bookkeeping.

LeNet (`lenet.py`) gives you one label for the whole image. This network hands you a small
tensor of predictions per cell — that's the big conceptual jump for detection.

Backbone: a shallow conv stack (think "baby Darknet"). Head: 1×1 conv to 5 + num_classes
channels per cell (center offsets, size, objectness, then class logits).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class TinyYoloV1(nn.Module):
    """
    Stack conv blocks + pooling until the activation map is S×S, then a 1×1 head.

    `decode_predictions` turns the raw head output into cx, cy, w, h in normalized image
    coords plus a confidence score and class logits.
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

        # With 224×224 input: first layer uses stride 2, then four 2×2 pools → 224 / (2×2^4) = 7×7 cells.
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
        Returns (N, 5 + C, S, S) — raw head activations before the loss pulls them apart.
        """
        out = self.net(x)
        if out.shape[2] != self.S or out.shape[3] != self.S:
            raise ValueError(
                f"Expected spatial output {self.S}x{self.S}, got {out.shape[2]}x{out.shape[3]}. "
                "Resize the input image so H and W match the architecture stride (224 for S=7)."
            )
        return out


def _make_grid(S: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    # gx, gy are just 0..S-1 repeated — we add them to sigmoid(tx), sigmoid(ty) so each cell
    # predicts a small offset *inside* its cell instead of the whole image at once.
    gy, gx = torch.meshgrid(
        torch.arange(S, device=device, dtype=dtype),
        torch.arange(S, device=device, dtype=dtype),
        indexing="ij",
    )
    return gx, gy


def decode_predictions(pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    pred: (N, 5+C, S, S) → decoded boxes in **normalized** [0,1] image coordinates.

    The math is the YOLOv1 flavor: sigmoid on center offsets, add cell index, divide by S;
    exp on log-size (clamped so nothing explodes); sigmoid on objectness; class logits stay
    logits for cross-entropy in the loss.
    """
    N, _, S, _ = pred.shape
    pred = pred.permute(0, 2, 3, 1).contiguous()  # (N, S, S, 5+C) — easier to read
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
    """cxcywh in [0,1] → corners xmin, ymin, xmax, ymax (still normalized)."""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)
