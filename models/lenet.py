"""
LeNet-5 — the classic "small CNN that actually worked" for digits (LeCun et al., 1998).

Back then people still leaned on hand-crafted features a lot. LeNet showed you could stack
convolutions + pooling and let gradients figure out useful patterns for whole-digit
recognition. We use ReLU instead of tanh/sigmoid because training is less finicky, but the
layout (conv → pool → conv → pool → FC) is still the LeNet story.

Compared to something like our tiny YOLO: LeNet ends with one vector of class scores for the
entire image — no grid of boxes, no "where is the thing?" head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    The usual teaching version of LeNet-5, wired for 28×28 grayscale (Fashion-MNIST here).

    Rough picture:
    - conv1 sees local 5×5 patches of the image and makes 6 feature maps (edges, strokes…).
    - pool1 shrinks the map so the model cares a bit less about exact pixel shifts.
    - conv2 / pool2 do the same story at a coarser resolution.
    - the three FC layers squash everything into class logits.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # 1 input channel (grayscale), 6 filters, 5×5 — still the standard LeNet opening
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # 28 → 14 after this pool
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # The original paper had a fancier "sparse" connection table here; the common shortcut
        # is a full 6→16 conv, which is what we do.
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # 14×14 → 10×10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 10 → 5

        # 16 maps × 5×5 spatial = 400 numbers going into the MLP
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Return logits only — CrossEntropyLoss applies softmax internally
        return self.fc3(x)
