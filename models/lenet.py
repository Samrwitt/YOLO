"""
LeNet-5 (LeCun et al., 1998) — gradient-based learning applied to document recognition.

Predecessors / context:
- Earlier work used hand-crafted features; LeNet showed that *convolutional* structure
  (local receptive fields, shared weights, subsampling) could learn hierarchical features
  end-to-end for digit recognition.

Compared to later CNNs (e.g. AlexNet, ResNet):
- Small depth (two conv groups + fully connected head), tanh/sigmoid-era activations
  (here we use ReLU for stable modern training while keeping the *topology* faithful).
- Spatial sizes shrink via pooling; modern nets often use strided convolutions instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    Classic LeNet-5 topology adapted for 1×28×28 (MNIST / Fashion-MNIST) input.

    Layer semantics:
    - C1: convolution extracts local edge/stroke patterns; weight sharing cuts parameters
      vs. a fully connected layer on the full image.
    - S2: subsampling (avg pool) reduces spatial resolution and adds slight translation
      tolerance.
    - C3, S4: repeat at a higher abstraction level.
    - F5, OUTPUT: flatten + MLP maps global representation to class logits.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # First conv block: 1 channel (grayscale) -> 6 feature maps, 5×5 kernels.
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # 28×28 -> pool 2×2 -> 14×14
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Second conv: 6 -> 16 maps (classic LeNet uses a specific connection table;
        # full 6→16 conv is the common simplified variant used in teaching).
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # 14→10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 10→5

        # Fully connected layers operate on flattened spatial features.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, 28, 28)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # No softmax here: CrossEntropyLoss expects raw logits.
        return self.fc3(x)
