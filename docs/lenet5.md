# LeNet-5 (1998) — Background, Motivation, and Tradeoffs

## Why LeNet-5 mattered

Before LeNet-style CNNs, many vision systems for handwriting/document recognition relied on **hand-engineered features** (edges, strokes, templates) plus shallow classifiers. LeNet-5 (LeCun et al., 1998) helped establish that a model with:

- **local receptive fields** (convolutions),
- **shared weights** (translation-equivariant feature detectors),
- **subsampling/pooling** (reduced resolution + partial translation tolerance),

could learn **hierarchical visual features end-to-end** from raw pixels and outperform feature engineering on digit recognition–style tasks.

## Predecessors and related ideas

LeNet did not appear in isolation. A few key predecessors/ideas that motivated it:

- **Neocognitron (Fukushima, 1980)**: introduced a layered, convolution-like hierarchy for pattern recognition with alternating “simple/complex” cells. It inspired the *concept* of hierarchical local feature extraction, but LeNet’s training was more strongly connected to gradient descent and modern backprop.
- **Weight sharing / time-delay neural networks (TDNNs)**: earlier work explored sharing parameters across positions to detect patterns regardless of location—one of the central ideas that convolution operationalizes.
- **Backpropagation-based neural networks**: LeNet’s key contribution was showing that a convolutional hierarchy could be trained effectively (and practically) with gradient-based methods on real tasks.

## Architecture (high level)

LeNet-5 is an early “small depth” CNN, originally designed for digit classification. The canonical pattern is:

- **Conv → Pool → Conv → Pool → Fully connected → Output**

In this repo, the implementation is adapted to **1×28×28** inputs (MNIST/Fashion-MNIST) and uses **ReLU** for stable modern training while keeping the **topology** faithful.

## Architectural innovations (for its time)

- **Convolutional feature extraction**: fewer parameters than fully connected layers on images, and better inductive bias for translation.
- **Pooling/subsampling**: reduces spatial resolution and sensitivity to small shifts.
- **End-to-end training**: feature learning and classifier training happen together.

## Strengths

- **Data efficiency**: fewer parameters than an MLP over pixels; strong inductive bias.
- **Fast inference**: shallow network with small compute footprint.
- **Interpretable building blocks**: easy to explain and visualize early layers (edges, strokes).

## Weaknesses and limitations

- **Limited capacity**: shallow depth and small channel counts can underfit harder datasets.
- **No explicit regularization tricks** from later eras (dropout, batch norm in the original).
- **Not scale-friendly** to high-resolution, large-class natural-image recognition without major changes.

## Computational tradeoffs

- **Parameter count vs fully connected**: convolutions are vastly cheaper than connecting every pixel to every hidden unit.
- **Compute vs accuracy**: LeNet is lightweight and fast, but later architectures (AlexNet, VGG, ResNet) scale depth/width to reach much higher accuracy on complex datasets at higher compute cost.

## What you should emphasize in your presentation

- **Motivation**: remove feature engineering; learn features from data.
- **Core idea**: local connectivity + weight sharing + pooling yields a strong vision prior.
- **Why it was influential**: established a practical recipe for CNNs long before modern large-scale datasets/GPUs.

## Where this appears in the repo

- Implementation: `models/lenet.py` (`LeNet5`).
- Train / evaluate / plots: `notebooks/lenet_train_eval.ipynb` (Fashion-MNIST).
- Written results template: `docs/assignment_findings.md`.
