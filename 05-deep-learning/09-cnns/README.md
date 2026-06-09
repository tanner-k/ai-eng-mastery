# Convolutional Neural Networks

## Overview

Convolutional neural networks use learned filters that slide across spatial dimensions. They exploit locality and weight sharing: the same small kernel is applied at many positions, making CNNs far more parameter-efficient than fully connected layers for images and other grid-like signals.

CNNs matter in AI engineering because they are still a core tool for vision, audio, medical imaging, robotics, and efficient feature extraction. Even when transformers are used for vision, convolutional ideas such as locality, receptive fields, stride, and translation equivariance remain foundational.

## Math / Derivation

For a 2D convolution with input `X` of shape `(B, Cin, H, W)` and kernels `K` of shape `(Cout, Cin, Kh, Kw)`, the output has shape:

```text
Y[b, co, i, j] =
  sum_ci sum_u sum_v X[b, ci, i*stride_h + u - pad_h, j*stride_w + v - pad_w] * K[co, ci, u, v]
  + bias[co]
```

Output height and width are:

```text
Hout = floor((H + 2*pad_h - dilation_h*(Kh - 1) - 1) / stride_h + 1)
Wout = floor((W + 2*pad_w - dilation_w*(Kw - 1) - 1) / stride_w + 1)
```

The same kernel weights are reused at every spatial location. That weight sharing is what makes convolution parameter-efficient and translation equivariant.

## Intuition

A convolutional filter is a small pattern detector. Early filters might detect edges or color transitions; deeper filters combine lower-level responses into more abstract features. Stacking convolutions increases the receptive field, so later units can depend on larger regions of the input.

Stride reduces spatial resolution. Padding controls whether border information is preserved. Pooling or strided convolutions aggregate local neighborhoods and trade spatial detail for invariance and efficiency.

## When & Why

Use CNNs when nearby positions have related meaning and the same pattern can appear in many locations. Images are the canonical example, but spectrograms, time-series windows, occupancy grids, and some biological sequences also fit.

Key design choices include kernel size, stride, padding, channel width, normalization, activation, pooling, and data augmentation. The engineering tradeoff is often between spatial resolution, receptive field, compute, and memory.

## Implementation

A later implementation pass should implement a small `conv2d` forward operation from tensor indexing or an `im2col` transformation, then compare results to PyTorch. It should compute output shapes, visualize or print receptive-field growth, and train a small CNN on synthetic image-like patterns.

The implementation should describe learner-created files only as future work and should avoid depending on repository datasets.

## Cross-links

- `[[broadcasting-in-nns]]` — convolutional biases and normalization parameters are broadcast over spatial axes.
- `[[activations-tanh-relu]]` — CNN blocks commonly combine convolution, normalization, and ReLU.
- `[[batch-normalization]]` — batch norm is often applied per channel in CNNs.
- `[[matrix-multiplication]]` — convolution can be lowered to matrix multiplication via `im2col`.
- `[[softmax]]` — classifiers often end CNN backbones with logits over classes.

## Resources

- LeCun et al., "Gradient-Based Learning Applied to Document Recognition." Proceedings of the IEEE, 1998.
- Krizhevsky, Sutskever, and Hinton, "ImageNet Classification with Deep Convolutional Neural Networks." NeurIPS 2012.
- PyTorch documentation, `torch.nn.Conv2d`. <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>
