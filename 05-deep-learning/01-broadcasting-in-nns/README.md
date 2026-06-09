# Broadcasting in Neural Networks

## Overview

Broadcasting is the set of tensor shape rules that lets neural-network code apply smaller tensors across larger tensors without explicitly copying data. It is why a bias vector of shape `(features,)` can be added to a mini-batch activation matrix of shape `(batch, features)`, why per-channel normalization parameters can be applied to image tensors, and why attention masks can be expanded across heads and query positions.

For AI engineers, broadcasting is not just syntactic convenience. It controls parameter sharing, memory use, gradient aggregation, and the difference between a correct model and a silent shape bug. Most production deep-learning failures involving tensors are not calculus mistakes; they are mistaken assumptions about which dimensions are being aligned.

## Math / Derivation

Two dimensions are broadcast-compatible when they are equal or one of them is `1`. Alignment starts from the trailing dimensions. For example:

```text
X shape:      (B, D)
b shape:          (D)
result shape: (B, D)
```

The bias is conceptually expanded to `(B, D)`, but frameworks usually implement this as a view with stride `0` along the broadcasted axis.

For affine layers:

```text
Z = XW + b
X in R^(B x Din)
W in R^(Din x Dout)
b in R^(Dout)
Z in R^(B x Dout)
```

The forward pass broadcasts `b` over the batch axis. During backpropagation, the gradient for a broadcasted tensor must sum over every axis that was expanded:

```text
dL/db = sum over batch of dL/dZ
```

More generally, if a tensor with shape `(1, D)` is broadcast to `(B, D)`, the reverse-mode gradient reduces the upstream gradient from `(B, D)` back to `(1, D)` by summing over axis `0`.

## Intuition

Broadcasting says, "use the same value everywhere along this axis." A scalar learning-rate multiplier applies to every parameter. A bias vector applies the same feature offset to every example in the batch. A per-channel scale applies the same channel multiplier at every pixel.

The key intuition is that broadcasting in the forward pass creates sharing, and sharing in the forward pass creates summation in the backward pass. If one parameter contributes to `B` different outputs, its gradient receives contributions from all `B` outputs.

## When & Why

Broadcasting shows up in almost every neural-network primitive:

- Adding bias after matrix multiplication.
- Applying layer, batch, or channel-wise normalization parameters.
- Creating masks for attention and sequence losses.
- Scaling logits by temperature or per-class weights.
- Mixing tensors with singleton dimensions such as `(B, 1)`, `(1, D)`, or `(B, C, 1, 1)`.

Use broadcasting when the semantics are "same parameter or value reused along an axis." Be suspicious when it happens accidentally, especially when a target tensor has shape `(B,)` and a prediction tensor has shape `(B, 1)`: the result may expand to `(B, B)` instead of comparing each example once.

## Implementation

A later implementation pass should build small PyTorch examples that manually reproduce broadcasting behavior for affine bias addition, per-channel image scaling, and reduction of gradients back to original parameter shapes. It should validate manual gradients against `torch.autograd` for cases such as `(B, D) + (D,)`, `(B, C, H, W) * (C, 1, 1)`, and a deliberately incorrect `(B, 1) - (B,)` loss shape.

The implementation should emphasize shape assertions, explicit `unsqueeze` calls where they clarify intent, and checks that expanded views do not allocate full repeated tensors.

## Cross-links

- `[[broadcasting]]` — the general tensor rule that neural-network layers rely on.
- `[[matrix-multiplication]]` — affine layers combine matrix products with broadcasted bias terms.
- `[[backpropagation]]` — broadcasted forward operations require summed gradients.
- `[[batch-normalization]]` — per-feature and per-channel scale/shift parameters are broadcast across batches and spatial axes.
- `[[cnns]]` — convolutional feature maps commonly use `(B, C, H, W)` broadcasting patterns.

## Resources

- NumPy documentation, "Broadcasting." <https://numpy.org/doc/stable/user/basics.broadcasting.html>
- PyTorch documentation, "Broadcasting semantics." <https://pytorch.org/docs/stable/notes/broadcasting.html>
- PyTorch documentation, `torch.Tensor.expand`. <https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html>
