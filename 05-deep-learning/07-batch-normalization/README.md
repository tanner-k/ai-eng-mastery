# Batch Normalization

## Overview

Batch normalization normalizes intermediate activations using mini-batch statistics, then applies learned scale and shift parameters. It often makes deep networks train faster and tolerate larger learning rates by stabilizing the distribution of inputs seen by later layers.

For AI engineers, batch norm is both a layer and an operating mode. It behaves differently during training and inference, depends on batch statistics, and can fail when batch sizes are too small or distribution shifts are large.

## Math / Derivation

For a feature `x` over a mini-batch:

```text
mu_B = (1/m) sum_i x_i
var_B = (1/m) sum_i (x_i - mu_B)^2
xhat_i = (x_i - mu_B) / sqrt(var_B + eps)
y_i = gamma xhat_i + beta
```

`gamma` and `beta` are learned parameters. For dense layers they usually have shape `(features,)`. For convolutional layers they usually have shape `(channels,)` and are broadcast across batch and spatial dimensions.

During inference, batch norm uses running estimates accumulated during training:

```text
running_mean <- momentum * running_mean + (1 - momentum) * batch_mean
running_var  <- momentum * running_var  + (1 - momentum) * batch_var
```

Exact momentum conventions vary by framework, so read the API carefully.

## Intuition

Batch norm keeps each feature's batch distribution near zero mean and unit variance, then lets the model learn the scale and offset it actually wants. The normalization makes downstream layers see a more stable input scale. The learned `gamma` and `beta` prevent normalization from permanently restricting what the network can represent.

Batch norm also adds noise because batch statistics vary across mini-batches. This can act as regularization, but it can be harmful with tiny or nonrepresentative batches.

## When & Why

Batch norm is common in CNNs and older feed-forward architectures. It is less common inside modern transformers, where layer normalization is typically preferred because it does not depend on batch statistics.

Use caution when:

- Batch size is very small.
- Training and inference distributions differ.
- Evaluation mode is not set correctly.
- Distributed training uses per-device batches without synchronized statistics.
- Fine-tuning with frozen or stale running statistics.

## Implementation

A later implementation pass should implement batch norm forward and backward for 2D dense activations and 4D convolutional activations. It should verify manual gradients against PyTorch, track running statistics across training steps, and demonstrate the difference between training and evaluation behavior.

The implementation should make broadcasting of `gamma` and `beta` explicit and should not claim topic-local implementation files already exist.

## Cross-links

- `[[broadcasting-in-nns]]` — `gamma` and `beta` are broadcast over batch and spatial axes.
- `[[vanishing-exploding-gradients]]` — normalization can stabilize signal scale.
- `[[activations-tanh-relu]]` — batch norm changes the distribution entering nonlinearities.
- `[[cnns]]` — batch norm is widely used after convolutional layers.
- `[[bias-variance]]` — mini-batch statistics introduce estimation noise.

## Resources

- Ioffe and Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." ICML 2015. <https://arxiv.org/abs/1502.03167>
- PyTorch documentation, `torch.nn.BatchNorm1d`. <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html>
- PyTorch documentation, `torch.nn.BatchNorm2d`. <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>
