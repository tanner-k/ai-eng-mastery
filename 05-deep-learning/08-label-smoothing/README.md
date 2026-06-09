# Label Smoothing

## Overview

Label smoothing replaces hard one-hot classification targets with slightly softened target distributions. Instead of assigning probability `1` to the correct class and `0` to all others, it assigns most mass to the correct class and spreads a small amount across the remaining classes.

This matters because modern neural networks can become overconfident. Label smoothing discourages infinite logit gaps, can improve generalization, and often improves calibration-like behavior, though it can also hurt tasks where exact confidence estimates or distillation targets are important.

## Math / Derivation

For `K` classes and smoothing parameter `epsilon`, a common convention is:

```text
y_smooth = (1 - epsilon) * one_hot + epsilon / K
```

So the correct class target becomes:

```text
1 - epsilon + epsilon / K
```

and every class, including incorrect classes, receives `epsilon / K`.

Cross-entropy with a soft target distribution `q` is:

```text
L = -sum_i q_i log p_i
```

With logits and softmax probabilities `p`, the logit gradient becomes:

```text
dL/dz = p - q
```

The model is no longer pushed toward probability `1` for the labeled class.

## Intuition

Hard labels say, "this class is certainly correct and every other class is impossible." Label smoothing says, "this class is correct, but do not become infinitely certain." It creates a small penalty for putting all probability mass on one class, which reduces overconfident logit gaps.

The target is still centered on the labeled class. Smoothing does not make the label ambiguous; it changes the training signal to be less extreme.

## When & Why

Label smoothing is common in image classification, sequence-to-sequence models, and large-vocabulary classifiers. It can improve validation accuracy and reduce overconfidence, especially when labels are noisy or class boundaries are not perfectly clean.

Use caution when:

- Training data has already-soft targets.
- You need probability estimates that preserve true empirical frequencies.
- You are doing knowledge distillation, where teacher probabilities carry information.
- The task has rare classes where smoothing may weaken an already sparse signal.

## Implementation

A later implementation pass should implement smoothed target construction and soft-target cross-entropy from logits. It should compare hard-label and smoothed-label gradients on synthetic logits and train a small classifier with different smoothing values.

The implementation should report accuracy, NLL, and confidence metrics so learners can see that smoothing changes more than top-1 accuracy.

## Cross-links

- `[[cross-entropy-nll]]` — label smoothing modifies the target distribution used by cross-entropy.
- `[[softmax]]` — smoothing changes the `p - y` logit gradient to `p - q`.
- `[[evaluation-metrics]]` — accuracy, NLL, and calibration can move differently.
- `[[loss-functions]]` — smoothing is a loss-target modification.
- `[[bias-variance]]` — smoothing acts like a regularizer that can reduce variance from hard labels.

## Resources

- Szegedy et al., "Rethinking the Inception Architecture for Computer Vision." CVPR 2016. <https://arxiv.org/abs/1512.00567>
- Muller, Kornblith, and Hinton, "When Does Label Smoothing Help?" NeurIPS 2019. <https://arxiv.org/abs/1906.02629>
- PyTorch documentation, `CrossEntropyLoss` label smoothing parameter. <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>
