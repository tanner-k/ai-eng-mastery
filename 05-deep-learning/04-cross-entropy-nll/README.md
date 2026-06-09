# Cross-Entropy and NLL Loss

## Overview

Cross-entropy is the standard loss for training classifiers that output logits over mutually exclusive classes. Negative log likelihood (NLL) is the same objective expressed as the negative log probability assigned to the correct class.

For AI engineers, the practical rule is simple: feed raw logits to a numerically stable cross-entropy implementation. Do not manually softmax first. The fused formulation avoids overflow, avoids taking `log(0)`, and gives the clean gradient that makes classifier training efficient.

## Math / Derivation

For class probabilities `p` and one-hot target `y`, cross-entropy is:

```text
H(y, p) = -sum_i y_i log p_i
```

If the correct class is `c`, this reduces to:

```text
L = -log p_c
```

With logits `z`, `p = softmax(z)`, so:

```text
L = -z_c + log(sum_j exp(z_j))
```

The stable version subtracts the maximum logit before the `logsumexp` computation. The gradient with respect to logits is:

```text
dL/dz_i = softmax(z)_i - y_i
```

For a mini-batch, losses are usually averaged:

```text
L_batch = (1/B) sum_b -log p_(b, c_b)
```

## Intuition

Cross-entropy rewards the model for assigning high probability to the correct class and penalizes confident wrong predictions heavily. A correct class probability of `0.9` gives a small loss; `0.01` gives a large loss. The logarithm makes the penalty sensitive to probability ratios rather than raw differences.

The loss is not just about the argmax. Two models can predict the same class but have very different losses if one is well-calibrated and the other is barely above its alternatives.

## When & Why

Use cross-entropy for single-label multiclass classification: image class, next token, intent label, or any mutually exclusive target. Use binary cross-entropy or multi-label losses when labels are independent rather than exclusive.

NLL appears when the model already outputs log probabilities, such as `log_softmax` followed by `NLLLoss`. Cross-entropy combines those steps from logits.

Be careful with class weights, ignored indices, label smoothing, and reduction mode (`sum` versus `mean`) because they change gradient scale.

## Implementation

A later implementation pass should implement stable `logsumexp`, log-softmax, NLL, and cross-entropy from logits. It should compare manual losses and gradients to PyTorch on batches, including class weights and ignored examples if included.

The implementation should include a demonstration that `softmax` followed by `log` is less stable than fused log-softmax, and it should keep all code references framed as learner-created future work.

## Cross-links

- `[[softmax]]` — converts logits to probabilities and gives the `p - y` gradient.
- `[[mle-and-nll]]` — NLL is maximum likelihood expressed as a loss.
- `[[label-smoothing]]` — changes hard one-hot targets into softened targets.
- `[[loss-functions]]` — cross-entropy is one member of the loss-function family.
- `[[evaluation-metrics]]` — loss and accuracy measure different properties.

## Resources

- PyTorch documentation, `torch.nn.CrossEntropyLoss`. <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>
- PyTorch documentation, `torch.nn.NLLLoss`. <https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html>
- Goodfellow, Bengio, and Courville, "Deep Learning", Chapter 6. <https://www.deeplearningbook.org/>
