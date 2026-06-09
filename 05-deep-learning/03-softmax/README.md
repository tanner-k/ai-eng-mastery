# Softmax

## Overview

Softmax converts a vector of real-valued logits into a categorical probability distribution. It is the standard output transform for multiclass classifiers, attention weights, and any model component that needs nonnegative values summing to one.

In AI engineering, softmax is important because it separates scores from probabilities. Neural networks usually produce logits because logits are unconstrained and easy to optimize. Softmax imposes the probability simplex only when needed for interpretation, sampling, attention, or a likelihood calculation.

## Math / Derivation

For logits `z in R^K`, softmax is:

```text
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

It is invariant to adding a constant to every logit:

```text
softmax(z) = softmax(z - c)
```

This gives the numerically stable form:

```text
p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
```

The Jacobian is:

```text
d p_i / d z_j = p_i * (1[i = j] - p_j)
```

For cross-entropy with one-hot target `y`, this simplifies to:

```text
dL/dz = p - y
```

That simplification is why practical frameworks combine softmax and cross-entropy into one stable operation.

## Intuition

Softmax is a smooth winner-take-more function. Increasing one logit raises that class probability and lowers all others because the probabilities must sum to one. The exponential makes differences matter: a logit gap of 2 means an odds ratio of about `exp(2)`, not a linear advantage of 2.

Temperature controls sharpness:

```text
softmax(z / T)
```

Low `T` makes the distribution sharper; high `T` makes it flatter.

## When & Why

Use softmax when outputs are mutually exclusive classes or normalized weights. Common cases include image classification, token prediction, attention, mixture weights, and policy distributions.

Do not apply softmax before a framework cross-entropy loss that expects logits. Passing probabilities instead of logits loses numerical stability and changes gradients. Also avoid interpreting softmax probabilities as calibrated confidence unless calibration has been measured.

## Implementation

A later implementation pass should implement stable softmax, softmax Jacobian-vector products, temperature scaling, and the combined softmax-cross-entropy gradient on synthetic logits. It should compare manual outputs and gradients to PyTorch operations and include examples showing overflow in the naive `exp(z)` formula.

The implementation should treat logits as the primary model output and only materialize probabilities when required for analysis.

## Cross-links

- `[[cross-entropy-nll]]` — cross-entropy consumes logits through a stable log-softmax.
- `[[label-smoothing]]` — modifies one-hot targets used with softmax classifiers.
- `[[backpropagation]]` — softmax derivatives are propagated backward through logits.
- `[[logistic-regression]]` — binary logistic regression is the two-class analogue.
- `[[evaluation-metrics]]` — predicted probabilities and top-k predictions feed classification metrics.

## Resources

- Goodfellow, Bengio, and Courville, "Deep Learning", Chapter 6. <https://www.deeplearningbook.org/>
- PyTorch documentation, `torch.nn.functional.softmax`. <https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html>
- PyTorch documentation, `torch.nn.CrossEntropyLoss`. <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>
