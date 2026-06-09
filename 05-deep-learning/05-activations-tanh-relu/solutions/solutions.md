# Activations: tanh and ReLU — Solutions

## Solution 1 — Derive the tanh derivative

Using the quotient rule:

```text
d/dx tanh(x) = d/dx [sinh(x) / cosh(x)]
             = (cosh(x)^2 - sinh(x)^2) / cosh(x)^2
             = 1 / cosh(x)^2
```

Since `1 - tanh(x)^2 = (cosh(x)^2 - sinh(x)^2) / cosh(x)^2`, the derivative is `1 - tanh(x)^2`.

## Solution 2 — Compute activation outputs and derivatives

ReLU outputs are `[0, 0, 0, 1, 3]` if the subgradient at zero is chosen as zero. Derivatives are `[0, 0, 0, 1, 1]` with that convention. Tanh derivatives are largest near input `0` and smallest for large positive or negative inputs where tanh saturates.

## Solution 3 — Explain why linear layers need nonlinear activations

Two linear layers give:

```text
y = (xW1 + b1)W2 + b2
  = x(W1W2) + (b1W2 + b2)
```

This is a single affine transformation with weight `W1W2` and bias `b1W2 + b2`.

## Solution 4 — Diagnose dead ReLUs

Symptoms include near-zero activations, zero gradients for affected unit weights, and little change in those parameters. Remedies include reducing learning rate, using better initialization, adding normalization, or replacing ReLU with leaky ReLU/PReLU.

## Solution 5 — Match activation and initialization

He initialization scales weights based on fan-in so variance is preserved through ReLU layers, accounting for roughly half the activations being zeroed. It prevents activations and gradients from shrinking or growing rapidly across depth, which naive small random initialization can cause.
