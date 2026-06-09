# Derivatives and Partials

## Overview

Derivatives measure local change. In AI engineering, that local change becomes the signal used to fit models: how much a loss changes when a weight, bias, activation, logit, or input feature changes. A derivative turns "this parameter is wrong" into "move this parameter in this direction by this amount."

Partial derivatives extend the same idea to multivariable functions. A neural-network loss usually depends on millions or billions of parameters, but each partial derivative isolates one coordinate while holding the others fixed. Backpropagation then organizes those partials efficiently instead of computing them one by one.

This topic is the entry point for [[chain-rule]], [[gradients-and-jacobians]], [[gradient-descent]], and [[backpropagation]]. If derivatives are unclear, optimizer behavior looks like a black box.

## Math / Derivation

For a scalar function `f: R -> R`, the derivative at `x` is the limit

```text
f'(x) = lim_{h -> 0} (f(x + h) - f(x)) / h
```

when that limit exists. It is the slope of the best local linear approximation:

```text
f(x + h) = f(x) + f'(x) h + o(h)
```

For a multivariable scalar function `f: R^d -> R`, the partial derivative with respect to coordinate `x_j` is

```text
partial f / partial x_j
  = lim_{h -> 0} (f(x_1, ..., x_j + h, ..., x_d) - f(x_1, ..., x_j, ..., x_d)) / h
```

The gradient collects all partial derivatives:

```text
grad f(x) = [partial f / partial x_1, ..., partial f / partial x_d]^T
```

Example: for squared error on one example,

```text
L(w, b) = (w^T x + b - y)^2
r = w^T x + b - y
```

then

```text
partial L / partial w_j = 2 r x_j
partial L / partial b   = 2 r
```

The partial with respect to `w_j` depends on the residual `r` and the feature value `x_j`. If `x_j = 0`, changing that coordinate cannot affect the prediction for this example.

## Intuition

A derivative answers a local "what if" question: if I nudge this value slightly, what happens immediately? It is not a global forecast. A positive derivative means increasing the input increases the output nearby; a negative derivative means increasing the input decreases the output nearby; a large magnitude means the function is sensitive at that point.

Partial derivatives are directional probes aligned with coordinate axes. You freeze every other input, nudge one coordinate, and observe the local response. In model training, each parameter receives exactly this kind of local responsibility signal.

Finite differences approximate derivatives by using a small but nonzero `h`. Autograd computes exact derivatives of the represented computation graph, up to floating-point error. Comparing the two is a practical way to catch shape mistakes, missing constants, and sign errors.

## When & Why

Use derivatives and partials when you need to reason about sensitivity, optimization, or attribution:

- Training: gradients of loss with respect to parameters drive [[gradient-descent]].
- Debugging: near-zero derivatives can explain stalled learning; very large derivatives can explain exploding updates.
- Input sensitivity: derivatives with respect to inputs show which features or pixels locally influence a prediction.
- Numerical checks: finite-difference derivatives validate manual formulas and custom autograd code.

The main limitation is locality. A derivative can tell you the best infinitesimal direction at the current point, but it does not guarantee that a large step will help or that the function is well behaved far away.

## Implementation

A later implementation pass should build a small PyTorch exercise that defines scalar and vector-valued toy functions, computes manual derivatives, and checks them against `torch.autograd`. It should also include a finite-difference checker that varies the step size `h` to show the tradeoff between truncation error and floating-point cancellation.

The implementation should validate:

1. Manual derivatives for simple scalar functions.
2. Manual partial derivatives for a squared-error linear model.
3. Agreement between finite differences and autograd over a useful range of `h`.
4. Failure cases where nondifferentiability or numerical precision makes derivative checks unreliable.

## Cross-links

- `[[chain-rule]]` — combines local derivatives through composed functions.
- `[[gradients-and-jacobians]]` — organizes partial derivatives for scalar and vector outputs.
- `[[gradient-descent]]` — uses derivatives to choose parameter updates.
- `[[backpropagation]]` — computes many derivatives efficiently through a computation graph.

## Resources

- Gilbert Strang, "Calculus." MIT OpenCourseWare. <https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/>
- Michael Spivak, "Calculus." Publish or Perish, 4th edition.
- Baydin et al., "Automatic Differentiation in Machine Learning: a Survey." JMLR 2018. <https://jmlr.org/papers/v18/17-468.html>
