# Derivatives and Partials — Solutions

## Solution 1 — Derive a scalar derivative from the limit

```text
f(x + h) = 3(x + h)^2 - 2(x + h) + 5
         = 3x^2 + 6xh + 3h^2 - 2x - 2h + 5

f(x + h) - f(x) = 6xh + 3h^2 - 2h
```

Divide by `h`:

```text
(f(x + h) - f(x)) / h = 6x + 3h - 2
```

Taking `h -> 0` gives:

```text
f'(x) = 6x - 2
```

## Solution 2 — Compute partial derivatives of a model loss

Let `r = w1 x1 + w2 x2 + b - y`, so `L = r^2`. By the chain rule:

```text
partial L / partial w1 = 2r x1
partial L / partial w2 = 2r x2
partial L / partial b  = 2r
```

At the given values:

```text
r = 0.5(2) + (-2)(-1) + 1 - 3 = 1
```

Therefore:

```text
partial L / partial w1 = 2(1)(2)  = 4
partial L / partial w2 = 2(1)(-1) = -2
partial L / partial b  = 2(1)     = 2
```

## Solution 3 — Diagnose a finite-difference check

`h = 1e-1` can be too large because the approximation is no longer very local; curvature terms contaminate the estimate. `h = 1e-12` can be too small because `f(x + h)` and `f(x - h)` become nearly equal in floating-point arithmetic, so subtraction loses significant digits.

In float64, a reasonable first sweep is often around `1e-3` to `1e-6`, with centered differences commonly working well near `1e-5` or `1e-6` for smooth functions of ordinary scale. The best value is empirical and depends on function scale.

## Solution 4 — Identify nondifferentiable points

1. `abs(x)` is not differentiable at `x = 0` because the left derivative is `-1` and the right derivative is `1`.
2. `max(0, x)` is not differentiable at `x = 0` because the left derivative is `0` and the right derivative is `1`.
3. `x^2 + abs(x - 1)` is not differentiable at `x = 1` because `abs(x - 1)` has a kink there. The smooth `x^2` term does not remove the kink.

ReLU has the same kink as `max(0, x)`. Deep-learning frameworks choose a subgradient at zero, but units can still become inactive over regions where the ReLU input is negative and the gradient is zero.

## Solution 5 — Interpret partial derivatives in context

`partial M / partial dropout = -0.8` means that, locally, increasing dropout on its raw scale is expected to decrease the metric. If larger `M` is better, this suggests reducing dropout may help nearby.

`partial M / partial lr = 12.0` means that, locally, increasing the learning rate on its raw scale is expected to increase the metric. The magnitudes are risky to compare directly because dropout and learning rate have different units, ranges, and nonlinear effects. A change of `0.01` in dropout is not comparable to a change of `0.01` in learning rate.
