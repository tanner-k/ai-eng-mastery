# Support Vector Machines — Solutions

## Solution 1 — Compute hinge loss

For margin `m = y f(x)`, hinge loss is `max(0, 1-m)`:

```
m = 2.0  -> 0
m = 1.0  -> 0
m = 0.4  -> 0.6
m = -0.5 -> 1.5
```

## Solution 2 — Relate norm and margin

The distance between the two margin hyperplanes is:

```
2 / ||w|| = 2 / 4 = 0.5
```

## Solution 3 — Interpret C

Very large `C` heavily penalizes violations, so the model tries hard to classify training examples correctly and may use a narrower margin. Very small `C` tolerates more violations, allowing a wider margin and stronger regularization, but it may underfit.

## Solution 4 — Compare logistic regression and SVM

Prefer logistic regression when calibrated probabilities and likelihood-based interpretation are important. Prefer a linear SVM when high-dimensional classification margin is the priority and probability calibration is not required.

## Solution 5 — Explain the kernel trick

A kernel computes the dot product that would result after mapping inputs into a transformed feature space. The SVM can learn a linear separator in that implicit space, which corresponds to a nonlinear boundary in the original input space, without explicitly materializing the transformed features.
