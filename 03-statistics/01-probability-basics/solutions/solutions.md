# Probability Basics — Solutions

## Solution 1 — Compute conditional probability

Use total probability:

```text
P(violation) = P(violation | flagged)P(flagged)
             + P(violation | unflagged)P(unflagged)
             = 0.30(0.04) + 0.01(0.96)
             = 0.012 + 0.0096
             = 0.0216
```

The overall violation probability is `2.16%`.

## Solution 2 — Use Bayes' rule

Let `D` be defective and `+` be detector fires.

```text
P(D | +) = P(+ | D)P(D) / P(+)
P(+) = P(+ | D)P(D) + P(+ | not D)P(not D)
     = 0.95(0.02) + 0.10(0.98)
     = 0.117
```

So:

```text
P(D | +) = 0.95(0.02) / 0.117 = 0.019 / 0.117 approx 0.162
```

Only about `16.2%` of fired detections are true defects.

## Solution 3 — Expectation and variance of a Bernoulli variable

For `X` taking value `1` with probability `p` and `0` otherwise:

```text
E[X] = 1*p + 0*(1 - p) = p
E[X^2] = 1^2*p + 0^2*(1 - p) = p
Var(X) = E[X^2] - E[X]^2 = p - p^2 = p(1 - p)
```

## Solution 4 — Diagnose independence

If independent, `P(A and B)` would equal:

```text
P(A)P(B) = 0.4 * 0.5 = 0.2
```

The actual joint probability is `0.25`, so the events are not independent.

```text
P(A | B) = P(A and B) / P(B) = 0.25 / 0.5 = 0.5
```

## Solution 5 — Interpret metric uncertainty

The observed difference is `15` examples out of `1000`, or `1.5` percentage points. Because validation accuracy is an estimate, this gap could be sampling noise, especially if the models are evaluated on different examples or the examples are not representative.

A good next check is a paired comparison on the same validation examples, such as a bootstrap interval or McNemar-style analysis of disagreements. Also inspect segment-level performance to ensure the lift is not concentrated in an unimportant or unstable slice.
