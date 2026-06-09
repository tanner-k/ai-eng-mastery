# Adam Optimization — Solutions

## Solution 1 — Compute the first Adam step

```
m_1 = 0.9 * 0 + 0.1 * 2 = 0.2
v_1 = 0.999 * 0 + 0.001 * 2^2 = 0.004
mhat_1 = 0.2 / (1 - 0.9) = 2
vhat_1 = 0.004 / (1 - 0.999) = 4
```

The update amount is:

```
0.001 * 2 / (sqrt(4) + 1e-8) ~= 0.001
```

The parameter moves by `-0.001` for a positive gradient.

## Solution 2 — Explain bias correction

For constant `g`, the recurrence gives:

```
m_t = (1 - beta1^t) g
```

because the EMA has only accumulated `t` terms and started at zero. Since `1 - beta1^t < 1` at finite early `t`, `m_t` is smaller than `g`. Dividing by `1 - beta1^t` removes that bias.

## Solution 3 — Compare Adam and AdamW

With L2-in-loss, the penalty adds `2 lambda w` to the gradient, and Adam divides that combined gradient by `sqrt(vhat)`. The regularization strength therefore depends on the adaptive second moment for each coordinate. AdamW applies decay directly to the parameter, so the shrinkage does not depend on gradient history.

## Solution 4 — Diagnose unstable Adam training

Likely causes include too-large learning rate, insufficient warmup, bad initialization, gradient spikes, unstable mixed precision, or an overly small batch. First changes: lower the learning rate, add or lengthen warmup, enable gradient clipping, and inspect gradient norms/loss scale.

## Solution 5 — Reason about beta values

Reducing `beta2` to `0.9` makes the squared-gradient estimate respond much faster to recent changes. This can help in nonstationary regimes, but it makes the denominator noisier, so effective learning rates fluctuate more. Training may become less stable unless the learning rate is reduced.
