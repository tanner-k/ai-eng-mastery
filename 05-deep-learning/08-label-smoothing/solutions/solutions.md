# Label Smoothing — Solutions

## Solution 1 — Construct a smoothed target

`epsilon / K = 0.1 / 5 = 0.02`. The correct class receives `1 - 0.1 + 0.02 = 0.92`. The target is:

```text
[0.02, 0.02, 0.92, 0.02, 0.02]
```

## Solution 2 — Compute soft-target cross-entropy

```text
L = -0.9 log(0.8) - 0.05 log(0.1) - 0.05 log(0.1)
  ~= 0.9 * 0.223 + 0.1 * 2.303
  ~= 0.431
```

## Solution 3 — Derive the gradient

For `L = -sum_i q_i log softmax(z)_i`, the derivative with respect to logit `z_j` is:

```text
dL/dz_j = p_j - q_j
```

where `p = softmax(z)`. This is the same form as hard-label cross-entropy with `q` replacing one-hot `y`.

## Solution 4 — Analyze confidence

The target no longer asks the model to assign probability `1` to the correct class, so the gradient stops pushing toward infinite logit gaps. This tends to reduce maximum softmax probability. It does not guarantee calibration because calibration depends on the relationship between predicted probabilities and empirical correctness, which must be measured.

## Solution 5 — Decide whether to use smoothing

1. Noisy image labels: likely helpful because it reduces overconfidence in imperfect labels.
2. Knowledge distillation: needs care or may be harmful because teacher targets are already informative soft distributions.
3. Large-vocabulary next-token prediction: often helpful, but tune epsilon and monitor NLL.
4. Medical diagnosis with probability outputs: needs care because smoothing can distort probability estimates; calibration metrics are essential.
