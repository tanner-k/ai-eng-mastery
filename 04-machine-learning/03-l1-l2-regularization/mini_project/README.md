# L1 and L2 Regularization — Mini-Project: Sparse Signal Recovery

## Goal

Train regularized linear models on synthetic high-dimensional data and compare coefficient recovery, sparsity, and validation error.

## Dataset

Generate synthetic data in the future script:

- 1,000 examples and 100 features
- only 8 true nonzero coefficients
- Gaussian feature matrix with standardized columns
- target `y = Xw + noise`

## Implementation tasks

Create `mini_project/sparse_recovery.py` when implementing the project. It should:

1. Generate train and validation splits.
2. Train linear models with no penalty, L1, L2, and Elastic Net.
3. Sweep at least five `lambda` values.
4. Report validation MSE, number of nonzero coefficients, and overlap with true signal features.
5. Plot coefficient magnitudes for the best L1 and L2 models.

## Expected workflow

After creating the script, run:

```bash
uv run python mini_project/sparse_recovery.py
```

## Expected outputs

- A table of penalty type, `lambda`, validation MSE, and sparsity.
- A short explanation of when L1 recovers the true sparse support and when it fails.
- A coefficient plot showing shrinkage behavior.

## Writeup prompt

Which penalty best recovered the true signal? Did it also achieve the lowest validation error? Explain why interpretability and predictive performance may not peak at the same `lambda`.

## Optional extensions

- Add correlated feature groups.
- Compare proximal L1 updates with naive subgradient descent.
- Repeat with more noise and fewer training examples.
