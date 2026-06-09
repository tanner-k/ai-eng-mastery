# Loss Functions — Mini-Project: Robust Regression Bake-Off

## Goal

Compare MSE, MAE, and Huber losses on the same synthetic regression problem with controlled outliers.

## Dataset

Generate a synthetic one-dimensional dataset inside the script you create:

- `x` sampled uniformly from `[-3, 3]`
- clean target `y = 2x - 0.5 + noise`
- Gaussian noise with standard deviation `0.3`
- replace 5% of labels with large outliers sampled from a wider distribution

## Implementation tasks

Create `mini_project/robust_losses.py` in a later implementation pass or as your own learner exercise. In that file:

1. Generate the synthetic data.
2. Train three linear models from the same initialization using MSE, MAE, and Huber loss.
3. Record training loss and validation MAE for each model.
4. Plot fitted lines against the noisy data.
5. Print the learned slope, intercept, and validation MAE.

## Expected workflow

After creating the script, run it from this topic directory with:

```bash
uv run python mini_project/robust_losses.py
```

## Expected outputs

- A table comparing learned parameters and validation MAE.
- A plot showing that MSE is pulled toward outliers more strongly than Huber or MAE.
- A short note explaining which loss best matched the clean underlying line.

## Writeup prompt

Explain why the lowest training loss does not necessarily imply the best validation behavior when labels contain outliers. Tie your answer to gradient size and the assumptions behind each loss.

## Optional extensions

- Sweep the Huber transition parameter.
- Add quantile loss for median or percentile regression.
- Repeat with asymmetric outliers above the true line only.
