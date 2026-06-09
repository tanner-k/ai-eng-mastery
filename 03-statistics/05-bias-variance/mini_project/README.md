# Bias-Variance Tradeoff — Mini-Project: Repeated-Sampling Decomposition

Build a repeated-sampling simulation that estimates bias and variance for models with different capacity.

---

## Goal

Empirically show how model capacity changes bias, variance, and expected test error.

---

## Data Setup

Generate synthetic regression data from a known function:

```text
f(x) = x^3 - x
y = f(x) + noise
```

For each trial, sample a new training set from `x in [-2, 2]` and add Gaussian noise. Use a fixed grid of test points to evaluate repeated predictions.

---

## Implementation Tasks

1. Create a future script such as `mini_project/bias_variance_sim.py`.
2. Fit polynomial models with degrees such as `1`, `3`, and `15`.
3. Repeat training across many random datasets.
4. At each test point, estimate mean prediction and prediction variance.
5. Estimate squared bias using the known true function.
6. Print average squared bias, variance, and test MSE by degree.
7. Optionally compare a simple ensemble average of high-degree models.

---

## Expected Workflow

After creating the script, run it from this topic directory with a command like:

```bash
uv run python mini_project/bias_variance_sim.py
```

The project should generate all data in memory.

---

## Expected Outputs

- A table showing high bias for low-degree models and high variance for high-degree models.
- A comparison showing the cubic model performs well on data generated from a cubic function.
- An explanation of how ensembling changes variance.

---

## Writeup Prompt

Write 6-8 sentences explaining which polynomial degree you would choose and why. Include how your answer would change if the training set became much larger or the noise variance increased.

---

## Optional Extensions

- Add ridge regularization.
- Plot prediction bands across repeated training sets.
- Demonstrate double-descent-like behavior with an overparameterized feature basis.
