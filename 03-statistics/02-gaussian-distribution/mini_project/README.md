# Gaussian Distribution — Mini-Project: Residual Gaussianity Check

Build a synthetic residual analysis that compares Gaussian and heavy-tailed noise.

---

## Goal

Show when Gaussian residual assumptions produce useful likelihood scores and when they understate tail risk.

---

## Data Setup

Generate two residual samples:

- Gaussian residuals from `N(0, 1)`.
- Heavy-tailed residuals from a Student-t-like or mixture distribution, such as 95% `N(0, 1)` and 5% `N(0, 10^2)`.

No external dataset is required.

---

## Implementation Tasks

1. Create a future script such as `mini_project/gaussian_residual_check.py`.
2. Estimate mean and variance for each residual sample.
3. Compute Gaussian NLL for each residual.
4. Count how often observations exceed `2` and `3` estimated standard deviations.
5. Print summary statistics for both samples.
6. Optionally save a simple histogram if plotting tools are already available.

---

## Expected Workflow

After creating the script, run it from this topic directory with a command like:

```bash
uv run python mini_project/gaussian_residual_check.py
```

The project should generate its own residuals.

---

## Expected Outputs

- Estimated mean and variance for both residual sources.
- Tail counts beyond `2` and `3` standard deviations.
- A short comparison explaining why Gaussian likelihood treats the heavy-tailed sample poorly.

---

## Writeup Prompt

Write 5-7 sentences explaining whether a Gaussian residual model is appropriate for each sample. Include one production risk of using Gaussian anomaly thresholds on heavy-tailed latency or error data.

---

## Optional Extensions

- Compare Gaussian NLL with absolute-error scoring.
- Add a multivariate diagonal Gaussian example.
- Compute Mahalanobis distances for a 2D correlated Gaussian.
