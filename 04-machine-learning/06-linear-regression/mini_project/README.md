# Linear Regression — Mini-Project: OLS vs Ridge Under Collinearity

## Goal

Compare ordinary least squares and ridge regression on synthetic data with correlated features.

## Dataset

Generate 800 examples with 10 features. Make features 0 and 1 highly correlated by creating one as the other plus small noise. Use a known coefficient vector and Gaussian target noise.

## Implementation tasks

Create `mini_project/ols_vs_ridge.py` in a future implementation pass. It should:

1. Generate train and validation splits.
2. Fit OLS using a stable linear solve.
3. Fit ridge regression for several `lambda` values.
4. Train the same model with gradient descent as a comparison.
5. Report coefficient error, validation MAE, RMSE, and R^2.
6. Plot true coefficients versus fitted OLS and ridge coefficients.

## Expected workflow

After creating the script, run:

```bash
uv run python mini_project/ols_vs_ridge.py
```

## Expected outputs

- A metrics table for OLS, ridge values, and gradient descent.
- Coefficient plots showing how ridge stabilizes correlated features.
- A note about whether prediction quality or coefficient recovery improved more.

## Writeup prompt

Explain why OLS coefficients become unstable under collinearity even when validation error is acceptable. Describe how ridge changes the optimization problem.

## Optional extensions

- Increase feature dimension beyond sample count.
- Add lasso or Elastic Net.
- Compare standardized and unstandardized features.
