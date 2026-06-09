# Random Forests and XGBoost — Mini-Project: Ensemble Comparison

## Goal

Compare a single decision tree, a random forest, and a simple gradient boosting regressor on synthetic tabular data.

## Dataset

Generate 2,000 examples with nonlinear interactions, feature noise, and a continuous target. Include at least one irrelevant feature and one correlated feature.

## Implementation tasks

Create `mini_project/ensemble_comparison.py` in a future implementation pass. It should:

1. Generate train and validation splits.
2. Train a single tree at several depths.
3. Train a random forest with bootstrap samples and feature subsampling.
4. Train a simple squared-error gradient boosting model.
5. Report train and validation RMSE.
6. Plot validation RMSE versus number of trees for forest and boosting.

## Expected workflow

After creating the script, run:

```bash
uv run python mini_project/ensemble_comparison.py
```

## Expected outputs

- A metrics table comparing model families.
- A curve showing forest stabilization and boosting's early-stopping point.
- A short recommendation for the best model under validation RMSE.

## Writeup prompt

Explain why the random forest improved over one tree and why boosting may require stricter validation control.

## Optional extensions

- Add classification and PR-AUC.
- Add permutation feature importance.
- Add a leakage feature and show how validation can be fooled.
