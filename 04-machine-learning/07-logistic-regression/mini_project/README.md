# Logistic Regression — Mini-Project: Rare Event Risk Model

## Goal

Train a logistic regression model on synthetic imbalanced data and choose an operating threshold based on recall and precision.

## Dataset

Generate 5,000 examples with 12 features. Use a known linear logit function, then sample labels from Bernoulli probabilities. Shift the intercept so positives are about 3% of examples.

## Implementation tasks

Create `mini_project/rare_event_logistic.py` in a future implementation pass. It should:

1. Generate train and validation data.
2. Fit logistic regression with BCE-with-logits and L2 regularization.
3. Validate analytic gradients against autograd for a small batch.
4. Sweep classification thresholds.
5. Report accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and calibration bins.

## Expected workflow

After creating the script, run:

```bash
uv run python mini_project/rare_event_logistic.py
```

## Expected outputs

- Learned coefficients compared to the true coefficients.
- A threshold table.
- A recommended threshold for at least 75% recall.
- A calibration summary.

## Writeup prompt

Explain why the model's probability ranking and deployed threshold are separate decisions. Describe how class imbalance affected accuracy, precision, and recall.

## Optional extensions

- Add class-weighted BCE.
- Compare L1 and L2 regularization.
- Add polynomial features to test misspecification.
