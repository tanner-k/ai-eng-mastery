# Evaluation Metrics — Mini-Project: Imbalanced Classifier Report

## Goal

Create an evaluation report for a synthetic rare-event classifier and show how threshold choice changes operational behavior.

## Dataset

Generate 10,000 synthetic examples with 1% positives. Create model scores by sampling positives from a distribution centered higher than negatives, with overlap so the classifier is imperfect.

## Implementation tasks

Create `mini_project/metric_report.py` in a future implementation pass. It should:

1. Generate labels and scores.
2. Compute confusion matrices at thresholds from `0.05` to `0.95`.
3. Report accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.
4. Plot precision-recall and ROC curves.
5. Select a threshold that reaches at least 80% recall and maximizes precision.

## Expected workflow

After creating the script, run:

```bash
uv run python mini_project/metric_report.py
```

## Expected outputs

- A threshold table with classification metrics.
- PR and ROC curves.
- A recommendation for an operating threshold.

## Writeup prompt

Explain why the chosen threshold is appropriate for the rare-event setting. Include one paragraph on why accuracy would have led to a poor decision.

## Optional extensions

- Add calibration bins and a Brier score.
- Bootstrap confidence intervals for PR-AUC.
- Add subgroup metrics for two synthetic populations.
