# kNN and Naive Bayes — Mini-Project: Baseline Duel

## Goal

Compare kNN and Naive Bayes on synthetic datasets designed to favor different assumptions.

## Dataset

Generate two datasets in the future script:

- A geometric 2D dataset where local neighborhoods determine the label.
- A sparse count dataset where class-specific token rates determine the label.

## Implementation tasks

Create `mini_project/baseline_duel.py` in a future implementation pass. It should:

1. Implement vectorized kNN classification with configurable `k`.
2. Implement multinomial Naive Bayes with Laplace smoothing.
3. Evaluate both models on both datasets.
4. Show the effect of feature scaling on kNN.
5. Show the effect of smoothing strength on Naive Bayes.
6. Report accuracy, precision, recall, and confusion matrices.

## Expected workflow

After creating the script, run:

```bash
uv run python mini_project/baseline_duel.py
```

## Expected outputs

- A table comparing both models on both datasets.
- A short note on which assumptions matched each dataset.
- A plot of kNN decision regions for the 2D dataset.

## Writeup prompt

Explain why each model won on one dataset and lost on the other. Include the operational tradeoff between training cost and prediction cost.

## Optional extensions

- Add distance-weighted kNN.
- Add Gaussian Naive Bayes.
- Add approximate nearest-neighbor discussion for large embedding sets.
