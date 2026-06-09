# Support Vector Machines — Mini-Project: Margin Visualizer

## Goal

Train a linear SVM on synthetic 2D data and visualize how `C` changes the margin and violations.

## Dataset

Generate two Gaussian clusters in 2D with adjustable overlap. Use labels `-1` and `1`.

## Implementation tasks

Create `mini_project/margin_visualizer.py` in a future implementation pass. It should:

1. Generate separable and overlapping datasets.
2. Implement linear scores and hinge loss with L2 penalty.
3. Train with subgradient descent for several `C` values.
4. Plot the decision boundary, margin lines, support vectors, and violations.
5. Report train accuracy and validation accuracy.

## Expected workflow

After creating the script, run:

```bash
uv run python mini_project/margin_visualizer.py
```

## Expected outputs

- Boundary plots for at least three `C` values.
- A table of margin width and validation accuracy.
- A note explaining which `C` generalized best.

## Writeup prompt

Describe how increasing `C` changed the margin width, number of violations, and validation behavior.

## Optional extensions

- Add logistic regression on the same data.
- Add an RBF feature map approximation.
- Calibrate SVM scores with a held-out logistic layer.
