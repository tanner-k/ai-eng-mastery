# Decision Trees — Mini-Project: Build a Tiny CART Classifier

## Goal

Implement a small decision tree classifier and study how depth affects train and validation performance.

## Dataset

Generate a two-dimensional synthetic classification dataset with nonlinear structure, such as an XOR-style rule with label noise.

## Implementation tasks

Create `mini_project/tiny_cart.py` in a future implementation pass. It should:

1. Generate train and validation data.
2. Implement Gini impurity and weighted split gain.
3. Search numeric thresholds for each feature.
4. Grow a binary tree with `max_depth` and `min_samples_leaf`.
5. Compare depths from 1 through 8.
6. Print train and validation accuracy for each depth.

## Expected workflow

After creating the script, run:

```bash
uv run python mini_project/tiny_cart.py
```

## Expected outputs

- A table showing underfitting at shallow depth and overfitting at excessive depth.
- Printed rules or a simple text tree for the best depth.
- A short validation-based choice of tree depth.

## Writeup prompt

Explain why deeper trees reduce training error monotonically but do not necessarily improve validation error.

## Optional extensions

- Add entropy as an alternative criterion.
- Add regression-tree support.
- Add cost-complexity pruning.
