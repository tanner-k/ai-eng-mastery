# Label Smoothing — Mini-Project: Confidence and Calibration Sweep

## Goal

Build a small classifier experiment that compares hard labels with several label-smoothing values and reports accuracy, NLL, and confidence statistics.

## Dataset

Generate a synthetic multiclass dataset:

- 1,500 examples with 10 features.
- 4 classes generated from linear scores plus noise.
- Optional controlled label noise by flipping a small percentage of labels.

No external dataset is required.

## Implementation Tasks

1. Implement smoothed target construction for integer labels.
2. Implement soft-target cross-entropy from logits.
3. Train identical classifiers with `epsilon` values such as `0`, `0.05`, `0.1`, and `0.2`.
4. Report validation accuracy, NLL, average max probability, and a simple calibration table.
5. Compare behavior with and without injected label noise.

## Expected Workflow

After creating a script such as `mini_project/label_smoothing_sweep.py`, the learner should be able to run:

```bash
uv run python mini_project/label_smoothing_sweep.py
```

This command is a future workflow for learner-created code.

## Expected Outputs

- A table of metrics by smoothing value.
- Evidence that smoothing usually lowers average confidence.
- A short discussion of when smoothing improved or harmed validation NLL.

## Writeup Prompt

Choose the best smoothing value for the synthetic task and justify it using more than accuracy. Include one reason you might avoid label smoothing in a production model.

## Optional Extensions

- Add temperature scaling after training and compare it to smoothing.
- Evaluate class-wise effects when one class is rare.
