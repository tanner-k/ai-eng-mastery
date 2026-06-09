# Activations: tanh and ReLU — Mini-Project: Activation Diagnostics

## Goal

Build a small experiment comparing tanh and ReLU in a multilayer perceptron trained on synthetic data, with emphasis on activation and gradient statistics.

## Dataset

Generate a synthetic binary classification dataset:

- 1,000 examples with 20 features.
- Labels from a nonlinear rule using products or thresholded sums.
- Train/validation split created in the script.

## Implementation Tasks

1. Build two MLPs with the same layer widths, one using tanh and one using ReLU.
2. Train both on the same synthetic data.
3. Log activation means, activation standard deviations, and fraction of zero ReLU outputs by layer.
4. Log gradient norms by layer.
5. Compare loss curves and validation accuracy.

## Expected Workflow

After creating a script such as `mini_project/activation_diagnostics.py`, the learner should be able to run:

```bash
uv run python mini_project/activation_diagnostics.py
```

This command is for future learner-created code and does not imply the script exists now.

## Expected Outputs

- Loss and accuracy summaries for tanh and ReLU models.
- Per-layer activation and gradient statistics.
- A short report identifying saturation or dead-unit behavior.

## Writeup Prompt

Which activation trained more easily on your synthetic task? Use gradient norms and activation statistics to support the answer.

## Optional Extensions

- Add leaky ReLU and compare dead-unit rates.
- Repeat with different initialization scales.
