# Batch Normalization — Mini-Project: Train/Eval Mode Audit

## Goal

Build a small experiment showing how batch normalization changes training dynamics and why train/eval mode matters.

## Dataset

Generate synthetic image-like tensors:

- `X` with shape `(512, 3, 16, 16)`.
- Labels from simple channel statistics, such as whether channel 0 mean exceeds channel 1 mean.

No external dataset is required.

## Implementation Tasks

1. Build a small CNN or MLP with and without batch norm.
2. Train both models on the synthetic task.
3. Log running means and variances over time.
4. Evaluate the batch-norm model in both training mode and evaluation mode on the same validation data.
5. Repeat with a very small batch size and compare stability.

## Expected Workflow

After creating a script such as `mini_project/batchnorm_audit.py`, the learner should be able to run:

```bash
uv run python mini_project/batchnorm_audit.py
```

This is a future command for learner-created code.

## Expected Outputs

- Loss curves with and without batch norm.
- Running-statistics summaries.
- Validation metrics comparing train mode and eval mode.
- Notes about small-batch instability.

## Writeup Prompt

Explain the largest difference you observed between training and evaluation behavior. Include one operational safeguard you would add before deploying a batch-norm model.

## Optional Extensions

- Compare batch norm to layer norm or group norm.
- Simulate distribution shift and observe stale running statistics.
