# Cross-Entropy and NLL Loss — Mini-Project: Stable Classifier Loss

## Goal

Build a script that implements cross-entropy from logits and compares it to PyTorch on synthetic multiclass data.

## Dataset

Generate synthetic logits and labels:

- `logits` with shape `(128, 5)`.
- Integer class targets with shape `(128,)`.
- A second set of extreme logits to test numerical stability.

No external dataset is required.

## Implementation Tasks

1. Implement stable `logsumexp`.
2. Implement `log_softmax` from logits.
3. Implement NLL loss from log probabilities.
4. Implement cross-entropy directly from logits.
5. Compare manual losses and gradients to PyTorch.
6. Demonstrate how `mean` and `sum` reductions change gradient scale.

## Expected Workflow

After creating a script such as `mini_project/stable_cross_entropy.py`, the learner should be able to run:

```bash
uv run python mini_project/stable_cross_entropy.py
```

This is a future command for learner-created code, not a file added in this pass.

## Expected Outputs

- Matching manual and PyTorch losses for normal logits.
- Stable finite losses for extreme logits.
- Maximum absolute gradient difference.
- A short table comparing reduction modes.

## Writeup Prompt

Explain why a cross-entropy API should accept logits rather than probabilities. Include one metric that should be monitored alongside loss.

## Optional Extensions

- Add class weights and verify their effect on selected examples.
- Add an ignored target index for padded sequence examples.
