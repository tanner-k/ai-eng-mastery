# Softmax — Mini-Project: Stable Logit Playground

## Goal

Build a small script that explores softmax stability, temperature scaling, and the difference between logits and probabilities.

## Dataset

Use synthetic logits generated in the script:

- Small logits for hand-checkable probabilities.
- Large logits such as `[1000, 1001, 1002]`.
- Random batches with shape `(32, 10)`.

No external data is required.

## Implementation Tasks

1. Implement naive softmax and stable softmax.
2. Show that the naive version overflows or produces invalid values on large logits.
3. Compute softmax at several temperatures and measure entropy.
4. Derive or implement a softmax Jacobian for one example.
5. Compare manual stable softmax to PyTorch for a batch of logits.

## Expected Workflow

After creating a script such as `mini_project/stable_softmax.py`, the learner should be able to run:

```bash
uv run python mini_project/stable_softmax.py
```

This command describes future learner-created work; the script is not part of this content pass.

## Expected Outputs

- Probability vectors that sum to one.
- A demonstration of overflow in the naive implementation.
- Entropy values showing lower temperature gives sharper distributions.
- Maximum absolute difference between manual and PyTorch softmax.

## Writeup Prompt

Explain why logits are the right interface between a classifier head and a cross-entropy loss. Include one example where applying softmax too early would be harmful.

## Optional Extensions

- Add masked softmax for attention-style inputs.
- Plot entropy as a function of temperature for a fixed logit vector.
