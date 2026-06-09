# Adam Optimization — Mini-Project: Moment Diagnostics

## Goal

Build an Adam/AdamW diagnostic script that visualizes moment estimates, effective learning rates, and convergence on a controlled synthetic problem.

## Dataset

Generate a two-feature regression dataset where one feature has much larger scale than the other. Standardize one copy and leave another copy unstandardized so optimizer behavior can be compared.

## Implementation tasks

Create `mini_project/adam_diagnostics.py` in a future implementation pass. It should:

1. Implement Adam and AdamW update equations directly.
2. Train the same linear model with SGD, Adam, and AdamW.
3. Log `mhat`, `vhat`, parameter values, loss, and effective step size.
4. Compare runs with and without bias correction.
5. Compare Adam with L2-in-loss against AdamW decoupled weight decay.

## Expected workflow

After creating the script, run:

```bash
uv run python mini_project/adam_diagnostics.py
```

## Expected outputs

- Loss curves for SGD, Adam, and AdamW.
- A table of final parameters and validation loss.
- A plot or printed trace of early-step bias correction effects.
- A short comparison of L2 penalty versus decoupled weight decay.

## Writeup prompt

Explain where Adam's adaptivity helped, where it made little difference, and why AdamW produced different parameter norms than Adam with an L2 penalty.

## Optional extensions

- Add learning-rate warmup.
- Sweep `beta2` values.
- Add gradient clipping and induce a gradient spike.
