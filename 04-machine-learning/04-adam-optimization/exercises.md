# Adam Optimization — Exercises

Worked solutions are in `solutions/solutions.md`.

## Exercise 1 — Compute the first Adam step

Let `g_1 = 2`, `m_0 = 0`, `v_0 = 0`, `beta1 = 0.9`, `beta2 = 0.999`, `alpha = 0.001`, and `epsilon = 1e-8`. Compute `m_1`, `v_1`, `mhat_1`, `vhat_1`, and the update amount.

## Exercise 2 — Explain bias correction

For a constant gradient `g_t = g`, show why `m_t` underestimates `g` at early steps before bias correction.

## Exercise 3 — Compare Adam and AdamW

Explain why adding `lambda ||w||_2^2` to the loss is not equivalent to decoupled weight decay when Adam normalizes gradients by `sqrt(vhat)`.

## Exercise 4 — Diagnose unstable Adam training

A transformer training run with AdamW shows loss spikes in the first 500 steps. Name three likely causes and two first changes you would try.

## Exercise 5 — Reason about beta values

What happens if `beta2` is reduced from `0.999` to `0.9`? Discuss responsiveness, noise, and effective learning-rate stability.
