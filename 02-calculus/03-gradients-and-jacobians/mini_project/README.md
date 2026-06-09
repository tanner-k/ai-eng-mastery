# Gradients and Jacobians — Mini-Project: Local Linearization Probe

Build a small probe that compares full Jacobians, vector-Jacobian products, and Jacobian-vector products on toy functions.

---

## Goal

Show that a Jacobian is a local linear approximation, then show why production training usually uses products with Jacobians rather than explicit matrices.

---

## Data Setup

Use generated tensors only. Suggested functions:

- `F: R^2 -> R^3`, `F(x1, x2) = [x1 + x2, x1*x2, exp(x1 - x2)]`
- A tiny linear layer from `R^4 -> R^3`
- A softmax over three logits

---

## Implementation Tasks

1. Create a future script such as `mini_project/jacobian_probe.py`.
2. Compute a manual Jacobian for the `R^2 -> R^3` function.
3. Use PyTorch autograd utilities to compute the same full Jacobian.
4. Use the Jacobian to predict output changes for several small perturbations.
5. Compare full-Jacobian multiplication with a VJP for a scalar loss.
6. Compare full-Jacobian multiplication with a JVP for an input perturbation.
7. Print the number of entries in full Jacobians for larger hypothetical layer sizes.

---

## Expected Workflow

After creating the script, run it from this topic directory with a command like:

```bash
uv run python mini_project/jacobian_probe.py
```

The script should not require external datasets or new dependencies beyond PyTorch.

---

## Expected Outputs

- Matching manual and autograd Jacobians for the toy function.
- A table showing local approximation error for perturbations of different sizes.
- A comparison showing VJP/JVP results match full-Jacobian products on small examples.
- A short memory estimate explaining why full Jacobians are avoided at realistic dimensions.

---

## Writeup Prompt

Write 6-8 sentences explaining when you would materialize a full Jacobian and when you would use VJPs or JVPs instead. Include one example from training and one from model analysis.

---

## Optional Extensions

- Compute the softmax Jacobian manually and compare it with autograd.
- Estimate the largest singular value of a small Jacobian.
- Add batching and describe whether examples are coupled or independent.
