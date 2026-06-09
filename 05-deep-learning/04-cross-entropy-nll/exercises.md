# Cross-Entropy and NLL Loss — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Compute NLL from probabilities

A classifier assigns probabilities `[0.1, 0.7, 0.2]` and the correct class is class `1`. Compute the negative log likelihood.

## Exercise 2 — Derive cross-entropy from logits

Starting with `p = softmax(z)` and `L = -log p_c`, show that `L = -z_c + log(sum_j exp(z_j))`.

## Exercise 3 — Derive the logit gradient

For one-hot target `y`, derive `dL/dz = softmax(z) - y`.

## Exercise 4 — Compare reduction modes

A batch has per-example losses `[0.2, 0.4, 1.4, 2.0]`. Compute the `sum` and `mean` reductions. How does switching from `mean` to `sum` affect gradient scale?

## Exercise 5 — Choose the right loss

For each task, choose cross-entropy, binary cross-entropy, or NLL loss and explain why:

1. Next-token prediction over a vocabulary.
2. Image tagging where each image can have multiple labels.
3. A model whose final layer already returns log probabilities for one class among five.
4. A binary fraud detector with one sigmoid output.
