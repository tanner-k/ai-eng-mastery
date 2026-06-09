# Label Smoothing — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Construct a smoothed target

For `K = 5`, correct class `2`, and `epsilon = 0.1`, construct the smoothed target using `y_smooth = (1 - epsilon) * one_hot + epsilon / K`.

## Exercise 2 — Compute soft-target cross-entropy

Given target `q = [0.9, 0.05, 0.05]` and predicted probabilities `p = [0.8, 0.1, 0.1]`, compute cross-entropy `-sum_i q_i log p_i` approximately.

## Exercise 3 — Derive the gradient

For soft target distribution `q` and logits `z`, derive the gradient of cross-entropy with respect to logits.

## Exercise 4 — Analyze confidence

Explain why label smoothing tends to reduce the maximum softmax probability learned by a classifier. Does this guarantee calibrated probabilities?

## Exercise 5 — Decide whether to use smoothing

For each setting, state whether label smoothing is likely helpful, harmful, or needs care:

1. Noisy image classification labels.
2. Knowledge distillation from a teacher model.
3. Next-token prediction with a very large vocabulary.
4. Medical diagnosis where calibrated probabilities are the primary output.
