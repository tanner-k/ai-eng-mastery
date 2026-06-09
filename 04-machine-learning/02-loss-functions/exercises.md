# Loss Functions — Exercises

Worked solutions are in `solutions/solutions.md`.

## Exercise 1 — Derive MSE and MAE gradients

For a single scalar prediction `a = yhat - y`, derive the derivative of MSE `a^2` and a valid subgradient of MAE `|a|`. Explain what happens at `a = 0` for MAE.

## Exercise 2 — Compute binary cross-entropy

For labels `y = [1, 0, 1]` and predicted probabilities `p = [0.9, 0.2, 0.4]`, compute the mean binary cross-entropy:

```
-(1/n) sum_i [y_i log p_i + (1-y_i) log(1-p_i)]
```

Use natural logarithms.

## Exercise 3 — Show the softmax cross-entropy gradient

For logits `z`, probabilities `p = softmax(z)`, and one-hot label vector `y`, show that the gradient of `-sum_k y_k log p_k` with respect to logits is `p - y`.

## Exercise 4 — Choose a loss for noisy regression

A sensor regression dataset has mostly small Gaussian noise but 3% of labels are corrupted by extreme measurement failures. Which loss would you start with: MSE, MAE, or Huber? Explain the tradeoff.

## Exercise 5 — Diagnose loss and metric mismatch

A fraud classifier is trained with plain binary cross-entropy on a dataset with 0.2% positives. Validation accuracy is 99.8%, but recall is near zero. Explain why the loss/metric setup can produce this result and name two changes you would try.
