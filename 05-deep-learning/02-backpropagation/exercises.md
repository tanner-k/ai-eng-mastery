# Backpropagation — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Derive gradients for a one-hidden-layer network

For one example, let:

```text
h = tanh(xW1 + b1)
y_hat = hW2 + b2
L = 0.5 * ||y_hat - y||^2
```

Derive gradients for `W2`, `b2`, `W1`, and `b1`.

## Exercise 2 — Add the batch dimension

Extend Exercise 1 to a mini-batch `X` with shape `(B, Din)`, hidden activations `H` with shape `(B, Hdim)`, and output `Yhat` with shape `(B, Dout)`. State the shapes of every gradient.

## Exercise 3 — Count work for finite differences

A neural network has 10 million trainable parameters. You estimate gradients by central finite differences, requiring two forward passes per parameter. If one forward pass takes 20 milliseconds, estimate the time for one gradient evaluation. Compare this with backpropagation.

## Exercise 4 — Identify cached values

For the operations `Z = XW + b`, `A = relu(Z)`, and `L = mean(A)`, list which values should be cached during the forward pass to compute the backward pass manually.

## Exercise 5 — Diagnose a broken gradient

During training, all gradients for the first layer are exactly zero, while later-layer gradients are nonzero. Give three plausible causes and one diagnostic check for each.
