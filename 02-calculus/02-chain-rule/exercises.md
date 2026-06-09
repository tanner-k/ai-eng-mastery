# The Chain Rule — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Differentiate a nested scalar function

Let

```text
f(x) = exp(3x^2 - 2x)
```

Use the chain rule to derive `f'(x)`. Evaluate the derivative at `x = 1`.

---

## Exercise 2 — Derive a sigmoid squared-error gradient

For one example, define

```text
a = wx + b
p = sigmoid(a)
L = (p - y)^2
```

Derive `dL/dw` and `dL/db`. Keep the answer in terms of `p`, `y`, and `x` where possible.

---

## Exercise 3 — Trace gradient flow through branches

Let

```text
u = x^2
v = 3x
z = u + v
L = z^2
```

Derive `dL/dx` two ways: by expanding `L` directly and by summing the two branch contributions through `u` and `v`.

---

## Exercise 4 — Explain vanishing through repeated multiplication

Suppose a depth-`k` scalar network has local derivative `0.6` at every layer. What is the derivative of the output with respect to the input? Compute it for `k = 5`, `10`, and `30`. Interpret the result.

---

## Exercise 5 — Find the missing factor

A candidate derives the following for `L = log(1 + exp(wx + b))`:

```text
dL/dw = exp(wx + b) x
```

Identify the mistake and give the correct derivative.
