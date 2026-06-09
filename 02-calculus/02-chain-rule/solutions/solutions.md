# The Chain Rule — Solutions

## Solution 1 — Differentiate a nested scalar function

Let `u = 3x^2 - 2x`, so `f(x) = exp(u)`.

```text
du/dx = 6x - 2
df/du = exp(u)
```

Therefore:

```text
f'(x) = exp(3x^2 - 2x)(6x - 2)
```

At `x = 1`, `u = 1`, so:

```text
f'(1) = exp(1)(4) = 4e
```

## Solution 2 — Derive a sigmoid squared-error gradient

The local derivatives are:

```text
dL/dp = 2(p - y)
dp/da = p(1 - p)
da/dw = x
da/db = 1
```

By the chain rule:

```text
dL/dw = 2(p - y) p(1 - p) x
dL/db = 2(p - y) p(1 - p)
```

## Solution 3 — Trace gradient flow through branches

Direct expansion:

```text
L = (x^2 + 3x)^2
dL/dx = 2(x^2 + 3x)(2x + 3)
```

Branch method:

```text
dL/dz = 2z
dz/du = 1
dz/dv = 1
du/dx = 2x
dv/dx = 3
```

So:

```text
dL/dx = (dL/dz)(dz/du)(du/dx) + (dL/dz)(dz/dv)(dv/dx)
       = 2z(2x) + 2z(3)
       = 2(x^2 + 3x)(2x + 3)
```

The two methods match.

## Solution 4 — Explain vanishing through repeated multiplication

If every local derivative is `0.6`, the total derivative is:

```text
0.6^k
```

Values:

```text
k = 5:  0.6^5  = 0.07776
k = 10: 0.6^10 = 0.0060466176
k = 30: 0.6^30 approx 0.000000221
```

The signal decays exponentially. By depth 30, the input receives almost no gradient signal from the output.

## Solution 5 — Find the missing factor

Let `a = wx + b`. The candidate differentiated `exp(a)` but forgot the derivative of `log(1 + exp(a))` with respect to its inside:

```text
d/da log(1 + exp(a)) = exp(a) / (1 + exp(a))
```

Then `da/dw = x`, so:

```text
dL/dw = [exp(wx + b) / (1 + exp(wx + b))] x
```

Equivalently:

```text
dL/dw = sigmoid(wx + b) x
```
