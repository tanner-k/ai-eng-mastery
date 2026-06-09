# The Chain Rule — Mini-Project: Tiny Reverse-Mode Autodiff

Build a small scalar reverse-mode autodiff engine and use it to differentiate composed functions.

---

## Goal

Make the chain rule executable. The project should show how a computation graph stores local operations during the forward pass and propagates gradients backward from a scalar loss.

---

## Data Setup

No dataset is required. Use scalar examples generated in the script:

- `f(x) = exp(3x^2 - 2x)`
- `L = (sigmoid(wx + b) - y)^2`
- A branched expression such as `z = x*x + 3*x`, `L = z*z`

---

## Implementation Tasks

1. Create a future script such as `mini_project/tiny_autodiff.py`.
2. Define a scalar `Value` object that stores data, gradient, parents, and a backward callback.
3. Implement addition, multiplication, negation, subtraction, power by a scalar, `exp`, `tanh`, and `sigmoid`.
4. Topologically sort the graph from the final loss.
5. Seed the final loss gradient with `1.0` and call backward callbacks in reverse topological order.
6. Compare the computed gradients against hand-derived formulas for the examples.
7. Include one branched graph to prove that gradient contributions accumulate.

---

## Expected Workflow

After creating the script, run it from this topic directory with a command like:

```bash
uv run python mini_project/tiny_autodiff.py
```

The script should print each expression, each variable's gradient, and the matching manual value.

---

## Expected Outputs

- Correct gradients for nested scalar functions.
- Correct gradient accumulation for a reused variable.
- A short comparison explaining why one backward pass can compute gradients for multiple inputs.

---

## Writeup Prompt

Write 6-8 sentences explaining how reverse-mode autodiff differs from symbolic differentiation and finite differences. Include why reverse mode is efficient for a scalar loss with many parameters.

---

## Optional Extensions

- Add division and `log`.
- Add a graph visualization text dump.
- Compare your scalar engine's results with PyTorch autograd on the same expressions.
