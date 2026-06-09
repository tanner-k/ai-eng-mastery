# Broadcasting — Mini-Project: Broadcast Debugger

Build a small shape debugger that predicts broadcast results and validates them against PyTorch.

## Goal

Create a script that takes pairs of tensor shapes, determines whether they are broadcast-compatible, and explains the result. Then use synthetic tensors to show how broadcasted gradients reduce over expanded axes.

## Dataset

No dataset is needed. Use hard-coded shape cases and synthetic tensors:

```python
shape_cases = [
    ((32, 128), (128,)),
    ((10, 1, 5), (1, 7, 5)),
    ((4, 3), (4,)),
    ((2, 1, 8, 1), (3, 1, 5)),
]
```

For gradient checks, generate small tensors with `torch.randn`.

## Implementation Tasks

Create a future file such as `mini_project/broadcast_debugger.py` and implement:

1. A function that applies the right-aligned broadcasting rules to two shape tuples.
2. Human-readable explanations for compatible and incompatible dimensions.
3. Validation against `torch.broadcast_shapes` when available.
4. A gradient demo for `Y = X + b`, where X has shape `(5, 3)` and b has shape `(3,)`.
5. A comparison showing that `b.grad` equals the row-wise sum of the upstream gradient.
6. A short memory comparison between `expand` and `repeat`.

## Expected Workflow

After creating the script, run it from this topic directory:

```bash
uv run python mini_project/broadcast_debugger.py
```

## Expected Outputs

The script should print:

- Each shape pair and either the broadcasted result shape or the incompatible dimension.
- PyTorch validation results.
- The manually computed and autograd-computed bias gradient for the gradient demo.
- The storage-size difference between an expanded view and a repeated tensor.

## Writeup Prompt

Write 5-8 sentences answering:

1. Which example was most likely to surprise you and why?
2. Why does the bias gradient sum over rows?
3. When would you use explicit `unsqueeze` even if broadcasting would work implicitly?
4. Why can `repeat` become a memory problem?

## Optional Extensions

- Support three or more input shapes.
- Add named-axis labels to explanations.
- Include an attention mask example with shape `(batch, 1, 1, tokens)`.
