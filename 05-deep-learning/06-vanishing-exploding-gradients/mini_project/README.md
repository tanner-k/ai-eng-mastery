# Vanishing and Exploding Gradients — Mini-Project: Gradient Flow Profiler

## Goal

Build a synthetic experiment that measures gradient norms across depth for different activations, initializations, and clipping settings.

## Dataset

Use random inputs and synthetic targets generated in the script:

- `X` with shape `(128, 32)`.
- Regression or classification targets generated from a shallow teacher model.
- MLP depths such as 5, 20, and 50 layers.

## Implementation Tasks

1. Build configurable deep MLPs with tanh or ReLU.
2. Compare small random, Xavier, and He-style initialization.
3. Run one forward/backward step and record gradient norm per layer.
4. Train briefly and record loss stability.
5. Add global norm clipping and compare before/after gradient norms.

## Expected Workflow

After creating a script such as `mini_project/gradient_flow_profiler.py`, the learner should be able to run:

```bash
uv run python mini_project/gradient_flow_profiler.py
```

This is a future workflow for learner-created code.

## Expected Outputs

- A layer-by-layer gradient norm table.
- Examples of vanishing and exploding gradient profiles.
- A comparison showing how initialization and clipping change behavior.

## Writeup Prompt

Identify which configuration had the healthiest gradient flow. Explain using layer-wise gradient norms, not only final loss.

## Optional Extensions

- Add residual connections and compare the gradient profile.
- Add activation histograms to distinguish saturation from exploding activations.
