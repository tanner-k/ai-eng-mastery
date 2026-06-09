# Probability Basics — Mini-Project: Metric Uncertainty Simulator

Build a simulation that shows how validation accuracy varies across repeated samples.

---

## Goal

Estimate how much a model metric can move purely from sampling noise, then use that intuition to compare two close models.

---

## Data Setup

Generate synthetic correctness indicators:

- Model A has true accuracy `0.82`.
- Model B has true accuracy `0.835`.
- Validation set sizes: `100`, `1,000`, and `10,000`.

Each simulated validation run should sample Bernoulli outcomes for each model.

---

## Implementation Tasks

1. Create a future script such as `mini_project/metric_uncertainty.py`.
2. Simulate at least `5,000` validation sets for each sample size.
3. Record sampled accuracy for both models.
4. Estimate how often B appears better than A.
5. Compute approximate 95% intervals for each model's sampled accuracy.
6. Print a concise table by validation size.

---

## Expected Workflow

After creating the script, run it from this topic directory with a command like:

```bash
uv run python mini_project/metric_uncertainty.py
```

The script should use generated data only.

---

## Expected Outputs

- A table showing that small validation sets produce wide metric variation.
- The estimated probability that B beats A in the sample at each validation size.
- A short explanation of why large validation sets are needed for small metric differences.

---

## Writeup Prompt

Write 5-7 sentences explaining whether a 1.5 percentage-point improvement is convincing at each validation size. Include one recommendation for an offline evaluation process.

---

## Optional Extensions

- Add paired outcomes where both models are evaluated on the same examples.
- Estimate uncertainty for precision on a rare positive class.
- Add a simple bootstrap interval.
