# Bayesian Statistics — Mini-Project: Bayesian Rate Tracker

Build a small Beta-Bernoulli tracker for uncertain success rates.

---

## Goal

Track a binary event rate over time and compare MLE, posterior mean, MAP, and uncertainty intervals.

---

## Data Setup

Generate synthetic binary outcomes for two variants:

- Variant A true success rate: `0.10`
- Variant B true success rate: `0.12`

Simulate small and large sample regimes, such as `20`, `200`, and `2,000` observations per variant.

---

## Implementation Tasks

1. Create a future script such as `mini_project/bayesian_rate_tracker.py`.
2. Implement Beta prior updates for successes and failures.
3. Compare MLE and posterior mean after each sample size.
4. Estimate credible intervals by sampling from the Beta posterior or using available quantile utilities.
5. Estimate `P(p_B > p_A)` by posterior sampling.
6. Print a table showing how uncertainty shrinks with more observations.

---

## Expected Workflow

After creating the script, run it from this topic directory with a command like:

```bash
uv run python mini_project/bayesian_rate_tracker.py
```

The project should generate all outcomes in the script.

---

## Expected Outputs

- Posterior parameters for each variant and sample size.
- MLE, posterior mean, and credible interval summaries.
- An estimated probability that B is better than A.
- A short explanation of how prior strength affects early estimates.

---

## Writeup Prompt

Write 6-8 sentences explaining whether you would ship Variant B at each sample size. Include how posterior uncertainty changes the decision compared with MLE alone.

---

## Optional Extensions

- Try several priors, such as `Beta(1, 1)`, `Beta(5, 5)`, and `Beta(50, 50)`.
- Add Thompson sampling for traffic allocation.
- Simulate delayed feedback and discuss how it affects updates.
