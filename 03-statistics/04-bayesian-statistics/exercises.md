# Bayesian Statistics — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Beta-Bernoulli update

Start with `p ~ Beta(2, 2)`. You observe `7` successes and `3` failures. What is the posterior distribution? What is the posterior mean?

---

## Exercise 2 — Compare MLE and posterior mean

For the data in Exercise 1, compute the Bernoulli MLE. Compare it with the posterior mean and explain the difference.

---

## Exercise 3 — MAP as regularized likelihood

Show that a Gaussian prior `w ~ N(0, tau^2)` adds a penalty proportional to `w^2` to the negative log posterior for a model parameter `w`.

---

## Exercise 4 — Interpret posterior uncertainty

Two models estimate a defect rate. Model A has posterior `Beta(60, 40)`. Model B has posterior `Beta(6, 4)`. Both have the same posterior mean. Which one has more uncertainty and why?

---

## Exercise 5 — Use posterior predictive reasoning

For `p ~ Beta(alpha, beta)` and Bernoulli observations, the posterior predictive probability of success is the posterior mean. Using the posterior from Exercise 1, what probability should be assigned to the next success?
