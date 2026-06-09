# Probability Basics — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Compute conditional probability

A classifier flags 4% of requests for review. Among flagged requests, 30% are truly policy violations. Among unflagged requests, 1% are truly violations. What is the overall probability that a random request is a violation?

---

## Exercise 2 — Use Bayes' rule

A rare defect appears in 2% of generated samples. A detector has 95% true-positive rate and 10% false-positive rate. If the detector fires, what is the probability that the sample is actually defective?

---

## Exercise 3 — Expectation and variance of a Bernoulli variable

Let `X ~ Bernoulli(p)`. Derive `E[X]` and `Var(X)`.

---

## Exercise 4 — Diagnose independence

Two binary events have `P(A) = 0.4`, `P(B) = 0.5`, and `P(A and B) = 0.25`. Are they independent? What is `P(A | B)`?

---

## Exercise 5 — Interpret metric uncertainty

Model A gets 820 correct predictions out of 1000 validation examples. Model B gets 835 correct out of 1000. Explain why this may not be enough evidence to declare B better in production. What additional check would you run?
