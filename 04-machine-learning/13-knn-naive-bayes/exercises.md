# kNN and Naive Bayes — Exercises

Worked solutions are in `solutions/solutions.md`.

## Exercise 1 — Run a kNN vote

A query point has nearest neighbor labels `[1, 0, 1, 1, 0]` for `k=5`. What class does unweighted kNN predict?

## Exercise 2 — Show scale sensitivity

Two features are age in years and income in dollars. Explain why Euclidean kNN may be dominated by income and what preprocessing helps.

## Exercise 3 — Compute Laplace-smoothed probability

In class `spam`, token `"free"` appears 8 times, total token count is 100, vocabulary size is 50, and `alpha = 1`. Compute `p("free" | spam)`.

## Exercise 4 — Use Naive Bayes log scores

Why do Naive Bayes implementations usually sum log probabilities instead of multiplying probabilities directly?

## Exercise 5 — Compare model assumptions

Give one situation where kNN is likely better than Naive Bayes and one situation where Naive Bayes is likely better than kNN.
