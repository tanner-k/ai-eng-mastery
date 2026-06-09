# Softmax — Exercises

Worked solutions for all exercises are in `solutions/solutions.md`.

---

## Exercise 1 — Compute softmax by hand

Compute softmax for logits `[1, 2, 4]`. Give probabilities rounded to three decimals.

## Exercise 2 — Prove shift invariance

Show that `softmax(z)_i = softmax(z - c)_i` for any scalar `c`.

## Exercise 3 — Derive the Jacobian

Derive `d p_i / d z_j` for `p = softmax(z)`. Express the result for the cases `i = j` and `i != j`.

## Exercise 4 — Diagnose numerical overflow

Explain why computing `exp([1000, 1001, 1002])` directly is unsafe. Show the stable logits used before exponentiation.

## Exercise 5 — Temperature behavior

For logits `[2, 1, 0]`, compare qualitatively what happens to the softmax distribution when `T = 0.5`, `T = 1`, and `T = 2`. Which is sharpest and which is flattest?
