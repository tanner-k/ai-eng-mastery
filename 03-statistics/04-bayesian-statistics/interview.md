# Bayesian Statistics — Interview Prep

## Q&A

1. **Q: What is the core Bayesian update?**
   **A:** Posterior is proportional to likelihood times prior: `p(theta | D) proportional p(D | theta)p(theta)`.

2. **Q: What is the difference between a prior and a likelihood?**
   **A:** The prior encodes belief about parameters before seeing the current data. The likelihood describes how probable the observed data is for each parameter value.

3. **Q: What is the evidence term?**
   **A:** The evidence `p(D)` normalizes the posterior so it integrates to one. It can also be used for model comparison, though it is often hard to compute.

4. **Q: How does MAP differ from MLE?**
   **A:** MLE maximizes likelihood. MAP maximizes likelihood times prior, or equivalently log-likelihood plus log-prior. MAP returns a point estimate but includes prior preference.

5. **Q: How does L2 regularization relate to Bayesian inference?**
   **A:** L2 regularization corresponds to a zero-mean Gaussian prior on weights. The squared penalty comes from the negative log of that prior.

6. **Q: What is a posterior predictive distribution?**
   **A:** It predicts new outcomes by integrating over parameter uncertainty under the posterior, rather than plugging in a single point estimate.

7. **Q: Why are Bayesian methods useful for small data?**
   **A:** Priors can stabilize estimates when the likelihood is weak. The posterior also exposes uncertainty instead of overconfidently returning a noisy point estimate.

8. **Q: Why is full Bayesian inference hard for neural networks?**
   **A:** The parameter space is huge and the posterior is complex. Exact integration is intractable, so approximations such as variational inference, Laplace approximations, MCMC on smaller models, or ensembles are used.

## Explain it like a principal

Bayesian statistics is most useful as an uncertainty discipline. A principal engineer should know when a point estimate is not enough: cold-start ranking, rare-event monitoring, experimentation, and safety decisions all need calibrated uncertainty. The practical tradeoff is computational cost versus decision quality. Often the best production answer is not "make the whole model Bayesian," but "put Bayesian updates around the component where uncertainty drives decisions."

## Gotchas & follow-ups

- **"Bayesian means subjective."** Priors can encode domain knowledge, regularization, or weak default assumptions; they should be inspected and stress-tested.
- **"MAP is full Bayesian inference."** MAP is a point estimate. It does not preserve posterior uncertainty.
- **"With enough data, priors never matter."** Usually the likelihood dominates, but model misspecification and prior support can still matter.
- **"Credible intervals are confidence intervals."** They answer different questions: Bayesian credible intervals describe posterior probability; frequentist confidence intervals describe long-run procedure coverage.
- **Follow-up:** How would you use Bayesian updating to rank two experiments with different sample sizes?
