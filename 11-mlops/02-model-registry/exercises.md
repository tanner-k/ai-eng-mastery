# Model Registry — Exercises

These prompts define the first-pass practice structure. Full worked derivations and executable checks should be expanded during the implementation pass.

## Exercise 1 — Define the contract

List the inputs, outputs, assumptions, and invariants for model registry. Include the shapes, state, or interface boundaries that an implementation would need to enforce.

## Exercise 2 — Work a tiny example

Create a hand-checkable toy example for model versions, stages, approvals, artifacts, signatures, lineage, and rollback. Compute the key intermediate values manually and note where approximation or implementation details could change the result.

## Exercise 3 — Compare two design choices

Choose two plausible approaches for this topic and compare their tradeoffs in accuracy, latency, memory, data requirements, observability, and failure recovery.

## Exercise 4 — Diagnose a failure

Describe a realistic failure mode for model registry. Specify the symptoms, the most likely root causes, and the telemetry or tests you would use to distinguish them.

## Exercise 5 — Plan the implementation lab

Turn this build target into a concrete implementation plan: model a registry promotion flow for candidate and production models. Define the files, assertions, plots or tables, and pass/fail checks the later implementation should include.
