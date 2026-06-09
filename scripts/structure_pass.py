"""Replace placeholder curriculum stubs with durable topic outlines."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.curriculum import SECTIONS, Section, Topic


STUB_MARKERS = (
    "🚧 Stub",
    "not yet written",
    "_What this is and why it matters._",
    "_Practice problems",
    "_Concise, senior-level framing._",
)


SECTION_FRAMING: dict[int, str] = {
    6: (
        "LLM engineering connects transformer mechanics to the practical work of "
        "tokenization, adaptation, alignment, inference, and evaluation."
    ),
    7: (
        "Reinforcement learning covers sequential decision-making, value estimation, "
        "bootstrapping, exploration, and policy optimization."
    ),
    8: (
        "Retrieval and RAG covers the indexing, ranking, context construction, and "
        "evaluation layer that makes knowledge-grounded AI systems useful."
    ),
    9: (
        "Graphs covers relational structure, graph storage, traversal, and graph-shaped "
        "memory for retrieval and agent systems."
    ),
    10: (
        "Agentic systems covers tool use, orchestration frameworks, stateful workflows, "
        "and memory patterns for production AI agents."
    ),
    11: (
        "MLOps covers the lifecycle controls that make model development reproducible, "
        "deployable, observable, and governable."
    ),
    12: (
        "Cloud and infrastructure covers the runtime substrate for AI systems: packaging, "
        "orchestration, accelerators, serverless patterns, autoscaling, and cost."
    ),
    13: (
        "AI system design integrates modeling, data, retrieval, serving, reliability, "
        "evaluation, and cost into interview-ready architecture decisions."
    ),
}


SECTION_LABS: dict[int, str] = {
    6: "Build a small language-model engineering lab with synthetic prompts and model-free simulations.",
    7: "Build a tabular environment and compare value-based and policy-based learning behavior.",
    8: "Build a small local retrieval pipeline over synthetic documents and evaluate answer grounding.",
    9: "Build a graph-backed knowledge and memory prototype over synthetic entities.",
    10: "Build a small tool-using agent loop with explicit state, safety checks, and memory.",
    11: "Build a model lifecycle pipeline with experiment metadata, registry handoff, serving, monitoring, and CI checks.",
    12: "Build an infrastructure plan for serving a small AI workload under latency, reliability, and cost constraints.",
    13: "Run a timed AI system-design case from requirements through architecture, tradeoffs, and rollout plan.",
}


TOPIC_FOCUS: dict[str, tuple[str, str, str]] = {
    "transformers-attention": (
        "attention scores, masking, positional information, and transformer block structure",
        "manual scaled dot-product attention on toy tensors",
        "[[tokenization-embeddings]], [[fine-tuning-lora-peft]], [[llm-evals]]",
    ),
    "tokenization-embeddings": (
        "token boundaries, vocabulary design, embedding lookup, and similarity geometry",
        "tokenize a toy corpus and inspect embedding-neighbor behavior",
        "[[transformers-attention]], [[semantic-retrieval]], [[vectors-and-norms]]",
    ),
    "fine-tuning-lora-peft": (
        "parameter-efficient adaptation, LoRA rank, freezing, adapters, and data quality",
        "simulate adapter updates and compare trainable parameter budgets",
        "[[transformers-attention]], [[rlhf-dpo]], [[llm-evals]]",
    ),
    "rlhf-dpo": (
        "preference data, reward modeling, policy optimization, and direct preference objectives",
        "fit a toy pairwise preference model and reason about chosen/rejected examples",
        "[[fine-tuning-lora-peft]], [[policy-gradients]], [[llm-evals]]",
    ),
    "quantization-inference": (
        "precision, memory bandwidth, latency, KV cache, batching, and serving tradeoffs",
        "estimate memory and latency for toy model configurations",
        "[[transformers-attention]], [[scaling-inference]], [[llm-evals]]",
    ),
    "llm-evals": (
        "task metrics, judge design, regression suites, calibration, and release gates",
        "design and score a synthetic prompt evaluation suite",
        "[[rag-evaluation]], [[fine-tuning-lora-peft]], [[quantization-inference]]",
    ),
    "mdps": (
        "states, actions, transitions, rewards, policies, returns, and discounting",
        "define a tiny gridworld MDP and compute returns by hand",
        "[[value-functions]], [[q-learning]], [[policy-gradients]]",
    ),
    "value-functions": (
        "state value, action value, Bellman expectation equations, and dynamic programming",
        "evaluate a fixed policy in a tiny MDP",
        "[[mdps]], [[q-learning]], [[policy-gradients]]",
    ),
    "q-learning": (
        "temporal-difference targets, bootstrapping, off-policy control, and exploration",
        "train tabular Q-learning in a small deterministic environment",
        "[[mdps]], [[value-functions]], [[policy-gradients]]",
    ),
    "policy-gradients": (
        "stochastic policies, score-function gradients, baselines, variance, and advantage estimates",
        "implement a toy REINFORCE update over sampled trajectories",
        "[[mdps]], [[value-functions]], [[rlhf-dpo]]",
    ),
    "rag-architecture": (
        "query analysis, retrieval, reranking, context assembly, generation, and citations",
        "design a toy RAG pipeline over local synthetic documents",
        "[[bm25]], [[semantic-retrieval]], [[rag-evaluation]]",
    ),
    "bm25": (
        "term frequency, inverse document frequency, length normalization, and sparse retrieval",
        "rank synthetic documents with a hand-computed BM25 score",
        "[[rag-architecture]], [[hybrid-fusion]], [[reranking]]",
    ),
    "semantic-retrieval": (
        "dense embeddings, similarity search, recall, indexing, and embedding drift",
        "embed toy passages and compare nearest neighbors",
        "[[tokenization-embeddings]], [[hybrid-fusion]], [[vector-databases]]",
    ),
    "hybrid-fusion": (
        "sparse/dense score fusion, normalization, reciprocal rank fusion, and recall tradeoffs",
        "combine BM25 and dense rankings for ambiguous synthetic queries",
        "[[bm25]], [[semantic-retrieval]], [[reranking]]",
    ),
    "reranking": (
        "cross-encoders, late interaction, pairwise ranking, latency, and top-k tradeoffs",
        "rerank retrieved candidates with a toy relevance function",
        "[[hybrid-fusion]], [[rag-architecture]], [[rag-evaluation]]",
    ),
    "vector-databases": (
        "ANN indexes, metadata filters, persistence, recall/latency, and operational constraints",
        "compare exact and approximate nearest-neighbor behavior conceptually",
        "[[semantic-retrieval]], [[chunking-strategies]], [[rag-evaluation]]",
    ),
    "chunking-strategies": (
        "chunk size, overlap, semantic boundaries, metadata, and context-window economics",
        "chunk synthetic documents and evaluate retrieval misses",
        "[[rag-architecture]], [[vector-databases]], [[rag-evaluation]]",
    ),
    "rag-evaluation": (
        "retrieval recall, groundedness, faithfulness, answer quality, and regression testing",
        "score a synthetic RAG trace for retrieval and answer failures",
        "[[rag-architecture]], [[llm-evals]], [[chunking-strategies]]",
    ),
    "graph-algorithms": (
        "traversal, shortest paths, centrality, connected components, and graph complexity",
        "run BFS/shortest-path reasoning on a toy dependency graph",
        "[[graph-databases]], [[graphrag-agent-memory]], [[rag-architecture]]",
    ),
    "graph-databases": (
        "nodes, relationships, property graphs, queries, indexing, and transactional graph storage",
        "model synthetic entities and write query plans conceptually",
        "[[graph-algorithms]], [[graphrag-agent-memory]], [[agent-memory]]",
    ),
    "graphrag-agent-memory": (
        "entity extraction, relation updates, graph retrieval, memory consolidation, and provenance",
        "design a graph memory from synthetic conversations",
        "[[graph-algorithms]], [[graph-databases]], [[agent-memory]]",
    ),
    "mcp-servers-clients": (
        "tool contracts, resource exposure, client/server boundaries, and safe tool invocation",
        "design a toy MCP server contract for local project search",
        "[[langchain]], [[langgraph]], [[agent-memory]]",
    ),
    "langchain": (
        "chains, tools, retrievers, prompt composition, observability, and framework tradeoffs",
        "outline a LangChain-style retrieval and tool-use flow",
        "[[rag-architecture]], [[mcp-servers-clients]], [[langgraph]]",
    ),
    "langgraph": (
        "state graphs, nodes, edges, checkpoints, control flow, and human-in-the-loop recovery",
        "design a state-machine agent workflow with retry and review paths",
        "[[langchain]], [[agent-memory]], [[mcp-servers-clients]]",
    ),
    "agent-memory": (
        "short-term state, long-term memory, retrieval, summarization, write policies, and forgetting",
        "design a memory store and retrieval policy for a synthetic assistant",
        "[[graphrag-agent-memory]], [[langgraph]], [[rag-architecture]]",
    ),
    "experiment-tracking": (
        "run metadata, parameters, metrics, artifacts, lineage, and reproducibility",
        "design a lightweight experiment tracker over synthetic training runs",
        "[[model-registry]], [[monitoring-drift]], [[cicd-for-ml]]",
    ),
    "model-registry": (
        "model versions, stages, approvals, artifacts, signatures, lineage, and rollback",
        "model a registry promotion flow for candidate and production models",
        "[[experiment-tracking]], [[serving]], [[cicd-for-ml]]",
    ),
    "serving": (
        "online inference APIs, batching, latency budgets, model loading, rollout, and rollback",
        "design a toy model-serving contract with canary and fallback behavior",
        "[[model-registry]], [[monitoring-drift]], [[serverless-inference]]",
    ),
    "monitoring-drift": (
        "data drift, concept drift, prediction monitoring, alerting, and feedback loops",
        "define monitoring checks for synthetic production predictions",
        "[[serving]], [[evaluation-metrics]], [[cicd-for-ml]]",
    ),
    "cicd-for-ml": (
        "pipeline automation, data validation, model tests, deployment gates, and rollback policy",
        "design a CI/CD gate sequence for a synthetic model release",
        "[[experiment-tracking]], [[model-registry]], [[monitoring-drift]]",
    ),
    "terraform-iac": (
        "declarative infrastructure, state, modules, environments, drift, and review workflows",
        "draft an infrastructure-as-code plan for a small model-serving stack",
        "[[docker]], [[kubernetes]], [[autoscaling-cost]]",
    ),
    "docker": (
        "container images, layers, reproducible runtime environments, security, and build artifacts",
        "design a Docker packaging plan for a deterministic inference service",
        "[[serving]], [[kubernetes]], [[cicd-for-ml]]",
    ),
    "kubernetes": (
        "pods, deployments, services, config, scheduling, rollout, and health management",
        "map a model-serving workload onto Kubernetes deployment primitives",
        "[[docker]], [[gpu-provisioning]], [[autoscaling-cost]]",
    ),
    "gpu-provisioning": (
        "accelerator selection, scheduling, memory capacity, utilization, quota, and workload placement",
        "estimate GPU capacity for synthetic training and inference workloads",
        "[[kubernetes]], [[quantization-inference]], [[autoscaling-cost]]",
    ),
    "serverless-inference": (
        "scale-to-zero, cold starts, concurrency, timeout limits, and stateless inference contracts",
        "design a serverless inference path for bursty synthetic traffic",
        "[[serving]], [[autoscaling-cost]], [[quantization-inference]]",
    ),
    "autoscaling-cost": (
        "traffic forecasting, horizontal scaling, queueing, utilization, SLOs, and cost controls",
        "create a cost and scaling model for synthetic inference traffic",
        "[[serving]], [[gpu-provisioning]], [[serverless-inference]]",
    ),
    "design-framework": (
        "requirements gathering, constraints, data flow, serving path, reliability, evaluation, and tradeoff defense",
        "write a complete design rubric and apply it to a small AI product prompt",
        "[[recsys]], [[search-ranking]], [[scaling-inference]]",
    ),
    "recsys": (
        "candidate generation, ranking, features, feedback loops, exploration, and evaluation",
        "design a recommendation system for synthetic user-item events",
        "[[feature-stores]], [[evaluation-metrics]], [[search-ranking]]",
    ),
    "search-ranking": (
        "query understanding, candidate retrieval, ranking features, learning-to-rank, and online evaluation",
        "design a search ranking system for synthetic queries and documents",
        "[[rag-architecture]], [[reranking]], [[feature-stores]]",
    ),
    "feature-stores": (
        "offline/online feature consistency, freshness, point-in-time correctness, and serving contracts",
        "design feature definitions and freshness checks for a synthetic ranking system",
        "[[experiment-tracking]], [[serving]], [[search-ranking]]",
    ),
    "scaling-inference": (
        "latency budgets, batching, caching, model replicas, accelerator use, and degradation strategies",
        "design a scalable inference plan for synthetic burst and steady-state traffic",
        "[[quantization-inference]], [[serving]], [[autoscaling-cost]]",
    ),
}


def _topic_dir(root: Path, section: Section, index: int, topic: Topic) -> Path:
    return root / section.dirname / f"{index:02d}-{topic.slug}"


def _is_placeholder(path: Path) -> bool:
    if not path.exists():
        return True
    text = path.read_text(encoding="utf-8")
    return any(marker in text for marker in STUB_MARKERS)


def _focus(topic: Topic) -> tuple[str, str, str]:
    return TOPIC_FOCUS[topic.slug]


def _section_readme(section: Section) -> str:
    lines = [
        f"# {section.number:02d} — {section.title}",
        "",
        SECTION_FRAMING[section.number],
        "",
        "## Learning Path",
        "",
    ]
    for i, topic in enumerate(section.topics, start=1):
        focus, _, _ = _focus(topic)
        lines.append(f"{i}. [{topic.title}]({i:02d}-{topic.slug}/README.md) — {focus}.")
    lines.extend([
        "",
        "## Section Build Target",
        "",
        SECTION_LABS[section.number],
        "",
        "## Completion Criteria",
        "",
        "- Topic notes explain the core concepts and production tradeoffs.",
        "- Exercises have matching solution outlines.",
        "- Interview prep covers senior-level design and debugging questions.",
        "- Mini-projects are self-contained and do not depend on hidden assets.",
        "- Implementation notebooks and tests are deferred to the next pass.",
        "",
    ])
    return "\n".join(lines)


def _readme(section: Section, topic: Topic) -> str:
    focus, lab, links = _focus(topic)
    return f"""# {topic.title}

> Structure pass: outline ready; full implementation notebook, code module, and tests are deferred.

## Overview

This topic covers {focus}. In the AI engineering curriculum, it connects the preceding foundations to the practical systems work needed to build, evaluate, and operate modern AI applications.

## Learning Objectives

- Define the core vocabulary and data shapes behind {topic.title.lower()}.
- Explain the main algorithmic or systems tradeoffs without relying on framework magic.
- Identify failure modes that show up in real AI engineering work.
- Describe what a from-scratch implementation should validate in a later pass.

## Math / Formalism To Cover

- Core objects, notation, and inputs/outputs.
- The key equation, update rule, scoring function, protocol, or state transition.
- Complexity, scaling, or statistical assumptions where they matter.
- Edge cases that change behavior in production settings.

## Intuition

The working mental model should answer: what signal moves through the system, what gets optimized or retrieved, what state is preserved, and what can break when scale, data quality, or latency constraints change.

## When & Why

Use this topic when you need to reason about {topic.title.lower()} at implementation, evaluation, and system-design levels. The practical emphasis is not only how the method works, but when it is the wrong abstraction and what telemetry would reveal that mismatch.

## Future Implementation

A later implementation pass should {lab}. That pass should validate behavior against small hand-checkable examples before adding larger experiments or framework comparisons.

## Cross-links

Related topics: {links}.

## Resources To Collect

- A primary paper, standard reference, or official documentation page.
- One practical engineering article or framework guide.
- One failure-mode or evaluation reference for production use.
"""


def _exercises(topic: Topic) -> str:
    focus, lab, _ = _focus(topic)
    return f"""# {topic.title} — Exercises

These prompts define the first-pass practice structure. Full worked derivations and executable checks should be expanded during the implementation pass.

## Exercise 1 — Define the contract

List the inputs, outputs, assumptions, and invariants for {topic.title.lower()}. Include the shapes, state, or interface boundaries that an implementation would need to enforce.

## Exercise 2 — Work a tiny example

Create a hand-checkable toy example for {focus}. Compute the key intermediate values manually and note where approximation or implementation details could change the result.

## Exercise 3 — Compare two design choices

Choose two plausible approaches for this topic and compare their tradeoffs in accuracy, latency, memory, data requirements, observability, and failure recovery.

## Exercise 4 — Diagnose a failure

Describe a realistic failure mode for {topic.title.lower()}. Specify the symptoms, the most likely root causes, and the telemetry or tests you would use to distinguish them.

## Exercise 5 — Plan the implementation lab

Turn this build target into a concrete implementation plan: {lab}. Define the files, assertions, plots or tables, and pass/fail checks the later implementation should include.
"""


def _solutions(topic: Topic) -> str:
    focus, lab, _ = _focus(topic)
    return f"""# {topic.title} — Solutions

These are solution outlines for the structure pass. Replace them with fuller worked answers once the implementation notebook and code module exist.

## Solution 1 — Define the contract

A complete answer should name the data model, allowed inputs, expected outputs, state or configuration, and invariants. For {topic.title.lower()}, pay special attention to the assumptions behind {focus}.

## Solution 2 — Work a tiny example

The toy example should be small enough to verify without a framework. It should expose the central computation or system boundary, include at least one edge case, and state which result would be asserted in a future test.

## Solution 3 — Compare two design choices

A strong comparison separates quality, cost, latency, operational complexity, and debugging surface. It should also state which choice is preferred under a concrete constraint rather than claiming one approach is universally better.

## Solution 4 — Diagnose a failure

The diagnosis should connect an observable symptom to at least two plausible causes and then propose a discriminating measurement. Good answers include both data-quality and implementation-quality explanations.

## Solution 5 — Plan the implementation lab

The plan should turn \"{lab}\" into a runnable lab with deterministic toy inputs, explicit assertions, and a short written interpretation. It should avoid hidden datasets or external services unless they are optional extensions.
"""


def _interview(topic: Topic) -> str:
    focus, _, links = _focus(topic)
    return f"""# {topic.title} — Interview Prep

## Q&A Outline

1. **Q: What problem does {topic.title.lower()} solve?**
   **A:** A complete answer should connect the concept to {focus}.

2. **Q: What are the core inputs and outputs?**
   **A:** Name the data, state, configuration, and result contract precisely.

3. **Q: What is the main tradeoff?**
   **A:** Discuss quality, latency, memory, data requirements, and operational complexity.

4. **Q: What is the smallest example you can compute by hand?**
   **A:** Use a toy case that exposes the key mechanism without hiding behind a library.

5. **Q: What breaks at production scale?**
   **A:** Cover data drift, observability, cost, latency, and interface mismatch.

6. **Q: How would you evaluate it?**
   **A:** Include offline metrics, regression checks, and qualitative inspection where appropriate.

7. **Q: What would you implement from scratch first?**
   **A:** Start with deterministic toy inputs and assertions before adding larger experiments.

8. **Q: How does this connect to adjacent topics?**
   **A:** Useful cross-links include {links}.

## Explain it like a principal

At principal level, the answer should go beyond the textbook definition. Explain the abstraction boundary, the operating constraints, the failure modes, and the evidence you would collect before choosing or rejecting this approach in a real AI system.

## Gotchas & follow-ups

- Do not confuse the conceptual method with a framework's default API.
- Do not discuss quality without also discussing cost, latency, and debuggability.
- Do not assume benchmark behavior transfers to a new data distribution.
- Follow-up: what is the first measurement you would add if this component behaved correctly on toy data but failed in a production workflow?
"""


def _mini_project(section: Section, topic: Topic) -> str:
    _, lab, _ = _focus(topic)
    return f"""# {topic.title} — Mini-Project

## Goal

{lab}

## Setup

Use synthetic, locally generated inputs. Do not require external services, hidden datasets, or files that are not created by the learner as part of the project.

## Tasks

1. Define the minimal data model and configuration.
2. Create a hand-checkable toy example.
3. Implement the core computation or workflow in a single script or notebook.
4. Add assertions for the toy example.
5. Write a short analysis of observed behavior, failure modes, and next improvements.

## Expected Outputs

- A small runnable artifact created by the learner.
- A printed table, trace, or metric summary.
- A short writeup explaining whether the result matches the expected behavior.

## Extension Ideas

- Add a harder edge case.
- Compare two design choices.
- Add lightweight observability around intermediate values.
- Connect the result to another topic in `{section.dirname}`.
"""


def _write_if_placeholder(path: Path, content: str) -> bool:
    if not _is_placeholder(path):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def _update_progress(root: Path, sections: set[int]) -> bool:
    path = root / "PROGRESS.md"
    text = path.read_text(encoding="utf-8")
    changed = False
    for section in SECTIONS:
        if section.number not in sections:
            continue
        for topic in section.topics:
            old = f"| {topic.title} | ⬜ | ⬜ | ⬜ | ⬜ |"
            new = f"| {topic.title} | 🟡 | ⬜ | 🟡 | 🟡 |"
            if old in text:
                text = text.replace(old, new)
                changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def build(root: Path, start: int, end: int) -> list[Path]:
    changed: list[Path] = []
    selected = {s.number for s in SECTIONS if start <= s.number <= end}
    for section in SECTIONS:
        if section.number not in selected:
            continue
        sec_path = root / section.dirname / "README.md"
        sec_path.write_text(_section_readme(section), encoding="utf-8")
        changed.append(sec_path)
        for i, topic in enumerate(section.topics, start=1):
            tdir = _topic_dir(root, section, i, topic)
            targets = {
                tdir / "README.md": _readme(section, topic),
                tdir / "exercises.md": _exercises(topic),
                tdir / "solutions" / "solutions.md": _solutions(topic),
                tdir / "interview.md": _interview(topic),
                tdir / "mini_project" / "README.md": _mini_project(section, topic),
            }
            for path, content in targets.items():
                if _write_if_placeholder(path, content):
                    changed.append(path)
    if _update_progress(root, selected):
        changed.append(root / "PROGRESS.md")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    changed = build(root, args.start, args.end)
    for path in changed:
        print(path.relative_to(root))
    print(f"Updated {len(changed)} files")


if __name__ == "__main__":
    main()
