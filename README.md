# AI Engineering Mastery

A build-it-yourself curriculum for the math, modeling, and systems behind modern AI — implemented **from scratch in PyTorch**, then contrasted with the idiomatic production API. Built as deep interview prep for **principal / lead AI Engineering** roles, and shared openly in case it's useful to anyone walking a similar path.

> 🚧 **Status: under active construction.** The structure and learning roadmap are set; topics are being filled in one at a time. The first fully built topic is the gradient-descent exemplar in `04-machine-learning/` — every other topic follows the same template.

## Philosophy

- **Build it from scratch first.** You don't really understand backprop, attention, or Adam until you've implemented them by hand. Each core topic derives the math and codes it on raw `torch.Tensor`s (no `.backward()` for the thing being taught), then **validates against `torch.autograd`**.
- **Then show the production way.** Where it helps, the from-scratch version sits side by side with the idiomatic PyTorch equivalent — because that's what the job actually uses.
- **Interview-ready, not just notebook-ready.** Every topic carries an interview Q&A bank and "explain it like a principal" talking points.

## Curriculum

Sections are numbered as a default learning path: foundations → core ML/DL → modern AI systems → production → capstone.

| # | Section | Focus |
|---|---------|-------|
| 01 | Linear Algebra | matrix multiplication, broadcasting, norms, eigendecomposition/SVD |
| 02 | Calculus | derivatives & partials, the chain rule, gradients & Jacobians |
| 03 | Statistics | Gaussian distribution, MLE & negative log-likelihood, Bayesian stats, bias–variance |
| 04 | Machine Learning | **gradient descent (exemplar)**, loss functions, L1/L2, Adam, regression, trees/XGBoost, SVM, k-means, PCA |
| 05 | Deep Learning | backprop, softmax, cross-entropy, tanh/ReLU, vanishing/exploding gradients, batch norm, CNNs |
| 06 | LLM Engineering | transformers & attention, tokenization & embeddings, fine-tuning (LoRA/PEFT), RLHF/DPO, quantization & inference, evals |
| 07 | Reinforcement Learning | MDPs, value functions, Q-learning, policy gradients |
| 08 | Retrieval & RAG | BM25, semantic retrieval, hybrid fusion, reranking, vector databases, chunking, RAG evaluation |
| 09 | Graphs | graph algorithms, graph databases, GraphRAG / agent-memory graphs |
| 10 | Agentic Systems | MCP servers/clients, LangChain, LangGraph, agent memory |
| 11 | MLOps | experiment tracking, model registry, serving, monitoring/drift, CI/CD for ML |
| 12 | Cloud & Infrastructure | Terraform/IaC, Docker, Kubernetes, GPU provisioning, serverless inference, autoscaling, cost |
| 13 | AI System Design | design framework + worked cases (recsys, search ranking, feature stores, scaling inference) — capstone |

## How each topic is structured

Every topic folder follows one repeatable template:

```
NN-topic-name/
  README.md          theory + math/derivation + intuition + when/why + resources
  implementation.*   from-scratch PyTorch (math topics → notebook; systems topics → tested module)
  interview.md       Q&A bank + principal-level talking points + gotchas
  exercises.md       practice problems (worked solutions in solutions/)
  mini_project/      optional small "build it" applying the concept
```

## Tech stack

Python · **PyTorch** · Jupyter · pytest · [uv](https://github.com/astral-ai/uv) for packaging · ruff. Code is device-aware (`cpu` / `cuda` / `mps`).

## Getting started

```bash
uv sync          # install dependencies
jupyter lab      # open the notebooks
pytest           # run tests for the systems sections
```

Track progress in [`PROGRESS.md`](PROGRESS.md) and see the full learning path — with the prerequisite graph and milestone phases — in [`ROADMAP.md`](ROADMAP.md).

## License

[MIT](LICENSE) — free to use, learn from, and adapt.
