"""Single source of truth for the curriculum: sections and their topics."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    """One learning topic within a section."""

    slug: str
    title: str


@dataclass(frozen=True)
class Section:
    """One numbered section of the curriculum."""

    number: int
    slug: str
    title: str
    topics: tuple[Topic, ...]

    @property
    def dirname(self) -> str:
        """Zero-padded directory name, e.g. ``04-machine-learning``."""
        return f"{self.number:02d}-{self.slug}"


def _t(slug: str, title: str) -> Topic:
    return Topic(slug=slug, title=title)


SECTIONS: tuple[Section, ...] = (
    Section(1, "linear-algebra", "Linear Algebra", (
        _t("matrix-multiplication", "Matrix Multiplication"),
        _t("broadcasting", "Broadcasting"),
        _t("vectors-and-norms", "Vectors and Norms"),
        _t("eigendecomposition-svd", "Eigendecomposition and SVD"),
    )),
    Section(2, "calculus", "Calculus", (
        _t("derivatives-and-partials", "Derivatives and Partials"),
        _t("chain-rule", "The Chain Rule"),
        _t("gradients-and-jacobians", "Gradients and Jacobians"),
    )),
    Section(3, "statistics", "Statistics", (
        _t("probability-basics", "Probability Basics"),
        _t("gaussian-distribution", "Gaussian Distribution"),
        _t("mle-and-nll", "MLE and Negative Log-Likelihood"),
        _t("bayesian-statistics", "Bayesian Statistics"),
        _t("bias-variance", "Bias-Variance Tradeoff"),
    )),
    Section(4, "machine-learning", "Machine Learning", (
        _t("gradient-descent", "Gradient Descent"),
        _t("loss-functions", "Loss Functions"),
        _t("l1-l2-regularization", "L1 and L2 Regularization"),
        _t("adam-optimization", "Adam Optimization"),
        _t("evaluation-metrics", "Evaluation Metrics"),
        _t("linear-regression", "Linear Regression"),
        _t("logistic-regression", "Logistic Regression"),
        _t("decision-trees", "Decision Trees"),
        _t("random-forests-xgboost", "Random Forests and XGBoost"),
        _t("svm", "Support Vector Machines"),
        _t("kmeans", "K-Means Clustering"),
        _t("pca", "Principal Component Analysis"),
        _t("knn-naive-bayes", "kNN and Naive Bayes"),
    )),
    Section(5, "deep-learning", "Deep Learning", (
        _t("broadcasting-in-nns", "Broadcasting in Neural Networks"),
        _t("backpropagation", "Backpropagation"),
        _t("softmax", "Softmax"),
        _t("cross-entropy-nll", "Cross-Entropy and NLL Loss"),
        _t("activations-tanh-relu", "Activations: tanh and ReLU"),
        _t("vanishing-exploding-gradients", "Vanishing and Exploding Gradients"),
        _t("batch-normalization", "Batch Normalization"),
        _t("label-smoothing", "Label Smoothing"),
        _t("cnns", "Convolutional Neural Networks"),
    )),
    Section(6, "llm-engineering", "LLM Engineering", (
        _t("transformers-attention", "Transformers and Attention"),
        _t("tokenization-embeddings", "Tokenization and Embeddings"),
        _t("fine-tuning-lora-peft", "Fine-Tuning (LoRA/PEFT)"),
        _t("rlhf-dpo", "RLHF and DPO"),
        _t("quantization-inference", "Quantization and Inference Optimization"),
        _t("llm-evals", "LLM Evals"),
    )),
    Section(7, "reinforcement-learning", "Reinforcement Learning", (
        _t("mdps", "Markov Decision Processes"),
        _t("value-functions", "Value Functions"),
        _t("q-learning", "Q-Learning"),
        _t("policy-gradients", "Policy Gradients"),
    )),
    Section(8, "retrieval-rag", "Retrieval and RAG", (
        _t("rag-architecture", "RAG Architecture"),
        _t("bm25", "BM25"),
        _t("semantic-retrieval", "Semantic Retrieval"),
        _t("hybrid-fusion", "Hybrid Fusion"),
        _t("reranking", "Reranking"),
        _t("vector-databases", "Vector Databases"),
        _t("chunking-strategies", "Chunking Strategies"),
        _t("rag-evaluation", "RAG Evaluation"),
    )),
    Section(9, "graphs", "Graphs", (
        _t("graph-algorithms", "Graph Algorithms"),
        _t("graph-databases", "Graph Databases"),
        _t("graphrag-agent-memory", "GraphRAG and Agent Memory"),
    )),
    Section(10, "agentic-systems", "Agentic Systems", (
        _t("mcp-servers-clients", "MCP Servers and Clients"),
        _t("langchain", "LangChain"),
        _t("langgraph", "LangGraph"),
        _t("agent-memory", "Memory for AI Agents"),
    )),
    Section(11, "mlops", "MLOps", (
        _t("experiment-tracking", "Experiment Tracking"),
        _t("model-registry", "Model Registry"),
        _t("serving", "Model Serving"),
        _t("monitoring-drift", "Monitoring and Drift"),
        _t("cicd-for-ml", "CI/CD for ML"),
    )),
    Section(12, "cloud-infrastructure", "Cloud and Infrastructure", (
        _t("terraform-iac", "Terraform and IaC"),
        _t("docker", "Docker"),
        _t("kubernetes", "Kubernetes"),
        _t("gpu-provisioning", "GPU Provisioning"),
        _t("serverless-inference", "Serverless Inference"),
        _t("autoscaling-cost", "Autoscaling and Cost"),
    )),
    Section(13, "ai-system-design", "AI System Design", (
        _t("design-framework", "Design Framework"),
        _t("recsys", "Recommendation Systems"),
        _t("search-ranking", "Search Ranking"),
        _t("feature-stores", "Feature Stores"),
        _t("scaling-inference", "Scaling Inference"),
    )),
)
