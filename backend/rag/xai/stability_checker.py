# Kuhn et al. (2023) ICLR - "Semantic Uncertainty: Linguistic Invariances for Uncertainty
# Estimation in Natural Language Generation" — semantic clustering of answer variants
# Fomicheva et al. (2020) TACL - consistency under paraphrase as confidence proxy
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

PARAPHRASE_TEMPLATES = [
    "{query}",
    "What is the value of {query}?",
    "Please provide the figures for {query} from the latest report.",
]


def _stable_threshold() -> float:
    raw = os.getenv("XAI_STABILITY_THRESHOLD", "0.82")
    try:
        return float(raw)
    except ValueError:
        return 0.82


@dataclass
class StabilityResult:
    mean_similarity: float
    min_similarity: float
    variance: float
    is_stable: bool
    answer_embeddings: List


class StabilityChecker:
    def __init__(self, embed_fn: Callable[[List[str]], np.ndarray], rag_answer_fn: Callable):
        """
        embed_fn: maps list of answer strings -> (n, dim) float32 numpy array
        rag_answer_fn: fn(query: str) -> str — RAG answer text only (may use LLM for generation).
        """
        self.embed_fn = embed_fn
        self.rag_answer_fn = rag_answer_fn

    def check(self, query: str, n_paraphrases: int = 3) -> StabilityResult:
        thr = _stable_threshold()
        templates = PARAPHRASE_TEMPLATES[: max(1, n_paraphrases)]
        queries = [t.format(query=query) for t in templates]

        answers: List[str] = []
        for q in queries:
            try:
                ans = self.rag_answer_fn(q)
                answers.append(ans or "")
            except Exception:
                answers.append("")

        answers = [a for a in answers if a.strip()]
        if len(answers) < 2:
            return StabilityResult(0.0, 0.0, 1.0, False, [])

        embeddings = np.asarray(self.embed_fn(answers), dtype=np.float64)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(answers):
            return StabilityResult(0.0, 0.0, 1.0, False, [])

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / (norms + 1e-9)
        sim_matrix = normed @ normed.T

        n = len(answers)
        pairs = [float(sim_matrix[i, j]) for i in range(n) for j in range(i + 1, n)]

        mean_sim = float(np.mean(pairs))
        min_sim = float(np.min(pairs))
        variance = float(np.var(pairs))

        return StabilityResult(
            mean_similarity=mean_sim,
            min_similarity=min_sim,
            variance=variance,
            is_stable=mean_sim >= thr,
            answer_embeddings=embeddings.tolist(),
        )
