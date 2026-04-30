"""Paraphrase resistance: cosine + citation overlap + verdict agreement (Kuhn et al. 2023)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, List

import numpy as np

PARAPHRASE_TEMPLATES = [
    "{query}",
    "What is {query}?",
    "Please provide the figures for {query} from the latest RBI report.",
    "According to RBI data, what are the details on {query}?",
    "Can you tell me the latest numbers for {query}?",
]

STABLE_THRESHOLD = float(os.getenv("XAI_STABILITY_THRESHOLD", "0.82"))


@dataclass
class ResistanceResult:
    mean_similarity: float
    min_similarity: float
    citation_overlap_score: float
    verdict_agreement_score: float
    is_stable: bool
    num_variants: int
    verdicts: List[str]


class ParaphraseResistanceTester:
    def __init__(self, embedder: Any, full_pipeline_fn: Callable[[str], Any]):
        self.embedder = embedder
        self.pipeline_fn = full_pipeline_fn

    def test(self, query: str, n: int = 5) -> ResistanceResult:
        queries = [t.format(query=query) for t in PARAPHRASE_TEMPLATES[: max(1, n)]]
        results = []
        for q in queries:
            try:
                results.append(self.pipeline_fn(q))
            except Exception:
                pass

        if len(results) < 2:
            return ResistanceResult(0.0, 0.0, 0.0, 0.0, False, len(results), [])

        answers = [r.artifact.get("final_answer", "") for r in results]
        embeddings = self.embedder.encode(answers)
        emb = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        normed = emb / (norms + 1e-9)
        sim_matrix = normed @ normed.T
        n_r = len(results)
        pairs = [float(sim_matrix[i, j]) for i in range(n_r) for j in range(i + 1, n_r)]
        mean_sim = float(np.mean(pairs))
        min_sim = float(np.min(pairs))

        all_source_sets = []
        for r in results:
            docs = set(
                s.get("doc_id")
                for s in r.artifact.get("retrieval_explanation", [])
                if s.get("doc_id")
            )
            all_source_sets.append(docs)
        if all_source_sets:
            common = set.intersection(*all_source_sets)
            union = set.union(*all_source_sets)
            citation_overlap = len(common) / len(union) if union else 1.0
        else:
            citation_overlap = 0.0

        verdicts = [r.verdict.gate.value for r in results]
        most_common = max(set(verdicts), key=verdicts.count)
        verdict_agreement = verdicts.count(most_common) / len(verdicts)

        is_stable = (
            mean_sim >= STABLE_THRESHOLD
            and verdict_agreement >= 0.8
            and citation_overlap >= 0.5
        )

        return ResistanceResult(
            mean_similarity=mean_sim,
            min_similarity=min_sim,
            citation_overlap_score=citation_overlap,
            verdict_agreement_score=verdict_agreement,
            is_stable=is_stable,
            num_variants=len(results),
            verdicts=verdicts,
        )
