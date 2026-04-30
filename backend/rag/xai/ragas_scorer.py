"""Deterministic RAGAS-style proxies (no LLM judge). Es et al. 2023 framing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RAGASScorecard:
    context_relevance: float
    faithfulness: float
    citation_precision: float
    conflict_risk: float
    paraphrase_stability: float
    overall_trust_score: float


WEIGHTS = {
    "context_relevance": 0.20,
    "faithfulness": 0.35,
    "citation_precision": 0.20,
    "conflict_risk": 0.10,
    "paraphrase_stability": 0.15,
}


class RAGASScorer:
    def __init__(self, embedder: Any):
        self.embedder = embedder

    def score(
        self,
        query: str,
        chunks: List[Dict],
        nli_result: Any,
        conflict_report: Any,
        attributions: List[Any],
        resistance_result: Optional[Any] = None,
    ) -> RAGASScorecard:
        context_relevance = self._context_relevance(query, chunks)
        faithfulness = float(getattr(nli_result, "answer_entailment_score", 0.0))
        citation_precision = self._citation_precision(chunks, attributions)
        conflict_risk = self._conflict_risk(conflict_report)
        stability = (
            float(resistance_result.mean_similarity)
            if resistance_result is not None
            else 1.0
        )

        overall = (
            WEIGHTS["context_relevance"] * context_relevance
            + WEIGHTS["faithfulness"] * faithfulness
            + WEIGHTS["citation_precision"] * citation_precision
            + WEIGHTS["conflict_risk"] * (1.0 - conflict_risk)
            + WEIGHTS["paraphrase_stability"] * stability
        )

        return RAGASScorecard(
            context_relevance=round(context_relevance, 4),
            faithfulness=round(faithfulness, 4),
            citation_precision=round(citation_precision, 4),
            conflict_risk=round(conflict_risk, 4),
            paraphrase_stability=round(stability, 4),
            overall_trust_score=round(overall, 4),
        )

    def _context_relevance(self, query: str, chunks: List[Dict]) -> float:
        if not chunks:
            return 0.0
        qe = self.embedder.encode(query)
        texts = [c.get("text", "")[:200] for c in chunks]
        ce = self.embedder.encode(texts)
        qn = np.asarray(qe, dtype=np.float64).ravel()
        qn = qn / (np.linalg.norm(qn) + 1e-9)
        C = np.asarray(ce, dtype=np.float64)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        sims = cn @ qn
        return float(np.mean(sims))

    def _citation_precision(self, chunks: List[Dict], attributions: List[Any]) -> float:
        if not chunks or not attributions:
            return 0.0
        attributed_docs = set(
            a.source_doc_id for a in attributions if a.is_attributed and a.source_doc_id
        )
        retrieved_docs = set(c.get("doc_id") for c in chunks if c.get("doc_id"))
        if not retrieved_docs:
            return 0.0
        return len(attributed_docs & retrieved_docs) / len(retrieved_docs)

    def _conflict_risk(self, conflict_report: Any) -> float:
        n = len(getattr(conflict_report, "conflicts", []) or [])
        return min(1.0, n * 0.25)
