from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .aggregator import AggregatedVerdict, aggregate
from .artifact import build_artifact
from .claim_splitter import AtomicClaimSplitter
from .edition_conflict import EditionConflictDetector
from .nli_verifier import TwoStageNLIVerifier
from .attribution_engine import AttributionEngine


@dataclass
class XAIResult:
    verdict: AggregatedVerdict
    artifact: Dict[str, Any]


class XAIPipeline:
    def __init__(
        self,
        embed_fn: Callable[[str], Any],
        qdrant_client: Any,
        collection_name: str,
        llm_fn: Optional[Callable[[str], str]] = None,
        second_pass_fn: Optional[Callable[[str], List[str]]] = None,
        db_conn: Any = None,
        run_ragas: Optional[bool] = None,
    ):
        self._embed_fn = embed_fn
        self.splitter = AtomicClaimSplitter()
        self.attribution = AttributionEngine(embed_fn, qdrant_client, collection_name)
        self.nli = TwoStageNLIVerifier(llm_fn=llm_fn, second_pass_fn=second_pass_fn)
        self.edition = EditionConflictDetector(
            db_conn=db_conn,
            nli_fn=lambda p, h: self.nli.stage1_label_scores(p, h),
        )
        self.llm_fn = llm_fn
        self._run_ragas = (
            run_ragas
            if run_ragas is not None
            else os.getenv("XAI_RAGAS", "0").lower() in ("1", "true", "yes")
        )

    def run(self, query: str, answer: str, chunks: List[Dict]) -> XAIResult:
        fast = os.getenv("XAI_FAST", "0").lower() in ("1", "true", "yes")
        run_ragas_effective = bool(self._run_ragas) and not fast
        decompose_llm = None
        if not fast and os.getenv("XAI_CLAIM_LLM", "1").lower() in ("1", "true", "yes"):
            decompose_llm = self.llm_fn
        claims = self.splitter.split(answer, llm_fn=decompose_llm)
        if not claims:
            from .claim_splitter import Claim

            txt = (answer or "").strip() or "."
            claims = [
                Claim(
                    id=0,
                    text=txt[:4000],
                    is_numerical=False,
                    entity=None,
                    metric=None,
                    original_sentence=txt,
                )
            ]
        attributions = self.attribution.attribute_all(claims)
        nli_result = self.nli.verify_all(claims, attributions)
        conflict_report = self.edition.detect(chunks, answer)
        verdict = aggregate(nli_result, conflict_report, attributions)

        ragas_sc = None
        resistance = None
        if run_ragas_effective:
            try:
                from .ragas_scorer import RAGASScorer

                ragas_sc = RAGASScorer(self._embed_adapter()).score(
                    query,
                    chunks,
                    nli_result,
                    conflict_report,
                    attributions,
                    resistance_result=None,
                )
            except Exception:
                pass

        artifact = build_artifact(
            query,
            answer,
            chunks,
            claims,
            attributions,
            nli_result,
            conflict_report,
            verdict,
            ragas_scorecard=ragas_sc,
            resistance_result=resistance,
        )
        return XAIResult(verdict=verdict, artifact=artifact)

    def _embed_adapter(self):
        class _E:
            def __init__(self, fn: Callable[[str], Any]):
                self._fn = fn

            def encode(self, x: Any):
                from pipeline.embedder import encode_texts_for_xai
                import numpy as np

                if isinstance(x, str):
                    v = self._fn(x)
                    return np.asarray(v, dtype=np.float32)
                return encode_texts_for_xai(list(x))

        return _E(self._embed_fn)
