"""
Per-claim dense retrieval against Qdrant (Karpukhin et al. 2020 DPR-style).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

ATTRIBUTION_THRESHOLD = float(os.getenv("XAI_ATTRIBUTION_THRESHOLD", "0.78"))
TOP_K_CANDIDATES = int(os.getenv("XAI_ATTRIBUTION_TOP_K", "3"))


def _doc_id_from_payload(payload: dict) -> str:
    rt = (payload.get("report_type") or "").strip()
    ed = (payload.get("edition_date") or "").strip()
    if rt and ed and str(ed).lower() != "unknown":
        return f"{rt} ({ed})"
    return rt or ""


def _section_from_payload(payload: dict) -> str:
    sid = (payload.get("section_id") or "").strip()
    st = (payload.get("section_title") or "").strip()
    if sid and st:
        return f"{sid} {st}"
    return sid or st or ""


@dataclass
class Attribution:
    claim_id: int
    claim_text: str
    is_attributed: bool
    source_doc_id: Optional[str]
    source_section: Optional[str]
    source_passage: Optional[str]
    source_page: Optional[int]
    similarity_score: float
    edition_date: Optional[str]


class AttributionEngine:
    def __init__(self, embed_fn, qdrant_client: Any, collection_name: str):
        """
        embed_fn: callable taking one str -> 1d float32 numpy vector (same space as index).
        qdrant_client: QdrantClient with ``query_points``.
        """
        self.embed_fn = embed_fn
        self.qdrant = qdrant_client
        self.collection = collection_name

    def attribute_all(self, claims: List[Any]) -> List[Attribution]:
        if not claims:
            return []
        by_id: dict = {}
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {ex.submit(self._attribute_one, c): c.id for c in claims}
            for fut in as_completed(futs):
                cid = futs[fut]
                by_id[cid] = fut.result()
        return [by_id[c.id] for c in sorted(claims, key=lambda x: x.id)]

    def _attribute_one(self, claim: Any) -> Attribution:
        try:
            claim_vector = self.embed_fn(claim.text)
            if hasattr(claim_vector, "tolist"):
                qv = claim_vector.tolist()
            else:
                qv = list(claim_vector)

            try:
                from pipeline.chunk_status_filter import qdrant_exclude_superseded_filter
            except ImportError:
                qdrant_exclude_superseded_filter = lambda: None  # type: ignore
            qf = qdrant_exclude_superseded_filter()
            qkwargs = {
                "collection_name": self.collection,
                "query": qv,
                "limit": TOP_K_CANDIDATES,
                "with_payload": True,
            }
            if qf is not None:
                qkwargs["query_filter"] = qf
            res = self.qdrant.query_points(**qkwargs)
            hits = list(res.points) if res and res.points else []
            if not hits:
                return self._unattributed(claim)

            # --- Top-2 Aggregation Logic ---
            best = hits[0]
            score = float(best.score or 0.0)
            
            # If top-1 is weak, try aggregating with top-2
            if score < ATTRIBUTION_THRESHOLD and len(hits) >= 2:
                next_best = hits[1]
                next_score = float(next_best.score or 0.0)
                
                # Check if they are from the same document/section (likely contiguous)
                same_doc = _doc_id_from_payload(best.payload) == _doc_id_from_payload(next_best.payload)
                if same_doc:
                    # Aggregated view
                    combined_text = (best.payload.get("text") or "")[:300] + " [...] " + (next_best.payload.get("text") or "")[:300]
                    # Since we can't easily re-embed the "combined" text here without overhead, 
                    # we boost the score slightly if both are somewhat relevant
                    boosted_score = score + (next_score * 0.1) 
                    
                    if boosted_score >= ATTRIBUTION_THRESHOLD:
                        return Attribution(
                            claim_id=claim.id,
                            claim_text=claim.text,
                            is_attributed=True,
                            source_doc_id=_doc_id_from_payload(best.payload),
                            source_section=_section_from_payload(best.payload),
                            source_passage=combined_text,
                            source_page=best.payload.get("page_number"),
                            similarity_score=boosted_score,
                            edition_date=best.payload.get("edition_date"),
                        )

            if score < ATTRIBUTION_THRESHOLD:
                return self._unattributed(claim, best_score=score)

            payload = best.payload or {}
            text = (payload.get("text") or "")[:300]
            return Attribution(
                claim_id=claim.id,
                claim_text=claim.text,
                is_attributed=True,
                source_doc_id=_doc_id_from_payload(payload),
                source_section=_section_from_payload(payload),
                source_passage=text,
                source_page=payload.get("page_number"),
                similarity_score=score,
                edition_date=payload.get("edition_date"),
            )
        except Exception:
            return self._unattributed(claim)


    def _unattributed(self, claim: Any, best_score: float = 0.0) -> Attribution:
        return Attribution(
            claim_id=claim.id,
            claim_text=claim.text,
            is_attributed=False,
            source_doc_id=None,
            source_section=None,
            source_passage=None,
            source_page=None,
            similarity_score=best_score,
            edition_date=None,
        )

    def unattributed_fraction(self, attributions: List[Attribution]) -> float:
        if not attributions:
            return 1.0
        return sum(1 for a in attributions if not a.is_attributed) / len(attributions)
