from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple


def _chunk_lineage_fields(c: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Any]:
    md = c.get("metadata") or {}
    doc_id = c.get("doc_id") if c.get("doc_id") is not None else md.get("doc_id")
    edition_date = c.get("edition_date") if c.get("edition_date") is not None else md.get("edition_date")
    report_type = c.get("report_type") if c.get("report_type") is not None else md.get("report_type")
    section = c.get("section") if c.get("section") is not None else md.get("section_title") or md.get("section_id")
    page = c.get("page") if c.get("page") is not None else c.get("page_number")
    return doc_id, edition_date, report_type, section, page


def build_version_history(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Distinct regulatory editions / documents represented in retrieved context."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for c in chunks:
        doc_id, edition_date, report_type, section, _ = _chunk_lineage_fields(c)
        key = (str(doc_id or ""), str(edition_date or ""), str(report_type or ""))
        if key in seen:
            continue
        if not any(key):
            continue
        seen.add(key)
        label = doc_id or (f"{report_type or 'Document'} ({edition_date or 'n/a'})")
        out.append(
            {
                "document_label": label,
                "doc_id": doc_id,
                "report_type": report_type,
                "edition_date": edition_date,
                "sample_section": section,
            }
        )
    return out


def build_supporting_evidence(chunks: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    """Retrieved passages the RAG layer used (excerpts + metadata)."""
    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(chunks[:limit]):
        doc_id, edition_date, report_type, section, page = _chunk_lineage_fields(c)
        text = (c.get("text") or "").strip()
        rows.append(
            {
                "rank": i + 1,
                "doc_id": doc_id,
                "report_type": report_type,
                "edition_date": edition_date,
                "section": section,
                "page": page,
                "rerank_score": c.get("rerank_score"),
                "excerpt": text[:600] + ("…" if len(text) > 600 else ""),
            }
        )
    return rows


def build_artifact(
    query: str,
    answer: str,
    chunks: List[Dict],
    claims: List[Any],
    attributions: List[Any],
    nli_result: Any,
    conflict_report: Any,
    verdict: Any,
    *,
    ragas_scorecard: Any = None,
    resistance_result: Any = None,
) -> Dict[str, Any]:
    attr_map = {a.claim_id: a for a in attributions}
    nli_map = {v.claim_id: v for v in nli_result.verdicts}

    claims_out = []
    for i, claim in enumerate(claims):
        attr = attr_map.get(claim.id)
        nv = nli_map.get(claim.id)
        cs = None
        if i < len(verdict.per_claim_scores):
            cs = round(verdict.per_claim_scores[i], 4)
        claims_out.append(
            {
                "id": claim.id,
                "text": claim.text,
                "is_numerical": claim.is_numerical,
                "entity": claim.entity,
                "metric": claim.metric,
                "attribution": (
                    {
                        "is_attributed": attr.is_attributed,
                        "source_doc": attr.source_doc_id,
                        "source_section": attr.source_section,
                        "source_page": attr.source_page,
                        "similarity_score": round(attr.similarity_score, 4),
                        "passage_excerpt": (attr.source_passage or "")[:150],
                    }
                    if attr
                    else None
                ),
                "nli": (
                    {
                        "label": nv.label,
                        "confidence": round(nv.confidence, 4),
                        "stage": nv.stage,
                        "reasoning": nv.reasoning,
                        "second_pass_used": nv.second_pass_used,
                        "is_hallucination": nv.is_hallucination,
                    }
                    if nv
                    else None
                ),
                "claim_score": cs,
            }
        )

    art: Dict[str, Any] = {
        "query": query,
        "trust_gate": verdict.gate.value,
        "confidence": round(verdict.confidence, 4),
        "reasoning": verdict.reasoning,
        "weakest_claim": verdict.weakest_claim,
        "hallucination_detected": verdict.hallucination_detected,
        "retrieval_explanation": [
            {
                "rank": i + 1,
                "doc_id": c.get("doc_id"),
                "section": c.get("section"),
                "page": c.get("page"),
                "rerank_score": c.get("rerank_score"),
                "edition_date": c.get("edition_date"),
                "why_retrieved": (
                    f"Cross-encoder score {c.get('rerank_score', 0) or 0:.3f}; "
                    f"RRF rank #{i + 1} from hybrid Qdrant+ES fusion"
                ),
            }
            for i, c in enumerate(chunks)
        ],
        "claims": claims_out,
        "edition_conflict": {
            "method": getattr(conflict_report, "resolution_method", "regex_fallback"),
            "has_conflict": conflict_report.has_conflict,
            "num_conflicts": len(conflict_report.conflicts),
            "conflicts": conflict_report.conflicts,
            "superseded_chunks": conflict_report.superseded_chunks,
            "recommended_edition": conflict_report.recommended_edition,
            "complementary_updates": getattr(conflict_report, "complementary_updates", []),
        },
        "temporal_completeness": {
            "completeness_score": round(getattr(verdict, "completeness_score", 1.0), 4),
            "num_complementary_ignored": getattr(verdict, "num_complementary_ignored", 0),
        },
        "trust_breakdown": {
            "raw_score": round(verdict.raw_score, 4),
            "unattributed_penalty": round(getattr(verdict, "penalty_unattributed", 0.0), 4),
            "conflict_penalty": round(getattr(verdict, "penalty_conflict", 0.0), 4),
            "completeness_penalty": round(getattr(verdict, "penalty_completeness", 0.0), 4),
            "rag_context_boost": round(getattr(verdict, "rag_prior_boost", 0.0), 4),
            "linear_score": round(getattr(verdict, "linear_score", 0.0), 4),
            "final_score": round(verdict.confidence, 4),
            "frac_unattributed": round(verdict.frac_unattributed, 4),
        },
        "version_history": build_version_history(chunks),
        "supporting_evidence": build_supporting_evidence(chunks),
        "failure_analysis": verdict.failure_analysis,
        "recommendations": verdict.recommendations,
        "final_answer": answer,
        "complementary_updates": getattr(conflict_report, "complementary_updates", []),
    }

    if ragas_scorecard is not None:
        art["ragas_scorecard"] = {
            "context_relevance": ragas_scorecard.context_relevance,
            "faithfulness": ragas_scorecard.faithfulness,
            "citation_precision": ragas_scorecard.citation_precision,
            "conflict_risk": ragas_scorecard.conflict_risk,
            "paraphrase_stability": ragas_scorecard.paraphrase_stability,
            "overall_trust_score": ragas_scorecard.overall_trust_score,
        }

    if resistance_result is not None:
        art["paraphrase_resistance"] = {
            "mean_similarity": resistance_result.mean_similarity,
            "min_similarity": resistance_result.min_similarity,
            "citation_overlap_score": resistance_result.citation_overlap_score,
            "verdict_agreement_score": resistance_result.verdict_agreement_score,
            "is_stable": resistance_result.is_stable,
            "num_variants": resistance_result.num_variants,
            "verdicts": resistance_result.verdicts,
        }

    return art


def print_artifact(artifact: Dict[str, Any]) -> None:
    print("\n" + "=" * 70, flush=True)
    print(
        f"  TRUST GATE : {artifact['trust_gate']}  |  Score: {artifact['confidence']:.2%}",
        flush=True,
    )
    print(
        f"  HALLUCINATION: {'YES' if artifact.get('hallucination_detected') else 'NO'}",
        flush=True,
    )
    print("=" * 70, flush=True)
    print(f"  {artifact['reasoning']}", flush=True)
    
    inf = artifact.get("inferred_findings")
    if inf:
        print("\n  " + "!" * 10 + " INFERRED REGULATORY REASONING " + "!" * 10, flush=True)
        print(f"  EXPLICIT FINDING:", flush=True)
        print(f"    {inf.get('explicit_finding')}", flush=True)
        print(f"\n  INFERRED REQUIREMENT:", flush=True)
        print(f"    {inf.get('inferred_requirement')}", flush=True)
        print(f"\n  Reasoning:", flush=True)
        for step in inf.get("reasoning_steps", []):
            print(f"    - {step}", flush=True)
        print("  " + "!" * 51 + "\n", flush=True)
    
    fa = artifact.get("failure_analysis", [])
    if fa:
        print("\n  FAILURE ANALYSIS:", flush=True)
        for f in fa:
            print(f"    - {f}", flush=True)
            
    recs = artifact.get("recommendations", [])
    if recs:
        print("\n  RECOMMENDED ACTIONS:", flush=True)
        for r in recs:
            print(f"    - {r}", flush=True)

    print(flush=True)
    for c in artifact.get("claims", []):
        nli = c.get("nli") or {}
        attr = c.get("attribution") or {}
        icon = "(V)" if nli.get("label") == "entailment" else "(X)"
        attrib_icon = "A" if attr.get("is_attributed") else "U"
        txt = (c.get("text") or "")[:80]
        print(
            f"  [{attrib_icon}]{icon} [{nli.get('confidence', 0):.2f}|S{nli.get('stage', 0)}] {txt}",
            flush=True,
        )
        if attr.get("source_doc"):
            print(
                f"      -> {attr['source_doc']} p.{attr.get('source_page')} "
                f"sim={attr.get('similarity_score', 0):.3f}",
                flush=True,
            )
    print(flush=True)
    ed = artifact.get("edition_conflict", {})
    if ed.get("has_conflict"):
        print(f"  EDITION CONFLICTS: {ed.get('num_conflicts', 0)}", flush=True)
        for cf in ed.get("conflicts", []):
            stale = " [ANSWER STALE]" if cf.get("answer_uses_older") else ""
            print(
                f"    ! {cf.get('metric')} {cf.get('entity')}: "
                f"{cf.get('older_edition')}={cf.get('older_value')} "
                f"vs {cf.get('newer_edition')}={cf.get('newer_value')}{stale}",
                flush=True,
            )

    tc = artifact.get("temporal_completeness", {})
    if tc.get("num_complementary_ignored", 0) > 0:
        print(f"\n  TEMPORAL INCOMPLETENESS DETECTED:", flush=True)
        print(f"    Completeness score: {tc.get('completeness_score', 1.0):.2%}", flush=True)
        comp_updates = artifact.get("complementary_updates", [])
        for cu in comp_updates:
            if not cu.get("answer_includes_newer", True):
                print(
                    f"    >> IGNORED: {cu.get('newer_edition')} data not in answer "
                    f"(type: {cu.get('classification', 'COMPLEMENTARY')})",
                    flush=True,
                )

    tb = artifact.get("trust_breakdown", {})
    print(
        f"\n  Score breakdown: raw={tb.get('raw_score', 0):.3f} "
        f"- unattrib={tb.get('unattributed_penalty', 0):.3f} "
        f"- conflict={tb.get('conflict_penalty', 0):.3f} "
        f"- completeness={tb.get('completeness_penalty', 0):.3f} "
        f"+ rag_prior={tb.get('rag_context_boost', 0):.3f} "
        f"=> linear={tb.get('linear_score', 0):.3f} "
        f"=> normalized={tb.get('final_score', 0):.3f}",
        flush=True,
    )
    if artifact.get("ragas_scorecard"):
        r = artifact["ragas_scorecard"]
        print(
            f"\n  RAGAS-style card: overall={r.get('overall_trust_score')} "
            f"(faithfulness={r.get('faithfulness')}, context={r.get('context_relevance')})",
            flush=True,
        )
    print("=" * 70 + "\n", flush=True)



def write_artifact_json(path: str, artifact: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False, default=str)


def maybe_write_artifact_json(artifact: Dict[str, Any]) -> None:
    path = (os.getenv("XAI_ARTIFACT_JSON") or "").strip()
    if path:
        write_artifact_json(path, artifact)


def format_analytics_report(artifact: Dict[str, Any]) -> str:
    """Compatibility: formatted dump of trust + claims + breakdown."""
    lines = []
    lines.append(str(artifact.get("reasoning", "")))
    for c in artifact.get("claims", []):
        lines.append(str(c))
    return "\n".join(lines)
