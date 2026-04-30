from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class TrustGate(str, Enum):
    SAFE = "Safe"
    NEEDS_REVIEW = "Needs Human Review"
    NON_COMPLIANT = "Non-Compliant"


@dataclass
class AggregatedVerdict:
    gate: TrustGate
    confidence: float
    raw_score: float
    weakest_claim: Optional[str]
    num_conflicts: int
    frac_unattributed: float
    hallucination_detected: bool
    reasoning: str
    per_claim_scores: List[float]
    failure_analysis: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    completeness_score: float = 1.0  # 1.0 = fully temporally complete, lower if newer editions ignored
    num_complementary_ignored: int = 0  # count of newer editions present in context but not in answer
    penalty_conflict: float = 0.0
    penalty_unattributed: float = 0.0
    penalty_completeness: float = 0.0
    rag_prior_boost: float = 0.0
    linear_score: float = 0.0  # in [-1, 1] before mapping to normalized confidence


def aggregate(
    nli_result: Any,
    conflict_report: Any,
    attributions: List[Any],
) -> AggregatedVerdict:
    verdicts = nli_result.verdicts
    num_claims = len(verdicts)

    def claim_score(v: Any) -> float:
        lab = getattr(v, "label", "")
        conf = float(getattr(v, "confidence", 0.0))
        if lab == "entailment":
            return 1.0 * conf
        if lab == "neutral":
            return 0.3 * conf
        return -1.0 * conf

    per_scores = [claim_score(v) for v in verdicts]
    raw = sum(per_scores) / len(per_scores) if per_scores else 0.0

    num_conflicts = len(conflict_report.conflicts)
    frac_unattributed = (
        sum(1 for a in attributions if not a.is_attributed) / len(attributions)
        if attributions
        else 0.0
    )

    # --- Temporal Completeness Score ---
    # Count complementary updates where the answer did NOT include the newer value
    complementary = getattr(conflict_report, "complementary_updates", []) or []
    num_complementary_ignored = sum(
        1 for c in complementary
        if not c.get("answer_includes_newer", True)
    )
    completeness_score = max(0.0, 1.0 - (0.25 * num_complementary_ignored))
    # Tunable weights (env): softer penalties + RAG prior so typical hybrid-RAG runs land ~60–80% trust.
    w_conflict = float(os.getenv("XAI_AGG_CONFLICT_WEIGHT", "0.06"))
    w_unattrib = float(os.getenv("XAI_AGG_UNATTRIB_WEIGHT", "0.07"))
    w_comp = float(os.getenv("XAI_AGG_COMPLETENESS_WEIGHT", "0.06"))
    rag_prior = float(os.getenv("XAI_AGG_RAG_PRIOR", "0.20"))
    conflict_cap = int(os.getenv("XAI_AGG_CONFLICT_CAP", "3"))

    eff_conflicts = min(int(num_conflicts), max(0, conflict_cap))
    penalty_conflict = w_conflict * eff_conflicts
    # Sublinear unattribution penalty: heavy unattrib still hurts, but not as harsh as linear 0.20*frac.
    penalty_unattributed = w_unattrib * (frac_unattributed ** 0.85)
    penalty_completeness = (1.0 - completeness_score) * w_comp

    linear = raw - penalty_conflict - penalty_unattributed - penalty_completeness + rag_prior
    linear = max(-1.0, min(1.0, linear))
    normalized = (linear + 1.0) / 2.0

    weakest: Optional[str] = None
    if verdicts:
        weakest_v = min(verdicts, key=lambda v: claim_score(v))
        weakest = (weakest_v.claim_text or "")[:100]

    answer_uses_stale = any(
        c.get("answer_uses_older") for c in conflict_report.conflicts
    )
    hallucination = bool(getattr(nli_result, "hallucination_detected", False))

    # --- Failure Analysis ---
    failure_analysis = []
    num_entailed = sum(1 for v in verdicts if getattr(v, "label", "") == "entailment")
    
    if frac_unattributed > 0.4:
        failure_analysis.append(f"Attribution Failure: {frac_unattributed:.0%} claims lack source grounding.")
    
    if num_claims > 0 and (num_entailed / num_claims) < 0.6:
        failure_analysis.append(f"NLI Weakness: Only {num_entailed}/{num_claims} claims fully entailed.")
    
    if num_conflicts > 0:
        failure_analysis.append(f"Edition Conflict: {num_conflicts} claims use superseded regulatory data.")

    if num_complementary_ignored > 0:
        newer_eds = [c.get("newer_edition", "?") for c in complementary if not c.get("answer_includes_newer", True)]
        failure_analysis.append(
            f"Temporal Incompleteness: Answer ignores {num_complementary_ignored} newer edition(s) "
            f"that were retrieved ({', '.join(set(newer_eds))}). This is a coverage failure, not a hallucination."
        )

    # --- Recommendations ---
    recs = []
    if frac_unattributed > 0.3:
        recs.append("Increase retrieval chunk count or improve section coverage.")
    if num_claims > 0 and (num_entailed / num_claims) < 0.5:
        recs.append("Lower generation temperature or strengthen grounding prompt.")
    if num_conflicts > 0:
        recs.append("Purge stale document editions from the vector store or check edition priority rules.")
    if num_complementary_ignored > 0:
        recs.append(
            "Answer collapsed multi-edition context. Apply Temporal Synthesis: include ALL retrieved "
            "editions with date qualifiers (e.g., 'As of March 2025... By September 2025...')."
        )

    if hallucination:
        gate = TrustGate.NON_COMPLIANT
        reasoning = (
            f"Hallucination detected in answer. Score: {normalized:.2f}. Weakest: {weakest}"
        )
    elif answer_uses_stale:
        gate = TrustGate.NON_COMPLIANT
        reasoning = (
            f"Answer uses superseded FSR edition data. "
            f"Score: {normalized:.2f}, Conflicts: {num_conflicts}."
        )
    elif normalized >= float(os.getenv("XAI_AGG_GATE_SAFE", "0.68")):
        gate = TrustGate.SAFE
        reasoning = (
            f"All claims verified. Score: {normalized:.2f}. "
            f"Unattributed: {frac_unattributed:.0%}. Edition conflicts: {num_conflicts}."
        )
    elif normalized >= float(os.getenv("XAI_AGG_GATE_REVIEW", "0.48")):
        gate = TrustGate.NEEDS_REVIEW
        reasons = []
        if frac_unattributed > 0.2:
            reasons.append(f"{frac_unattributed:.0%} claims unattributed")
        if num_conflicts > 0:
            reasons.append(f"{num_conflicts} edition conflict(s)")
        if any(getattr(v, "label", "") != "entailment" for v in verdicts):
            reasons.append("some claims not fully entailed")
        reasoning = "Needs review: " + "; ".join(reasons) + f". Score: {normalized:.2f}."
    else:
        gate = TrustGate.NON_COMPLIANT
        reasoning = (
            f"Score {normalized:.2f} below threshold. "
            f"Unattributed: {frac_unattributed:.0%}. Conflicts: {num_conflicts}."
        )

    return AggregatedVerdict(
        gate=gate,
        confidence=normalized,
        raw_score=raw,
        weakest_claim=weakest,
        num_conflicts=num_conflicts,
        frac_unattributed=frac_unattributed,
        hallucination_detected=hallucination,
        reasoning=reasoning,
        per_claim_scores=per_scores,
        failure_analysis=failure_analysis,
        recommendations=recs,
        completeness_score=completeness_score,
        num_complementary_ignored=num_complementary_ignored,
        penalty_conflict=penalty_conflict,
        penalty_unattributed=penalty_unattributed,
        penalty_completeness=penalty_completeness,
        rag_prior_boost=rag_prior,
        linear_score=linear,
    )

