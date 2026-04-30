"""Explainable AI: claims, attribution, two-stage NLI, edition rules, RAGAS proxies, audit."""

from .artifact import build_artifact, format_analytics_report, print_artifact, write_artifact_json
from .audit_logger import AuditLogger
from .claim_splitter import AtomicClaimSplitter, Claim
from .paraphrase_resistance import ParaphraseResistanceTester, ResistanceResult
from .pipeline import XAIPipeline, XAIResult
from .ragas_scorer import RAGASScorecard, RAGASScorer

__all__ = [
    "XAIPipeline",
    "XAIResult",
    "build_artifact",
    "format_analytics_report",
    "print_artifact",
    "write_artifact_json",
    "AuditLogger",
    "AtomicClaimSplitter",
    "Claim",
    "ParaphraseResistanceTester",
    "ResistanceResult",
    "RAGASScorecard",
    "RAGASScorer",
]
