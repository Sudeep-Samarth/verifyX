"""
Quick smoke test — run with:
  cd backend/rag && python -m pytest xai/test_xai.py -v
NLI cases download ``cross-encoder/nli-deberta-v3-large`` on first run (~1.5GB).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

_RAG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

from xai.aggregator import TrustGate, aggregate
from xai.edition_conflict import EditionConflictDetector


def test_edition_conflict():
    det = EditionConflictDetector()
    chunks = [
        {"doc_id": "FSR (2025-06)", "text": "PSBs CRAR was 13.9 per cent."},
        {"doc_id": "FSR (2025-12)", "text": "PSBs CRAR was 14.5 per cent."},
    ]
    answer = "CRAR of PSBs is 13.9 per cent."
    report = det.detect(chunks, answer)
    assert report.has_conflict
    assert any(c["answer_uses_older"] for c in report.conflicts)


def test_gate_safe():
    cv = SimpleNamespace(label="entailment", confidence=1.0, claim_text="ok")
    nli = SimpleNamespace(verdicts=[cv], hallucination_detected=False)
    cr = SimpleNamespace(conflicts=[], has_conflict=False)
    attr = [SimpleNamespace(is_attributed=True)]
    v = aggregate(nli, cr, attr)
    assert v.gate == TrustGate.SAFE


def test_gate_non_compliant_stale():
    cv = SimpleNamespace(label="entailment", confidence=0.9, claim_text="ok")
    nli = SimpleNamespace(verdicts=[cv], hallucination_detected=False)
    cr = SimpleNamespace(
        conflicts=[{"answer_uses_older": True}],
        has_conflict=True,
    )
    attr = [SimpleNamespace(is_attributed=True)]
    v = aggregate(nli, cr, attr)
    assert v.gate == TrustGate.NON_COMPLIANT


@pytest.mark.slow
def test_nli_entailed():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from xai.nli_verifier import SentenceLevelNLIVerifier

    v = SentenceLevelNLIVerifier()
    chunk = "PSBs reported a CRAR of 14.5 per cent as of September 2025."
    answer = "The CRAR of PSBs was 14.5 per cent."
    result = v.verify(answer, [chunk])
    assert result.answer_entailment_score > 0.7


@pytest.mark.slow
def test_nli_hallucination():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from xai.nli_verifier import SentenceLevelNLIVerifier

    v = SentenceLevelNLIVerifier()
    chunk = "PSBs reported a CRAR of 14.5 per cent as of September 2025."
    answer = "The CRAR of PSBs was 8.2 per cent."
    result = v.verify(answer, [chunk])
    assert result.answer_entailment_score < 0.6
