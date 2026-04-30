"""
End-to-end BRD test: load test_data/sample_brd.txt and run run_brd_pipeline.

- test_sample_brd_file_exists: no network
- test_run_brd_pipeline_full: needs GROQ_API_KEY, Qdrant, ES, models (integration)

Run all:
  cd backend && set PYTHONPATH=rag;ingestion && pytest rag/brd/test_brd_e2e.py -v

Integration only:
  pytest rag/brd/test_brd_e2e.py -v -m integration
"""
from __future__ import annotations

import os
import sys

import pytest

_BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAG = os.path.join(_BACKEND, "rag")
_ING = os.path.join(_BACKEND, "ingestion")
for p in (_RAG, _ING):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_BACKEND, ".env"))
except ImportError:
    pass


def _sample_path() -> str:
    return os.path.join(os.path.dirname(__file__), "test_data", "sample_brd.txt")


def test_sample_brd_file_exists():
    p = _sample_path()
    assert os.path.isfile(p), f"Missing {p}"
    text = open(p, encoding="utf-8").read()
    assert "personal loans" in text.lower() or "50" in text
    assert len(text) > 50


@pytest.mark.integration
def test_run_brd_pipeline_full():
    if not (os.getenv("GROQ_API_KEY") or "").strip():
        pytest.skip("GROQ_API_KEY not set")

    from brd.pipeline import run_brd_pipeline

    text = open(_sample_path(), encoding="utf-8").read()
    out = run_brd_pipeline(text, brd_filename="sample_brd.txt")

    assert "brd_id" in out
    assert out.get("trust_status") in ("SAFE", "NEEDS_REVIEW", "NON_COMPLIANT")
    assert "requirements" in out
    assert isinstance(out["requirements"], list)
    assert "compliance_score" in out
