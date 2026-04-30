"""
Run the full BRD pipeline on the bundled sample document.

Usage (from repo root):
  python backend/rag/brd/run_sample.py

Or from backend with PYTHONPATH:
  cd backend && set PYTHONPATH=rag;ingestion && python rag/brd/run_sample.py

Waits for Enter before starting (same as “submit” in a UI flow).
Set BRD_SKIP_PROMPT=1 to run immediately (CI / scripts).
"""
from __future__ import annotations

import json
import os
import sys

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

from brd.pipeline import run_brd_pipeline  # noqa: E402


def _sample_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data", "sample_brd.txt")


def main() -> None:
    sample = _sample_path()
    if not os.path.isfile(sample):
        print("Missing sample file:", sample, file=sys.stderr)
        sys.exit(1)

    if os.getenv("BRD_SKIP_PROMPT", "").lower() not in ("1", "true", "yes"):
        input("Press Enter to run the full BRD pipeline on sample_brd.txt...\n")

    with open(sample, encoding="utf-8") as f:
        text = f.read()

    print("Running BRD pipeline (Groq + Qdrant + ES + NLI)...", flush=True)
    out = run_brd_pipeline(text, brd_filename="sample_brd.txt")
    # Compact console summary; full JSON available in artifact-style dict
    summary = {
        "brd_id": out.get("brd_id"),
        "trust_status": out.get("trust_status"),
        "compliance_score": out.get("compliance_score"),
        "requirements_count": len(out.get("requirements") or []),
        "violations_count": len(out.get("violations") or []),
    }
    print(json.dumps(summary, indent=2))
    print("\n--- requirements (rolled up) ---")
    for r in out.get("requirements") or []:
        print(f"  {r.get('req_id')}: {r.get('status')} — {str(r.get('req_text', ''))[:80]}...")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_brd_run_summary.json")
    with open(out_path, "w", encoding="utf-8") as wf:
        json.dump(
            {
                "summary": summary,
                "requirements": out.get("requirements"),
                "violations": out.get("violations"),
                "remediation_suggestions": out.get("remediation_suggestions"),
            },
            wf,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
