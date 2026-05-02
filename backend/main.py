"""
FastAPI entrypoint for Char-Chatore RAG + full XAI pipeline.

Run (pick one; do **not** add ``--reload`` unless you need it — reload restarts reload models):
- From this directory (`backend/`):  uvicorn main:app
- From repo root:                     uvicorn main:app --app-dir backend

If you use ``--reload``, exclude noisy paths, e.g.
``uvicorn main:app --app-dir backend --reload --reload-exclude "*.json"``.

Keep XAI artifacts outside the repo (``XAI_ARTIFACT_JSON``) so writes do not trigger reloads.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(BACKEND_DIR, "rag")
INGESTION_DIR = os.path.join(BACKEND_DIR, "ingestion")
for p in (RAG_DIR, INGESTION_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

load_dotenv(os.path.join(BACKEND_DIR, ".env"))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Char-Chatore", version="1.0")

_cors = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").strip()
_origins = [o.strip() for o in _cors.split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class QueryBody(BaseModel):
    query: str
    mode: str = "query"
    model: str = "openai/gpt-oss-120b"


class BrdBody(BaseModel):
    """Plain-text BRD content (upload PDF/DOCX separately and send extracted text, or use multipart later)."""

    text: str
    filename: str = "brd.txt"
    model: Optional[str] = None


from fastapi.responses import Response

@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "VerifyX Backend API is running. Frontend runs on port 3000."}

@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(content=b"", media_type="image/x-icon")

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/brd")
def brd_endpoint(body: BrdBody) -> Dict[str, Any]:
    """
    BRD mode: requirement extraction → validation → per-atomic hybrid RAG (see read.txt),
    rule extraction, attribution + NLI, compliance aggregation.
    """
    from brd import run_brd_pipeline

    t = (body.text or "").strip()
    if not t:
        raise HTTPException(status_code=400, detail="text required")
    try:
        return run_brd_pipeline(t, brd_filename=body.filename or "brd.txt", model=body.model)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/brd/analyze")
async def brd_analyze(
    file: UploadFile | None = File(None),
    pasted_brd: str = Form(""),
    user_query: str = Form(""),
    model: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    BRD upload: optional PDF/DOC/DOCX/TXT file plus optional user instructions.
    Prepend ``user_query`` to extracted document text, then run ``run_brd_pipeline``.
    """
    from brd import run_brd_pipeline
    from brd.pipeline import parse_brd_bytes

    raw = ""
    fname = "brd.txt"
    if file is not None and file.filename:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty file")
        fname = file.filename or fname
        try:
            raw = parse_brd_bytes(data, fname)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    elif (pasted_brd or "").strip():
        raw = pasted_brd.strip()
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide a BRD file or non-empty pasted_brd text",
        )

    uq = (user_query or "").strip()
    if uq:
        raw = f"{uq}\n\n---\n\n{raw}"

    try:
        return run_brd_pipeline(raw, brd_filename=fname, model=model)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/query")
def query_endpoint(body: QueryBody) -> Dict[str, Any]:
    """Non-streaming: RAG answer + full XAI artifact JSON."""
    import numpy as np
    from groq import Groq

    from chat import _complete_rag_answer_sync
    from config import COLLECTION_NAME, GROQ_API_KEY
    from pipeline.embedder import encode_query
    from pipeline.qdrant_cloud import client as qdrant_client
    from retriever import get_hybrid_rag_results
    from xai.pipeline import XAIPipeline

    q = body.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query required")

    mode_key = (body.mode or "query").lower()
    if mode_key not in ("query", "brd"):
        mode_key = "query"

    chunks = get_hybrid_rag_results(q, mode=mode_key)
    if not chunks:
        return {"answer": "", "artifact": None, "error": "no_chunks"}

    cap = int(os.getenv("RAG_LLM_CHUNK_LIMIT", "5" if mode_key == "query" else "8"))
    chunks = chunks[:cap]

    answer = _complete_rag_answer_sync(
        q,
        chunks,
        model=body.model,
        temperature=0.3,
        max_completion_tokens=4096,
        top_p=1.0,
    )

    def embed_fn(t: str):
        return np.asarray(encode_query(t), dtype=np.float32)

    def llm_sidecar(prompt: str) -> str:
        c = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else Groq()
        comp = c.chat.completions.create(
            model=body.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=512,
            stream=False,
        )
        return (comp.choices[0].message.content or "").strip()

    fast = os.getenv("XAI_FAST", "0").lower() in ("1", "true", "yes")
    stage2_on = os.getenv("XAI_NLI_STAGE2", "1").lower() in ("1", "true", "yes")
    claim_llm_on = os.getenv("XAI_CLAIM_LLM", "1").lower() in ("1", "true", "yes")
    llm_fn = llm_sidecar if (stage2_on or claim_llm_on) and not fast else None

    def second_pass_fn(claim_text: str) -> List[str]:
        chs = get_hybrid_rag_results(claim_text, mode=mode_key)
        return [(c.get("text") or "") for c in chs[:8]]

    gov_conn = None
    try:
        try:
            from governance_db import get_governance_db_connection

            gov_conn = get_governance_db_connection(fresh=True)
        except Exception:
            gov_conn = None
        xp = XAIPipeline(
            embed_fn=embed_fn,
            qdrant_client=qdrant_client,
            collection_name=COLLECTION_NAME,
            llm_fn=llm_fn,
            second_pass_fn=second_pass_fn,
            db_conn=gov_conn,
        )
        xai_result = xp.run(q, answer, chunks)
    finally:
        if gov_conn:
            try:
                gov_conn.close()
            except Exception:
                pass

    art = xai_result.artifact
    return {
        "answer": answer,
        "trust_gate": xai_result.verdict.gate.value,
        "confidence": xai_result.verdict.confidence,
        "artifact": art,
        "version_history": art.get("version_history"),
        "supporting_evidence": art.get("supporting_evidence"),
    }


@app.get("/audit/{session_id}")
def audit_endpoint(session_id: str) -> Dict[str, Any]:
    """Fetch audit row by session_id (SQLite ``XAI_AUDIT_SQLITE``)."""
    path = (os.getenv("XAI_AUDIT_SQLITE") or "").strip()
    if not path:
        raise HTTPException(status_code=501, detail="XAI_AUDIT_SQLITE not configured")
    import sqlite3

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM audit_log WHERE session_id = ? ORDER BY id DESC LIMIT 1",
        (session_id,),
    )
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))
