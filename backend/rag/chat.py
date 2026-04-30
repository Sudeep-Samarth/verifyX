"""
RAG chat: Qdrant retrieval + Groq streaming completion, with provenance appended from chunk payloads.
"""
import os
import sys
from typing import Iterator, List, Optional

from dotenv import load_dotenv
from groq import Groq

RAG_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(RAG_DIR)
INGESTION_DIR = os.path.join(BACKEND_DIR, "ingestion")
if INGESTION_DIR not in sys.path:
    sys.path.insert(0, INGESTION_DIR)

load_dotenv(os.path.join(BACKEND_DIR, ".env"))

from config import GROQ_API_KEY  # noqa: E402

EXCERPT_MAX_LEN = 280
# LLM context budget: on-demand Groq tiers often cap ~8k input tokens; full chunk text blows past that.
CONTEXT_CHARS_PER_CHUNK = int(os.getenv("RAG_CONTEXT_CHARS_PER_CHUNK", "900"))
# Deep-tier retrieval volume for high-recall grounding.
_DEFAULT_LLM_CHUNKS = {"query": 15, "brd": 6}

_rag_prewarm_checked = False


def _optional_prewarm_rag() -> None:
    """If RAG_PREWARM=1, load bi-encoder + CE once before first retrieval (API / long-running apps)."""
    global _rag_prewarm_checked
    if _rag_prewarm_checked:
        return
    _rag_prewarm_checked = True
    if os.getenv("RAG_PREWARM", "0").lower() not in ("1", "true", "yes"):
        return
    from retriever import prewarm_rag_models

    prewarm_rag_models()

SYSTEM_PROMPT = """You are a regulatory assistant expert in EXPLICIT DATA MAPPING and TEMPORAL SYNTHESIS.

STRICT GROUNDING RULES:
1. ONLY use information explicitly present in the provided numbered context passages.
2. CITATION MANDATE: Every single sentence in your answer MUST end with a source citation, e.g., [1] or [2].
3. ZERO-LEAKAGE: Do NOT add any external knowledge, entity names, URLs, or technical series unless they are explicitly written in the retrieved text.
4. If a piece of information is missing from the text, even if you know it's true, you MUST NOT include it.
5. LIMIT your answer to a maximum of 4 specific points.
6. If the documents don't provide the answer, say: "The retrieved documents provide limited information on this topic."

TEMPORAL SYNTHESIS MANDATE (CRITICAL FOR COMPLIANCE):
7. ALWAYS include the time qualifier ("as of [date]") for every numerical data point.
8. If multiple document editions are provided (e.g., FSR June 2025 AND FSR December 2025), you MUST synthesize BOTH:
   - Lead with the MOST RECENT data point.
   - Then describe the EARLIER context or evolution.
   - Explicitly note when data is segmented (e.g., PSBs vs PVBs) vs. aggregate (all SCBs).
9. DO NOT collapse multiple valid truths into a single snapshot.
   - WRONG: "CRAR was 17.3%."
   - CORRECT: "As of March 2025, CRAR for all SCBs was 17.3% [1]. By September 2025, PSBs stood at 16.0% and PVBs at 18.1% [9], indicating sustained capital adequacy."

Example Format:
"As of [date], [metric] was [value] [citation]. In a later update, as of [date], [metric] was [value] [citation], showing [trend]."
"""


def _document_label(chunk: dict) -> str:
    doc = (chunk.get("report_type") or "Unknown document").strip()
    ed = chunk.get("edition_date")
    if ed and str(ed).strip() and str(ed).strip().lower() != "unknown":
        return f"{doc} ({ed})"
    return doc


def _section_label(chunk: dict) -> str:
    sid = (chunk.get("section_id") or "").strip()
    stitle = (chunk.get("section_title") or "").strip()
    if sid and stitle:
        return f"{sid} {stitle}"
    return sid or stitle or "—"


def _excerpt(text: str) -> str:
    t = " ".join(text.split())
    if len(t) <= EXCERPT_MAX_LEN:
        return t
    return t[: EXCERPT_MAX_LEN - 3] + "..."


def _truncate_for_llm_context(text: str, limit: int = CONTEXT_CHARS_PER_CHUNK) -> str:
    t = " ".join((text or "").split())
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def format_sources_block(chunks: List[dict]) -> str:
    """Build the user-facing Sources section from retrieved chunk payloads (ground truth)."""
    lines = ["Sources:"]
    for i, c in enumerate(chunks, start=1):
        doc = _document_label(c)
        section = _section_label(c)
        page = c.get("page_number", "—")
        excerpt = _excerpt(c.get("text") or "")
        lines.append(f"[{i}] Document: {doc}")
        lines.append(f"    Section: {section}")
        lines.append(f"    Page: {page}")
        lines.append(f'    Excerpt: "{excerpt}"')
        lines.append("")
    return "\n".join(lines).rstrip()


def build_user_prompt(
    query: str,
    chunks: List[dict],
    *,
    chars_per_chunk: Optional[int] = None,
) -> str:
    """
    ``chars_per_chunk`` overrides ``RAG_CONTEXT_CHARS_PER_CHUNK`` for BRD / compact prompts.
    """
    limit = (
        int(chars_per_chunk)
        if chars_per_chunk is not None
        else CONTEXT_CHARS_PER_CHUNK
    )
    parts = [
        "Use the following context passages. Each block is labeled [n] for citation.",
        "",
    ]
    for i, c in enumerate(chunks, start=1):
        meta = (
            f"Document: {_document_label(c)} | "
            f"Section: {_section_label(c)} | Page: {c.get('page_number', '—')}"
        )
        parts.append(f"[{i}] ({meta})")
        parts.append(_truncate_for_llm_context(c.get("text") or "", limit=limit))
        parts.append("")
    parts.append(f"Question: {query}")
    return "\n".join(parts)


def _complete_rag_answer_sync(
    query: str,
    chunks: List[dict],
    *,
    model: str,
    temperature: float,
    max_completion_tokens: int,
    top_p: float,
    reasoning_effort: str | None = None,
) -> str:
    """Non-streaming Groq completion for XAI paraphrase stability (generation only)."""
    user_content = build_user_prompt(query, chunks)
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else Groq()
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "top_p": top_p,
        "stream": False,
        "stop": None,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
        
    completion = client.chat.completions.create(**kwargs)
    msg = completion.choices[0].message
    return (getattr(msg, "content", None) or "").strip()


def stream_rag_answer(
    query: str,
    *,
    mode: str = "query",
    llm_chunk_limit: Optional[int] = None,
    model: str = "openai/gpt-oss-120b",
    temperature: float = 1.0,
    max_completion_tokens: int = 4096,
    top_p: float = 1.0,
    reasoning_effort: str | None = None,
) -> Iterator[str]:
    """
    Retrieve with hybrid RRF (Qdrant + ES) + cross-encoder, then stream Groq.
    ``llm_chunk_limit`` caps how many reranked chunks are sent to the model (sources match that slice).
    """
    _optional_prewarm_rag()
    from retriever import get_hybrid_rag_results

    mode_key = (mode or "query").lower()
    if mode_key not in ("query", "brd"):
        mode_key = "query"

    chunks = get_hybrid_rag_results(query, mode=mode_key)
    if not chunks:
        yield (
            "No relevant passages were found in Qdrant for this query.\n\n"
            "Sources:\n(no matching chunks)"
        )
        return

    env_cap = os.getenv("RAG_LLM_CHUNK_LIMIT")
    if llm_chunk_limit is not None:
        cap = llm_chunk_limit
    elif env_cap and env_cap.isdigit():
        cap = int(env_cap)
    else:
        cap = _DEFAULT_LLM_CHUNKS[mode_key]
    chunks = chunks[:cap]

    print(f"\n[GROUNDING] {len(chunks)} chunks used for generation:", flush=True)
    for i, c in enumerate(chunks, start=1):
        txt = (c.get("text") or "")[:70].replace("\n", " ")
        print(f"  [{i}] {c.get('doc_id')} | {txt}...", flush=True)

    user_content = build_user_prompt(query, chunks)
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else Groq()

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "top_p": top_p,
        "stream": True,
        "stop": None,
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
        
    completion = client.chat.completions.create(**kwargs)

    streamed_parts: List[str] = []
    for event in completion:
        delta = event.choices[0].delta
        piece = getattr(delta, "content", None) or ""
        if piece:
            streamed_parts.append(piece)
            yield piece

    answer_body = "".join(streamed_parts).strip()
    xai_enabled = os.getenv("XAI_ENABLED", "1") == "1"
    xai_fast = os.getenv("XAI_FAST", "0") == "1"
    if xai_enabled and answer_body:
        try:
            import numpy as np

            from config import COLLECTION_NAME
            from pipeline.embedder import encode_query
            from pipeline.qdrant_cloud import client as qdrant_client
            from xai.artifact import maybe_write_artifact_json, print_artifact
            from xai.pipeline import XAIPipeline

            gov_conn = None
            try:
                try:
                    from governance_db import get_governance_db_connection

                    gov_conn = get_governance_db_connection(fresh=True)
                except Exception:
                    gov_conn = None

                def embed_fn(t: str):
                    return np.asarray(encode_query(t), dtype=np.float32)

                def llm_sidecar(prompt: str) -> str:
                    c = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else Groq()
                    comp = c.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_completion_tokens=512,
                        top_p=top_p,
                        stream=False,
                    )
                    return (comp.choices[0].message.content or "").strip()

                stage2_on = os.getenv("XAI_NLI_STAGE2", "1").lower() in ("1", "true", "yes")
                claim_llm_on = os.getenv("XAI_CLAIM_LLM", "1").lower() in ("1", "true", "yes")
                llm_fn = llm_sidecar if (stage2_on or claim_llm_on) and not xai_fast else None

                def second_pass_fn(claim_text: str) -> List[str]:
                    from retriever import get_hybrid_rag_results

                    chs = get_hybrid_rag_results(claim_text, mode=mode_key)
                    return [(c.get("text") or "") for c in chs[:8]]

                xai_pipeline = XAIPipeline(
                    embed_fn=embed_fn,
                    qdrant_client=qdrant_client,
                    collection_name=COLLECTION_NAME,
                    llm_fn=llm_fn,
                    second_pass_fn=second_pass_fn,
                    db_conn=gov_conn,
                )

                from xai.assistant import XAIAssistant

                assistant = XAIAssistant(xai_pipeline, llm_fn=llm_fn)
                xai_result = assistant.run_with_autofix(query, answer_body, chunks)

                print_artifact(xai_result.artifact)
                maybe_write_artifact_json(xai_result.artifact)

                if os.getenv("XAI_COUNTERFACTUAL", "0") == "1":
                    from xai.counterfactual import CounterfactualEngine

                    suggestions = []
                    if "if i implement" in query.lower():
                        s = query.lower().split("if i implement")[-1].strip("? .")
                        suggestions.append(s)
                    elif "if i add" in query.lower():
                        s = query.lower().split("if i add")[-1].strip("? .")
                        suggestions.append(s)

                    if suggestions:
                        cf_engine = CounterfactualEngine(xai_pipeline, llm_fn=llm_fn)
                        cf_results = cf_engine.simulate_impact(query, chunks, suggestions)

                        if cf_results:
                            print("\n" + "=" * 30 + " COUNTERFACTUAL ANALYSIS " + "=" * 30, flush=True)
                            for r in cf_results:
                                delta = r["impact_score"] - xai_result.verdict.confidence
                                print(f"  PROPOSED: {r['suggestion']}", flush=True)
                                print(
                                    f"  IMPACT  : {r['impact_verdict']} (Score: {r['impact_score']:.2f}, Delta={delta:+.2f})",
                                    flush=True,
                                )
                                print(
                                    f"  SIMULATED RATIONALE: {r['simulated_answer'][:100]}...",
                                    flush=True,
                                )
                            print("=" * 85 + "\n", flush=True)

                audit_sqlite = (os.getenv("XAI_AUDIT_SQLITE") or "").strip()
                if audit_sqlite:
                    from xai.audit_logger import AuditLogger

                    AuditLogger(sqlite_path=audit_sqlite).log(
                        query, answer_body, xai_result
                    )
            finally:
                if gov_conn:
                    try:
                        gov_conn.close()
                    except Exception:
                        pass
        except Exception as e:
            print(f"[XAI] Verification failed: {e}", flush=True)
            import traceback
            traceback.print_exc()

    yield "\n\n" + format_sources_block(chunks)



def answer_rag_sync(
    query: str,
    **kwargs,
) -> str:
    """Non-streaming helper: full reply including Sources."""
    return "".join(stream_rag_answer(query, **kwargs))


if __name__ == "__main__":
    from pipeline.embedder import embedding_backend_label, get_embed_provider

    _prov = get_embed_provider()
    print(
        f"Embedding: {embedding_backend_label()}  (set EMBED_PROVIDER=local|ollama in backend/.env)",
        flush=True,
    )
    if _prov == "local":
        print(
            "  -> Query vectors: PyTorch SentenceTransformer (MiniLM); first load ~10s on CPU.",
            flush=True,
        )
    else:
        _om = (os.getenv("OLLAMA_EMBED_MODEL") or "").lower()
        if "nomic" in _om:
            print(
                "  -> Ollama nomic = 768-d: only works after full ingest with nomic; else use all-minilm or EMBED_PROVIDER=local.",
                flush=True,
            )
        else:
            print(
                "  -> Query vectors: Ollama (no local MiniLM). If 404: ollama pull all-minilm",
                flush=True,
            )
    print(
        "  -> Reranking: CrossEncoder (PyTorch) — different model; loads on first search unless RAG_PREWARM_CROSS_ENCODER=1.",
        flush=True,
    )
    from retriever import prewarm_rag_models

    try:
        prewarm_rag_models()
    except RuntimeError as e:
        msg = str(e)
        if "dimension mismatch" in msg or "Embedding/Qdrant" in msg:
            print("\n--- Embedding / Qdrant setup error ---\n", msg, "\n", sep="", flush=True)
            sys.exit(2)
        raise

    argv = sys.argv[1:]
    mode = "query"
    if argv and argv[0].lower() in ("query", "brd"):
        mode = argv[0].lower()
        argv = argv[1:]
    q = (
        " ".join(argv)
        if argv
        else "What are the new guidelines for the CET1 ratio?"
    )
    for piece in stream_rag_answer(q, mode=mode):
        print(piece, end="", flush=True)
    print()
