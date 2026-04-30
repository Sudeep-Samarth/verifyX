import concurrent.futures
import os
import sys
import time
from typing import List, Optional

# Add ingestion directory to sys.path to easily load config and clients
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INGESTION_DIR = os.path.join(BASE_DIR, "ingestion")
if INGESTION_DIR not in sys.path:
    sys.path.append(INGESTION_DIR)

from config import COLLECTION_NAME
from pipeline.chunk_status_filter import (
    annotate_chunk_governance_flags,
    chunk_status_filter_enabled,
    elasticsearch_superseded_must_not,
    qdrant_exclude_superseded_filter,
)
from pipeline.embedder import encode_query, pick_ce_device, prewarm_embedding_model
from pipeline.qdrant_cloud import client as qdrant_client
from pipeline.elasticsearch_cloud import client as es_client, ES_INDEX_NAME

CE_MODEL_ID = os.getenv("CE_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CE_PREDICT_BATCH_SIZE = int(os.getenv("CE_PREDICT_BATCH_SIZE", "32"))

# Lazy import/load: avoid importing sentence_transformers until first rerank (saves seconds on Ollama-only startup).
_cross_encoder: Optional[object] = None


def get_cross_encoder():
    from sentence_transformers import CrossEncoder

    global _cross_encoder
    if _cross_encoder is None:
        device = pick_ce_device()
        backend = (os.getenv("CE_BACKEND") or "torch").strip().lower()
        if backend not in ("torch", "onnx", "openvino"):
            backend = "torch"
        try:
            _cross_encoder = CrossEncoder(
                CE_MODEL_ID, device=device, backend=backend
            )
        except Exception as e:
            if backend != "torch":
                print(f"[retriever] CE_BACKEND={backend} failed ({e}); using PyTorch.")
                _cross_encoder = CrossEncoder(
                    CE_MODEL_ID, device=device, backend="torch"
                )
            else:
                raise
        if os.getenv("CE_WARMUP", "1").lower() not in ("0", "false", "no"):
            _cross_encoder.predict([["warmup", "warmup"]], show_progress_bar=False)
    return _cross_encoder


def prewarm_rag_models() -> None:
    """
    Warm query embeddings. Cross-encoder is heavy; by default it loads on first retrieval
    (often in parallel with Qdrant/ES). Set RAG_PREWARM_CROSS_ENCODER=1 to load it here.
    """
    probe_vec = prewarm_embedding_model()
    if os.getenv("RAG_SKIP_QDRANT_DIM_CHECK", "").lower() not in ("1", "true", "yes"):
        _ensure_query_vector_matches_qdrant(probe_vec)
    if os.getenv("RAG_PREWARM_CROSS_ENCODER", "0").lower() in ("1", "true", "yes"):
        get_cross_encoder()


def _elasticsearch_search_safe(query_string: str, top_k: int):
    try:
        return query_elasticsearch(query_string, top_k=top_k)
    except Exception as e:
        print(f"Warning: Elasticsearch query failed (is it running?): {e}")
        return []


# Reciprocal Rank Fusion constant: score contribution per list = 1 / (RRF_K + rank)
# with rank starting at 1 for the top hit in that list.
RRF_K = 60


def enrich_chunks_for_xai(chunks: List[dict]) -> List[dict]:
    """
    Add XAI-facing keys: doc_id, section, page, rerank_score, metadata.
    ``doc_id`` is ``"{report_type} ({edition_date})"`` when edition_date is known
    (e.g. ``FSR (2025-12)``) for edition-conflict detection.
    """
    meta_keys = (
        "chunk_id",
        "report_type",
        "edition_date",
        "section_id",
        "section_title",
        "chunk_type",
        "parent_chunk_id",
        "footnote_ids",
        "cross_ref_ids",
        "rrf_score",
        "qdrant_rank",
        "es_rank",
        "_qdrant_score",
        "_es_score",
        "cross_encoder_score",
        "status",
        "superseded_partial",
        "governance_flags",
    )

    for c in chunks:
        rt = (c.get("report_type") or "").strip()
        ed = (c.get("edition_date") or "").strip()
        if rt and ed and str(ed).lower() != "unknown":
            c["doc_id"] = f"{rt} ({ed})"
        else:
            c["doc_id"] = rt or ""

        sid = (c.get("section_id") or "").strip()
        st = (c.get("section_title") or "").strip()
        if sid and st:
            c["section"] = f"{sid} {st}"
        else:
            c["section"] = sid or st or ""

        c["page"] = c.get("page_number")
        ce = c.get("cross_encoder_score")
        c["rerank_score"] = float(ce) if ce is not None else None

        annotate_chunk_governance_flags(c)
        c["metadata"] = {k: c.get(k) for k in meta_keys if k in c}

    return chunks

# Pull: per-channel top-k from Qdrant + ES.
# Fusion: max unique chunks after RRF (wide recall).
# ce_candidates: ONLY these top-RRF chunks go through the cross-encoder (latency control).
# top_after_ce: how many to return after sorting by CE score.
HYBRID_LIMITS = {
    "query": {
        "pull": 60,
        "fusion": 100,
        "ce_candidates": 24,
        "top_after_ce": 12,
    },
    # BRD: smaller prompts + fewer CE pairs = faster runs and fewer Groq 429s
    "brd": {
        "pull": 40,
        "fusion": 50,
        "ce_candidates": 24,
        "top_after_ce": 6,
    },
}


def _limits_for_mode(mode: str) -> dict:
    """Defaults above, override with RAG_PULL, RAG_FUSION, RAG_CE_CANDIDATES, RAG_TOP_AFTER_CE."""
    m = (mode or "query").lower()
    if m not in HYBRID_LIMITS:
        m = "query"
    L = dict(HYBRID_LIMITS[m])
    if os.getenv("RAG_PULL", "").strip().isdigit():
        L["pull"] = int(os.getenv("RAG_PULL"))
    if os.getenv("RAG_FUSION", "").strip().isdigit():
        L["fusion"] = int(os.getenv("RAG_FUSION"))
    if os.getenv("RAG_CE_CANDIDATES", "").strip().isdigit():
        L["ce_candidates"] = int(os.getenv("RAG_CE_CANDIDATES"))
    if os.getenv("RAG_TOP_AFTER_CE", "").strip().isdigit():
        L["top_after_ce"] = int(os.getenv("RAG_TOP_AFTER_CE"))
    return L

_cached_qdrant_vector_size: Optional[int] = None


def _get_qdrant_collection_vector_size() -> int:
    global _cached_qdrant_vector_size
    if _cached_qdrant_vector_size is not None:
        return _cached_qdrant_vector_size
    info = qdrant_client.get_collection(COLLECTION_NAME)
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        first = next(iter(vectors.values()))
        _cached_qdrant_vector_size = int(first.size)
    else:
        _cached_qdrant_vector_size = int(vectors.size)
    return _cached_qdrant_vector_size


def _ensure_query_vector_matches_qdrant(query_vector: list) -> None:
    expected = _get_qdrant_collection_vector_size()
    got = len(query_vector)
    if got == expected:
        return
    from pipeline.embedder import get_embed_provider

    prov = get_embed_provider()
    raise RuntimeError(
        f"Embedding/Qdrant dimension mismatch: query vector length is {got}, but Qdrant "
        f"collection {COLLECTION_NAME!r} expects size={expected}. "
        f"Current EMBED_PROVIDER={prov!r}. "
        "Fixes: (A) MiniLM / 384-d index: set EMBED_PROVIDER=local, or Ollama with "
        "OLLAMA_EMBED_MODEL=all-minilm (run: ollama pull all-minilm). "
        "(B) 768-d models (e.g. nomic-embed-text): re-run full ingest with that model so Qdrant is recreated."
    )


def query_qdrant(vector, top_k=50):
    """Fetch chunks using dense semantic similarity (Cosine)"""
    qf = qdrant_exclude_superseded_filter()
    kwargs = {
        "collection_name": COLLECTION_NAME,
        "query": vector,
        "limit": top_k,
    }
    if qf is not None:
        kwargs["query_filter"] = qf
    results = qdrant_client.query_points(**kwargs)
    # Convert payload into canonical chunk format
    chunks = []
    for point in results.points:
        chunk = point.payload.copy()
        chunk["_qdrant_score"] = point.score
        chunks.append(chunk)
    return chunks

def query_elasticsearch(query_string, top_k=50):
    """
    BM25 over indexed chunks. Uses ``multi_match`` on ``text`` + ``section_title`` — **not**
    ``query_string`` (Lucene treats ``?`` / ``*`` as wildcards; ``AND`` + boosted clauses
    killed recall on natural questions like “What measures has RBI taken…?”).
    """
    q = (query_string or "").strip()
    if not q:
        return []

    mm: dict = {
        "type": "best_fields",
        "fields": ["text^1.0", "section_title^1.3"],
        "query": q,
        "operator": "or",
        "minimum_should_match": (os.getenv("RAG_ES_MIN_SHOULD_MATCH") or "30%").strip(),
    }
    if os.getenv("RAG_ES_FUZZY", "0").lower() in ("1", "true", "yes"):
        mm["fuzziness"] = "AUTO"

    must_not = elasticsearch_superseded_must_not()
    if must_not:
        es_query: dict = {
            "bool": {
                "must": [{"multi_match": mm}],
                "must_not": must_not,
            }
        }
    else:
        es_query = {"multi_match": mm}

    try:
        response = es_client.search(
            index=ES_INDEX_NAME,
            body={"query": es_query, "size": top_k},
        )
    except Exception as e:
        if must_not and chunk_status_filter_enabled():
            print(
                f"[retriever] ES status filter failed ({e}); retrying without status must_not.",
                flush=True,
            )
            response = es_client.search(
                index=ES_INDEX_NAME,
                body={"query": {"multi_match": mm}, "size": top_k},
            )
        else:
            raise
    chunks = []
    for hit in response["hits"]["hits"]:
        chunk = hit["_source"].copy()
        chunk["_es_score"] = hit["_score"]
        chunks.append(chunk)
    return chunks

def deduplicate_chunks(chunk_lists: List[List[dict]]) -> List[dict]:
    """Helper to merge results from multiple query variants while keeping best scores."""
    seen = {}
    for clist in chunk_lists:
        for c in clist:
            cid = c.get("chunk_id")
            if not cid: continue
            if cid not in seen:
                seen[cid] = c
            else:
                # Keep existing, but maybe update internal technical scores if anyway relevant
                pass
    return list(seen.values())

def check_semantic_coverage(chunks: List[dict], core_terms: List[str]) -> bool:
    """True if core concepts are mentioned in at least 3 chunks among top-10 candidates."""
    if not core_terms or not chunks:
        return True
    
    count = 0
    top_texts = [c.get("text", "").lower() for c in chunks[:10]]
    for term in core_terms:
        # If any core term is found in at least 3 chunks
        mentions = sum(1 for t in top_texts if term in t)
        if mentions >= 3:
            count += 1
            
    # Success threshold: at least 2 core concepts well-represented
    return count >= 2

def get_hybrid_rag_results(query_text, mode="query"):
    """
    Enhanced Hybrid RAG:
    1. Query Expansion (Multi-query generation)
    2. Parallel search across all variants
    3. Global RRF fusion
    4. Semantic coverage check + Potential Fallback
    5. Cross-Encoder reranking
    """
    _mode = (mode or "query").lower()
    if _mode not in HYBRID_LIMITS:
        _mode = "query"
    limits = _limits_for_mode(_mode)
    pull_limit = limits["pull"]
    fusion_limit = limits["fusion"]
    ce_candidates = limits["ce_candidates"]
    top_after_ce = limits["top_after_ce"]

    expansion_enabled = os.getenv("RAG_QUERY_EXPANSION", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    # BRD does N atomics × retrieval; expansion multiplies embeddings + ES load. Opt-in only.
    if expansion_enabled and _mode == "brd":
        expansion_enabled = os.getenv("RAG_QUERY_EXPANSION_BRD", "0").lower() in (
            "1",
            "true",
            "yes",
        )

    queries = [query_text]
    core_terms = []

    if expansion_enabled:
        from query_expansion import QueryExpander

        expander = QueryExpander()
        max_variants = int(os.getenv("RAG_EXPAND_MAX_VARIANTS", "2"))
        max_variants = max(1, min(max_variants, 4))
        variants = expander.expand(query_text, max_variants=max_variants)
        queries.extend(variants[:max_variants])
        core_terms = expander.extract_core_terms(query_text)
        print(f"[{_mode}] Query expansion: {len(queries)} queries active.", flush=True)

    t0 = time.perf_counter()
    
    # 1. Gather all result candidates from all queries
    all_q_results = []
    all_e_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries) * 2) as pool:
        q_futs = []
        e_futs = []
        for q in queries:
            v = encode_query(q)
            q_futs.append(pool.submit(query_qdrant, v, pull_limit))
            e_futs.append(pool.submit(_elasticsearch_search_safe, q, pull_limit))
        
        for f in q_futs:
            all_q_results.append(f.result())
        for f in e_futs:
            all_e_results.append(f.result())

    # 2. Global Fusion: apply RRF across all lists
    # Note: RRF here needs to handle multiple pairs of lists. 
    # Current RRF takes (q_list, es_list). 
    # We'll flatten them and pass to a multi-ranker fuser.
    
    fused_scores = {}
    chunk_map = {}
    
    def apply_ranks(chunk_lists, origin_prefix):
        for i, clist in enumerate(chunk_lists):
            for rank, chunk in enumerate(clist):
                cid = chunk["chunk_id"]
                if cid not in fused_scores:
                    fused_scores[cid] = 0.0
                    chunk_map[cid] = chunk
                fused_scores[cid] += 1.0 / (RRF_K + rank + 1)
                chunk_map[cid][f"{origin_prefix}_{i}_rank"] = rank + 1

    apply_ranks(all_q_results, "qdrant")
    apply_ranks(all_e_results, "es")

    sorted_chunks = sorted(
        chunk_map.values(),
        key=lambda c: fused_scores[c["chunk_id"]],
        reverse=True
    )
    for c in sorted_chunks:
        c["rrf_score"] = fused_scores[c["chunk_id"]]

    fused_res = sorted_chunks[:fusion_limit]

    # 3. Semantic Coverage Check & Fallback
    if expansion_enabled and not check_semantic_coverage(fused_res, core_terms):
        print(f"[{_mode}] Coverage check FAILED. Triggering fallback expansion...", flush=True)
        # Simple fallback: broader search on ES with core terms only
        broad_query = " OR ".join(core_terms)
        es_fallback = _elasticsearch_search_safe(broad_query, pull_limit)
        # Merge into existing map and re-sort RRF
        apply_ranks([es_fallback], "fallback")
        sorted_chunks = sorted(
            chunk_map.values(),
            key=lambda c: fused_scores[c["chunk_id"]],
            reverse=True
        )
        fused_res = sorted_chunks[:fusion_limit]
    
    if not fused_res:
        return []

    # 4. Cross-Encoder Reranking
    ce_pool = fused_res[: max(1, min(ce_candidates, len(fused_res)))]
    pairs = [[query_text, c["text"]] for c in ce_pool]
    
    # Lazy load CE
    ce_model = get_cross_encoder()
    scores = ce_model.predict(pairs, batch_size=CE_PREDICT_BATCH_SIZE)

    for i, score in enumerate(scores):
        ce_pool[i]["cross_encoder_score"] = float(score)

    ce_ranked = sorted(
        ce_pool, key=lambda x: x["cross_encoder_score"], reverse=True
    )
    reranked = ce_ranked

    # --- 5. Optional relevance filter (easy to over-reject; never return empty) ---
    filter_enabled = os.getenv("RAG_RELEVANCE_FILTER", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    if filter_enabled:
        try:
            from query_expansion import QueryExpander
            from xai.relevance_filter import PassageFilter

            expander = QueryExpander()
            q_intent = expander.extract_intent(query_text)
            print(
                f"[{_mode}] Intent: domain={q_intent.get('domain')}, "
                f"keywords={q_intent.get('keywords')}",
                flush=True,
            )

            p_filter = PassageFilter()
            filtered_chunks = p_filter.filter_chunks(q_intent, list(ce_ranked))

            if len(filtered_chunks) < 3:
                print(
                    f"[{_mode}] Relevance filter kept {len(filtered_chunks)} chunks (<3) — "
                    f"falling back to cross-encoder order (no empty retrieval).",
                    flush=True,
                )
                reranked = ce_ranked
            else:
                num_rejected = len(ce_ranked) - len(filtered_chunks)
                if num_rejected > 0:
                    print(
                        f"[{_mode}] Filtered out {num_rejected} chunks (semantic drift).",
                        flush=True,
                    )
                reranked = filtered_chunks
        except Exception as e:
            print(f"[{_mode}] Relevance filter skipped: {e}", flush=True)
            reranked = ce_ranked

    if os.getenv("RAG_DEBUG_TIMING", "").lower() in ("1", "true", "yes"):
        print(f"[{_mode}] Enhanced retrieval wall: {(time.perf_counter() - t0) * 1000:.1f} ms", flush=True)

    return enrich_chunks_for_xai(reranked[:top_after_ce])

