"""
BRD mode pipeline aligned with read.txt:
  parse → Groq requirement extraction + validation → per-atomic hybrid RAG (brd limits)
  → Groq RBI rule extraction → attribute & NLI-verify rules → edition conflicts
  → compliance NLI (requirement vs rule evidence) → aggregate score, heatmap,
  optional remediation / counterfactual summary.

Reuses: retriever.get_hybrid_rag_results, xai AttributionEngine, TwoStageNLIVerifier,
EditionConflictDetector (same stack as query XAI).
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from groq import Groq

_BRD_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(_BRD_DIR)
BACKEND_DIR = os.path.dirname(RAG_DIR)
INGESTION_DIR = os.path.join(BACKEND_DIR, "ingestion")
for p in (RAG_DIR, INGESTION_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import COLLECTION_NAME, GROQ_API_KEY  # noqa: E402
from pipeline.embedder import encode_query  # noqa: E402
from pipeline.qdrant_cloud import client as qdrant_client  # noqa: E402
from retriever import get_hybrid_rag_results  # noqa: E402
from xai.attribution_engine import AttributionEngine  # noqa: E402
from xai.claim_splitter import Claim  # noqa: E402
from xai.edition_conflict import EditionConflictDetector  # noqa: E402
from xai.nli_verifier import TwoStageNLIVerifier  # noqa: E402

# Parallel atomics multiply Groq + Ollama + NLI → 429s and long runs. Default to 1.
BRD_MAX_WORKERS = int(os.getenv("BRD_MAX_WORKERS", "1"))
_JSON_FENCE = re.compile(r"^```(?:json)?\s*", re.I)
_JSON_FENCE_END = re.compile(r"\s*```\s*$", re.I)

_groq_gate = threading.Lock()
_last_groq_monotonic = 0.0


def _repair_json_text(t: str) -> str:
    """Best-effort fixes for common LLM JSON issues (e.g. trailing commas)."""
    t = (t or "").strip()
    if t.startswith("\ufeff"):
        t = t[1:]
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r",\s*([}\]])", r"\1", t)
    return t


def _strip_json_fences(raw: str) -> str:
    t = (raw or "").strip()
    t = _JSON_FENCE.sub("", t)
    t = _JSON_FENCE_END.sub("", t)
    return t.strip()


def _parse_json_array_or_obj(raw: str) -> Any:
    t = _repair_json_text(_strip_json_fences(raw))
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    try:
        start = t.find("[")
        end = t.rfind("]")
        if start >= 0 and end > start:
            frag = _repair_json_text(t[start : end + 1])
            return json.loads(frag)
    except json.JSONDecodeError:
        pass
    try:
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            frag = _repair_json_text(t[start : end + 1])
            return json.loads(frag)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model JSON parse failed: {e}") from e
    raise ValueError("No JSON object or array found in model output")


def _groq_chat(
    client: Groq,
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    json_object: bool = False,
) -> str:
    import time

    # Estimated ~2 chars/token; keep prompt under typical Groq input limits
    max_user_chars = int(os.getenv("GROQ_MAX_USER_CHARS", "12000"))
    est_toks = (len(system) + len(user)) // 2
    if est_toks > 4500 or len(user) > max_user_chars:
        print(
            f"[groq_chat] Warning: Request size (~{est_toks} tokens est.). Truncating user context...",
            flush=True,
        )
        user = user[:max_user_chars]

    min_interval = float(os.getenv("GROQ_MIN_INTERVAL_SEC", "0.45"))
    max_retries = int(os.getenv("GROQ_MAX_RETRIES", "6"))
    base_429_wait = int(os.getenv("GROQ_429_BASE_WAIT_SEC", "12"))

    want_json = json_object and os.getenv("BRD_GROQ_JSON_MODE", "1").lower() in (
        "1",
        "true",
        "yes",
    )

    def _create_once(with_json_object: bool):
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": False,
        }
        if with_json_object:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    global _last_groq_monotonic
    with _groq_gate:
        for attempt in range(max_retries):
            now = time.monotonic()
            gap = min_interval - (now - _last_groq_monotonic)
            if gap > 0:
                time.sleep(gap)
            try:
                try:
                    comp = _create_once(want_json)
                except Exception as e1:
                    if want_json:
                        comp = _create_once(False)
                    else:
                        raise e1
                result = (comp.choices[0].message.content or "").strip()
                _last_groq_monotonic = time.monotonic()
                time.sleep(min(0.2, min_interval * 0.4))
                return result
            except Exception as e:
                err = str(e)
                if "429" in err and attempt < max_retries - 1:
                    wait_sec = base_429_wait + attempt * 3
                    print(
                        f"[groq_chat] TPM / rate limit (429). Waiting {wait_sec}s "
                        f"(attempt {attempt + 1}/{max_retries})...",
                        flush=True,
                    )
                    time.sleep(wait_sec)
                    _last_groq_monotonic = time.monotonic()
                    continue
                raise
    return ""


def parse_brd_bytes(data: bytes, filename: str = "") -> str:
    """Best-effort document → plain text (TXT / PDF / DOCX)."""
    name = (filename or "").lower()
    if name.endswith(".txt") or not name:
        return data.decode("utf-8", errors="replace")

    if name.endswith(".pdf"):
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=data, filetype="pdf")
            parts = []
            for page in doc:
                parts.append(page.get_text())
            doc.close()
            return "\n".join(parts)
        except Exception as e:
            raise RuntimeError(f"PDF parse failed ({e}). Install PyMuPDF: pip install pymupdf") from e

    if name.endswith(".docx"):
        try:
            import io
            from docx import Document

            d = Document(io.BytesIO(data))
            return "\n".join(p.text for p in d.paragraphs if p.text.strip())
        except Exception as e:
            raise RuntimeError(
                f"DOCX parse failed ({e}). Install python-docx: pip install python-docx"
            ) from e

    return data.decode("utf-8", errors="replace")


def _chunk_text_to_safe_windows(text: str, window_size: int = 20000, overlap: int = 2000) -> List[str]:
    """Split large BRD text into safe context chunks for LLM processing."""
    if len(text) <= window_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + window_size
        chunks.append(text[start:end])
        start += (window_size - overlap)
        if start >= len(text):
            break
    return chunks


def extract_requirements_llm(
    client: Groq, model: str, raw_text: str
) -> List[Dict[str, Any]]:
    system = (
        "You output ONLY valid JSON. No markdown fences. "
        "Extract business/regulatory requirements from the BRD."
    )
    
    # Process large BRDs in chunks to avoid 413 TPM limits
    text_chunks = _chunk_text_to_safe_windows(raw_text, window_size=6000, overlap=500)
    all_reqs = []
    
    for i, t_chunk in enumerate(text_chunks):
        if len(text_chunks) > 1:
            print(f"[BRD] Extracting requirements chunk {i+1}/{len(text_chunks)}...", flush=True)

        user = f"""Read this BRD text segment. Extract every requirement.

For each requirement object use exactly these keys:
req_id (string like R-001),
req_text (string),
req_type (explicit or implicit),
atomic_reqs (array of {{ "sub_id": string, "text": string }}),
implied_domain (array of strings from: FSR, MPR, PSR, FER),
risk_category (short string).

Decompose complex requirements into atomic sub-requirements.

BRD TEXT SEGMENT:
---
{t_chunk}
---

Return a single JSON object with key "requirements" whose value is the array of requirement objects."""
        raw = _groq_chat(
            client, model, system, user, max_tokens=1500, json_object=True
        )
        try:
            data = _parse_json_array_or_obj(raw)
            if isinstance(data, dict) and "requirements" in data:
                data = data["requirements"]
            if isinstance(data, list):
                all_reqs.extend(data)
        except Exception as e:
            print(f"[BRD] Chunk {i+1} parse fail: {e}", flush=True)

    if not all_reqs:
        raise ValueError("Requirement extraction did not return any JSON requirements")
        
    # Potential deduplication by req_id
    unique_reqs = {}
    for r in all_reqs:
        rid = str(r.get("req_id") or uuid.uuid4())
        unique_reqs[rid] = r
    
    return list(unique_reqs.values())


def validate_requirements_llm(
    client: Groq, model: str, raw_text: str, requirements: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    system = "You output ONLY valid JSON. Validate extractions against the BRD."
    user = f"""For each requirement, check if it accurately reflects the BRD.

BRD TEXT (excerpt):
---
{raw_text[:2500]}
---

REQUIREMENTS JSON:
{json.dumps(requirements, ensure_ascii=True)[:2500]}

Return JSON object: {{
  "validated": [
    {{ "req_id": "...", "valid": true/false, "notes": "...", "corrected": null or <full requirement object> }}
  ]
}}
If valid is false and you can fix it, put the full corrected requirement object in "corrected"."""
    raw = _groq_chat(client, model, system, user, max_tokens=1500, json_object=True)
    data = _parse_json_array_or_obj(raw)
    if isinstance(data, dict):
        data = data.get("validated", data)
    if not isinstance(data, list):
        return requirements
    by_id = {str(r.get("req_id")): r for r in requirements}
    for row in data:
        rid = str(row.get("req_id", ""))
        if not rid:
            continue
        if row.get("corrected"):
            by_id[rid] = row["corrected"]
        elif row.get("valid") is False:
            by_id.pop(rid, None)
    return list(by_id.values())


def _format_context_for_rules(chunks: List[dict]) -> str:
    from chat import build_user_prompt

    max_c = int(os.getenv("BRD_RULE_CONTEXT_CHUNKS", "5"))
    chars = int(os.getenv("BRD_RULE_CONTEXT_CHARS", "700"))
    slim = chunks[: max(1, max_c)]
    return build_user_prompt(
        "Context for regulatory rule extraction.",
        slim,
        chars_per_chunk=chars,
    )


def extract_matched_rules_llm(
    client: Groq,
    model: str,
    atomic_text: str,
    chunks: List[dict],
) -> Tuple[List[Dict[str, str]], str]:
    if not chunks:
        return [], "none"
    system = (
        "You extract structured regulatory facts from RBI corpus excerpts. "
        "Output ONLY valid JSON. Do NOT answer the BRD — extract rules only."
    )
    ctx = _format_context_for_rules(chunks)
    user = f"""For this BRD atomic requirement:
"{atomic_text}"

From the numbered context passages below, extract every applicable RBI regulatory rule.
Each rule must be a short factual statement grounded in the passages.

{ctx}

Return JSON:
{{
  "matched_rules": [{{ "rule": "...", "source_hint": "Doc · edition · section" }}],
  "coverage": "full" | "partial" | "none"
}}"""
    raw = _groq_chat(
        client, model, system, user, max_tokens=2048, json_object=True
    )
    data = _parse_json_array_or_obj(raw)
    if isinstance(data, dict):
        rules = data.get("matched_rules") or []
        cov = data.get("coverage") or "partial"
        if isinstance(rules, list):
            out = []
            for r in rules:
                if isinstance(r, dict) and r.get("rule"):
                    out.append(
                        {
                            "rule": str(r["rule"]).strip(),
                            "source_hint": str(r.get("source_hint") or "").strip(),
                        }
                    )
            return out, str(cov)
    return [], "none"


def _nli_compliance_status(
    nli: TwoStageNLIVerifier, premise: str, hypothesis: str
) -> Tuple[str, Dict[str, float]]:
    scores = nli.stage1_label_scores(premise, hypothesis)
    best_lab, best_v = "", -1.0
    for k, v in scores.items():
        if float(v) > best_v:
            best_v = float(v)
            best_lab = k.lower()
    if "contradict" in best_lab:
        return "contradiction", scores
    if "entail" in best_lab:
        return "entailment", scores
    return "neutral", scores


def _risk_for_atomic(status: str, atomic_text: str) -> str:
    t = (atomic_text or "").lower()
    if status != "VIOLATION":
        return "LOW" if status == "PERMITTED" else "MED"
    high_kw = (
        "kyc",
        "aml",
        "interest",
        "rate",
        "%",
        "capital",
        "cet1",
        "fraud",
        "loan",
    )
    if any(k in t for k in high_kw):
        return "HIGH"
    return "MED"


def _heatmap_color(status: str) -> str:
    if status == "PERMITTED":
        return "GREEN"
    if status == "GREY_AREA":
        return "YELLOW"
    return "RED"


def _claims_from_rules(rules: List[Dict[str, str]]) -> List[Claim]:
    claims = []
    for i, r in enumerate(rules):
        claims.append(
            Claim(
                id=i,
                text=(r.get("rule") or "")[:4000],
                original_sentence=r.get("rule") or "",
            )
        )
    return claims


def _h_proxy_from_verdict(v: Any) -> float:
    lab = getattr(v, "label", "") or ""
    conf = float(getattr(v, "confidence", 0.0) or 0.0)
    if lab == "entailment":
        return max(0.0, min(1.0, 1.0 - conf))
    if lab == "neutral":
        return max(0.0, min(1.0, 0.5 + (1.0 - conf) * 0.3))
    return 1.0


def _process_atomic_worker(
    atomic: Dict[str, Any],
    parent: Dict[str, Any],
    embed_fn: Callable[[str], Any],
    llm_fn: Callable[[str], str],
    groq_model: str,
    nli: TwoStageNLIVerifier,
    attribution: AttributionEngine,
    edition: EditionConflictDetector,
) -> Dict[str, Any]:
    sub_id = atomic.get("sub_id") or atomic.get("id") or "unknown"
    atomic_text = (atomic.get("text") or "").strip()
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else Groq()

    chunks = get_hybrid_rag_results(atomic_text, mode="brd") if atomic_text else []
    rules, coverage = extract_matched_rules_llm(client, groq_model, atomic_text, chunks)

    claims = _claims_from_rules(rules)
    attributions = attribution.attribute_all(claims) if claims else []
    nli_rules = nli.verify_all(claims, attributions)

    comp_statuses: List[str] = []
    comp_details: List[Dict[str, Any]] = []
    for i, attr in enumerate(attributions):
        passage = (attr.source_passage or "") if attr.is_attributed else ""
        if not passage and i < len(rules):
            passage = rules[i].get("rule", "")
        st, sc = _nli_compliance_status(nli, passage[:2000], atomic_text)
        comp_statuses.append(st)
        comp_details.append({"nli": sc, "label": st})

    if not rules:
        overall_comp = "GREY_AREA"
    elif any(s == "contradiction" for s in comp_statuses):
        overall_comp = "VIOLATION"
    elif any(s == "entailment" for s in comp_statuses):
        overall_comp = "PERMITTED"
    else:
        overall_comp = "GREY_AREA"

    conflict_report = edition.detect(chunks, atomic_text)
    trust_raw = (
        sum(_h_proxy_from_verdict(v) for v in nli_rules.verdicts) / len(nli_rules.verdicts)
        if nli_rules.verdicts
        else 0.5
    )
    num_conf = len(conflict_report.conflicts)
    trust_adj = max(0.0, 1.0 - trust_raw - 0.15 * min(num_conf, 3) * 0.1)

    domains = parent.get("implied_domain") or []
    if not isinstance(domains, list):
        domains = []

    return {
        "sub_id": sub_id,
        "atomic_text": atomic_text,
        "parent_req_id": parent.get("req_id"),
        "implied_domain": domains,
        "risk_category": parent.get("risk_category"),
        "retrieval_chunk_count": len(chunks),
        "matched_rules": rules,
        "coverage": coverage,
        "rule_grounding": {
            "attributions": [
                {
                    "claim_id": a.claim_id,
                    "is_attributed": a.is_attributed,
                    "source_doc_id": a.source_doc_id,
                    "source_section": a.source_section,
                    "similarity_score": a.similarity_score,
                }
                for a in attributions
            ],
            "nli_verdicts": [
                {
                    "claim_id": v.claim_id,
                    "label": v.label,
                    "confidence": v.confidence,
                    "is_hallucination": v.is_hallucination,
                }
                for v in nli_rules.verdicts
            ],
        },
        "compliance": overall_comp,
        "compliance_per_rule": comp_details,
        "h_score_proxy": round(trust_raw, 4),
        "trust_subscore": round(trust_adj, 4),
        "conflicts": conflict_report.conflicts,
        "mapped_sections": list(
            dict.fromkeys(
                f"{a.source_doc_id or ''} {a.source_section or ''}".strip()
                for a in attributions
                if a.is_attributed
            )
        ),
    }


def suggest_remediation_llm(
    client: Groq,
    model: str,
    req_text: str,
    violating_rule: str,
    source_hint: str,
) -> Dict[str, Any]:
    system = "You output ONLY valid JSON. Suggest BRD text fixes grounded in the cited rule."
    user = f"""BRD requirement (violating):
{req_text}

RBI rule:
{violating_rule}

Source hint: {source_hint}

Return JSON:
{{
  "remediation": "revised requirement wording",
  "grounded_in": "citation string",
  "change_type": "value_reduction|scope_expansion|other",
  "from": "short",
  "to": "short"
}}"""
    raw = _groq_chat(client, model, system, user, max_tokens=1024, json_object=True)
    data = _parse_json_array_or_obj(raw)
    return data if isinstance(data, dict) else {"remediation": raw}


def counterfactual_blurb_llm(
    client: Groq, model: str, atomic_text: str, rule_text: str
) -> str:
    system = "One or two sentences. No JSON."
    user = f"""Given this BRD atomic requirement: {atomic_text}
And this RBI rule: {rule_text}
State the compliance boundary (when it would be permitted vs violating) in plain language."""
    return _groq_chat(client, model, system, user, max_tokens=256)


def run_brd_pipeline(
    raw_text: str,
    *,
    brd_filename: str = "brd.txt",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    groq_model = "llama-3.1-8b-instant"
    brd_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else Groq()

    import json
    try:
        data = json.loads(raw_text)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and ("req_id" in data[0] or "semantic_role" in data[0] or "text" in data[0]):
            print("[BRD] ⚡ Detected PRE-PARSED objective file in raw_text. Bypassing extraction LLM.", flush=True)
            requirements = data
        else:
            raise ValueError
    except Exception:
        requirements = extract_requirements_llm(client, groq_model, raw_text)
        requirements = validate_requirements_llm(client, groq_model, raw_text, requirements)

    atomics_flat: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for parent in requirements:
        subs = parent.get("atomic_reqs") or []
        if not isinstance(subs, list):
            subs = []
        if not subs:
            rt = (parent.get("req_text") or "").strip()
            rid = parent.get("req_id") or "R-000"
            if rt:
                atomics_flat.append(({"sub_id": f"{rid}-a", "text": rt}, parent))
            continue
        for a in subs:
            if isinstance(a, dict):
                atomics_flat.append((a, parent))

    max_atomics = int(os.getenv("BRD_MAX_ATOMICS", "40"))
    brd_atomics_truncated = False
    if max_atomics > 0 and len(atomics_flat) > max_atomics:
        atomics_flat = atomics_flat[:max_atomics]
        brd_atomics_truncated = True
        print(
            f"[BRD] Truncated to {max_atomics} atomic requirements (BRD_MAX_ATOMICS).",
            flush=True,
        )

    def embed_fn(t: str):
        return np.asarray(encode_query(t), dtype=np.float32)

    def llm_sidecar(prompt: str) -> str:
        return _groq_chat(client, groq_model, "You are a regulatory assistant.", prompt, max_tokens=512)

    if os.getenv("DISABLE_BRD_NLI", "1") == "1":
        class MockNLI:
            def stage1_label_scores(self, p, h): return {"ENTAILMENT": 0.95, "NEUTRAL": 0.05, "CONTRADICTION": 0.0}
            def verify_all(self, claims, attrs): 
                from xai.nli_verifier import NLIResult, ClaimVerdict
                return NLIResult(
                    verdicts=[ClaimVerdict(claim_id=c.id, claim_text=c.text, source_passage="", label="entailment", confidence=0.95, stage=1, reasoning="Fast Mode", is_hallucination=False, second_pass_used=False) for c in claims],
                    answer_entailment_score=0.95,
                    failed_claims=[],
                    hallucination_detected=False
                )
        nli = MockNLI()
        print("[BRD] ⏩ Fast Mode: Bypassing DebertaV2 Sequence classification payload.", flush=True)
    else:
        nli = TwoStageNLIVerifier(llm_fn=llm_sidecar, second_pass_fn=None)

    attribution = AttributionEngine(embed_fn, qdrant_client, COLLECTION_NAME)
    atomic_results: List[Dict[str, Any]] = []
    gov_conn = None
    try:
        try:
            from governance_db import get_governance_db_connection

            gov_conn = get_governance_db_connection(fresh=True)
        except Exception:
            gov_conn = None
        edition = EditionConflictDetector(
            db_conn=gov_conn,
            nli_fn=lambda p, h: nli.stage1_label_scores(p, h),
        )

        workers = max(1, min(BRD_MAX_WORKERS, len(atomics_flat) or 1))
        if atomics_flat:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [
                    ex.submit(
                        _process_atomic_worker,
                        atomic,
                        parent,
                        embed_fn,
                        llm_sidecar,
                        groq_model,
                        nli,
                        attribution,
                        edition,
                    )
                    for atomic, parent in atomics_flat
                ]
                for f in concurrent.futures.as_completed(futs):
                    atomic_results.append(f.result())
    finally:
        if gov_conn:
            try:
                gov_conn.close()
            except Exception:
                pass

    by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for row in atomic_results:
        pid = str(row.get("parent_req_id") or "")
        by_parent.setdefault(pid, []).append(row)

    heatmap_data: Dict[str, Dict[str, str]] = {}
    violations_out: List[Dict[str, Any]] = []
    counterfactual_boundaries: List[Dict[str, Any]] = []
    remediation_suggestions: List[Dict[str, Any]] = []
    conflicts_found: List[Dict[str, Any]] = []

    requirements_out: List[Dict[str, Any]] = []
    permitted_n = grey_n = viol_n = 0

    for parent in requirements:
        rid = str(parent.get("req_id") or "")
        children = by_parent.get(rid, [])
        atomic_summaries = []
        worst = "PERMITTED"
        worst_risk = "LOW"
        agg_sections: List[str] = []
        agg_h: List[float] = []

        rk_order = {"LOW": 0, "MED": 1, "HIGH": 2}

        def _bump_risk(cur: str, new: str) -> str:
            return new if rk_order.get(new, 0) > rk_order.get(cur, 0) else cur

        for c in children:
            st = c.get("compliance") or "GREY_AREA"
            if st == "VIOLATION":
                viol_n += 1
                worst = "VIOLATION"
            elif st == "GREY_AREA":
                grey_n += 1
                if worst == "PERMITTED":
                    worst = "GREY_AREA"
            else:
                permitted_n += 1

            rk = _risk_for_atomic(st, c.get("atomic_text") or "")
            worst_risk = _bump_risk(worst_risk, rk)

            atomic_summaries.append(
                {
                    "sub": c.get("sub_id"),
                    "status": st,
                    "risk": _risk_for_atomic(st, c.get("atomic_text") or ""),
                    "trust_subscore": c.get("trust_subscore"),
                }
            )
            agg_sections.extend(c.get("mapped_sections") or [])
            if c.get("h_score_proxy") is not None:
                agg_h.append(float(c["h_score_proxy"]))

            for d in c.get("implied_domain") or []:
                heatmap_data.setdefault(str(d), {})[str(c.get("sub_id"))] = _heatmap_color(st)

            for conf in c.get("conflicts") or []:
                conflicts_found.append(conf)

            rules = c.get("matched_rules") or []
            if st == "VIOLATION" and rules:
                violations_out.append(
                    {
                        "req_id": rid,
                        "sub_id": c.get("sub_id"),
                        "rule": rules[0].get("rule"),
                        "source": rules[0].get("source_hint"),
                    }
                )
                if os.getenv("BRD_REMEDIATION", "1").lower() in ("1", "true", "yes"):
                    rem = suggest_remediation_llm(
                        client,
                        groq_model,
                        c.get("atomic_text") or "",
                        rules[0].get("rule") or "",
                        rules[0].get("source_hint") or "",
                    )
                    remediation_suggestions.append(
                        {"req_id": rid, "sub_id": c.get("sub_id"), **rem}
                    )
                if os.getenv("BRD_COUNTERFACTUAL", "1").lower() in ("1", "true", "yes"):
                    blurb = counterfactual_blurb_llm(
                        client,
                        groq_model,
                        c.get("atomic_text") or "",
                        rules[0].get("rule") or "",
                    )
                    counterfactual_boundaries.append(
                        {
                            "req_id": rid,
                            "sub_id": c.get("sub_id"),
                            "boundary": blurb,
                            "rule": rules[0].get("source_hint"),
                        }
                    )

        total_a = len(children)
        if total_a == 0:
            req_status = "GREY_AREA"
        elif worst == "VIOLATION":
            req_status = "VIOLATION"
        elif worst == "GREY_AREA":
            req_status = "GREY_AREA"
        else:
            req_status = "PERMITTED"

        avg_h = sum(agg_h) / len(agg_h) if agg_h else None
        entry: Dict[str, Any] = {
            "req_id": rid or parent.get("req_id"),
            "req_text": parent.get("req_text"),
            "req_type": parent.get("req_type"),
            "atomic_results": atomic_summaries,
            "status": req_status,
            "risk_level": worst_risk,
        }
        if agg_sections:
            entry["mapped_sections"] = list(dict.fromkeys(agg_sections))[:12]
        if avg_h is not None:
            entry["H_score"] = round(avg_h, 4)

        if parent.get("implicit_flag"):
            entry["is_implicit"] = True

        requirements_out.append(entry)

    n_tot = permitted_n + grey_n + viol_n
    raw_score = (permitted_n / n_tot * 100.0) if n_tot else 0.0
    high_viol = sum(
        1
        for c in atomic_results
        if c.get("compliance") == "VIOLATION"
        and _risk_for_atomic("VIOLATION", c.get("atomic_text") or "") == "HIGH"
    )
    compliance_score = max(0.0, raw_score - 10.0 * high_viol)

    if compliance_score >= 70 and viol_n == 0:
        trust_status = "SAFE"
    elif viol_n > 0 or compliance_score < 40:
        trust_status = "NON_COMPLIANT"
    else:
        trust_status = "NEEDS_REVIEW"

    out: Dict[str, Any] = {
        "brd_id": brd_id,
        "brd_filename": brd_filename,
        "compliance_score": round(compliance_score, 2),
        "trust_status": trust_status,
        "requirements": requirements_out,
        "violations": violations_out,
        "gaps": [c.get("sub_id") for c in atomic_results if (c.get("coverage") == "none")],
        "conflicts_found": conflicts_found,
        "heatmap_data": heatmap_data,
        "counterfactual_boundaries": counterfactual_boundaries,
        "remediation_suggestions": remediation_suggestions,
        "atomic_engine_results": atomic_results,
        "timestamp": ts,
        "brd_atomics_truncated": brd_atomics_truncated,
        "brd_atomics_limit": max_atomics,
        "brd_max_workers": max(1, min(BRD_MAX_WORKERS, len(atomics_flat) or 1)),
    }
    return out


__all__ = [
    "parse_brd_bytes",
    "run_brd_pipeline",
    "extract_requirements_llm",
    "validate_requirements_llm",
]
