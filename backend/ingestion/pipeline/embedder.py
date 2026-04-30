import json
import os
import time
import urllib.error
import urllib.request
from functools import lru_cache
from typing import List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Values for EMBED_PROVIDER (and common aliases → canonical "local" | "ollama")
_EMBED_LOCAL_ALIASES = frozenset(
    {
        "",
        "local",
        "sentence_transformers",
        "sentence-transformers",
        "pytorch",
        "torch",
        "st",
        "transformers",
        "huggingface",
        "hf",
        "minilm",
        "sentence_transformer",
    }
)
_EMBED_OLLAMA_ALIASES = frozenset({"ollama", "ollama_api", "ollama-api"})


def get_embed_provider() -> str:
    """
    Which backend builds query/index vectors: ``local`` (SentenceTransformer / PyTorch)
    or ``ollama`` (HTTP /api/embed). Set ``EMBED_PROVIDER`` in ``backend/.env``.

    Read on each call so ``load_dotenv`` and shell exports apply even if this module
    imported early.
    """
    raw = (os.getenv("EMBED_PROVIDER") or "local").strip().lower()
    if not raw or raw in _EMBED_LOCAL_ALIASES:
        return "local"
    if raw in _EMBED_OLLAMA_ALIASES:
        return "ollama"
    print(
        f"[embedder] Unknown EMBED_PROVIDER={raw!r} — use 'local' or 'ollama'; defaulting to 'local'.",
        flush=True,
    )
    return "local"


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def _ollama_embed_model() -> str:
    # Default all-minilm = 384-d (same as sentence-transformers/all-MiniLM-L6-v2 style indexes).
    # nomic-embed-text / mxbai etc. use other sizes — set OLLAMA_EMBED_MODEL and match Qdrant (re-ingest if needed).
    return os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")


OLLAMA_EMBED_BATCH = max(1, int(os.getenv("OLLAMA_EMBED_BATCH", "32")))
OLLAMA_EMBED_TIMEOUT = int(os.getenv("OLLAMA_EMBED_TIMEOUT", "120"))

# Local (SentenceTransformer) — only used when get_embed_provider() != ollama
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "96"))
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_QUERY_MAX_CHARS = int(os.getenv("EMBED_QUERY_MAX_CHARS", "0"))

_torch_threads = int(os.getenv("TORCH_NUM_THREADS", str(os.cpu_count() or 4)))

_st_model: Optional[object] = None


def _use_ollama() -> bool:
    return get_embed_provider() == "ollama"


def _ollama_embed_api(texts: List[str]) -> List[List[float]]:
    """Call Ollama /api/embed. Same model must be used for ingest + query; re-index Qdrant if you switch."""
    if not texts:
        return []
    url = f"{_ollama_base_url()}/api/embed"
    model = _ollama_embed_model()
    
    # 🚨 Safety: all-minilm context limits vary by quantization and version in Ollama.
    # 500 chars ≈ 125 tokens is extremely safe for all MiniLM variants.
    _OLLAMA_MAX_CHARS = 500
    safe_texts = []
    for t in texts:
        t_clean = (t or "").strip()
        if len(t_clean) > _OLLAMA_MAX_CHARS:
            t_clean = t_clean[:_OLLAMA_MAX_CHARS - 3] + "..."
        safe_texts.append(t_clean)

    if os.getenv("RAG_DEBUG_TIMING", "").lower() in ("1", "true", "yes"):
        lengths = [len(s) for s in safe_texts]
        print(f"[embedder] Ollama batch: n={len(safe_texts)}, max_len={max(lengths) if lengths else 0}", flush=True)


    # Ollama accepts a string or a list of strings for `input`
    payload_obj = {"model": model, "input": safe_texts if len(safe_texts) > 1 else safe_texts[0]}
    payload = json.dumps(payload_obj).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_EMBED_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        model = _ollama_embed_model()
        if e.code == 404 and "not found" in body.lower():
            raise RuntimeError(
                f"Ollama returned 404: embedding model {model!r} is not installed locally.\n"
                f"  Fix: ollama pull {model}\n"
                f"  See what you have: ollama list\n"
                f"Notes:\n"
                f"  • Default all-minilm (384-d) matches Qdrant indexes built with SentenceTransformer MiniLM.\n"
                f"  • nomic-embed-text is 768-d — only if you re-ingested with it; set OLLAMA_EMBED_MODEL=nomic-embed-text.\n"
                f"  • If `ollama serve` fails with 'address already in use', Ollama is already running (Windows app) — that is normal."
            ) from e
        raise RuntimeError(
            f"Ollama embed HTTP {e.code}: {body}. Model: {model!r}. Try: ollama pull {model}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {_ollama_base_url()}: {e}. "
            "Start Ollama or set OLLAMA_BASE_URL."
        ) from e

    if "embeddings" in data:
        out = data["embeddings"]
    elif "embedding" in data:
        out = [data["embedding"]]
    else:
        raise RuntimeError(f"Unexpected Ollama /api/embed response keys: {data.keys()}")

    if len(out) != len(safe_texts):
        raise RuntimeError(
            f"Ollama returned {len(out)} embeddings for {len(safe_texts)} inputs"
        )
    return [list(map(float, row)) for row in out]


def _resolve_device_choice(raw: str) -> str:
    import torch

    raw = raw.strip().lower()
    if raw in ("cuda", "mps", "cpu"):
        if raw == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if raw == "mps" and not (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            return "cpu"
        return raw
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_embed_device() -> str:
    if _use_ollama():
        return "ollama"
    raw = (os.getenv("EMBED_DEVICE") or "auto").strip().lower()
    return _resolve_device_choice(raw)


def pick_ce_device() -> str:
    raw = (os.getenv("CE_DEVICE") or os.getenv("EMBED_DEVICE") or "auto").strip().lower()
    return _resolve_device_choice(raw)


def _build_sentence_transformer():
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(max(1, _torch_threads))

    device = pick_embed_device()
    backend = (os.getenv("EMBED_BACKEND") or "torch").strip().lower()
    if backend not in ("torch", "onnx", "openvino"):
        backend = "torch"

    model_kwargs = None
    if device == "cuda" and os.getenv("EMBED_FP16", "1").lower() not in ("0", "false", "no"):
        model_kwargs = {"torch_dtype": torch.float16}

    local_only = os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")

    def _load(b: str):
        kw = dict(device=device, backend=b, local_files_only=local_only)
        if model_kwargs is not None:
            kw["model_kwargs"] = model_kwargs
        return SentenceTransformer(EMBED_MODEL_ID, **kw)

    try:
        model = _load(backend)
    except Exception as e:
        if backend != "torch":
            print(f"[embedder] EMBED_BACKEND={backend} failed ({e}); using PyTorch backend.")
            model = _load("torch")
        else:
            raise

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if os.getenv("EMBED_WARMUP", "1").lower() not in ("0", "false", "no"):
        import torch as _t

        with _t.inference_mode():
            model.encode(
                "warmup",
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=False,
                convert_to_numpy=True,
            )
    return model


def _get_sentence_transformer():
    global _st_model
    if _st_model is None:
        _st_model = _build_sentence_transformer()
    return _st_model


def embed_chunks(chunks, batch_size=None):
    """Batch-encode chunk texts for Qdrant (local ST or Ollama)."""
    if _use_ollama():
        texts = [c["text"] for c in chunks]
        all_vec: List[List[float]] = []
        for i in range(0, len(texts), OLLAMA_EMBED_BATCH):
            batch = texts[i : i + OLLAMA_EMBED_BATCH]
            all_vec.extend(_ollama_embed_api(batch))
        for i, c in enumerate(chunks):
            c["vector"] = all_vec[i]
        return chunks

    import numpy as np
    import torch

    model = _get_sentence_transformer()
    bs = batch_size if batch_size is not None else EMBED_BATCH_SIZE
    texts = [c["text"] for c in chunks]
    n = len(texts)
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=n > 256,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )

    for i, c in enumerate(chunks):
        c["vector"] = np.asarray(embeddings[i], dtype=np.float32).reshape(-1).tolist()

    return chunks


def prewarm_embedding_model() -> List[float]:
    """
    Load weights / hit Ollama once. Returns the warmup embedding (for Qdrant dim check).
    """
    if _use_ollama():
        vec = _ollama_embed_api(["warmup"])[0]
        print(
            f"[embedder] Ollama embed ready: model={_ollama_embed_model()!r} @ {_ollama_base_url()} (dim={len(vec)})",
            flush=True,
        )
        if os.getenv("OLLAMA_SKIP_COMPAT_WARN", "").lower() not in ("1", "true", "yes"):
            print(
                "[embedder] Tip: use OLLAMA_EMBED_MODEL=all-minilm for 384-d Qdrant built with MiniLM; "
                "nomic-embed-text is 768-d and needs a re-ingested collection.",
                flush=True,
            )
        return vec

    import numpy as np
    import torch

    model = _get_sentence_transformer()
    with torch.inference_mode():
        emb = model.encode(
            "warmup",
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
    arr = np.asarray(emb, dtype=np.float32).reshape(-1)
    return arr.tolist()


def encode_texts_for_xai(texts: List[str]):
    """
    Batch-encode answer strings for XAI paraphrase stability (same space as ``encode_query``).
    Returns a float32 numpy array of shape (n, dim).
    """
    import numpy as np

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    cleaned = [(t or "").strip() or " " for t in texts]
    if _use_ollama():
        # Apply truncation safety for Ollama
        limit = EMBED_QUERY_MAX_CHARS if EMBED_QUERY_MAX_CHARS > 0 else 500
        cleaned = [t[:limit] for t in cleaned]

        all_vec: List[List[float]] = []
        for i in range(0, len(cleaned), OLLAMA_EMBED_BATCH):
            batch = cleaned[i : i + OLLAMA_EMBED_BATCH]
            all_vec.extend(_ollama_embed_api(batch))
        return np.asarray(all_vec, dtype=np.float32)

    import torch

    model = _get_sentence_transformer()
    with torch.inference_mode():
        emb = model.encode(
            cleaned,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
    return np.asarray(emb, dtype=np.float32)


@lru_cache(maxsize=256)
def encode_query(text: str):
    """Single-query vector for RAG (must match ingest provider + model)."""
    q = (text or "").strip()
    if not q:
        q = " "
    if _use_ollama():
        # Apply truncation safety for Ollama
        limit = EMBED_QUERY_MAX_CHARS if EMBED_QUERY_MAX_CHARS > 0 else 500
        if len(q) > limit:
            q = q[:limit]

        t0 = time.perf_counter()
        out = _ollama_embed_api([q])[0]
        if os.getenv("RAG_DEBUG_TIMING", "").lower() in ("1", "true", "yes"):
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"[embedder] encode_query (ollama): {ms:.1f} ms (chars={len(q)})", flush=True)
        return out

    import numpy as np
    import torch

    model = _get_sentence_transformer()
    t0 = time.perf_counter()
    single_thread = os.getenv("EMBED_QUERY_SINGLE_THREAD", "").lower() in (
        "1",
        "true",
        "yes",
    )
    old_threads = torch.get_num_threads()
    try:
        if single_thread:
            torch.set_num_threads(1)
        with torch.inference_mode():
            emb = model.encode(
                q,
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=False,
                convert_to_numpy=True,
            )
    finally:
        torch.set_num_threads(old_threads)

    arr = np.asarray(emb, dtype=np.float32).reshape(-1)
    out = arr.tolist()

    if os.getenv("RAG_DEBUG_TIMING", "").lower() in ("1", "true", "yes"):
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"[embedder] encode_query (local): {ms:.1f} ms (chars={len(q)})", flush=True)

    return out


def embedding_backend_label() -> str:
    """Human-readable label for logs (matches ``get_embed_provider()``)."""
    if _use_ollama():
        return f"ollama:{_ollama_embed_model()} @ {_ollama_base_url()}"
    return f"local:{EMBED_MODEL_ID}"
