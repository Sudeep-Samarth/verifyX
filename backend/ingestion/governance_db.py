"""
Supabase Postgres (or any PostgreSQL) for ``concept_index`` and ``conflicts``.

Connection: set ``SUPABASE_DB_URL`` or ``DATABASE_URL`` (Session pooler or direct URI
from Supabase project settings -> Database).

Optional: ``GOVERNANCE_PERSIST_CONFLICTS=1`` to INSERT into ``conflicts`` when
edition conflicts are detected (requires tables from schema/supabase_governance.sql).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_CONN = None


def governance_db_url() -> str:
    return (os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL") or "").strip()


def get_governance_db_connection(fresh: bool = False):
    """
    Return a psycopg2 connection or None if not configured / import fails.
    Caller should ``close()()`` when done unless using the module singleton (not recommended for FastAPI).
    """
    url = governance_db_url()
    if not url:
        return None
    if fresh:
        return _connect(url)
    global _CONN
    if _CONN is not None:
        try:
            if not _CONN.closed:
                return _CONN
        except Exception:
            pass
    _CONN = _connect(url)
    return _CONN


def _connect(url: str):
    try:
        import psycopg2
    except ImportError as e:
        raise RuntimeError(
            "psycopg2 required for governance DB. Install: pip install psycopg2-binary"
        ) from e
    return psycopg2.connect(url)


def persist_conflict_records(
    conn: Any,
    conflicts: List[Dict[str, Any]],
    *,
    default_action: str = "SUPERSEDED",
) -> int:
    """
    Insert rows into ``conflicts``. Returns number of rows inserted.
    """
    if not conflicts or not conn:
        return 0
    if os.getenv("GOVERNANCE_PERSIST_CONFLICTS", "0").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return 0
    cur = conn.cursor()
    n = 0
    ts = datetime.now(timezone.utc)
    for c in conflicts:
        action_taken = default_action
        if "action_taken" in c and c["action_taken"] is not None:
            action_taken = c["action_taken"]
        try:
            cur.execute(
                """
                INSERT INTO conflicts (
                    entity, claim_type,
                    older_chunk_id, older_edition, older_section, older_text,
                    newer_chunk_id, newer_edition, newer_section, newer_text,
                    nli_P_contradiction, cosine_sim, conflict_type,
                    action_taken, resolved_to, timestamp
                ) VALUES (
                    %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                """,
                (
                    c.get("entity") or "UNKNOWN",
                    c.get("claim_type") or c.get("metric") or "UNKNOWN",
                    c.get("older_chunk_id"),
                    _as_str(c.get("older_edition")),
                    _as_str(c.get("older_section")),
                    (_as_str(c.get("older_text")) or "")[:8000] or None,
                    c.get("newer_chunk_id"),
                    _as_str(c.get("newer_edition")),
                    _as_str(c.get("newer_section")),
                    (_as_str(c.get("newer_text")) or "")[:8000] or None,
                    c.get("nli_P_contradiction"),
                    c.get("cosine_sim"),
                    c.get("conflict_type") or "value_change",
                    action_taken,
                    c.get("resolved_to") or c.get("newer_chunk_id"),
                    c.get("timestamp") or ts,
                ),
            )
            n += 1
        except Exception as e:
            print(f"[governance_db] conflict insert skipped: {e}", flush=True)
    conn.commit()
    cur.close()
    return n


def _as_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    return str(v)
