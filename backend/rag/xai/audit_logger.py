"""Persist XAI artifacts — PostgreSQL if ``DATABASE_URL`` set, else SQLite."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLogger:
    def __init__(self, db_conn: Any = None, sqlite_path: Optional[str] = None):
        self.db = db_conn
        self._sqlite_path = sqlite_path or os.getenv("XAI_AUDIT_SQLITE", "").strip()

    def _ensure_sqlite(self) -> None:
        if self.db is not None or not self._sqlite_path:
            return
        self.db = sqlite3.connect(self._sqlite_path, check_same_thread=False)
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                created_at TEXT,
                query TEXT,
                answer TEXT,
                trust_gate TEXT,
                confidence REAL,
                hallucination INTEGER,
                claims_json TEXT,
                sources_json TEXT,
                nli_json TEXT,
                conflicts_json TEXT,
                ragas_json TEXT,
                artifact_json TEXT
            )
            """
        )
        self.db.commit()

    def log(
        self,
        query: str,
        answer: str,
        xai_result: Any,
        ragas_scorecard: Any = None,
        session_id: Optional[str] = None,
    ) -> str:
        self._ensure_sqlite()
        sid = session_id or str(uuid.uuid4())
        art = xai_result.artifact
        v = xai_result.verdict

        claims_json = json.dumps(art.get("claims", []), default=str)
        sources_json = json.dumps(art.get("retrieval_explanation", []), default=str)
        nli_json = json.dumps(
            [c.get("nli") for c in art.get("claims", [])], default=str
        )
        conflicts_json = json.dumps(art.get("edition_conflict", {}), default=str)
        ragas_json = json.dumps(
            art.get("ragas_scorecard") or (ragas_scorecard and ragas_scorecard.__dict__),
            default=str,
        )
        artifact_json = json.dumps(art, default=str)

        if self.db is None:
            return sid

        # psycopg2
        if hasattr(self.db, "cursor") and "sqlite3" not in type(self.db).__module__:
            cur = self.db.cursor()
            cur.execute(
                """
                INSERT INTO audit_log (
                    session_id, query, answer, trust_gate, confidence, hallucination,
                    claims_json, sources_json, nli_json, conflicts_json, ragas_json, artifact_json
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING session_id
                """,
                (
                    sid,
                    query,
                    answer,
                    v.gate.value,
                    v.confidence,
                    1 if art.get("hallucination_detected") else 0,
                    claims_json,
                    sources_json,
                    nli_json,
                    conflicts_json,
                    ragas_json,
                    artifact_json,
                ),
            )
            self.db.commit()
            cur.close()
            return sid

        self.db.execute(
            """
            INSERT INTO audit_log (
                session_id, created_at, query, answer, trust_gate, confidence, hallucination,
                claims_json, sources_json, nli_json, conflicts_json, ragas_json, artifact_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                sid,
                _utc_now(),
                query,
                answer,
                v.gate.value,
                v.confidence,
                1 if art.get("hallucination_detected") else 0,
                claims_json,
                sources_json,
                nli_json,
                conflicts_json,
                ragas_json,
                artifact_json,
            ),
        )
        self.db.commit()
        return sid
