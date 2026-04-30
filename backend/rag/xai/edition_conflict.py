from __future__ import annotations

import re
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

EDITION_PATTERN = re.compile(r"FSR\s*\((\d{4})-(\d{2})\)", re.IGNORECASE)
METRIC_PATTERNS = {
    "CRAR": r"CRAR[^\d]*(\d+\.?\d*)\s*(?:per\s*cent|%)",
    "GNPA": r"GNPA[^\d]*(\d+\.?\d*)\s*(?:per\s*cent|%)",
    "NNPA": r"NNPA[^\d]*(\d+\.?\d*)\s*(?:per\s*cent|%)",
    "ROA": r"ROA[^\d]*(\d+\.?\d*)\s*(?:per\s*cent|%)",
    "ROE": r"ROE[^\d]*(\d+\.?\d*)\s*(?:per\s*cent|%)",
    "NIM": r"NIM[^\d]*(\d+\.?\d*)\s*(?:per\s*cent|%)",
    "PCR": r"PCR[^\d]*(\d+\.?\d*)\s*(?:per\s*cent|%)",
    "SLR": r"SLR[^\d]*(\d+\.?\d*)\s*(?:per\s*cent|%)",
    "CET1": r"CET1[^\d]*(\d+\.?\d*)\s*(?:per\s*cent|%)",
    "LCR": r"LCR[^\d]*(\d+\.?\d*)\s*(?:per\s*cent|%)",
}
COHORT_TAGS = ["PSBs", "PVBs", "FBs", "SCBs", "All SCBs"]

# psycopg2 connections are not thread-safe; BRD runs parallel atomic workers.
_GOVERNANCE_DB_LOCK = threading.Lock()


@dataclass
class ConflictReport:
    has_conflict: bool
    conflicts: List[Dict[str, Any]]
    superseded_chunks: List[str]
    recommended_edition: Optional[str]
    resolution_method: str
    complementary_updates: List[Dict[str, Any]] = None  # Non-conflicting multi-edition facts

    def __post_init__(self):
        if self.complementary_updates is None:
            self.complementary_updates = []


def _nli_contradiction_prob(scores: Dict[str, float]) -> float:
    p = 0.0
    for k, v in scores.items():
        if "contrad" in str(k).lower():
            p = max(p, float(v))
    return p


def _action_from_nli(ctr: float) -> str:
    if ctr >= 0.75:
        return "SUPERSEDED"
    if ctr >= 0.40:
        return "SUPERSEDED_PARTIAL"
    return "NONE"


class EditionConflictDetector:
    def __init__(
        self,
        db_conn: Any = None,
        nli_fn: Optional[Callable[[str, str], Dict[str, float]]] = None,
    ):
        self.db = db_conn
        self.nli_fn = nli_fn

    def parse_edition_date(self, doc_id: str) -> Optional[datetime]:
        m = EDITION_PATTERN.search(doc_id or "")
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), 1)
        return None

    def detect(self, chunks: List[Dict], answer: str) -> ConflictReport:
        lock = _GOVERNANCE_DB_LOCK if self.db else nullcontext()
        with lock:
            report: ConflictReport
            if self.db:
                try:
                    report = self._detect_via_concept_index(chunks, answer)
                except Exception:
                    report = self._detect_via_regex(chunks, answer)
            else:
                report = self._detect_via_regex(chunks, answer)

            if report.has_conflict and self.db:
                try:
                    from governance_db import persist_conflict_records

                    persist_conflict_records(self.db, report.conflicts)
                except Exception as e:
                    print(f"[edition_conflict] persist skipped: {e}", flush=True)

            return report

    def _cursor_dict_rows(self, cursor: Any, sql: str, params: tuple) -> List[Dict[str, Any]]:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        if not rows:
            return []
        if isinstance(rows[0], dict):
            return rows
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, t)) for t in rows]

    def _detect_via_concept_index(self, chunks: List[Dict], answer: str) -> ConflictReport:
        conflicts: List[Dict[str, Any]] = []
        superseded: List[str] = []
        chunk_by_id = {c.get("chunk_id"): c for c in chunks if c.get("chunk_id")}
        chunk_ids = list(chunk_by_id.keys())
        if not chunk_ids:
            return self._detect_via_regex(chunks, answer)

        cursor = self.db.cursor()
        placeholders = ",".join(["%s"] * len(chunk_ids))
        sql = f"""
            SELECT entity, claim_type, value, condition, chunk_id, report_type, edition_date, section_id
            FROM concept_index
            WHERE chunk_id IN ({placeholders})
            ORDER BY entity, claim_type, edition_date NULLS LAST
        """
        try:
            rows = self._cursor_dict_rows(cursor, sql, tuple(chunk_ids))
        except Exception:
            cursor.close()
            raise

        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in rows:
            e = row.get("entity")
            ct = row.get("claim_type")
            if e is None or ct is None:
                continue
            key = (str(e), str(ct))
            groups.setdefault(key, []).append(row)

        for (entity, claim_type), entries in groups.items():
            if len(entries) < 2:
                continue
            entries_sorted = sorted(
                entries,
                key=lambda x: (str(x.get("edition_date") or ""), str(x.get("chunk_id") or "")),
            )
            oldest, newest = entries_sorted[0], entries_sorted[-1]
            if (oldest.get("edition_date") or "") == (newest.get("edition_date") or "") and oldest.get(
                "chunk_id"
            ) == newest.get("chunk_id"):
                continue

            old_chunk = chunk_by_id.get(oldest.get("chunk_id"))
            new_chunk = chunk_by_id.get(newest.get("chunk_id"))

            ctr = 0.0
            nli_confirms = False
            if self.nli_fn and old_chunk and new_chunk:
                result = self.nli_fn(
                    (new_chunk.get("text") or "")[:2000],
                    (old_chunk.get("text") or "")[:2000],
                )
                ctr = _nli_contradiction_prob(result)
                nli_confirms = ctr > 0.4

            if not nli_confirms:
                continue

            ov = oldest.get("value")
            answer_uses_older = False
            if ov is not None:
                answer_uses_older = str(ov) in answer or (
                    f"{float(ov):.1f}" in answer if isinstance(ov, (int, float)) else False
                )

            action_taken = _action_from_nli(ctr)
            old_txt = (old_chunk.get("text") if old_chunk else None) or ""
            new_txt = (new_chunk.get("text") if new_chunk else None) or ""
            old_sec = oldest.get("section_id")
            new_sec = newest.get("section_id")

            conflicts.append(
                {
                    "entity": entity,
                    "metric": claim_type,
                    "claim_type": claim_type,
                    "older_value": oldest.get("value"),
                    "older_chunk_id": oldest.get("chunk_id"),
                    "older_edition": oldest.get("edition_date"),
                    "older_section": old_sec,
                    "older_text": old_txt[:4000] if old_txt else None,
                    "newer_value": newest.get("value"),
                    "newer_chunk_id": newest.get("chunk_id"),
                    "newer_edition": newest.get("edition_date"),
                    "newer_section": new_sec,
                    "newer_text": new_txt[:4000] if new_txt else None,
                    "nli_P_contradiction": ctr,
                    "cosine_sim": None,
                    "conflict_type": "value_change",
                    "action_taken": action_taken,
                    "resolved_to": newest.get("chunk_id"),
                    "answer_uses_older": answer_uses_older,
                    "severity": "high" if answer_uses_older else "low",
                    "nli_confirmed": True,
                }
            )
            if answer_uses_older:
                superseded.append(str(oldest.get("chunk_id")))

        cursor.close()
        newest_edition = self._newest_doc_id(chunks)
        return ConflictReport(
            has_conflict=len(conflicts) > 0,
            conflicts=conflicts,
            superseded_chunks=list(set(superseded)),
            recommended_edition=newest_edition,
            resolution_method="concept_index",
        )

    def _newest_doc_id(self, chunks: List[Dict]) -> Optional[str]:
        editions: List[Tuple[datetime, str]] = []
        for c in chunks:
            did = c.get("doc_id") or ""
            dt = self.parse_edition_date(did)
            if dt:
                editions.append((dt, did))
        if not editions:
            return None
        return max(editions, key=lambda x: x[0])[1]

    def _detect_via_regex(self, chunks: List[Dict], answer: str) -> ConflictReport:
        all_metrics: List[Dict[str, Any]] = []
        for chunk in chunks:
            doc_id = chunk.get("doc_id", "") or ""
            text = chunk.get("text", "") or ""
            edition_date = self.parse_edition_date(doc_id)
            if edition_date is None:
                continue
            cohort = next((t for t in COHORT_TAGS if t in text), None)
            for metric_name, pattern in METRIC_PATTERNS.items():
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    try:
                        all_metrics.append(
                            {
                                "metric": metric_name,
                                "cohort": cohort,
                                "value": float(match.group(1)),
                                "edition_date": edition_date,
                                "doc_id": doc_id,
                                "text": text,
                            }
                        )
                    except ValueError:
                        pass

        groups: Dict[Tuple[str, Optional[str]], List[Dict[str, Any]]] = {}
        for m in all_metrics:
            key = (m["metric"], m["cohort"])
            groups.setdefault(key, []).append(m)

        conflicts: List[Dict[str, Any]] = []
        complementary_updates: List[Dict[str, Any]] = []
        superseded: List[str] = []

        for (metric, cohort), entries in groups.items():
            if len(entries) < 2:
                continue
            entries_sorted = sorted(entries, key=lambda x: x["edition_date"])
            oldest, newest = entries_sorted[0], entries_sorted[-1]

            # Same edition: skip
            if oldest["edition_date"] == newest["edition_date"]:
                continue

            value_delta = abs(oldest["value"] - newest["value"])
            answer_uses_older = str(oldest["value"]) in answer or (
                f"{oldest['value']:.1f}" in answer
            )
            answer_uses_newer = str(newest["value"]) in answer or (
                f"{newest['value']:.1f}" in answer
            )

            if value_delta > 0.1:
                # Values differ significantly → could be conflict OR complementary
                # Check: if the newer value appears in the answer, they may be aggregated differently
                # We treat as CONFLICT only if values completely disagree for same cohort/same period
                # Since we don't have period info here, treat large delta as conflict:
                conflicts.append(
                    {
                        "entity": cohort or f"metric:{metric}",
                        "metric": metric,
                        "claim_type": metric,
                        "older_value": oldest["value"],
                        "older_edition": oldest["doc_id"],
                        "newer_edition": newest["doc_id"],
                        "newer_value": newest["value"],
                        "answer_uses_older": answer_uses_older,
                        "severity": "high" if answer_uses_older else "low",
                        "nli_confirmed": False,
                        "action_taken": "NONE",
                        "nli_P_contradiction": None,
                        "cosine_sim": None,
                        "conflict_type": "value_change",
                    }
                )
                if answer_uses_older:
                    superseded.append(oldest["doc_id"])
            else:
                # Values are similar (within 0.1) but from different editions:
                # → Complementary temporal facts (different dates, same metric, same ball-park)
                # Example: 17.3% in June and 17.1% in Dec = same trend, not conflict.
                complementary_updates.append(
                    {
                        "entity": cohort or f"metric:{metric}",
                        "metric": metric,
                        "older_value": oldest["value"],
                        "older_edition": oldest["doc_id"],
                        "newer_value": newest["value"],
                        "newer_edition": newest["doc_id"],
                        "answer_includes_newer": answer_uses_newer,
                        "classification": "COMPLEMENTARY",
                    }
                )

        # Also detect multi-edition context from docs even without metric value differences
        # (different editions are present but metric data may not overlap exactly)
        all_doc_ids = list({c.get("doc_id") for c in chunks if c.get("doc_id")})
        unique_editions = []
        for did in all_doc_ids:
            dt = self.parse_edition_date(did)
            if dt:
                unique_editions.append((dt, did))
        unique_editions.sort(key=lambda x: x[0])

        # If multiple editions present but complementary_updates is empty, still signal it
        if len(unique_editions) >= 2 and not complementary_updates:
            oldest_ed = unique_editions[0]
            newest_ed = unique_editions[-1]
            # Check if the answer at least mentions the newest doc
            newest_mentioned = newest_ed[1] in answer or any(
                c.get("edition_date", "") in answer
                for c in chunks
                if c.get("doc_id") == newest_ed[1]
            )
            complementary_updates.append(
                {
                    "entity": "multi_edition_context",
                    "metric": "general",
                    "older_edition": oldest_ed[1],
                    "newer_edition": newest_ed[1],
                    "answer_includes_newer": newest_mentioned,
                    "classification": "TEMPORAL_EVOLUTION",
                }
            )

        return ConflictReport(
            has_conflict=len(conflicts) > 0,
            conflicts=conflicts,
            superseded_chunks=list(set(superseded)),
            recommended_edition=self._newest_doc_id(chunks),
            resolution_method="regex_fallback",
            complementary_updates=complementary_updates,
        )
