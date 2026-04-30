import os
import sys
import json
import time

# --- Path setup ---
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

from config import COLLECTION_NAME, GROQ_API_KEY
from pipeline.qdrant_cloud import client as qdrant_client
from xai.edition_conflict import EditionConflictDetector
from xai.nli_verifier import TwoStageNLIVerifier
from governance_db import get_governance_db_connection, governance_db_url

def check_supabase():
    print("--- Phase 1: Supabase / PostgreSQL Connectivity ---")
    url = governance_db_url()
    if not url or "[PASSWORD]" in url:
        print("[X] SUPABASE_DB_URL is not configured (or still has placeholders).")
        return False
    
    try:
        conn = get_governance_db_connection(fresh=True)
        if conn:
            print("[OK] Successfully connected to Supabase PostgreSQL.")
            cur = conn.cursor()
            
            # Check for tables
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            tables = [r[0] for r in cur.fetchall()]
            
            for t in ["concept_index", "conflicts"]:
                if t in tables:
                    print(f"[OK] Table found: {t}")
                    cur.execute(f"SELECT count(*) FROM {t}")
                    count = cur.fetchone()[0]
                    print(f"   (Contains {count} rows)")
                else:
                    print(f"[X] Table MISSING: {t}. Did you run the DDL in Supabase Editor?")
            
            conn.close()
            return True
        else:
            print("[X] Failed to establish a connection (get_governance_db_connection returned None).")
            return False
    except Exception as e:
        print(f"[X] Supabase Connection error: {e}")
        return False

def check_qdrant_status():
    print("\n--- Phase 2: Qdrant Status Filtering ---")
    filter_active = os.getenv("RAG_CHUNK_STATUS_FILTER", "1") == "1"
    print(f"RAG_CHUNK_STATUS_FILTER is {'ENABLED' if filter_active else 'DISABLED'}")
    
    try:
        results = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5,
            with_payload=True
        )
        points = results[0]
        if not points:
            print("[!] Qdrant collection is empty. Cannot verify status payloads.")
            return
            
        print(f"Verifying {len(points)} random points for 'status' payload...")
        has_status = 0
        superseded = 0
        for p in points:
            payload = p.payload
            status = payload.get("status", "NOT_SET")
            print(f" - Point {p.id}: status={status}")
            if "status" in payload:
                has_status += 1
            if status == "SUPERSEDED":
                superseded += 1
        
        if has_status > 0:
            print(f"[OK] Status payload found in {has_status}/{len(points)} sampled points.")
        else:
            print("[!] No status fields found in sampled points. Ingest defaults may not have run yet.")
            
    except Exception as e:
        print(f"[X] Qdrant verification failed: {e}")

def simulate_conflict():
    print("\n--- Phase 3: Edition Conflict Logic (Mock) ---")
    # Mock NLI that always contradicts (for testing detection)
    def mock_nli(p, h):
        return {"contradiction": 0.9, "entailment": 0.1, "neutral": 0.0}
    
    detector = EditionConflictDetector(db_conn=None, nli_fn=mock_nli)
    
    # Mock chunks
    chunks = [
        {
            "chunk_id": "old_1",
            "doc_id": "FSR (2022-06)",
            "text": "The CRAR of all SCBs was 16.7 per cent in March 2022.",
            "status": "ACTIVE"
        },
        {
            "chunk_id": "new_1",
            "doc_id": "FSR (2023-12)",
            "text": "The CRAR of all SCBs was 17.2 per cent in Sept 2023.",
            "status": "ACTIVE"
        }
    ]
    
    print("Simulating conflict detection between FSR 2022 and FSR 2023...")
    # The regex fallback should catch 'CRAR' and 'All SCBs' (All SCBs is in COHORT_TAGS)
    report = detector.detect(chunks, answer="The CRAR of all SCBs is 16.7 per cent.")
    
    if report.has_conflict:
        print("[OK] Conflict DETECTED correctly via regex/NLI.")
        for c in report.conflicts:
            print(f"   - Entity: {c['entity']}")
            print(f"   - Metric: {c['metric']}")
            print(f"   - Outcome: {c['action_taken']} (Older: {c['older_value']}, Newer: {c['newer_value']})")
            print(f"   - Answer uses older? {c['answer_uses_older']}")
    else:
        print("[X] Conflict NOT detected. Check regex patterns in edition_conflict.py.")

if __name__ == "__main__":
    print("==================================================")
    print("      REGULATORY CONFLICT PIPELINE DIAGNOSTIC     ")
    print("==================================================\n")
    
    success = check_supabase()
    check_qdrant_status()
    simulate_conflict()
    
    print("\n==================================================")
    print("Diagnostic Complete.")
