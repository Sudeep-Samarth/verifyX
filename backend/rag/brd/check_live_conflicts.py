import os
import sys
from dotenv import load_dotenv

# The file is in backend/rag/brd/check_live_conflicts.py
_RAG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND = os.path.dirname(_RAG)
if _RAG not in sys.path:
    sys.path.insert(0, _RAG)

load_dotenv(os.path.join(_BACKEND, ".env"))

from chat import answer_rag_sync

def test_live_conflict():
    print("==================================================")
    print("        LIVE END-TO-END CONFLICT CHECK            ")
    print("==================================================")
    
    # RAG_CHUNK_STATUS_FILTER=0 must be set in .env for this to find historical conflicts
    filter_val = os.getenv("RAG_CHUNK_STATUS_FILTER", "1")
    if filter_val == "1":
        print("[!] Note: RAG_CHUNK_STATUS_FILTER is ENABLED.")
        print("    The retriever will only see the latest data, so no conflicts will likely trigger.")
        print("    To test detection logic, set it to 0 in .env temporarily.\n")

    # Ask a question known to appear in multiple editions (e.g., CRAR of SCBs)
    query = "What was the CRAR of all SCBs?"
    print(f"Querying: '{query}'...")
    
    # Run the full pipeline
    output_text = answer_rag_sync(query)
    
    print("\n--- Answer ---")
    print(output_text)
    
    print("\n--- Conflict Analysis (from Artifact JSON) ---")
    artifact_path = os.getenv("XAI_ARTIFACT_JSON", "backend/xai_last_run.json")
    if os.path.exists(artifact_path):
        import json
        with open(artifact_path, "r", encoding="utf-8") as f:
            artifact = json.load(f)
        
        conflicts = artifact.get("edition_conflicts", [])
        if conflicts:
            print(f"[!] DETECTED {len(conflicts)} CONFLICTS in final output:")
            for i, c in enumerate(conflicts, 1):
                print(f"\nConflict #{i}:")
                print(f"  - Entity: {c.get('entity')}")
                print(f"  - Metric: {c.get('metric')}")
                print(f"  - Claim: {c.get('older_value')} vs {c.get('newer_value')}")
        else:
            print("[OK] No edition conflicts detected in XAI artifact.")
    else:
        print("[!] XAI artifact JSON not found. Check if XAI_ENABLED=1.")

    print("\n==================================================")

if __name__ == "__main__":
    test_live_conflict()
