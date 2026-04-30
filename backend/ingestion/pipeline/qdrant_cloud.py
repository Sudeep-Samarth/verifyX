from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME
import time

import os
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
client = QdrantClient(
    path=os.path.join(_BASE, "local_qdrant")
)

def init_collection(vector_size, force_recreate: bool = False):
    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if COLLECTION_NAME in names:
        if not force_recreate:
            print(f"[qdrant] Collection '{COLLECTION_NAME}' already exists. Skipping recreation.")
            return
        print(f"[qdrant] Force-recreating collection '{COLLECTION_NAME}'...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    # Ensure payload indexes exist for efficient filtering
    from qdrant_client.models import PayloadSchemaType
    for field in ["report_type", "edition_date", "status"]:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD
            )
        except Exception:
            pass 
    
    # Optional stabilization sleep for Qdrant Cloud indexing
    print("[qdrant] Waiting for indexes to stabilize...", flush=True)
    time.sleep(5) 

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def upload(chunks, batch_size=100, max_workers=1):
    """Parallel batch upload to Qdrant Cloud."""
    points = []
    for c in chunks:
        points.append(
            PointStruct(
                id=c["chunk_id"],
                vector=c["vector"],
                payload={
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "report_type": c["report_type"],
                    "edition_date": c["edition_date"],
                    "section_id": c["section_id"],
                    "section_title": c["section_title"],
                    "chunk_type": c["chunk_type"],
                    "page_number": c["page_number"],
                    "parent_chunk_id": c["parent_chunk_id"],
                    "footnote_ids": c["footnote_ids"],
                    "cross_ref_ids": c["cross_ref_ids"]
                }
            )
        )

    # 1. Prepare batches
    batches = [points[i:i + batch_size] for i in range(0, len(points), batch_size)]
    
    # 2. Parallel upload with Progress Bar
    print(f"  -> Starting upload...", flush=True)
    
    def _upsert_batch(batch):
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            return True
        except Exception as e:
            print(f"\n[qdrant] Batch failed: {e}")
            return False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(
            executor.map(_upsert_batch, batches),
            total=len(batches),
            desc="  -> Qdrant Upload",
            unit="batch"
        ))

    print(f"  → Uploaded {len(points)} vectors successfully.", flush=True)
