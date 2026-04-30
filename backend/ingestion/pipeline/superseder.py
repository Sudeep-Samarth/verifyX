from __future__ import annotations
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from elasticsearch import Elasticsearch
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, ELASTICSEARCH_URL, ELASTICSEARCH_API_KEY

def supersede_older_editions(new_report_type: str, new_edition_date: str):
    """
    Finds all 'ACTIVE' chunks of the same report type with an older edition date
    and marks them as 'SUPERSEDED' in both Qdrant and Elasticsearch.
    """
    print(f"\n--- Superseding Logic: {new_report_type} ({new_edition_date}) ---", flush=True)

    # 1. Qdrant Update
    from pipeline.qdrant_cloud import client as q_client
    
    q_filter = Filter(
        must=[
            FieldCondition(key="report_type", match=MatchValue(value=new_report_type)),
        ],
        must_not=[
            FieldCondition(key="edition_date", match=MatchValue(value=new_edition_date)),
        ]
    )

    try:
        res = q_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=q_filter,
            limit=10000,
            with_payload=False
        )
        old_ids = [p.id for p in res[0]]
        
        if old_ids:
            print(f"  -> Qdrant: Marking {len(old_ids)} older chunks as SUPERSEDED...", flush=True)
            q_client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"status": "SUPERSEDED"},
                points=old_ids,
                wait=True
            )
        else:
            print("  -> Qdrant: No older editions found to supersede.", flush=True)
    except Exception as e:
        print(f"  -> Qdrant Supersede Error: {e}", flush=True)

    # 2. Elasticsearch Update
    if ELASTICSEARCH_URL and ELASTICSEARCH_API_KEY:
        es_client = Elasticsearch(ELASTICSEARCH_URL, api_key=ELASTICSEARCH_API_KEY, verify_certs=False)
        es_index = COLLECTION_NAME.lower()
        
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"report_type": new_report_type}}
                    ],
                    "must_not": [
                        {"term": {"edition_date": new_edition_date}}
                    ]
                }
            },
            "script": {
                "source": "ctx._source.status = 'SUPERSEDED'",
                "lang": "painless"
            }
        }
        
        try:
            if es_client.indices.exists(index=es_index):
                res = es_client.update_by_query(index=es_index, body=query, wait_for_completion=True)
                updated = res.get("updated", 0)
                if updated > 0:
                    print(f"  -> Elasticsearch: Updated {updated} older chunks to SUPERSEDED.", flush=True)
        except Exception as e:
            print(f"  -> Elasticsearch Supersede Error: {e}", flush=True)
