from elasticsearch import Elasticsearch
from config import ELASTICSEARCH_URL, ELASTICSEARCH_API_KEY, COLLECTION_NAME

ES_INDEX_NAME = COLLECTION_NAME.lower()

client = Elasticsearch(
    ELASTICSEARCH_URL,
    api_key=ELASTICSEARCH_API_KEY,
    request_timeout=30,
    retry_on_timeout=True,
    verify_certs=False
)

def init_es_index(force_recreate: bool = False):
    # 1. Check Connectivity
    print(f"  -> Connecting to Elasticsearch at {ELASTICSEARCH_URL}...", flush=True)
    try:
        if not client.ping():
            raise ConnectionError("Elasticsearch ping failed. Check your URL and API Key.")
        print("  -> Connection successful.", flush=True)
    except Exception as e:
        print(f"  -> Connection failed: {e}", flush=True)
        raise

    # 2. Check and Delete Index
    print(f"  -> Checking if index {ES_INDEX_NAME!r} exists...", flush=True)
    if client.indices.exists(index=ES_INDEX_NAME):
        if not force_recreate:
            print(f"  -> Index {ES_INDEX_NAME!r} already exists. Skipping recreation.")
            return
        print(f"  -> Force-recreating index {ES_INDEX_NAME!r}...", flush=True)
        client.indices.delete(index=ES_INDEX_NAME)
    
    # 3. Create Index
    print(f"  -> Creating new index {ES_INDEX_NAME!r} with BM25 mappings...", flush=True)
    client.indices.create(
        index=ES_INDEX_NAME,
        body={
            "mappings": {
                "properties": {
                    "text": {"type": "text", "analyzer": "standard"},
                    "chunk_id": {"type": "keyword"},
                    "report_type": {"type": "keyword"},
                    "edition_date": {"type": "keyword"},
                    "status": {"type": "keyword"},
                    "section_id": {"type": "keyword"},
                    "section_title": {"type": "text"},
                    "chunk_type": {"type": "keyword"},
                    "page_number": {"type": "integer"}
                }
            }
        }
    )
    print(f"  -> Index {ES_INDEX_NAME!r} initialized successfully.", flush=True)


def upload_to_es(chunks):
    from elasticsearch.helpers import bulk
    
    actions = []
    for c in chunks:
        doc = {
            "_index": ES_INDEX_NAME,
            "_id": c["chunk_id"],
            "_source": {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "report_type": c.get("report_type", ""),
                "edition_date": c.get("edition_date", ""),
                "section_id": c.get("section_id", ""),
                "section_title": c.get("section_title", ""),
                "chunk_type": c.get("chunk_type", ""),
                "page_number": c.get("page_number", 0)
            }
        }
        actions.append(doc)
    
    success, failed = bulk(client, actions)
    return success
