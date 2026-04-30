from qdrant_client.models import Filter, FieldCondition, MatchValue
import os

def chunk_status_filter_enabled() -> bool:
    """Check if we should filter out superseded records from RAG retrieval."""
    return os.getenv("RAG_CHUNK_STATUS_FILTER", "1") != "0"

def qdrant_exclude_superseded_filter() -> Filter | None:
    """
    Returns a Qdrant filter to exclude chunks marked 'SUPERSEDED'.
    Used primarily in retriever.py.
    """
    if not chunk_status_filter_enabled():
        return None
        
    return Filter(
        must_not=[
            FieldCondition(
                key="status",
                match=MatchValue(value="SUPERSEDED")
            )
        ]
    )

def elasticsearch_superseded_must_not() -> list | None:
    """
    Returns an Elasticsearch must_not term list to exclude 'SUPERSEDED' documents.
    """
    if not chunk_status_filter_enabled():
        return None
        
    return [{"term": {"status": "SUPERSEDED"}}]

def annotate_chunk_governance_flags(chunk: dict) -> None:
    """
    Enrich a chunk with descriptive boolean flags based on its regulatory status.
    This helps the RAG / XAI layers explain WHY a piece of information is flagged.
    """
    status = chunk.get("status", "ACTIVE")
    
    # Example: mark chunks that are only partially superseded (e.g., pending final review)
    # or identify specific governance issues.
    chunk["superseded_partial"] = (status == "SUPERSEDED_PARTIAL")
    
    flags = []
    if status == "SUPERSEDED":
        flags.append("SUPERSEDED_BY_NEWER_REPORT")
    elif status == "SUPERSEDED_PARTIAL":
        flags.append("PENDING_REGULATORY_PHASE_OUT")
        
    chunk["governance_flags"] = flags
