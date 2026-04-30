import os
from tqdm import tqdm

from config import DATA_PATH
from pipeline.pdf_reader import read_pdf
from pipeline.chunker import chunk_text
from pipeline.embedder import embed_chunks
from pipeline.qdrant_cloud import init_collection, upload
from pipeline.elasticsearch_cloud import init_es_index, upload_to_es


def run(force_recreate: bool = False):
    all_chunks = []

    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]

    print(f"Found {len(files)} PDFs")

    for file in tqdm(files):
        path = os.path.join(DATA_PATH, file)

        name_parts = file.replace(".pdf", "").split("_")
        report_type = name_parts[0] if len(name_parts) > 0 else "Unknown"
        edition_date = f"{name_parts[1]}-{name_parts[2]}" if len(name_parts) > 2 else "Unknown"

        semantic_blocks = read_pdf(path)
        chunks = chunk_text(semantic_blocks)

        for c in chunks:
            c["report_type"] = report_type
            c["edition_date"] = edition_date
            c["status"] = "ACTIVE"

        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    from pipeline.embedder import embedding_backend_label, get_embed_provider
    print(
        f"Embedding provider: {get_embed_provider()} - {embedding_backend_label()}",
        flush=True,
    )
    if get_embed_provider() == "ollama":
        print(
            "  (Qdrant vector size follows Ollama model, e.g. all-minilm~384, nomic-embed-text~768 - match chat .env.)",
            flush=True,
        )

    embedded = embed_chunks(all_chunks)

    # init qdrant
    vector_size = len(embedded[0]["vector"])
    init_collection(vector_size, force_recreate=force_recreate)

    # init elasticsearch
    print("\n--- Elasticsearch Sync ---")
    es_available = False
    try:
        init_es_index(force_recreate=force_recreate)
        es_available = True
    except Exception as e:
        print(f"Warning: Elasticsearch initialization failed. Error: {e}")

    # upload qdrant
    print("\n--- Qdrant Sync ---")
    print(f"Uploading {len(embedded)} vectors to Qdrant...")
    upload(embedded)
    print("Done.")

    # Superseding logic (Clean up older editions)
    try:
        from pipeline.superseder import supersede_older_editions
        # We process unique report_type/edition pairs from this run across ALL chunks
        processed_pairs = set((c["report_type"], c["edition_date"]) for c in all_chunks)
        for r_type, e_date in processed_pairs:
            supersede_older_editions(r_type, e_date)
    except Exception as e:
        print(f"Warning: Superseding cleanup failed: {e}")
    
    # upload elasticsearch
    if es_available:
        print("\n--- Elasticsearch Upload ---")
        print(f"Indexing {len(embedded)} chunks to Elasticsearch...")
        try:
            es_success = upload_to_es(embedded)
            print(f"Successfully indexed {es_success} chunks.")
        except Exception as e:
            print("Warning: Elasticsearch upload failed. Error:", e)
    else:
        print("\nSkipping Elasticsearch upload (Initialization failed).")

    print("\nINGESTION COMPLETE -> DATA STORED IN QDRANT CLOUD & ELASTICSEARCH")



if __name__ == "__main__":
    run()