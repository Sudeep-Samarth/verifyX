from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from layer0_ingestrion.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def search_query(query: str, top_k: int = 5):
    vector = model.encode(query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True
    )

    output = []
    for r in results.points:
        output.append({
            "score": r.score,
            "text": r.payload["text"]
        })

    return output


if __name__ == "__main__":
    query = "what is the main concept of the documents?"
    results = search_query(query)

    for r in results:
        print("\nScore:", r["score"])
        print("Text:", r["text"][:300])