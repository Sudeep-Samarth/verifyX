import os
from dotenv import load_dotenv

# Always load backend/.env regardless of cwd (e.g. python ingest.py from backend/ingestion).
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE, ".env"))
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "rbi_chunks"

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Embeddings: set EMBED_PROVIDER in this directory's parent .env (backend/.env): local | ollama.
# Runtime API: pipeline.embedder.get_embed_provider()

DATA_PATH = os.path.join(_BASE, "data", "pdfs")