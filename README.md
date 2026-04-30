# VerifyX - Compliance RAG Engine 🛡️

**VerifyX** (formerly Char-Chatore) is a production-grade Regulatory RAG (Retrieval-Augmented Generation) system designed to transform raw regulatory documents (RBI, SEBI, etc.) into high-trust compliance engines.

It specializes in strict grounding, explainable attribution, and analyzing massive Business Requirement Documents (BRDs) against complex regulatory guidelines to highlight violations and provide remediations.

## 🌟 Key Features

* **Query Bot**: Ask natural language questions about your regulatory corpus. It uses a Hybrid RAG approach (Qdrant for semantic search + Elasticsearch for exact keyword matching) to guarantee high recall.
* **BRD Analyzer**: Upload your Business Requirement Documents (.pdf, .docx, or .txt). The engine automatically extracts atomic requirements, validates them, and checks them against rules retrieved from the vector database.
* **XAI & Compliance Heatmaps**: Visually map which requirements pass, fail, or fall into a grey area. Powered by cross-encoder NLI (Natural Language Inference) models.
* **Fully Local-Ready**: Supports running local embedding models via Ollama to keep your proprietary documents completely private.
* **Supabase Governance**: Persists chats, UI snapshots, and metadata via PostgreSQL.

---

## 🚀 Setup & Installation

Please refer to the `setup_instructions.txt` file in this repository for a complete, step-by-step guide on how to configure Elasticsearch, Qdrant, Supabase, Ollama, and Groq to get this running locally.

### Quick Start (Once Infrastructure is Configured)

1. **Start the Backend:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn main:app
   ```

2. **Start the Frontend:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Access the App:** Open `http://localhost:3000`

---

## 🏗️ Architecture

- **Frontend**: Next.js 14 (App Router), React, TailwindCSS, Framer Motion.
- **Backend**: Python, FastAPI, Qdrant (Vector DB), Elasticsearch (Lexical DB).
- **AI Models**: Groq (Llama-3.1-8b) for ultra-fast reasoning, Ollama (all-minilm) for embeddings, DeBERTa (Cross-Encoder) for NLI validation.
- **Database**: Supabase (PostgreSQL) for user auth and artifact storage.

---

## 📄 License

This project is licensed under the MIT License.
