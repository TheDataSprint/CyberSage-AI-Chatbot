# 🛡️ CyberSage AI

**A Retrieval-Augmented Generation (RAG) cybersecurity knowledge and threat-awareness assistant.**

CyberSage AI answers cybersecurity learning questions from PDFs that you provide. It combines semantic search (FastEmbed + Qdrant) with keyword search (Rank-BM25), merges the results using Reciprocal Rank Fusion, and uses NVIDIA Nemotron to generate grounded answers with source/page references. A LangGraph workflow coordinates retrieval and generation. Streamlit provides the user interface and FastAPI exposes the backend.

> Scope: defensive security education, prevention, detection, secure development and incident-response awareness. The assistant is not a live security scanner and does not replace professional incident response.

## 1. Technology stack

- **Python** — application language
- **Streamlit** — interactive frontend
- **FastAPI + Uvicorn** — backend API
- **LangChain + PyPDF** — PDF loading and text chunking
- **FastEmbed** — `BAAI/bge-small-en-v1.5` embeddings (384 dimensions)
- **Qdrant** — local persistent vector database
- **Rank-BM25** — keyword retrieval
- **LangGraph** — RAG workflow orchestration
- **OpenAI SDK + NVIDIA API** — NVIDIA Nemotron model connection
- **SSE-Starlette** — Server-Sent Events endpoint
- **Pydantic, python-dotenv, HTTPX/Requests, Tenacity** — configuration, validation, HTTP and retries

The dependency list is based on the technology set in the reference project. This implementation uses the components needed for the CyberSage features; a package being listed in requirements does not mean every package is imported by every module.

## 2. Architecture

```text
Cybersecurity PDF documents
          │
          ▼
       PyPDF
          │
          ▼
LangChain RecursiveCharacterTextSplitter
          │
          ├──────────────► chunks.jsonl ───────► Rank-BM25
          │
          ▼
      FastEmbed
          │
          ▼
        Qdrant
          │
          ▼
  Hybrid retrieval (semantic + keyword)
          │
          ▼
 Reciprocal Rank Fusion (result merging)
          │
          ▼
      LangGraph
          │
          ▼
 NVIDIA Nemotron through NVIDIA API
          │
          ▼
   Answer + source/page references
          │
          ▼
       Streamlit UI

FastAPI exposes upload, indexing, health, chat and SSE endpoints.
```

## 3. Project structure

```text
CyberSage-AI-Chatbot/
├── app.py                       # Streamlit frontend
├── api/
│   └── main.py                  # FastAPI endpoints
├── cybersage/
│   ├── config.py                # Environment and paths
│   ├── embeddings.py            # FastEmbed model
│   ├── vector_store.py          # Persistent Qdrant client
│   ├── ingestion.py             # PDF extraction, chunking and indexing
│   ├── retrieval.py             # Qdrant + BM25 hybrid search
│   ├── llm.py                   # NVIDIA Nemotron answer generation
│   └── workflow.py              # LangGraph RAG workflow
├── data/
│   ├── raw/                     # Add source PDFs here
│   └── processed/               # Generated chunks.jsonl
├── qdrant_db/                   # Local Qdrant storage
├── tests/                       # Small deterministic tests
├── .env.example
├── requirements.txt
├── KNOWLEDGE_SOURCES.md
└── README.md
```

## 4. Setup on Windows + Ubuntu (WSL)

Open Ubuntu (WSL). Change directory to wherever you extracted the project. Example:

```bash
cd "/mnt/d/Documents/Projects/CyberSage-AI-Chatbot"
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If the `venv` module is missing on Ubuntu:

```bash
sudo apt update
sudo apt install python3-venv
```

Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first embedding-model use downloads `BAAI/bge-small-en-v1.5` to the local FastEmbed model cache. Make sure the machine has internet access for the first run.

## 5. Configure NVIDIA API

1. Copy `.env.example` to `.env`:

   ```bash
   cp .env.example .env
   ```
2. Open `.env` in VS Code.
3. Replace `PASTE_YOUR_NVIDIA_API_KEY_HERE` with your private key from NVIDIA Build.
4. Keep `.env` private. It is excluded from Git.

The default model is the same Nemotron model identified in the reference backend: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`. NVIDIA documents an OpenAI-compatible endpoint at `https://integrate.api.nvidia.com/v1/chat/completions` and supports this model through that endpoint. If model availability or your account's access changes, update `NVIDIA_MODEL` in `.env` to a model available to your NVIDIA API key.

## 6. Add cybersecurity PDFs

Place 3–5 trusted, searchable cybersecurity PDFs in `data/raw/`. Suggestions and official source links are in `KNOWLEDGE_SOURCES.md`.

The initial chunking parameters mirror the reference implementation:

- Chunk size: 500 characters
- Chunk overlap: 50 characters
- Embedding model: `BAAI/bge-small-en-v1.5`
- Vector dimension: 384
- Default retrieval count: 5

## 7. Run the project

The Streamlit frontend calls the FastAPI backend. Run them in **two separate Ubuntu terminals**. Do not start multiple backend processes against the same local Qdrant folder.

### Terminal 1 — Backend

```bash
cd "/mnt/d/Documents/Projects/CyberSage-AI-Chatbot"
source .venv/bin/activate
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Backend health: http://127.0.0.1:8000/health  
Interactive API docs: http://127.0.0.1:8000/docs

### Terminal 2 — Frontend

```bash
cd "/mnt/d/Documents/Projects/CyberSage-AI-Chatbot"
source .venv/bin/activate
python -m streamlit run app.py
```

Frontend: http://localhost:8501

### First-time indexing

1. Open the Streamlit page.
2. Select **Knowledge Base**.
3. Upload one or more PDF files.
4. Select **Index all PDFs in the knowledge base**.
5. Wait for embedding and Qdrant indexing to finish.
6. Open **Ask CyberSage** and ask a question covered by the PDFs.

## 8. API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Backend and index status |
| GET | `/documents` | List uploaded PDFs and indexed chunk count |
| POST | `/documents/upload` | Upload one PDF (`multipart/form-data`, field `file`) |
| POST | `/documents/index` | Rebuild the collection from all PDFs in `data/raw/` |
| POST | `/chat` | Ask a question; accepts JSON question and optional history |
| POST | `/chat/stream` | Return the answer over Server-Sent Events |

Example chat request:

```json
{
  "question": "What are common warning signs of a phishing message?",
  "history": []
}
```

## 9. Run tests and compile checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py api cybersage
```

## 10. Important limitations

- Answers depend on the PDFs you upload and index. No relevant document means no grounded answer.
- Text extraction may be poor for scanned/image-only PDFs. This version does not add OCR.
- The knowledge base is not automatically updated. Re-index when you add or replace PDFs.
- Qdrant runs in local persistent mode. Run one API backend process for a given `qdrant_db` directory.
- NVIDIA API access requires a valid key and available model entitlement. API usage may have rate limits or costs depending on the account and service terms.
- This application is an educational RAG assistant, not a threat-intelligence platform, vulnerability scanner, or incident-response service.

## 11. Suggested future improvements within the same stack

- Add per-document deletion and incremental indexing.
- Add retrieval evaluation questions and precision/recall measurements.
- Add conversation persistence with LangGraph checkpointing.
- Add richer SSE token streaming.
- Add a structured cyber-awareness quiz bank with progress tracking.
- Add role-based access and authentication before public deployment.

## 12. Credits and references

- Qdrant Python client: https://github.com/qdrant/qdrant-client
- LangGraph StateGraph: https://reference.langchain.com/python/langgraph/graph/state/StateGraph
- NVIDIA Nemotron model API: https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning/apireference
- Suggested cyber knowledge sources: see `KNOWLEDGE_SOURCES.md`
