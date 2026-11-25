"""FastAPI backend for CyberSage AI."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from cybersage.config import RAW_DIR, CHUNKS_FILE, QDRANT_COLLECTION
from cybersage.ingestion import index_pdf_directory
from cybersage.vector_store import get_qdrant_client, collection_exists
from cybersage.workflow import answer_question

app = FastAPI(
    title="CyberSage AI API",
    description="Document-grounded cybersecurity awareness and learning assistant.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)


@app.get("/health")
def health_check():
    indexed_chunks = 0
    if CHUNKS_FILE.exists():
        with CHUNKS_FILE.open("r", encoding="utf-8") as handle:
            indexed_chunks = sum(1 for line in handle if line.strip())
    return {
        "status": "ok",
        "service": "CyberSage AI API",
        "collection": QDRANT_COLLECTION,
        "collection_ready": collection_exists(),
        "indexed_chunks": indexed_chunks,
    }


@app.get("/documents")
def list_documents():
    files = sorted(path.name for path in RAW_DIR.glob("*.pdf"))
    chunks = 0
    if CHUNKS_FILE.exists():
        with CHUNKS_FILE.open("r", encoding="utf-8") as handle:
            chunks = sum(1 for line in handle if line.strip())
    return {"pdf_files": files, "indexed_chunks": chunks}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    original_name = Path(file.filename or "upload.pdf").name
    if Path(original_name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", original_name)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF size limit is 30 MB per file.")
    target = RAW_DIR / safe_name
    target.write_bytes(content)
    return {"message": "PDF uploaded. Click Index documents to build the knowledge base.", "filename": safe_name}


@app.post("/documents/index")
def index_documents():
    try:
        return index_pdf_directory()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc


def _run_chat(request: ChatRequest) -> dict:
    history = [turn.model_dump() for turn in request.history]
    try:
        if not collection_exists():
            raise HTTPException(
                status_code=400,
                detail="The knowledge base is empty. Upload cybersecurity PDFs and index them first.",
            )
        return answer_question(request.question, history)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat request failed: {exc}") from exc


@app.post("/chat")
def chat(request: ChatRequest):
    return _run_chat(request)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Return the completed answer as an SSE event for clients needing SSE transport."""
    result = _run_chat(request)

    async def events():
        yield {"event": "answer", "data": json.dumps(result, ensure_ascii=False)}
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(events())
