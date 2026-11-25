
"""PDF ingestion: extract, split, embed, and index cybersecurity documents."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct

from cybersage.config import (
    RAW_DIR,
    CHUNKS_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    QDRANT_COLLECTION,
)
from cybersage.embeddings import embed_texts
from cybersage.vector_store import (
    get_qdrant_client,
    create_or_reset_collection,
)


NAMESPACE = uuid.UUID("e82d44ab-5d8a-4e82-b16e-9ba7a0b52c42")


def get_source_name(pdf_path: Path) -> str:
    """
    Return a readable path relative to data/raw.

    This helps distinguish files with identical filenames
    stored in different folders.
    """
    try:
        return pdf_path.resolve().relative_to(RAW_DIR.resolve()).as_posix()
    except ValueError:
        return pdf_path.name


def calculate_file_hash(pdf_path: Path) -> str:
    """Calculate SHA-256 hash to identify exact duplicate PDFs."""
    sha256 = hashlib.sha256()

    with pdf_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def discover_pdf_files(
    pdf_paths: list[Path] | None = None,
) -> tuple[list[Path], int]:
    """
    Find PDFs recursively, including all subfolders.

    Exact duplicate files are skipped using their SHA-256 hashes.
    """
    if pdf_paths is None:
        candidates = list(RAW_DIR.rglob("*"))
    else:
        candidates = [Path(path) for path in pdf_paths]

    pdf_files = sorted(
        {
            path.resolve()
            for path in candidates
            if path.is_file() and path.suffix.lower() == ".pdf"
        },
        key=lambda path: path.as_posix().lower(),
    )

    discovered_count = len(pdf_files)

    unique_files: list[Path] = []
    seen_hashes: set[str] = set()

    for pdf_path in pdf_files:
        try:
            file_hash = calculate_file_hash(pdf_path)
        except OSError as exc:
            raise RuntimeError(
                f"Could not read PDF file {pdf_path}: {exc}"
            ) from exc

        if file_hash in seen_hashes:
            continue

        seen_hashes.add(file_hash)
        unique_files.append(pdf_path)

    duplicate_count = discovered_count - len(unique_files)

    return unique_files, duplicate_count


def extract_pdf_documents(pdf_path: Path) -> list[Document]:
    """
    Extract one LangChain Document per non-empty PDF page.

    Metadata preserves the source filename/path and page number.
    """
    reader = PdfReader(str(pdf_path))
    documents: list[Document] = []

    source = get_source_name(pdf_path)

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "page": page_number,
                },
            )
        )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split extracted pages into smaller text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_documents(documents)


def index_pdf_directory(
    pdf_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """
    Rebuild the Qdrant collection using every unique PDF.

    Searches recursively through data/raw and all its subfolders.
    """
    files, duplicate_count = discover_pdf_files(pdf_paths)

    if not files:
        raise ValueError(
            f"No PDF files found. Add PDFs to: {RAW_DIR}"
        )

    page_documents: list[Document] = []
    errors: list[str] = []
    successfully_read_sources: set[str] = set()

    # --------------------------------------------------------
    # 1. EXTRACT TEXT FROM EVERY PDF
    # --------------------------------------------------------
    for pdf_path in files:
        source = get_source_name(pdf_path)

        try:
            extracted_documents = extract_pdf_documents(pdf_path)

            if not extracted_documents:
                errors.append(
                    f"{source}: No extractable text found. "
                    "The PDF may be scanned or image-based."
                )
                continue

            page_documents.extend(extracted_documents)
            successfully_read_sources.add(source)

        except Exception as exc:
            errors.append(f"{source}: {exc}")

    if not page_documents:
        details = "; ".join(errors) if errors else "No readable PDF text."
        raise ValueError(
            f"No usable PDF text could be extracted. {details}"
        )

    # --------------------------------------------------------
    # 2. SPLIT DOCUMENTS INTO CHUNKS
    # --------------------------------------------------------
    chunks = split_documents(page_documents)

    if not chunks:
        raise ValueError("PDF text was extracted, but no chunks were created.")

    # --------------------------------------------------------
    # 3. PREPARE CHUNK RECORDS
    # --------------------------------------------------------
    records: list[dict[str, Any]] = []

    for index, document in enumerate(chunks):
        source = str(document.metadata.get("source", "unknown.pdf"))
        page = int(document.metadata.get("page", 0))
        content = document.page_content.strip()

        if not content:
            continue

        chunk_id = str(
            uuid.uuid5(
                NAMESPACE,
                f"{source}|{page}|{index}|{content}",
            )
        )

        records.append(
            {
                "chunk_id": chunk_id,
                "text": content,
                "source": source,
                "page": page,
                "chunk_number": index + 1,
            }
        )

    if not records:
        raise ValueError("No valid text chunks were generated.")

    # --------------------------------------------------------
    # 4. GENERATE EMBEDDINGS AND PREPARE QDRANT POINTS
    # --------------------------------------------------------
    batch_size = 64
    prepared_batches: list[list[PointStruct]] = []

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]

        vectors = embed_texts(
            [item["text"] for item in batch]
        )

        if len(vectors) != len(batch):
            raise RuntimeError(
                "Embedding count does not match the number of text chunks."
            )

        points: list[PointStruct] = []

        for item, vector in zip(batch, vectors):
            if len(vector) != 384:
                raise RuntimeError(
                    f"Expected 384-dimensional embedding, "
                    f"received {len(vector)} dimensions."
                )

            points.append(
                PointStruct(
                    id=item["chunk_id"],
                    vector=vector,
                    payload={
                        "chunk_id": item["chunk_id"],
                        "text": item["text"],
                        "source": item["source"],
                        "page": item["page"],
                        "chunk_number": item["chunk_number"],
                    },
                )
            )

        prepared_batches.append(points)

    # --------------------------------------------------------
    # 5. SAVE TEXT CORPUS FOR BM25
    # --------------------------------------------------------
    CHUNKS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = CHUNKS_FILE.with_suffix(".jsonl.tmp")

    with temp_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # --------------------------------------------------------
    # 6. RESET QDRANT COLLECTION AND INDEX ALL DOCUMENTS
    # --------------------------------------------------------
    create_or_reset_collection(reset=True)

    client = get_qdrant_client()

    for points in prepared_batches:
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
            wait=True,
        )

    # Save BM25 corpus after successful vector indexing.
    temp_file.replace(CHUNKS_FILE)

    # --------------------------------------------------------
    # 7. RETURN INDEXING REPORT
    # --------------------------------------------------------
    return {
        "files_seen": len(files),
        "files_found": len(files) + duplicate_count,
        "files_indexed": len(successfully_read_sources),
        "duplicate_files_skipped": duplicate_count,
        "pages_with_text": len(page_documents),
        "chunks_indexed": len(records),
        "collection": QDRANT_COLLECTION,
        "sources_indexed": sorted(successfully_read_sources),
        "errors": errors,
    }
