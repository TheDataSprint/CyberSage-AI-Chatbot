"""Hybrid retrieval using Qdrant semantic search and Rank-BM25 keyword search."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from cybersage.config import CHUNKS_FILE, QDRANT_COLLECTION, RETRIEVAL_TOP_K
from cybersage.embeddings import embed_query
from cybersage.vector_store import get_qdrant_client, collection_exists

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def load_chunk_records(path: Path = CHUNKS_FILE) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]], k: int = 60
) -> list[dict[str, Any]]:
    """Merge result lists without comparing raw dense and BM25 scores."""
    merged: dict[str, dict[str, Any]] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = str(item["chunk_id"])
            if key not in merged:
                merged[key] = dict(item)
                merged[key]["rrf_score"] = 0.0
            merged[key]["rrf_score"] += 1.0 / (k + rank)
    return sorted(merged.values(), key=lambda item: item["rrf_score"], reverse=True)


def retrieve_relevant_chunks(
    query: str, top_k: int = RETRIEVAL_TOP_K
) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    if not collection_exists():
        return []

    dense_vector = embed_query(query)
    client = get_qdrant_client()
    dense_response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=dense_vector,
        limit=max(top_k * 3, 10),
        with_payload=True,
    )
    dense_results: list[dict[str, Any]] = []
    for point in dense_response.points:
        payload = point.payload or {}
        dense_results.append(
            {
                "chunk_id": str(payload.get("chunk_id", point.id)),
                "text": str(payload.get("text", "")),
                "source": str(payload.get("source", "Unknown source")),
                "page": int(payload.get("page", 0)),
                "dense_score": float(point.score or 0.0),
            }
        )

    records = load_chunk_records()
    keyword_results: list[dict[str, Any]] = []
    tokenized_corpus = [tokenize(item.get("text", "")) for item in records]
    if records and any(tokenized_corpus):
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        for index in ranked_indices[: max(top_k * 3, 10)]:
            item = records[index]
            keyword_results.append(
                {
                    "chunk_id": str(item["chunk_id"]),
                    "text": item["text"],
                    "source": item["source"],
                    "page": int(item["page"]),
                    "bm25_score": float(scores[index]),
                }
            )

    results = reciprocal_rank_fusion([dense_results, keyword_results])
    return results[:top_k]
