"""Persistent local Qdrant client and collection helpers."""
from __future__ import annotations
from functools import lru_cache
from qdrant_client import QdrantClient, models
from cybersage.config import QDRANT_PATH, QDRANT_COLLECTION, VECTOR_SIZE


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Keep one local-mode client per process to avoid storage-lock conflicts."""
    return QdrantClient(path=str(QDRANT_PATH))


def collection_exists() -> bool:
    return get_qdrant_client().collection_exists(collection_name=QDRANT_COLLECTION)


def create_or_reset_collection(reset: bool = False) -> None:
    client = get_qdrant_client()
    exists = client.collection_exists(collection_name=QDRANT_COLLECTION)
    if exists and reset:
        client.delete_collection(collection_name=QDRANT_COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
