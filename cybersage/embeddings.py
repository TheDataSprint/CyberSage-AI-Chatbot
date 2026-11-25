"""FastEmbed helpers."""
from __future__ import annotations
from functools import lru_cache
from fastembed import TextEmbedding
from cybersage.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    """Load the embedding model once per process."""
    return TextEmbedding(model_name=EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert text strings into ordinary Python vectors."""
    if not texts:
        return []
    model = get_embedding_model()
    return [vector.tolist() for vector in model.embed(texts)]


def embed_query(query: str) -> list[float]:
    vectors = embed_texts([query])
    if not vectors:
        raise ValueError("The query is empty; an embedding could not be generated.")
    return vectors[0]
