"""
Task 5 - Semantic search on the local vector index.

Uses the same deterministic hashing embedding created in Task 4 and searches
the local JSONL vector store at data/index/chunks.jsonl.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

try:
    from .task4_chunking_indexing import INDEX_PATH, _hash_embedding, run_pipeline
except ImportError:  # Allows running this file directly: python src/task5_semantic_search.py
    from task4_chunking_indexing import INDEX_PATH, _hash_embedding, run_pipeline


def _load_index() -> list[dict]:
    """Load chunks from the local JSONL vector index."""
    if not INDEX_PATH.exists():
        run_pipeline()

    chunks: list[dict] = []
    with INDEX_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _cosine_similarity(query_vector: list[float], doc_vector: list[float]) -> float:
    if not query_vector or not doc_vector:
        return 0.0

    dot = sum(a * b for a, b in zip(query_vector, doc_vector))
    query_norm = math.sqrt(sum(value * value for value in query_vector))
    doc_norm = math.sqrt(sum(value * value for value in doc_vector))
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return dot / (query_norm * doc_norm)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search semantically using vector similarity.

    Args:
        query: User query.
        top_k: Maximum number of results.

    Returns:
        List of {
            "content": str,
            "score": float,
            "metadata": dict,
        }, sorted by score descending.
    """
    if top_k <= 0:
        return []

    query_embedding = _hash_embedding(query)
    indexed_chunks = _load_index()

    results = []
    for chunk in indexed_chunks:
        score = _cosine_similarity(query_embedding, chunk.get("embedding", []))
        results.append(
            {
                "content": chunk.get("content", ""),
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
                "source": "semantic",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    results = semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5)
    for r in results:
        preview = r["content"][:100].encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"[{r['score']:.3f}] {preview}...")
