"""
Task 4 - Chunking and local indexing.

This module reads Markdown files from data/standardized/, chunks them with a
recursive character splitter, creates lightweight deterministic embeddings, and
writes a local JSONL index for later retrieval tasks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
INDEX_PATH = INDEX_DIR / "chunks.jsonl"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Recursive character chunking is conservative for mixed legal/news Markdown:
# it prefers paragraph boundaries, then lines, then sentence/word boundaries.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Default to a deterministic local hashing embedding so the pipeline runs in
# class/offline environments. It can be replaced by sentence-transformers later.
EMBEDDING_MODEL = "local-hashing-v1"
EMBEDDING_DIM = 384

# Store vectors locally as JSONL. This avoids requiring a running Weaviate
# instance during individual tasks while keeping the same chunk metadata shape.
VECTOR_STORE = "local_jsonl"


# =============================================================================
# DOCUMENT LOADING
# =============================================================================

def load_documents() -> list[dict]:
    """
    Read all Markdown files from data/standardized/.

    Returns:
        List of {
            "content": str,
            "metadata": {
                "source": str,
                "path": str,
                "type": "legal" | "news" | "unknown",
            }
        }
    """
    if not STANDARDIZED_DIR.exists():
        return []

    documents: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        first_part = relative_path.parts[0] if relative_path.parts else ""
        doc_type = first_part if first_part in {"legal", "news"} else "unknown"

        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(relative_path).replace("\\", "/"),
                    "type": doc_type,
                },
            }
        )

    return documents


# =============================================================================
# CHUNKING
# =============================================================================

def _split_long_text(text: str, chunk_size: int) -> list[str]:
    """Hard split text that has no useful separators."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _recursive_split(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Split text recursively by progressively smaller separators.

    This mirrors the practical behavior of RecursiveCharacterTextSplitter while
    avoiding an import-time dependency in environments missing LangChain.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    for separator in separators:
        if separator not in text:
            continue

        parts = [part.strip() for part in text.split(separator) if part.strip()]
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{separator}{part}" if current else part
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.extend(_recursive_split(current, chunk_size))
            current = part

        if current:
            chunks.extend(_recursive_split(current, chunk_size))

        if chunks:
            return chunks

    return _split_long_text(text, chunk_size)


def _add_overlap(chunks: list[str], overlap: int = CHUNK_OVERLAP) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        prefix = previous[-overlap:].strip()
        merged = f"{prefix}\n{current}" if prefix else current
        if len(merged) > CHUNK_SIZE:
            merged = merged[-CHUNK_SIZE:]
        overlapped.append(merged.strip())
    return overlapped


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents using the configured recursive strategy.

    Returns:
        List of {"content": str, "metadata": dict}
    """
    chunks: list[dict] = []

    for doc_index, doc in enumerate(documents):
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        splits = _add_overlap(_recursive_split(content))

        for chunk_index, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "doc_id": doc_index,
                        "chunk_index": chunk_index,
                        "chunking_method": CHUNKING_METHOD,
                    },
                }
            )

    return chunks


# =============================================================================
# EMBEDDING AND INDEXING
# =============================================================================

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    for token in _tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an embedding vector to each chunk.

    The default local hashing embedding is deterministic, fast, and works
    offline. It is enough for the class pipeline and can be swapped out later.
    """
    embedded = []
    for chunk in chunks:
        embedded_chunk = dict(chunk)
        embedded_chunk["embedding"] = _hash_embedding(chunk.get("content", ""))
        embedded.append(embedded_chunk)
    return embedded


def index_to_vectorstore(chunks: list[dict]) -> Path:
    """
    Persist chunks to a local JSONL vector index.

    Returns:
        Path to the created index file.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return INDEX_PATH


def run_pipeline() -> Path:
    """Run the full pipeline: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_path = index_to_vectorstore(chunks)
    print(f"Indexed to vector store: {index_path}")
    return index_path


if __name__ == "__main__":
    run_pipeline()
