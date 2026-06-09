"""
Task 6 - Lexical search with BM25.

The corpus is loaded from the local chunk index generated in Task 4
(`data/index/chunks.jsonl`). A small pure-Python BM25 implementation is used so
the task works even when `rank_bm25` is not installed.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

try:
    from .task4_chunking_indexing import INDEX_PATH, run_pipeline
except ImportError:  # Allows: python src/task6_lexical_search.py
    from task4_chunking_indexing import INDEX_PATH, run_pipeline


CORPUS: list[dict] = []


def _tokenize(text: str) -> list[str]:
    """Simple Unicode-aware tokenizer suitable for Vietnamese text."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _load_corpus() -> list[dict]:
    """Load content and metadata from the local JSONL index."""
    global CORPUS
    if CORPUS:
        return CORPUS

    if not INDEX_PATH.exists():
        run_pipeline()

    corpus: list[dict] = []
    with INDEX_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            corpus.append(
                {
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                }
            )

    CORPUS = corpus
    return CORPUS


class SimpleBM25:
    """BM25Okapi-style scorer implemented with standard Python."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_count = len(tokenized_corpus)
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avg_doc_len = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )
        self.term_freqs = [Counter(doc) for doc in tokenized_corpus]
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        doc_freq: Counter[str] = Counter()
        for doc in self.tokenized_corpus:
            doc_freq.update(set(doc))

        idf = {}
        for term, freq in doc_freq.items():
            # Robertson/Sparck Jones IDF with +1 smoothing.
            idf[term] = math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
        return idf

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.doc_count
        if not query_tokens or not self.doc_count:
            return scores

        for doc_index, term_freq in enumerate(self.term_freqs):
            doc_len = self.doc_lengths[doc_index] or 1
            score = 0.0
            for term in query_tokens:
                frequency = term_freq.get(term, 0)
                if frequency == 0:
                    continue

                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * doc_len / (self.avg_doc_len or 1)
                )
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1) / denominator
                )
            scores[doc_index] = score

        return scores


def build_bm25_index(corpus: list[dict]) -> SimpleBM25:
    """
    Build a BM25 index from a corpus.

    Args:
        corpus: List of {"content": str, "metadata": dict}
    """
    tokenized_corpus = [_tokenize(doc.get("content", "")) for doc in corpus]
    return SimpleBM25(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks by exact keyword overlap using BM25.

    Returns:
        List of {"content": str, "score": float, "metadata": dict}, sorted by
        BM25 score descending.
    """
    if top_k <= 0:
        return []

    corpus = _load_corpus()
    if not corpus:
        return []

    bm25 = build_bm25_index(corpus)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []
    for index in ranked_indices[:top_k]:
        doc = corpus[index]
        results.append(
            {
                "content": doc["content"],
                "score": float(scores[index]),
                "metadata": doc.get("metadata", {}),
                "source": "lexical",
            }
        )

    return results


if __name__ == "__main__":
    results = lexical_search("Dieu 248 tang tru trai phep chat ma tuy", top_k=5)
    for r in results:
        preview = r["content"][:100].encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"[{r['score']:.3f}] {preview}...")
