"""
Task 7 - Reranking module.

This implementation keeps the public interfaces from the assignment but uses
local, deterministic reranking methods so it works without API keys or model
downloads:
    - rerank_cross_encoder: lexical relevance heuristic blended with base score
    - rerank_mmr: Maximal Marginal Relevance for embedded candidates
    - rerank_rrf: Reciprocal Rank Fusion for multiple ranked lists
"""

from __future__ import annotations

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _lexical_relevance(query: str, text: str) -> float:
    query_terms = _tokenize(query)
    if not query_terms:
        return 0.0

    text_counts = Counter(_tokenize(text))
    if not text_counts:
        return 0.0

    score = 0.0
    for term in query_terms:
        if term in text_counts:
            # Saturate repeated matches so one common term cannot dominate.
            score += 1.0 + math.log(text_counts[term])

    return score / len(query_terms)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates with a local cross-encoder-style heuristic.

    A real cross-encoder scores the query/document pair jointly. Here we mimic
    that behavior locally by combining query-term relevance with the candidate's
    original retrieval score.
    """
    if top_k <= 0 or not candidates:
        return []

    rescored = []
    for candidate in candidates:
        base_score = float(candidate.get("score", 0.0))
        relevance = _lexical_relevance(query, candidate.get("content", ""))
        rerank_score = 0.7 * relevance + 0.3 * base_score

        item = candidate.copy()
        item["score"] = float(rerank_score)
        item["rerank_method"] = "local_cross_encoder"
        rescored.append(item)

    rescored.sort(key=lambda item: item["score"], reverse=True)
    return rescored[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance.

    MMR = lambda * sim(query, doc) - (1 - lambda) * max(sim(doc, selected_docs))
    """
    if top_k <= 0 or not candidates:
        return []

    lambda_param = max(0.0, min(1.0, lambda_param))
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            candidate_embedding = candidates[idx].get("embedding", [])
            relevance = _cosine_sim(query_embedding, candidate_embedding)

            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    _cosine_sim(candidate_embedding, candidates[sel_idx].get("embedding", []))
                    for sel_idx in selected
                )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        if "score" not in item:
            item["score"] = float(_cosine_sim(query_embedding, item.get("embedding", [])))
        item["rerank_method"] = "mmr"
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion.

    RRF(d) = sum(1 / (k + rank_r(d))) across rankers.
    """
    if top_k <= 0:
        return []

    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        item["rerank_method"] = "rrf"
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Unified reranking interface."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        try:
            from .task4_chunking_indexing import _hash_embedding
        except ImportError:
            from task4_chunking_indexing import _hash_embedding

        query_embedding = _hash_embedding(query)
        return rerank_mmr(query_embedding, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k)

    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Dieu 248: Toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Nghe si bi bat vi su dung ma tuy", "score": 0.7, "metadata": {}},
        {"content": "Hinh phat tu 2-7 nam cho toi tang tru", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hinh phat tang tru ma tuy", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
