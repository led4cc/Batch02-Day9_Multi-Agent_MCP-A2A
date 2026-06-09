"""
Task 9 - Complete retrieval pipeline.

Combines semantic search, lexical search, RRF fusion, reranking, and a
PageIndex-style vectorless fallback into one retrieval function.
"""

from __future__ import annotations

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:  # Allows: python src/task9_retrieval_pipeline.py
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def _safe_search(search_fn, query: str, top_k: int) -> list[dict]:
    """Run a search function without letting one retriever break the pipeline."""
    try:
        return search_fn(query, top_k=top_k)
    except Exception as exc:
        print(f"{search_fn.__name__} failed: {exc}")
        return []


def _as_hybrid_results(results: list[dict]) -> list[dict]:
    hybrid = []
    for result in results:
        item = result.copy()
        item["source"] = "hybrid"
        item.setdefault("metadata", {})
        hybrid.append(item)
    return hybrid


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Complete retrieval pipeline with fallback logic.

    Steps:
        1. Run semantic and lexical search.
        2. Merge ranked lists with RRF.
        3. Rerank merged results.
        4. If hybrid confidence is too low, fallback to PageIndex-style search.

    Returns:
        List of {
            "content": str,
            "score": float,
            "metadata": dict,
            "source": "hybrid" | "pageindex",
        }
    """
    if top_k <= 0:
        return []

    search_k = max(top_k * 2, top_k)
    dense_results = _safe_search(semantic_search, query, search_k)
    sparse_results = _safe_search(lexical_search, query, search_k)

    ranked_lists = [results for results in [dense_results, sparse_results] if results]
    if ranked_lists:
        merged = rerank_rrf(ranked_lists, top_k=search_k)
        merged = _as_hybrid_results(merged)
    else:
        merged = []

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        final_results = _as_hybrid_results(final_results)
    else:
        final_results = merged[:top_k]

    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        return pageindex_search(query, top_k=top_k)

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy",
        "Nghe si nao bi bat vi su dung ma tuy nam 2024",
        "Luat phong chong ma tuy 2021 quy dinh gi ve cai nghien",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            preview = r["content"][:80].encode("ascii", errors="backslashreplace").decode("ascii")
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {preview}...")
