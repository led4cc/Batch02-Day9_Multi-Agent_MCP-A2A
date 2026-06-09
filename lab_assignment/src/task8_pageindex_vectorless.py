"""
Task 8 - PageIndex vectorless fallback.

If PAGEINDEX_API_KEY and the PageIndex SDK are available, this module can be
extended to upload/query the cloud service. For this course repo, the default
implementation provides a local vectorless fallback over the Markdown/chunk
index and marks every result with source="pageindex".
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

try:
    from .task6_lexical_search import lexical_search
except ImportError:  # Allows: python src/task8_pageindex_vectorless.py
    from task6_lexical_search import lexical_search

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents() -> list[dict]:
    """
    Prepare local Markdown documents for vectorless fallback search.

    Returns metadata for discovered documents. In a real PageIndex deployment,
    this is where files would be uploaded to PageIndex.
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        documents.append(
            {
                "filename": md_file.name,
                "path": str(md_file.relative_to(STANDARDIZED_DIR)).replace("\\", "/"),
                "type": md_file.parent.name,
            }
        )
    return documents


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval fallback.

    The local implementation reuses lexical/BM25 retrieval, which does not use
    dense vectors at query time. Results are marked as `source="pageindex"` so
    Task 9 can distinguish fallback retrieval from hybrid search.
    """
    if top_k <= 0:
        return []

    results = lexical_search(query, top_k=top_k)
    pageindex_results = []

    for result in results:
        item = result.copy()
        item["source"] = "pageindex"
        metadata = dict(item.get("metadata", {}))
        metadata["fallback"] = "local_bm25_vectorless"
        item["metadata"] = metadata
        pageindex_results.append(item)

    return pageindex_results


if __name__ == "__main__":
    documents = upload_documents()
    print(f"Prepared {len(documents)} local documents for PageIndex fallback")

    results = pageindex_search("hinh phat su dung ma tuy", top_k=3)
    for r in results:
        preview = r["content"][:100].encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"[{r['score']:.3f}] {preview}...")
