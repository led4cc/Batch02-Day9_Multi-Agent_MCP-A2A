"""
Task 10 - Generation with citation.

Pipeline:
    1. Retrieve relevant chunks
    2. Reorder chunks to reduce "lost in the middle"
    3. Format context with source labels
    4. Call OpenRouter for answer generation
    5. Return answer with citations and sources
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:  # Allows: python src/task10_generation.py
    from task9_retrieval_pipeline import retrieve

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# top_k: number of chunks passed to the model.
TOP_K = 5

# top_p: nucleus sampling. 0.9 keeps some flexibility without becoming too loose.
TOP_P = 0.9

# temperature: low value because RAG answers should be factual.
TEMPERATURE = 0.3

# OpenRouter config. Uses the same env var convention as the main project.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "1000"))


def _safe_text(value) -> str:
    return str(value).encode("ascii", errors="backslashreplace").decode("ascii")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Tra loi bang tieng Viet cho muc dich giao duc, phap ly,
bao chi, va phong chong tac hai. Khong dua ra huong dan thuc hien hanh vi
bat hop phap; chi giai thich quy dinh, su kien, va thong tin da co trong
context.

Bat buoc citation:
- Moi nhan dinh su kien/phap ly phai co citation ngay trong cung cau.
- Citation phai dung Source trong context, vi du [Luat-73-2021-QH14.md] hoac
  [article_01_...md].
- Neu context khong noi ro, tra loi: "Toi khong the xac minh thong tin nay tu
  nguon hien co".

Rules:
- Chi su dung thong tin trong context.
- Khong doan, khong them kien thuc ben ngoai.
- Neu khong du bang chung, noi ro phan nao khong xac minh duoc.
- Tra loi ngan gon, co cau truc ro rang."""


# =============================================================================
# DOCUMENT REORDERING
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to reduce the "lost in the middle" effect.

    Input order is assumed to be sorted by relevance descending.
    Example: [1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks

    front = chunks[::2]
    back = chunks[1::2][::-1]
    return front + back


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks as source-labeled context for citation-aware generation.
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        source = metadata.get("source") or metadata.get("title") or f"Source {i}"
        doc_type = metadata.get("type") or metadata.get("doc_type") or "unknown"
        page = metadata.get("page")
        article = metadata.get("article") or metadata.get("section")

        labels = [f"Document {i}", f"Source: {source}", f"Type: {doc_type}"]
        if page is not None:
            labels.append(f"Page: {page}")
        if article is not None:
            labels.append(f"Article/Section: {article}")

        context_parts.append(
            f"[{' | '.join(labels)}]\n"
            f"{chunk.get('content', '')}\n"
        )

    return "\n---\n".join(context_parts)


def _call_openrouter_chat(system_prompt: str, user_message: str) -> str:
    """Call OpenRouter's OpenAI-compatible API and return the response text."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    llm = ChatOpenAI(
        model=OPENROUTER_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=OPENROUTER_MAX_TOKENS,
    )
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
    )
    answer = str(response.content).strip()
    if not answer:
        raise RuntimeError("OpenRouter returned an empty response")
    return answer


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation with citation using OpenRouter.

    Returns:
        {
            "answer": str,
            "sources": list[dict],
            "retrieval_source": str,
            "model": str,
        }
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    user_message = f"""Context:
{context}

---

Question: {query}

Answer requirements:
- Cite each factual sentence using the source filename shown in the context.
- If sources conflict or are insufficient, say what cannot be verified."""

    answer = _call_openrouter_chat(SYSTEM_PROMPT, user_message)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "model": OPENROUTER_MODEL,
    }


if __name__ == "__main__":
    test_queries = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy theo phap luat Viet Nam?",
        "Nhung nghe si nao da bi bat vi lien quan toi ma tuy?",
        "Quy trinh cai nghien bat buoc theo Luat Phong chong ma tuy 2021?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {_safe_text(result['answer'])}")
        print(
            f"\n[Sources: {len(result['sources'])} chunks "
            f"| via {result['retrieval_source']} | model {result['model']}]"
        )
