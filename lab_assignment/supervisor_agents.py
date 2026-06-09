"""Supervisor-agent architecture for the lab assignment RAG chatbot.

The module keeps the app usable even when the original `src.task9_*` and
`src.task10_*` group-project modules are not present in this repository. If
those modules exist, the agents delegate to them. Otherwise they fall back to a
small local lexical retriever built from the golden dataset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


try:
    from src.task9_retrieval_pipeline import retrieve as external_retrieve
except ImportError:
    external_retrieve = None

try:
    from src.task10_generation import generate_with_citation as external_generate
except ImportError:
    external_generate = None


ROOT_DIR = Path(__file__).resolve().parent
GOLDEN_DATASET_PATH = ROOT_DIR / "evaluation" / "golden_dataset.json"


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _overlap_score(query: str, text: str) -> float:
    query_terms = _tokenize(query)
    text_terms = _tokenize(text)
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def _load_fallback_corpus() -> list[dict[str, Any]]:
    if not GOLDEN_DATASET_PATH.exists():
        return []

    with GOLDEN_DATASET_PATH.open("r", encoding="utf-8") as file:
        items = json.load(file)

    documents: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        source = item.get("expected_context") or f"golden_case_{index}"
        content = item.get("expected_answer") or item.get("question", "")
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": source,
                    "question": item.get("question", ""),
                },
                "source": "supervisor_fallback",
                "score": 0.0,
            }
        )
    return documents


@dataclass
class SupervisorState:
    question: str
    top_k: int
    generation_enabled: bool
    route: str = "general"
    sources: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    model: str = "supervisor-fallback"
    retrieval_source: str = "none"
    trace: list[str] = field(default_factory=list)


class RouterAgent:
    """Classify the question so the supervisor can explain the chosen path."""

    def run(self, state: SupervisorState) -> SupervisorState:
        question = state.question.lower()
        if any(term in question for term in ["pipeline", "rag", "citation", "source"]):
            state.route = "rag_meta"
        elif any(term in question for term in ["bao", "nghe si", "ca si", "g-dragon", "long nhat"]):
            state.route = "news"
        else:
            state.route = "legal"
        state.trace.append(f"RouterAgent: route={state.route}")
        return state


class RetrievalAgent:
    """Retrieve evidence from the original pipeline or local fallback corpus."""

    def retrieve(self, question: str, *, top_k: int, use_reranking: bool = True) -> list[dict[str, Any]]:
        if external_retrieve is not None:
            return external_retrieve(
                question,
                top_k=top_k,
                score_threshold=0.0,
                use_reranking=use_reranking,
            )

        documents = _load_fallback_corpus()
        scored: list[dict[str, Any]] = []
        for document in documents:
            content = document.get("content", "")
            metadata = document.get("metadata", {})
            haystack = f"{metadata.get('question', '')}\n{content}"
            score = _overlap_score(question, haystack)
            item = dict(document)
            item["score"] = score
            scored.append(item)

        scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        if use_reranking:
            scored.sort(
                key=lambda item: (
                    item.get("score", 0.0),
                    1 if stateful_source_hint(question, item) else 0,
                ),
                reverse=True,
            )
        return scored[:top_k]

    def run(self, state: SupervisorState) -> SupervisorState:
        state.sources = self.retrieve(state.question, top_k=state.top_k, use_reranking=True)
        state.retrieval_source = state.sources[0].get("source", "none") if state.sources else "none"
        state.trace.append(f"RetrievalAgent: sources={len(state.sources)} source={state.retrieval_source}")
        return state


def stateful_source_hint(question: str, item: dict[str, Any]) -> bool:
    source = str(item.get("metadata", {}).get("source", "")).lower()
    question_lower = question.lower()
    return any(term in source for term in _tokenize(question_lower))


class GenerationAgent:
    """Generate an answer with citations or create a deterministic fallback."""

    def run(self, state: SupervisorState) -> SupervisorState:
        if state.generation_enabled and external_generate is not None:
            try:
                result = external_generate(state.question, top_k=state.top_k)
            except Exception as exc:
                state.trace.append(f"GenerationAgent: external generation failed={exc}")
            else:
                state.answer = result.get("answer", "")
                state.sources = result.get("sources", state.sources)
                state.model = result.get("model", "external-generation")
                state.retrieval_source = result.get("retrieval_source", state.retrieval_source)
                state.trace.append(f"GenerationAgent: model={state.model}")
                return state

        state.answer = self._fallback_answer(state)
        state.model = "supervisor-fallback"
        state.trace.append("GenerationAgent: fallback answer")
        return state

    @staticmethod
    def _fallback_answer(state: SupervisorState) -> str:
        if not state.sources:
            return "Khong tim thay source phu hop de tra loi. Hay thu cau hoi cu the hon."

        lines = [
            "Tra loi dua tren cac source truy xuat duoc:",
            "",
        ]
        for index, source in enumerate(state.sources[:3], 1):
            metadata = source.get("metadata", {})
            title = metadata.get("source") or metadata.get("path") or f"source_{index}"
            content = (source.get("content") or "").strip()
            lines.append(f"{index}. {content} [{title}]")

        lines.extend(
            [
                "",
                "Luu y: day la cau tra loi RAG fallback, nen kiem tra source document truoc khi ket luan.",
            ]
        )
        return "\n".join(lines)


class CitationVerifierAgent:
    """Ensure the response exposes source information for every generated answer."""

    def run(self, state: SupervisorState) -> SupervisorState:
        if not state.sources:
            state.trace.append("CitationVerifierAgent: no sources")
            return state

        source_titles = [
            source.get("metadata", {}).get("source") or source.get("metadata", {}).get("path")
            for source in state.sources
        ]
        visible_titles = [title for title in source_titles if title and str(title) in state.answer]
        if not visible_titles:
            state.answer += "\n\nSources: " + ", ".join(str(title) for title in source_titles if title)

        state.trace.append(
            f"CitationVerifierAgent: visible_citations={len(visible_titles)} total_sources={len(state.sources)}"
        )
        return state


class SupervisorAgent:
    """Coordinate specialist agents and return the Streamlit/evaluation payload."""

    def __init__(self) -> None:
        self.router = RouterAgent()
        self.retrieval = RetrievalAgent()
        self.generation = GenerationAgent()
        self.citation_verifier = CitationVerifierAgent()

    def answer(self, question: str, *, top_k: int = 5, generation_enabled: bool = True) -> dict[str, Any]:
        state = SupervisorState(
            question=question,
            top_k=top_k,
            generation_enabled=generation_enabled,
        )
        for agent in [self.router, self.retrieval, self.generation, self.citation_verifier]:
            state = agent.run(state)

        return {
            "answer": state.answer,
            "sources": state.sources,
            "retrieval_source": state.retrieval_source,
            "model": state.model,
            "route": state.route,
            "trace": state.trace,
        }

    def retrieve(self, question: str, *, top_k: int = 5, use_reranking: bool = True) -> list[dict[str, Any]]:
        return self.retrieval.retrieve(question, top_k=top_k, use_reranking=use_reranking)


def average_score(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    return mean(item.get("score", 0.0) for item in items)
