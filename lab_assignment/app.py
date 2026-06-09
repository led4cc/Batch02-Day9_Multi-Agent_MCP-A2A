"""
Streamlit RAG chatbot with a supervisor-agent architecture.

Run:
    streamlit run lab_assignment/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lab_assignment.supervisor_agents import SupervisorAgent


st.set_page_config(
    page_title="Drug Law RAG Chatbot",
    page_icon="",
    layout="wide",
)


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Nhap cau hoi ve phap luat ma tuy hoac tin tuc lien quan. Cau tra loi se kem citation va source.",
                "sources": [],
            }
        ]


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander("Sources used", expanded=True):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            title = metadata.get("source") or metadata.get("path") or f"Source {index}"
            score = source.get("score", 0.0)
            retrieval_source = source.get("source", "unknown")
            st.markdown(f"**{index}. {title}**  \nScore: `{score:.3f}` | Retrieval: `{retrieval_source}`")
            st.caption((source.get("content") or "")[:900])


def _answer_question(question: str, top_k: int, generation_enabled: bool) -> dict:
    supervisor = SupervisorAgent()
    return supervisor.answer(question, top_k=top_k, generation_enabled=generation_enabled)


_init_state()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top K sources", min_value=1, max_value=8, value=5)
    generation_enabled = st.toggle("Use Ollama generation", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("Drug Law RAG Chatbot")
st.caption("Supervisor agent: router -> retrieval -> generation -> citation verifier")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("sources", []))

prompt = st.chat_input("Hoi ve Luat Phong chong ma tuy, cai nghien, tin tuc nghe si...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Dang truy van pipeline..."):
            result = _answer_question(prompt, top_k=top_k, generation_enabled=generation_enabled)
        st.markdown(result["answer"])
        with st.expander("Supervisor trace", expanded=False):
            for step in result.get("trace", []):
                st.write(step)
        _render_sources(result.get("sources", []))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "trace": result.get("trace", []),
        }
    )
