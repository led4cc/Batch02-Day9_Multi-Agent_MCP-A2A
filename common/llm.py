"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var. Output length is capped
with OPENROUTER_MAX_TOKENS to avoid reserving an unaffordable response size.
"""

import os

from langchain_openai import ChatOpenAI


DEFAULT_MAX_TOKENS = 1000


def _get_max_tokens() -> int:
    raw_value = os.getenv("OPENROUTER_MAX_TOKENS")
    if not raw_value:
        return DEFAULT_MAX_TOKENS

    try:
        max_tokens = int(raw_value)
    except ValueError:
        return DEFAULT_MAX_TOKENS

    return max_tokens if max_tokens > 0 else DEFAULT_MAX_TOKENS


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        max_tokens=_get_max_tokens(),
        temperature=0.3,  # Low temperature for more focused, deterministic responses
    )
