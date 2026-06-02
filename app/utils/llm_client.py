"""
LLM client utility.
Supports Groq (default) and OpenAI backends with automatic retry.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import GROQ_API_KEY, LLM_MODEL, LLM_PROVIDER, OPENAI_API_KEY


def _build_groq_client():
    try:
        from groq import Groq
        return Groq(api_key=GROQ_API_KEY)
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")


def _build_openai_client():
    try:
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    json_mode: bool = False,
) -> str:
    """
    Send messages to the configured LLM and return the text response.

    Args:
        messages:    List of {role, content} dicts.
        model:       Override the default model.
        temperature: Sampling temperature.
        max_tokens:  Max tokens in the response.
        json_mode:   If True, instruct the model to return valid JSON.

    Returns:
        The assistant's reply as a plain string.
    """
    _model = model or LLM_MODEL

    if json_mode:
        messages = [
            *messages,
            {
                "role": "system",
                "content": "Respond ONLY with valid JSON. No markdown fences, no commentary.",
            },
        ]

    try:
        if LLM_PROVIDER == "groq":
            client = _build_groq_client()
            response = client.chat.completions.create(
                model=_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            client = _build_openai_client()
            kwargs: dict[str, Any] = dict(
                model=_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)

        return response.choices[0].message.content.strip()

    except Exception as exc:
        logger.error(f"LLM call failed: {exc}")
        raise


def safe_json_llm(messages: list[dict[str, str]], **kwargs) -> dict | list:
    """Call the LLM and safely parse JSON from the response."""
    raw = chat_completion(messages, json_mode=True, **kwargs)
    # Strip accidental markdown fences
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(raw)
