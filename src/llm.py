"""Thin LLM wrapper over the Groq API (OpenAI-compatible chat completions).

Handles: JSON-mode responses, simple retry/backoff for the free-tier rate limit,
and a tiny on-disk cache so re-runs of the eval don't re-spend the quota.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from groq import Groq

from src.config import DATA, LLM_MODEL, require_llm

_CACHE_DIR = DATA / ".llm_cache"
_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=require_llm())
    return _client


def _cache_key(model: str, messages: list[dict], json_mode: bool, temperature: float) -> Path:
    blob = json.dumps([model, messages, json_mode, temperature], sort_keys=True)
    h = hashlib.sha256(blob.encode()).hexdigest()[:24]
    return _CACHE_DIR / f"{h}.json"


def chat(
    messages: list[dict],
    *,
    model: str = LLM_MODEL,
    json_mode: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 700,
    use_cache: bool = True,
    max_retries: int = 6,
) -> str:
    ck = _cache_key(model, messages, json_mode, temperature)
    if use_cache and ck.exists():
        return json.loads(ck.read_text())["content"]

    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = _get_client().chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            if use_cache:
                _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                ck.write_text(json.dumps({"content": content, "model": model}))
            return content
        except Exception as e:  # noqa: BLE001 - free tier: rate limit / transient
            msg = str(e).lower()
            if attempt == max_retries - 1 or not (
                "rate" in msg or "429" in msg or "503" in msg or "timeout" in msg
            ):
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")


def chat_json(messages: list[dict], **kw) -> dict:
    raw = chat(messages, json_mode=True, **kw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        return json.loads(raw[start : end + 1])
