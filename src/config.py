"""Central config: paths, model names, env loading."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

load_dotenv(ROOT / ".env")

# --- LLM: Groq free tier (OpenAI-compatible). Judge is a different model lineage
#     than the drafter to reduce self-preference bias. ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")

# --- Embeddings: local sentence-transformers, no API key, deterministic ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

BRAND = os.getenv("BRAND", "")

RAW_CSV = DATA / "twcs.csv"
SEED = 13

CORPUS_THREADS = 6000
EVAL_HOLDOUT_THREADS = 1200


def require_llm() -> str:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and add it to .env"
        )
    return GROQ_API_KEY
