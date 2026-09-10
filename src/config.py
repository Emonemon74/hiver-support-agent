"""Central config: paths, model names, env loading."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
BRAND = os.getenv("BRAND", "")

# Raw dataset location (twcs.csv from the Kaggle "Customer Support on Twitter" set)
RAW_CSV = DATA / "twcs.csv"

# Reproducibility
SEED = 13

# Subsample sizes (kept small so the pipeline runs in <15 min)
CORPUS_THREADS = 6000
EVAL_HOLDOUT_THREADS = 1200


def require_openai() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return OPENAI_API_KEY
