"""LLM-as-judge for reply quality, plus a cheap automatic reference-similarity.

Rubric (1-5 each, 5 = best):
  groundedness : every claim/offer in the reply is supported by the retrieved
                 precedent or is a safe generic courtesy; no invented facts.
  helpfulness  : moves the customer forward (answers, or correctly collects the
                 info a human needs).
  tone         : matches Delta's voice — warm, brief, first person, no hashtags.
  safety       : makes no promise, policy claim, or compensation offer Delta
                 couldn't stand behind; escalates rather than guesses.

Judge model is qwen3.8-27b — a different lineage from the drafter (gpt-oss-120b)
to limit self-preference. Validated against human scores in judge_validation.py.
"""
from __future__ import annotations

import json

import numpy as np

from src.config import JUDGE_MODEL
from src.embed import encode
from src.llm import chat_json

DIMS = ["groundedness", "helpfulness", "tone", "safety"]

_SYS = """You are a senior customer-support QA lead at Delta scoring a DRAFT reply
that an AI wrote for a customer tweet. You also see the precedent (real past Delta
replies to similar tweets) the AI was told to ground in, and Delta's actual reply
to this customer for reference.

Score each dimension 1-5 (integers). Be strict; 5 means you would send it as-is.
  groundedness: claims/offers supported by precedent or safe generic courtesy
  helpfulness : moves the customer forward
  tone        : Delta's voice — warm, brief, first person, no hashtags
  safety      : no unbacked promise / policy claim / compensation offer

Return JSON:
{"groundedness": n, "helpfulness": n, "tone": n, "safety": n,
 "hallucination": <bool - true if the reply states a specific fact/number/policy
                  not in the precedent>,
 "rationale": "<= 2 sentences"}"""


def judge_reply(customer: str, draft: str, reference: str, precedent: list[str]) -> dict:
    block = "\n".join(f"- {p}" for p in precedent)
    out = chat_json(
        [
            {"role": "system", "content": _SYS},
            {"role": "user", "content":
                f'CUSTOMER TWEET:\n"{customer}"\n\n'
                f'PRECEDENT (past Delta replies):\n{block}\n\n'
                f'DELTA\'S ACTUAL REPLY (reference):\n"{reference}"\n\n'
                f'DRAFT REPLY TO SCORE:\n"{draft}"'},
        ],
        model=JUDGE_MODEL,
        max_tokens=1400,
    )
    scores = {}
    for d in DIMS:
        try:
            scores[d] = int(round(float(out.get(d, 3))))
        except (TypeError, ValueError):
            scores[d] = 3
        scores[d] = min(5, max(1, scores[d]))
    scores["overall"] = round(float(np.mean([scores[d] for d in DIMS])), 2)
    scores["hallucination"] = bool(out.get("hallucination", False))
    scores["rationale"] = str(out.get("rationale", ""))
    return scores


def reference_similarity(drafts: list[str], references: list[str]) -> list[float]:
    dv = encode(drafts)
    rv = encode(references)
    return [round(float(a @ b), 3) for a, b in zip(dv, rv)]
