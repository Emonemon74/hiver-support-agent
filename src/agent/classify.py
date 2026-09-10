"""Intent classification.

- `llm_classify`  : the headline classifier (gpt-oss-120b, JSON, few-shot).
- `knn_classify`  : a simple, training-free baseline — nearest neighbours among a
                    set of already-labelled examples (used in the eval harness
                    with leave-one-out over the golden set).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.embed import encode
from src.intents import INTENTS, LABELS
from src.llm import chat_json

_TAXONOMY = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())

_SYS = f"""You classify a customer's opening tweet to @Delta into exactly one intent.

{_TAXONOMY}

Rules:
- Choose the customer's PRIMARY ask. Pure venting with no request -> complaint.
- Praise -> compliment. Off-topic / spam / news / not for support -> other.
Return JSON: {{"intent": "<label>", "confidence": <0-1>}}"""

_FEWSHOT = [
    ("My bag never showed up at baggage claim in ATL, help!", "baggage"),
    ("Flight 1920 delayed 3 hrs, I'll miss my connection to SAV — what are my options?", "flight_disruption"),
    ("Huge thanks to the crew on DL245, best service I've had in years", "compliment"),
    ("Why won't the app let me pick a seat even though there are open ones?", "seat_upgrade"),
    ("My SkyMiles from last week's flight still haven't posted", "loyalty_miles"),
]


@dataclass
class IntentPred:
    intent: str
    confidence: float


def llm_classify(message: str) -> IntentPred:
    msgs = [{"role": "system", "content": _SYS}]
    for text, label in _FEWSHOT:
        msgs.append({"role": "user", "content": f'Tweet: "{text}"'})
        msgs.append({"role": "assistant", "content": f'{{"intent": "{label}", "confidence": 0.9}}'})
    msgs.append({"role": "user", "content": f'Tweet: "{message}"'})

    out = chat_json(msgs, max_tokens=900)
    intent = out.get("intent", "other")
    if intent not in LABELS:
        intent = "other"
    try:
        conf = float(out.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return IntentPred(intent, max(0.0, min(1.0, conf)))


def knn_classify(
    message: str, ref_texts: list[str], ref_labels: list[str], k: int = 15
) -> IntentPred:
    ref_vecs = _cached_matrix(tuple(ref_texts))
    q = encode([message])[0]
    sims = ref_vecs @ q
    top = np.argsort(sims)[-k:][::-1]
    votes: dict[str, float] = {}
    for j in top:
        votes[ref_labels[j]] = votes.get(ref_labels[j], 0.0) + float(sims[j])
    best = max(votes, key=votes.get)
    conf = votes[best] / sum(votes.values())
    return IntentPred(best, conf)


_MATRIX_CACHE: dict[tuple, np.ndarray] = {}


def _cached_matrix(texts: tuple) -> np.ndarray:
    if texts not in _MATRIX_CACHE:
        _MATRIX_CACHE[texts] = encode(list(texts))
    return _MATRIX_CACHE[texts]
