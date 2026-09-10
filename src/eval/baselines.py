"""Baselines for each task: one trivial, one simple. The agent is the headline.

Classification
  trivial : majority class
  simple  : embedding kNN, leave-one-out over the golden set (training-free)
Routing
  trivial : always-escalate  (also report always-auto)
  simple  : escalate iff the intent's risk class is 'high'
Reply
  trivial : one canned holding reply
  simple  : the single nearest past Delta reply, verbatim
"""
from __future__ import annotations

import numpy as np

from src.embed import encode
from src.intents import RISK_CLASS

CANNED_REPLY = (
    "Thanks for reaching out. A member of our team will follow up with you shortly."
)


def majority_class(labels: list[str]) -> list[str]:
    top = max(set(labels), key=labels.count)
    return [top] * len(labels)


def knn_loo(texts: list[str], labels: list[str], k: int = 15) -> list[str]:
    vecs = encode(texts)
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1.0)
    preds = []
    for i in range(len(texts)):
        top = np.argsort(sims[i])[-k:][::-1]
        votes: dict[str, float] = {}
        for j in top:
            votes[labels[j]] = votes.get(labels[j], 0.0) + float(sims[i, j])
        preds.append(max(votes, key=votes.get))
    return preds


def route_risk_rule(intents: list[str]) -> list[str]:
    return ["escalate" if RISK_CLASS[i] == "high" else "auto" for i in intents]
