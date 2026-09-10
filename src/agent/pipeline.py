"""End-to-end agent: message -> intent -> precedent -> draft -> route.

CLI:  python -m src.agent.pipeline "my bag didn't arrive in ATL, help"
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass

from src.agent.classify import IntentPred, llm_classify
from src.agent.draft import Draft, draft_reply
from src.agent.retrieve import get_retriever
from src.agent.route import Route, route


@dataclass
class AgentResult:
    message: str
    intent: str
    intent_confidence: float
    reply: str
    reply_grounded: bool
    decision: str
    reason: str
    trigger: str
    signals: dict
    precedent: list[dict]


def run(message: str, k: int = 5) -> AgentResult:
    intent: IntentPred = llm_classify(message)
    precedents = get_retriever().search(message, k=k)
    d: Draft = draft_reply(message, precedents)
    r: Route = route(message, intent, precedents, d)
    return AgentResult(
        message=message,
        intent=intent.intent,
        intent_confidence=round(intent.confidence, 3),
        reply=d.reply,
        reply_grounded=d.grounded,
        decision=r.decision,
        reason=r.reason,
        trigger=r.trigger,
        signals=r.signals,
        precedent=[{"opening": p.opening, "reply": p.reply, "score": round(p.score, 3)}
                   for p in precedents],
    )


if __name__ == "__main__":
    msg = " ".join(sys.argv[1:]) or "My flight got cancelled and I need to get to ATL tonight"
    print(json.dumps(asdict(run(msg)), indent=2))
