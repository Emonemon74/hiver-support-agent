"""Auto-handle vs. escalate decision, with a stated reason.

A deterministic rule layer makes the call from a handful of signals; the LLM then
writes the one-sentence reason and may *upgrade* auto->escalate if it spots
something the rules missed (it can never downgrade). This keeps the decision
auditable while still catching long-tail risks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.agent.classify import IntentPred
from src.agent.draft import Draft
from src.agent.retrieve import Precedent
from src.golden.corrections import REASONS
from src.intents import RISK_CLASS
from src.llm import chat_json

REASONS = {**REASONS,
           "E-LOWCONF": "Low classifier confidence or weak precedent — not safe to auto-send.",
           "A-POLICY": "General question answerable from past replies; safe to auto-send.",
           "A-FEEDBACK": "Feedback or off-topic with no actionable request; safe to auto-send.",
           "A-COMPLIMENT": "Compliment — safe to thank publicly and forward internally."}

MONEY = re.compile(r"\b(refund|reimburse|compensat|credit|voucher|charged?|fee|gouorg|overcharg|money back)\b", re.I)
SAFETY = re.compile(r"\b(assault|injur|medical|emergency|wheelchair|disab|discriminat|racist|unsafe|sue|lawyer|attorney|lawsuit|funeral|bereavement|died|passed away)\b", re.I)
CHURN = re.compile(r"\b(never (fly|flying|again)|done with|switching to|worst airline|#dontfly\w*|BBB|DOT complaint|file a complaint|diamond|platinum|medallion)\b", re.I)
DISRUPT = re.compile(r"\b(missed? (my )?connection|stranded|stuck|rebook|re-?accommodat|reroute|next flight|other (flight|routing)|deplane)\b", re.I)
LIVEDATA = re.compile(r"\b(nonstop|non-stop|do you (still )?fly|route|schedule|on time|status of|is (flight|DL)\s*\d|in the air|taxiing|seats? available|standby room|weather waiver)\b", re.I)

CONF_FLOOR = 0.55
SIM_FLOOR = 0.45


@dataclass
class Route:
    decision: str          # "auto" | "escalate"
    reason: str
    trigger: str           # short code
    signals: dict


def _rule_layer(message: str, intent: IntentPred, top_sim: float, draft: Draft) -> tuple[str, str, dict]:
    sig = {
        "intent": intent.intent,
        "intent_risk": RISK_CLASS[intent.intent],
        "intent_confidence": round(intent.confidence, 2),
        "top_similarity": round(top_sim, 2),
        "draft_grounded": draft.grounded,
        "money": bool(MONEY.search(message)),
        "safety": bool(SAFETY.search(message)),
        "churn": bool(CHURN.search(message)),
        "disruption_urgency": bool(DISRUPT.search(message)),
        "needs_live_data": bool(LIVEDATA.search(message)),
    }
    if sig["safety"]:
        return "escalate", "E-SAFETY", sig
    if sig["money"]:
        return "escalate", "E-MONEY", sig
    if sig["disruption_urgency"] or (intent.intent == "flight_disruption" and sig["churn"]):
        return "escalate", "E-DISRUPTION", sig
    if sig["churn"]:
        return "escalate", "E-CHURN", sig
    if LIVEDATA.search(message):
        return "escalate", "E-LIVEDATA", sig
    if RISK_CLASS[intent.intent] == "high":
        return "escalate", "E-ACCOUNT", sig
    if intent.confidence < CONF_FLOOR or top_sim < SIM_FLOOR or not draft.grounded:
        return "escalate", "E-LOWCONF", sig
    return "auto", {"compliment": "A-COMPLIMENT", "other": "A-FEEDBACK"}.get(
        intent.intent, "A-POLICY"
    ), sig


_LLM_SYS = """You are the final safety check on an automated support reply for @Delta.
Given the customer message, the proposed reply, and a preliminary routing decision,
respond with JSON:
{"final_decision": "auto|escalate", "reason": "<one sentence>"}
You may change "auto" to "escalate" if the reply could mislead the customer, make a
promise Delta can't keep, or the situation clearly needs a human. You may NOT change
"escalate" to "auto". Keep the reason specific and customer-facing."""


def route(message: str, intent: IntentPred, precedents: list[Precedent], draft: Draft) -> Route:
    top_sim = precedents[0].score if precedents else 0.0
    decision, trigger, sig = _rule_layer(message, intent, top_sim, draft)

    out = chat_json(
        [
            {"role": "system", "content": _LLM_SYS},
            {"role": "user", "content":
                f'CUSTOMER: "{message}"\nPROPOSED REPLY: "{draft.reply}"\n'
                f'PRELIMINARY: {decision} ({trigger})'},
        ],
        max_tokens=700,
    )
    if decision == "escalate":
        # rules are decisive; ignore an LLM "reason" that argues for auto
        return Route("escalate", REASONS[trigger], trigger, sig)

    if out.get("final_decision") == "escalate":
        reason = str(out.get("reason", "")).strip() or "The drafted reply could mislead or overpromise; a human should review."
        return Route("escalate", reason, "E-LLM", sig)

    reason = str(out.get("reason", "")).strip() or REASONS[trigger]
    return Route("auto", reason, trigger, sig)
