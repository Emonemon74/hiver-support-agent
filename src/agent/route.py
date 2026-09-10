"""Auto-handle vs. escalate decision, with a stated reason.

A deterministic rule layer makes the call from a handful of signals; the LLM then
writes the one-sentence reason and may *upgrade* auto->escalate if it spots
something the rules missed (it can never downgrade). This keeps the decision
auditable while still catching long-tail risks.

`route_rules` is a pure function (no LLM) so the routing eval can be replayed
offline from cached signals — see src/eval/rerun_routing.py.
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
           "E-INCIDENT": "Specific personal service incident that needs a human to follow up.",
           "E-LOWCONF": "Low classifier confidence or weak precedent — not safe to auto-send.",
           "A-POLICY": "General question answerable from past replies; safe to auto-send.",
           "A-FEEDBACK": "Feedback or off-topic with no actionable request; safe to auto-send.",
           "A-COMPLIMENT": "Compliment — safe to thank publicly and forward internally."}

MONEY = re.compile(
    r"\b(refund|reimburse|reimbursement|compensat\w*|credit|voucher|charg(e|ed|es|ing)|"
    r"over-?charg\w*|fee|fees|\$\s?\d|\d+\s?(dollars|usd)|bill(ed|ing)?|paid|pay(ing)?\s+for|"
    r"price|priced|pricing|gouging|money back|partial refund)\b", re.I)
SAFETY = re.compile(
    r"\b(assault\w*|injur\w*|medical|emergency|wheelchair|disab\w*|discriminat\w*|racist|"
    r"unsafe|harass\w*|sue|lawyer|attorney|lawsuit|legal action|funeral|bereavement|"
    r"died|passed away|surgery)\b", re.I)
CHURN = re.compile(
    r"\b(never (fly|flying|book)\w*( with)?( you| delta| again)|done with (you|delta)|"
    r"switch\w* to|worst airline|#dontfly\w*|\bBBB\b|DOT complaint|"
    r"filing? a (formal )?complaint|lost a (loyal )?customer|you'?ve lost (a|me)|"
    r"loyal customer .*(10|ten) years|diamond|platinum|medallion)\b", re.I)
DISRUPT = re.compile(
    r"\b(missed? (my |our )?connection|stranded|stuck|rebook\w*|re-?accommodat\w*|reroute\w*|"
    r"next (available )?flight|other (flight|routing)s?|deplane|call ?back|callback|on hold|"
    r"hold (time|for)|change my flight|earlier flight|get me (home|to)|takes off in)\b", re.I)
LIVEDATA = re.compile(
    r"\b(nonstop|non-stop|do you (still )?fly|what (time|gate)|schedule|on time|status of|"
    r"is (flight|DL)\s*#?\s*\d|in the air|taxiing|seats? (currently )?available|standby room|"
    r"weather waiver|wifi[- ]?enabled|is .*\bwifi\b)\b", re.I)
# a concrete, personal, actionable service incident (vs. generic gripe / opinion)
INCIDENT = re.compile(
    r"\b(my |our |the )?(gate agent|gate staff|flight attendant|the crew|the rep|the pilot|"
    r"reservation|confirmation|itinerary|my seat|my bag|my luggage|my flight|my ticket|"
    r"my (connecting |return )?flight|boarding pass|seat (was |got )?(given|taken|changed))\b|"
    r"\b(DL|flight)\s*#?\s?\d{2,4}\b|\b(today|yesterday|this morning|last (week|night)|recently|just now|earlier)\b",
    re.I)

# intents whose resolution essentially always needs account/PNR access
ACCOUNT_INTENTS = {"flight_disruption", "baggage", "booking_reservation",
                   "seat_upgrade", "loyalty_miles", "checkin_boarding"}

CONF_FLOOR = 0.55
SIM_FLOOR = 0.42


@dataclass
class Route:
    decision: str          # "auto" | "escalate"
    reason: str
    trigger: str
    signals: dict


def route_rules(
    message: str, intent: str, intent_confidence: float, top_sim: float, grounded: bool
) -> tuple[str, str, dict]:
    m = message
    sig = {
        "intent": intent,
        "intent_risk": RISK_CLASS[intent],
        "intent_confidence": round(intent_confidence, 2),
        "top_similarity": round(top_sim, 2),
        "draft_grounded": grounded,
        "money": bool(MONEY.search(m)),
        "safety": bool(SAFETY.search(m)),
        "churn": bool(CHURN.search(m)),
        "disruption_urgency": bool(DISRUPT.search(m)),
        "needs_live_data": bool(LIVEDATA.search(m)),
        "specific_incident": bool(INCIDENT.search(m)),
    }
    if sig["safety"]:
        return "escalate", "E-SAFETY", sig
    if sig["money"]:
        return "escalate", "E-MONEY", sig
    if sig["disruption_urgency"] or (intent == "flight_disruption" and sig["churn"]):
        return "escalate", "E-DISRUPTION", sig
    if sig["churn"]:
        return "escalate", "E-CHURN", sig
    if sig["needs_live_data"]:
        return "escalate", "E-LIVEDATA", sig
    if intent in ACCOUNT_INTENTS:
        return "escalate", "E-ACCOUNT", sig
    # complaint / other: escalate only when it's a concrete personal incident
    if intent in ("complaint", "other") and sig["specific_incident"]:
        return "escalate", "E-INCIDENT", sig
    if intent_confidence < CONF_FLOOR or top_sim < SIM_FLOOR or not grounded:
        return "escalate", "E-LOWCONF", sig
    return "auto", {"compliment": "A-COMPLIMENT", "other": "A-FEEDBACK",
                    "complaint": "A-FEEDBACK"}.get(intent, "A-POLICY"), sig


_LLM_SYS = """You are the final safety check on an automated support reply for @Delta.
Given the customer message, the proposed reply, and a preliminary routing decision,
respond with JSON:
{"final_decision": "auto|escalate", "reason": "<one sentence>"}
You may change "auto" to "escalate" if the reply could mislead the customer, make a
promise Delta can't keep, or the situation clearly needs a human. You may NOT change
"escalate" to "auto". Keep the reason specific and customer-facing."""


def route(message: str, intent: IntentPred, precedents: list[Precedent], draft: Draft) -> Route:
    top_sim = precedents[0].score if precedents else 0.0
    decision, trigger, sig = route_rules(
        message, intent.intent, intent.confidence, top_sim, draft.grounded
    )

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
        return Route("escalate", REASONS[trigger], trigger, sig)
    if out.get("final_decision") == "escalate":
        reason = str(out.get("reason", "")).strip() or \
            "The drafted reply could mislead or overpromise; a human should review."
        return Route("escalate", reason, "E-LLM", sig)
    reason = str(out.get("reason", "")).strip() or REASONS[trigger]
    return Route("auto", reason, trigger, sig)
