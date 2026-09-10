"""Reply drafting, grounded in retrieved precedent.

The model may only use facts/phrasing that appear in the retrieved past replies.
If the precedent is weak or contradictory it must abstain (`grounded=false`), which
the router treats as a signal to escalate.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.agent.retrieve import Precedent
from src.llm import chat_json

_SYS = """You draft a public reply tweet for @Delta's support team.

You are given the customer's message and up to 5 examples of how Delta replied to
similar past messages. Rules:
- Ground the reply ONLY in the style and content of the past replies. Do not
  invent policies, numbers, compensation, or promises not present in them.
- Match Delta's voice: warm, brief, first person ("I"), apologise once if
  relevant, no hashtags. <= 280 characters.
- If the past replies ask the customer to DM a confirmation number, do the same.
- If the precedent does not support a confident, safe reply, set grounded=false
  and return a minimal holding reply.

Return JSON:
{"reply": "<tweet>", "grounded": <bool>,
 "used_precedent": [<indices of the examples you leaned on>],
 "notes": "<one short reason>"}"""


@dataclass
class Draft:
    reply: str
    grounded: bool
    used_precedent: list[int]
    notes: str


def draft_reply(message: str, precedents: list[Precedent]) -> Draft:
    block = "\n".join(
        f"[{i}] (sim={p.score:.2f}) CUSTOMER: {p.opening}\n     DELTA: {p.reply}"
        for i, p in enumerate(precedents)
    )
    out = chat_json(
        [
            {"role": "system", "content": _SYS},
            {"role": "user", "content": f'CUSTOMER MESSAGE:\n"{message}"\n\nPAST REPLIES:\n{block}'},
        ],
        max_tokens=1200,
    )
    used = out.get("used_precedent", []) or []
    if not isinstance(used, list):
        used = []
    return Draft(
        reply=str(out.get("reply", "")).strip(),
        grounded=bool(out.get("grounded", False)),
        used_precedent=[int(x) for x in used if isinstance(x, (int, float))],
        notes=str(out.get("notes", "")),
    )
