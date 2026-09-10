"""Run the full agent over every golden example and cache the outputs.

Output: data/agent_outputs.jsonl  (one line per golden thread_id)
Idempotent: already-processed thread_ids are skipped, and every LLM call is
cached by src.llm, so re-runs are cheap.
"""
from __future__ import annotations

import json

from tqdm import tqdm

from src.agent.pipeline import run
from src.config import DATA

GOLDEN = DATA / "golden.jsonl"
OUT = DATA / "agent_outputs.jsonl"


def main() -> None:
    golden = [json.loads(l) for l in GOLDEN.read_text().splitlines()]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["thread_id"] for l in OUT.read_text().splitlines()}

    with OUT.open("a") as f:
        for g in tqdm(golden, desc="agent"):
            if g["thread_id"] in done:
                continue
            r = run(g["customer_opening"])
            f.write(json.dumps({
                "thread_id": g["thread_id"],
                "intent_pred": r.intent,
                "intent_confidence": r.intent_confidence,
                "reply": r.reply,
                "reply_grounded": r.reply_grounded,
                "decision": r.decision,
                "trigger": r.trigger,
                "reason": r.reason,
                "signals": r.signals,
                "top_precedent_reply": r.precedent[0]["reply"] if r.precedent else "",
                "top_precedent_score": r.precedent[0]["score"] if r.precedent else 0.0,
            }) + "\n")
    print(f"done -> {OUT} ({sum(1 for _ in OUT.open())} rows)")


if __name__ == "__main__":
    main()
