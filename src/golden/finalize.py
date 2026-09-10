"""Phase 4c: turn the hand-reviewed CSV into the immutable golden set.

Reads data/golden_review.csv (after a human has set `reviewed=True` and corrected
intent / route / route_reason / difficulty), validates it, and writes
data/golden.jsonl — the file every evaluation runs against.

Run: python -m src.golden.finalize
"""
from __future__ import annotations

import json

import pandas as pd

from src.config import DATA
from src.intents import LABELS

ROUTES = {"auto", "escalate"}
DIFF = {"easy", "ambiguous"}


def main() -> None:
    df = pd.read_csv(DATA / "golden_review.csv")

    unreviewed = (~df.reviewed.astype(bool)).sum()
    if unreviewed:
        raise SystemExit(f"{unreviewed} rows still have reviewed=False — finish the review first")

    bad_intent = set(df.intent) - set(LABELS)
    bad_route = set(df.route) - ROUTES
    bad_diff = set(df.difficulty) - DIFF
    assert not bad_intent, f"unknown intents: {bad_intent}"
    assert not bad_route, f"unknown routes: {bad_route}"
    assert not bad_diff, f"unknown difficulty: {bad_diff}"
    assert df.route_reason.str.len().min() > 5, "every row needs a route_reason"

    out = DATA / "golden.jsonl"
    with out.open("w") as f:
        for r in df.itertuples(index=False):
            f.write(json.dumps({
                "thread_id": int(r.thread_id),
                "created_at": r.created_at,
                "customer_opening": r.customer_opening,
                "reference_reply": r.reference_reply,
                "intent": r.intent,
                "route": r.route,
                "route_reason": r.route_reason,
                "difficulty": r.difficulty,
                "notes": "" if pd.isna(r.notes) else str(r.notes),
            }) + "\n")

    print(f"wrote {out} — {len(df)} examples")
    print("intent:\n", df.intent.value_counts().to_string())
    print("\nroute:", df.route.value_counts().to_dict())
    print("difficulty:", df.difficulty.value_counts().to_dict())
    agree = (df.intent == df.pred_intent).mean()
    r_agree = (df.route == df.pred_route).mean()
    print(f"\npre-label vs final agreement — intent {agree:.1%}, route {r_agree:.1%}")


if __name__ == "__main__":
    main()
