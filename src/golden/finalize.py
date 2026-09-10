"""Phase 4c: apply the reviewer's decisions (src/golden/corrections.py) on top of
the model pre-labels and write the immutable golden set data/golden.jsonl.

Run: python -m src.golden.finalize
"""
from __future__ import annotations

import json

import pandas as pd

from src.config import DATA
from src.golden.corrections import AMBIGUOUS, INTENT_FIX, REASONS, TRIGGER
from src.intents import LABELS


def main() -> None:
    pre = [json.loads(l) for l in (DATA / "golden_prelabel.jsonl").read_text().splitlines()]
    ids = {r["thread_id"] for r in pre}

    missing = ids - set(TRIGGER)
    assert not missing, f"{len(missing)} threads have no reviewer trigger: {sorted(missing)[:10]}"

    rows = []
    for r in pre:
        tid = r["thread_id"]
        code = TRIGGER[tid]
        route = "escalate" if code.startswith("E-") else "auto"
        intent = INTENT_FIX.get(tid, r["pred_intent"])
        assert intent in LABELS, f"{tid}: bad intent {intent}"
        rows.append({
            "thread_id": tid,
            "created_at": r["created_at"],
            "customer_opening": r["customer_opening"],
            "reference_reply": r["reference_reply"],
            "intent": intent,
            "route": route,
            "route_trigger": code,
            "route_reason": REASONS[code],
            "difficulty": "ambiguous" if tid in AMBIGUOUS else "easy",
        })

    df = pd.DataFrame(rows).sort_values("created_at")
    out = DATA / "golden.jsonl"
    with out.open("w") as f:
        for _, r in df.iterrows():
            f.write(json.dumps(r.to_dict()) + "\n")

    print(f"wrote {out} — {len(df)} examples\n")
    print("intent:\n", df.intent.value_counts().to_string())
    print("\nroute:", df.route.value_counts().to_dict())
    print("difficulty:", df.difficulty.value_counts().to_dict())
    print("\nescalate triggers:\n",
          df[df.route == "escalate"].route_trigger.value_counts().to_string())

    pre_df = pd.DataFrame(pre)
    m = df.merge(pre_df, on="thread_id")
    print(f"\nreviewer vs model pre-label — "
          f"intent kept {(m.intent == m.pred_intent).mean():.1%}, "
          f"route kept {(m.route == m.pred_route).mean():.1%}")


if __name__ == "__main__":
    main()
