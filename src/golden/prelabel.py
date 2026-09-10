"""Phase 4b: LLM pre-labels the sampled threads (intent + route + reason) to speed
up human review. These are SUGGESTIONS ONLY — every row is then checked by hand
and corrected in data/golden.jsonl. The pre-label model is the same family we
later evaluate, so we never treat these as ground truth.

Run: python -m src.golden.prelabel
Output: data/golden_prelabel.jsonl  and  data/golden_review.csv (for hand review)
"""
from __future__ import annotations

import json

import pandas as pd
from tqdm import tqdm

from src.config import DATA
from src.intents import INTENTS
from src.llm import chat_json

TAXONOMY = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())

SYS = f"""You label historical customer-support tweets sent to @Delta (airline).

INTENTS (choose exactly one):
{TAXONOMY}

ROUTE decision — would it be safe for an automated agent to send a public reply,
or must a human handle it?
- "auto": generic info, policy pointers, acknowledgements, thanks for compliments,
  or asking the customer to DM details. No account action, no promises, low risk.
- "escalate": needs account/PNR lookup or action, a refund/compensation decision,
  a safety/legal/medical issue, an angry customer threatening to leave, press, or
  anything where a wrong public answer would harm the customer or the brand.

Return JSON:
{{"intent": "<label>", "intent_confidence": <0-1>,
  "route": "auto|escalate", "route_reason": "<one sentence>",
  "difficulty": "easy|ambiguous"}}"""


def main() -> None:
    df = pd.read_parquet(DATA / "golden_sample.parquet")
    out_path = DATA / "golden_prelabel.jsonl"
    rows = []
    with out_path.open("w") as f:
        for r in tqdm(df.itertuples(index=False), total=len(df)):
            pred = chat_json(
                [
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": f'Customer tweet: "{r.customer_opening}"'},
                ],
                max_tokens=1200,
            )
            rec = {
                "thread_id": int(r.thread_id),
                "created_at": str(r.created_at),
                "customer_opening": r.customer_opening,
                "reference_reply": r.first_brand_reply,
                "reply_is_handoff": bool(r.reply_is_handoff),
                "pred_intent": pred.get("intent"),
                "pred_intent_confidence": pred.get("intent_confidence"),
                "pred_route": pred.get("route"),
                "pred_route_reason": pred.get("route_reason"),
                "pred_difficulty": pred.get("difficulty"),
            }
            f.write(json.dumps(rec) + "\n")
            rows.append(rec)

    review = pd.DataFrame(rows)
    # Human fills these; pre-filled with the prediction as a starting point.
    review["intent"] = review["pred_intent"]
    review["route"] = review["pred_route"]
    review["route_reason"] = review["pred_route_reason"]
    review["difficulty"] = review["pred_difficulty"]
    review["notes"] = ""
    review["reviewed"] = False
    review.to_csv(DATA / "golden_review.csv", index=False)
    print(f"wrote {out_path} and data/golden_review.csv ({len(review)} rows)")
    print(review.pred_intent.value_counts().to_string())
    print("\nroute:", review.pred_route.value_counts().to_dict())


if __name__ == "__main__":
    main()
