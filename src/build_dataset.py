"""Phase 3: filter twcs to the chosen brand, clean, and split into a retrieval
corpus and a held-out pool (the golden set is later sampled from the holdout).

Time-based split => a golden example's own resolution is never in the corpus.

Run: python -m src.build_dataset
Outputs: data/corpus.parquet, data/holdout.parquet, data/samples/<brand>_sample.csv
"""
from __future__ import annotations

import re

import langid
import pandas as pd

from src.clean import clean_text
from src.config import BRAND, CORPUS_THREADS, DATA, EVAL_HOLDOUT_THREADS, SEED
from src.threads import build_threads, load_raw

HANDOFF = re.compile(r"\b(DM|direct message|private message|send us|share .*(confirmation|record locator)|via DM)\b", re.I)


def thread_records(brand: str) -> pd.DataFrame:
    df = load_raw()
    rows = []
    for t in build_threads(df):
        if t.brand != brand or not t.first_brand_reply:
            continue
        opening = clean_text(t.customer_opening)
        reply = clean_text(t.first_brand_reply)
        if len(opening) < 10 or len(reply) < 10:
            continue
        if langid.classify(opening)[0] != "en":
            continue
        rows.append(
            {
                "thread_id": t.thread_id,
                "created_at": t.created_at[0],
                "n_turns": t.n_turns,
                "customer_opening": opening,
                "first_brand_reply": reply,
                "transcript": clean_text(t.transcript(), drop_lead_handle=False),
                "reply_is_handoff": bool(HANDOFF.search(reply)),
            }
        )
    out = pd.DataFrame(rows).drop_duplicates("customer_opening")
    return out.sort_values("created_at").reset_index(drop=True)


def main() -> None:
    assert BRAND, "set BRAND in .env"
    df = thread_records(BRAND)
    print(f"{BRAND}: {len(df):,} clean english threads with a brand reply")
    print(f"  date range {df.created_at.min()} .. {df.created_at.max()}")
    print(f"  handoff replies: {df.reply_is_handoff.mean():.1%}")

    holdout = df.tail(EVAL_HOLDOUT_THREADS).copy()
    corpus = df.iloc[: -EVAL_HOLDOUT_THREADS]
    if len(corpus) > CORPUS_THREADS:
        corpus = corpus.sample(CORPUS_THREADS, random_state=SEED).sort_values("created_at")

    corpus.to_parquet(DATA / "corpus.parquet")
    holdout.to_parquet(DATA / "holdout.parquet")
    (DATA / "samples").mkdir(exist_ok=True)
    corpus.head(50).to_csv(DATA / "samples" / f"{BRAND}_sample.csv", index=False)
    print(f"  corpus  -> {len(corpus):,} threads (data/corpus.parquet)")
    print(f"  holdout -> {len(holdout):,} threads (data/holdout.parquet)")


if __name__ == "__main__":
    main()
