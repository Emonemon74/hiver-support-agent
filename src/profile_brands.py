"""Phase 2: per-brand metrics to choose ONE brand for the agent.

Writes data/brand_profile.csv and prints a ranked table. Run: python -m src.profile_brands
"""
from __future__ import annotations

import re

import langid
import pandas as pd

from src.config import DATA, SEED
from src.threads import build_threads, load_raw

THANKS = re.compile(r"\b(thank(s| you)?|thx|cheers|appreciate|sorted|resolved|fixed)\b", re.I)
HANDOFF = re.compile(r"\b(DM|direct message|private message|send us a|link|form)\b", re.I)


def resolved_proxy(t) -> bool:
    """Weak 'looks resolved' signal: customer's last message thanks the brand."""
    for a_in, txt in zip(reversed(t.inbound), reversed(t.texts)):
        if a_in:
            return bool(THANKS.search(txt))
    return False


def main() -> None:
    df = load_raw()
    threads = build_threads(df)
    print(f"{len(threads):,} threads reconstructed")

    rows = []
    single = [t for t in threads if t.brand]
    by_brand: dict[str, list] = {}
    for t in single:
        by_brand.setdefault(t.brand, []).append(t)

    for brand, ts in by_brand.items():
        if len(ts) < 500:
            continue
        n = len(ts)
        with_reply = [t for t in ts if t.first_brand_reply]
        turns = pd.Series([t.n_turns for t in ts])
        # english share on a sample of openings
        samp = pd.Series([t.customer_opening for t in ts]).sample(
            min(400, n), random_state=SEED
        )
        en = sum(langid.classify(x)[0] == "en" for x in samp) / len(samp)
        replies = [t.first_brand_reply or "" for t in with_reply]
        handoff = sum(bool(HANDOFF.search(r)) for r in replies) / max(len(replies), 1)
        rows.append(
            {
                "brand": brand,
                "threads": n,
                "pct_with_brand_reply": round(len(with_reply) / n, 3),
                "median_turns": int(turns.median()),
                "p90_turns": int(turns.quantile(0.9)),
                "pct_resolved_proxy": round(sum(resolved_proxy(t) for t in ts) / n, 3),
                "pct_reply_is_handoff": round(handoff, 3),
                "english_share": round(en, 3),
            }
        )

    prof = pd.DataFrame(rows).sort_values("threads", ascending=False)
    out = DATA / "brand_profile.csv"
    prof.to_csv(out, index=False)
    print(f"\nwrote {out}\n")
    print(prof.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
