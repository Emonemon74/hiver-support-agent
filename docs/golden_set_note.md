# Golden evaluation set — how it was built

**File:** `data/golden.jsonl` · **Size:** ~200 examples · **Brand:** @Delta

## What each example contains

| field | meaning |
|-------|---------|
| `customer_opening` | the customer's first tweet (cleaned: handles/URLs masked) |
| `reference_reply` | Delta's *actual* first reply in that thread — used as a reference for reply eval, **not** as a gold "correct" answer (see caveats) |
| `intent` | one of the 8 taxonomy labels + `other` (`INTENTS.md`) |
| `route` | `auto` or `escalate` — the decision a trustworthy agent should make |
| `route_reason` | one sentence justifying the route label |
| `difficulty` | `easy` or `ambiguous` — labeller's own call, for slicing metrics |
| `notes` | free-text ambiguity flags |

## Sampling (`src/golden/sample.py`)

1. Pool = the **1,200 most recent** Delta threads (`holdout.parquet`). These are
   time-split *after* the retrieval corpus, so a golden example's own resolution is
   never available to the RAG agent — no leakage.
2. Openings embedded locally (`bge-small-en-v1.5`), **KMeans k=10** for topical
   spread.
3. Crossed with structural strata: question vs statement · contains a link ·
   length bucket · whether Delta's real reply was a "DM us" hand-off.
4. Proportional allocation across clusters with a **floor of 12 per cluster** so
   rare topics (e.g. `loyalty_miles`) are represented.
5. Near-duplicates removed (cosine > 0.95) — Twitter has many almost-identical
   complaints.
6. Seed = 13 throughout; the sample is reproducible.

## Labelling

- **Pre-labelling (`prelabel.py`):** `gpt-oss-120b` proposed `intent` + `route` +
  reason for each row. This is a *time-saver only*.
- **Human review:** every row was read and corrected in `golden_review.csv`
  (`reviewed=True` per row). Pre-label→final agreement is reported by
  `finalize.py` and in the report — where it is *low* the class is genuinely hard.
- **Guidelines used:**
  - Intent = the customer's *primary* ask. Multi-intent tweets take the most
    actionable one; pure venting with no ask → `complaint`.
  - `route=escalate` whenever a correct reply needs information the agent cannot
    see (PNR, ticket status, bag tag), or money/compensation is at stake, or the
    customer is angry enough to churn, or there's a safety/legal/medical angle.
  - "Please DM us your confirmation number" counts as `auto` — it's the standard
    safe holding reply and needs no account access.
- **Consistency check:** ~20 examples were re-labelled blind a day later;
  disagreements are listed in the report's failure section.
- Single labeller (the author). This is the main threat to the headline number and
  is called out explicitly in the report.
