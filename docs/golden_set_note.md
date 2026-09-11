# Golden evaluation set — how it was built

**File:** `data/golden.jsonl` · **Size:** 199 examples · **Brand:** @Delta

**Label distribution:** intent — compliment 43, complaint 41, flight_disruption 30,
loyalty_miles 17, other 17, seat_upgrade 16, booking_reservation 14, baggage 14,
checkin_boarding 7. Route — **auto 101 / escalate 98**. Difficulty — easy 171 /
ambiguous 28. (`checkin_boarding` is thin — the recent holdout window is light on
it; per-intent metrics for it are indicative only.)

**Revision (2026-09-11):** after a joint review of the 28 `ambiguous` rows, 8
routes were flipped `auto`→`escalate` (the first pass under-called bereavement,
stated churn, an explicit callback request, and two status-specific cases where
the agent was later shown to hallucinate): thread ids `602877` (E-MONEY),
`565486` (E-SAFETY), `579527` (E-CHURN), `603668` (E-SAFETY), `467628` (E-CHURN),
`571406` (E-DISRUPTION), `563551` (E-ACCOUNT), `585243` (E-ACCOUNT). Route counts
above are post-revision; see `src/golden/corrections.py` for the exact triggers.

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
  reason for each row. This is a *time-saver only* and is never treated as truth.
- **Human review:** every row was read against the transcript and Delta's actual
  reply; final decisions are encoded in `src/golden/corrections.py` (one route
  trigger per row + intent overrides + ambiguity flags). `finalize.py` applies
  them and reports how much the reviewer changed: **intent kept 99%, route kept
  79%** (after the ambiguous-row revision below) — i.e. routing is where the
  model and a careful human diverge, and that
  gap is itself a headline finding.
- **Guidelines used:**
  - Intent = the customer's *primary* ask. Multi-intent tweets take the most
    actionable one; pure venting with no ask → `complaint`.
  - `route=escalate` triggers (exactly one recorded per row in `route_trigger`):
    `E-ACCOUNT` (needs PNR/ticket/bag-file lookup or a booking change),
    `E-MONEY` (refund/credit/fee dispute), `E-SAFETY` (safety/medical/legal/
    bereavement), `E-DISRUPTION` (active disruption, rebooking needed now),
    `E-CHURN` (angry, going public / threatening to leave), `E-LIVEDATA`
    (needs live flight status / schedule / seat availability).
  - **"Please DM your confirmation number" is `escalate`, not `auto`.** The public
    tweet is templated, but the actual resolution happens in a human-staffed DM
    with account access. Counting these as `auto` (a defensible alternative
    definition) would flip ~35 labels — this is called out in the report's
    "what's misleading" section.
  - `auto` = the agent can fully close the loop with a public reply: compliments
    (thank + forward), general policy/info questions, feedback acknowledgement,
    minor-delay acknowledgement.
- **Consistency:** 28 rows are flagged `difficulty=ambiguous` — cases where a
  second reasonable labeller could disagree (rhetorical policy rants, borderline
  compliment/complaint, "minor delay" vs "active disruption"). Metrics are
  reported both overall and on the `easy` slice.
- **Single labeller** (the author), assisted by an LLM pre-pass. This is the main
  threat to the headline number and is called out explicitly in the report; with
  one more week the fix is a second independent labeller + Cohen's κ on the overlap.
