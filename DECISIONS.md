# Decision log

Non-obvious choices and why. (Numbers filled from `reports/results.json`.)

1. **Brand = Delta.** Of the high-volume brands, Delta pairs low deflection (22% of
   first replies are "DM us") with high English share (96%) and a bounded intent
   set. AmazonHelp is 26% English-noise and mostly needs account lookups (routing
   would collapse to "always escalate"); AppleSupport deflects 54% of the time, so
   "ground the reply in past resolutions" would just reproduce deflections.

2. **8 intents + `other`, not 20.** The golden set is ~200 examples; more classes
   makes per-intent F1 uninterpretable. Booking info-vs-change is *one* intent —
   whether the customer has an existing PNR is a routing signal, not a topic.

3. **Time-based train/test split.** The retrieval corpus is the older 6k threads;
   the golden set is sampled from the newest 1.2k. A golden example's own
   resolution is therefore never retrievable — prevents the agent from "cheating"
   by matching the exact thread.

4. **"Please DM your confirmation number" is labelled `escalate`.** The public
   tweet is templated but the resolution happens in a human-staffed DM with
   account access. Counting it as `auto` (defensible) would move ~35 labels and
   roughly halve the apparent escalation problem — see the report's misleading-
   number section.

5. **One route trigger recorded per golden row** (`E-ACCOUNT`, `E-MONEY`,
   `E-SAFETY`, `E-DISRUPTION`, `E-CHURN`, `E-LIVEDATA`). Lets failure analysis say
   *which kind* of routing call the agent gets wrong, not just how often.

6. **Free model stack (Groq + local embeddings).** The OpenAI account had no
   credit. `gpt-oss-120b` drafts/classifies; `qwen3.8-27b` judges (different
   lineage → less self-preference); `bge-small-en-v1.5` embeds locally (also makes
   retrieval reproducible with no key). Cost: $0.

7. **Judge model ≠ drafter model.** Self-preference is a known LLM-judge failure;
   using Qwen to judge GPT-OSS output removes the most obvious form of it.

8. **Deterministic rule layer for routing, LLM as upgrade-only.** The routing
   decision must be auditable, so rules (keyword + intent-risk + confidence +
   grounding) decide, and the LLM may only turn `auto`→`escalate`, never the
   reverse. This bounds the worst case (a chatty model auto-sending something
   risky).

9. **The router encodes the same rubric used to label golden routing.** So
   routing accuracy partly measures "did I implement my own policy". The
   always-escalate and LLM-only baselines are the independent reference points;
   the honest read is escalate-*recall* and false-auto-rate, not raw accuracy.

9b. **Router revised after error analysis.** The first cut (false-auto 0.38) lost to the one-line
   risk rule. v2 added `checkin_boarding` to the account-access set, made
   `complaint`/`other` escalate on a concrete-personal-incident signal
   (`E-INCIDENT`) rather than defaulting to auto, and broadened the money /
   disruption / live-data patterns → false-auto 0.09, recall 0.91. Because this
   was tuned on the eval subsample, it is re-checked with gold intent over all 199
   (0.71 acc, 0.07 false-auto) — `reports/routing_v2.json`. The routing decision is
   recomputed offline by `make routing` (`src/eval/rerun_routing.py`) without
   re-running the pipeline.

10. **Draft step must cite precedent and may abstain** (`grounded=false`). An
    abstention is fed to the router as an escalation signal rather than shipping a
    guessed answer.

11. **kNN (leave-one-out) as the "simple" classifier baseline**, not a trained
    model. No labelled data exists outside the golden set; LOO-kNN over golden
    embeddings is training-free and a fair "non-trivial but dumb" bar.

12. **Reply baselines: canned holding reply (trivial) + verbatim nearest-neighbour
    reply (simple).** The NN reply is a strong baseline for a support setting
    where past replies are templated — beating it is the real test.

13. **Judge validation on a 39-row stratified sample**, quadratic-weighted Cohen's
    κ + Spearman per dimension. Single human scorer (the author) — flagged as the
    top limitation; the one-week fix is a second annotator.

14. **Subsample sizes: 6k corpus / 1.2k holdout / ~200 golden.** Chosen so
    `make eval` finishes in minutes once the LLM cache is warm, per the brief.

15. **`checkin_boarding` kept despite only 7 golden examples.** It's a real
    cluster in the data; its per-intent metric is reported as indicative only
    rather than dropping the class and hiding the gap.
