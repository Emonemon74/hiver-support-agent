# Hiver SDE Intern Take-Home — Implementation Plan

## 0. Guiding principle
"The proof is worth more than the system." Every hour goes to: a defensible eval,
honest failure analysis, and reproducibility — not to a fancy agent.

## 1. Environment & prerequisites (blockers to clear first)
- **No `kaggle` CLI, no dataset yet.** Install `kaggle`, place `~/.kaggle/kaggle.json`
  (user provides Kaggle API token). Fallback: manual download of `twcs.csv` zip.
- **No `OPENAI_API_KEY` in env.** User must supply one (`.env`, git-ignored).
- **Python 3.14 is too new** for some ML wheels. Create venv on **3.11 or 3.12**
  (`uv venv --python 3.12`). Deps: pandas, pyarrow, openai, scikit-learn,
  sentence-transformers (or OpenAI embeddings), faiss-cpu / chromadb, rich, tqdm,
  python-dotenv, pytest.

## 2. Data profiling → brand pick  (script: `src/profile_brands.py`)
1. Load `twcs.csv` (2.8M rows). Columns: tweet_id, author_id, inbound,
   created_at, text, response_tweet_id, in_response_to_tweet_id.
2. Reconstruct threads by following `in_response_to_tweet_id` / `response_tweet_id`.
3. Per brand metrics: # inbound customer msgs, # threads, % threads with a brand
   reply, median turns, % threads that look "resolved" (brand has last word / thanks
   token from customer), language = English share.
4. Output `data/brand_profile.csv` + a short markdown table.
5. **Recommend one brand** (candidates: AmazonHelp, AppleSupport, SpotifyCares,
   Uber_Support, Delta). Decision recorded in decision log. User confirms.

## 3. Dataset construction  (script: `src/build_dataset.py`)
- Filter to chosen brand. Build records:
  `{thread_id, customer_opening_msg, full_thread, brand_reply_1, resolved_flag, created_at}`
- Subsample: ~4–6k threads for the retrieval corpus, held-out slice for eval.
- Clean: strip @mentions, URLs → `<URL>`, de-dup, drop non-English (langid),
  drop threads with no brand reply.
- Persist as parquet in `data/`.

## 4. Intent taxonomy  (notebook + `src/intents.py`)
- Sample ~300 customer openings, cluster (embeddings + KMeans / manual pass),
  read clusters, define **6–9 intents** + an `other` bucket. Example for Amazon:
  `order_status`, `delivery_problem`, `refund_return`, `account_access`,
  `payment_billing`, `product_quality`, `cancellation`, `general_query`, `other`.
- Write taxonomy with 1-line definition + 2 real examples each → `INTENTS.md`.

## 5. Golden evaluation set (150–250 examples)  → `data/golden.jsonl`
- **Sampling:** stratified — cover every intent, oversample rare ones; mix
  easy/ambiguous; only customer *opening* messages; fixed random seed; time-split
  so golden threads are excluded from the retrieval corpus (no leakage).
- **Labels per example:**
  - `intent` (from taxonomy)
  - `route_gold` = auto | escalate  + `route_reason`
  - `reference_reply` = the brand's actual first reply (for reply grounding ref)
  - `notes` (ambiguity flags)
- **Process note** (`docs/golden_set_note.md`): who labeled (me), guidelines,
  how ambiguity resolved, ~20 examples double-passed for self-consistency.

## 6. The agent  (`src/agent/`)
- `classify.py` — LLM classifier (OpenAI, structured output / JSON schema),
  prompt includes taxonomy + few-shot. Also a cheap embedding-KNN classifier as
  the "simple baseline".
- `retrieve.py` — embed customer msg, retrieve top-k similar *resolved* past
  threads for the brand (FAISS). Return their customer msg + brand reply pairs.
- `draft.py` — LLM drafts reply grounded ONLY in retrieved precedent; must cite
  which precedent(s); instructed to abstain → escalate if precedent weak.
- `route.py` — decide auto vs escalate + reason. Signals: classifier confidence,
  retrieval similarity score, intent risk class (e.g. refunds/legal/safety =
  always escalate), presence of PII / anger / threat keywords, thread already
  long. Rule layer + LLM justification. Output structured reason.
- `pipeline.py` — orchestrates: message in → intent → retrieve → draft → route →
  JSON out. This is the "runnable pipeline" the README reproduces.

## 7. Baselines (report needs ≥2: trivial + simple)
- **Classification:** trivial = majority-class; simple = TF-IDF + logistic reg
  (or embedding-KNN). Headline = LLM classifier.
- **Reply:** trivial = one canned "A human will get back to you" reply;
  simple = return the brand reply of the single nearest neighbour verbatim.
  Headline = RAG-grounded LLM draft.
- **Routing:** trivial = always-escalate (and always-auto for contrast);
  simple = escalate if intent in risk-set. Headline = full signal router.

## 8. Evaluation harness  (`src/eval/`)
- `eval_classify.py` — accuracy, macro-F1, per-intent F1, confusion matrix vs golden.
- `eval_route.py` — precision/recall on "escalate" class, false-auto rate
  (dangerous), cost of over-escalation. Report the tradeoff curve.
- `eval_reply.py` — **LLM-as-judge** with an explicit rubric (1–5 each):
  groundedness (no claims outside precedent), helpfulness, tone/brand-voice,
  safety. Plus automatic: ROUGE-L / embedding-sim vs `reference_reply`,
  hallucination check (claims not in retrieved context).
- **Judge validation (the differentiator):** human-score ~40 replies myself on
  the same rubric, compute Cohen's κ / Spearman between judge and me, report it.
  Include a calibration paragraph on where judge and human diverge.
- `run_all.py` — one command → `reports/results.json` + printed tables. <15 min
  on the subsample.

## 9. Report  (`REPORT.md`, ≤6 pages)
Sections, in order:
1. Problem framing — what "good" means for this brand; explicit **non-goals**
   (no fine-tuning, no multi-turn dialogue state, no live posting, no non-English).
2. System overview (1 diagram).
3. Results vs baselines — tables for all 3 tasks.
4. Judge-vs-human agreement — κ, scatter, discussion.
5. Failure analysis — **top 5 modes**, each with a real example + hypothesis.
6. "What is misleading about my headline number?" — mandatory. (e.g. golden set
   labeled by one person; brand's historical reply treated as ground truth but
   it's often just "DM us"; test threads may overlap topically with corpus;
   English-only filter inflates quality; class imbalance flatters accuracy.)
7. What I'd do next with one more week.

## 10. Decision log  (`DECISIONS.md`) — 10–15 entries
Brand choice; intent count; "resolved" heuristic; golden set size & sampling;
treating first brand reply as reference; risk-set for routing; k for retrieval;
embedding model; judge model ≠ draft model (avoid self-preference); κ sample size;
subsample sizes; English-only; canned-vs-KNN baselines; structured-output vs
free-text parsing.

## 11. Repo layout
```
hiver-support-agent/
  README.md            # reproduce headline in <15 min
  REPORT.md  DECISIONS.md  INTENTS.md
  .env.example  requirements.txt  Makefile
  src/  (profile_brands, build_dataset, intents, agent/, eval/)
  data/    # git-ignored except tiny samples + golden.jsonl
  reports/ # committed results.json + figures
  tests/   # pytest: pipeline smoke, schema, no-leakage check
  docs/golden_set_note.md
```

## 12. Build order (checkpoints — stop for review at each ✋)
1. Env + deps + repo skeleton.  ✋
2. Data download + `profile_brands.py` → brand table. ✋ **confirm brand**
3. `build_dataset.py` + intent taxonomy → `INTENTS.md`. ✋
4. Golden set tooling + label 150–250 (may need user help / review). ✋
5. Agent pipeline (classify/retrieve/draft/route). ✋
6. Baselines + eval harness + judge validation. ✋
7. Report + decision log + README repro pass. ✋

## Open questions for user
- Kaggle API token available? OpenAI key + rough budget ceiling?
- OK with project at `~/pyprojects/hiver-support-agent`?
- Embeddings: OpenAI `text-embedding-3-small` (needs key, costs ~nothing) vs
  local `sentence-transformers`? (default: OpenAI small)
- How hands-on for golden labeling — I draft labels, you spot-check? (recommended)
