# Hiver Support Agent

An AI customer-support agent for **one Twitter brand**, built from the Kaggle
*Customer Support on Twitter* dataset. It (1) classifies an incoming customer
message into a data-derived intent, (2) drafts a reply grounded in how the brand
historically resolved similar issues, and (3) decides auto-handle vs. escalate
with a stated reason.

**The evaluation is the point.** See [`REPORT.md`](REPORT.md) for results,
baselines, judge-vs-human agreement, and failure analysis, and
[`DECISIONS.md`](DECISIONS.md) for the decision log.

## Reproduce the headline results (verified ~1 min, no API key)

```bash
uv venv --python 3.11 && source .venv/bin/activate
make install
make eval          # -> reports/results.json + the tables below
```

`make eval` replays from the committed caches (`data/golden.jsonl`,
`data/agent_outputs.jsonl`, `data/judge_scores.jsonl`, `data/eval_subset.json`).
On a fresh clone it downloads the ~130 MB embedding model once and then prints
every headline number in about a minute. **No `GROQ_API_KEY` needed for this
path.**

### Rebuild from scratch (needs data + a free Groq key)

```bash
cp .env.example .env                       # add GROQ_API_KEY (https://console.groq.com)
make data                                  # Kaggle API, or drop twcs.csv into ./data/
make dataset                               # Delta filter -> corpus + holdout (~2 min)
rm data/agent_outputs.jsonl data/judge_scores.jsonl
make agent && make eval && make judge-agreement && make routing
```

The from-scratch run is ~50 min and hits the Groq free-tier cap (200k
tokens/day/model), which is why the eval uses a **100-example stratified
subsample** (`data/eval_subset.json`); see REPORT.md.

**Headline:** intent classification **0.83 acc / 0.79 macro-F1** (vs 0.47 kNN);
routing **0.69 acc, 0.11 false-auto, 0.89 escalate-recall** (v2, after error
analysis — v1 was 0.59/0.38 and lost to a one-line rule; see REPORT.md §3b);
reply quality **4.13/5** LLM-judge (vs 3.73 nearest-neighbour), judge↔human
pooled κ **0.49**.

## Status

| Phase | State |
|-------|-------|
| 1. Env + skeleton | done |
| 2. Brand profiling → **Delta** | done |
| 3. Dataset + intent taxonomy | done |
| 4. Golden eval set (199, hand-labelled) | done |
| 5. Agent pipeline (classify/retrieve/draft/route) | done |
| 6. Baselines + eval harness + judge validation | done |
| 7. Report + decision log | done — `REPORT.md`, `DECISIONS.md` |

## Layout

```
src/
  config.py          paths, model names, seeds
  get_data.py        fetch twcs.csv
  profile_brands.py  thread reconstruction + per-brand metrics  (Phase 2)
  build_dataset.py   brand filter, clean, subsample             (Phase 3)
  llm.py             Groq chat wrapper (JSON mode, retry, cache)
  embed.py           local sentence-transformers + FAISS
  intents.py         the 8-intent taxonomy + router risk classes
  agent/             classify / retrieve / draft / route        (Phase 5)
  eval/              metrics + LLM-judge + judge validation      (Phase 6)
data/                git-ignored except data/samples/ and data/golden.jsonl
reports/             committed results.json + figures
docs/                golden_set_note.md, etc.
```
