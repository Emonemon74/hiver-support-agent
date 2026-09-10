# Hiver Support Agent

An AI customer-support agent for **one Twitter brand**, built from the Kaggle
*Customer Support on Twitter* dataset. It (1) classifies an incoming customer
message into a data-derived intent, (2) drafts a reply grounded in how the brand
historically resolved similar issues, and (3) decides auto-handle vs. escalate
with a stated reason.

**The evaluation is the point.** See [`REPORT.md`](REPORT.md) for results,
baselines, judge-vs-human agreement, and failure analysis, and
[`DECISIONS.md`](DECISIONS.md) for the decision log.

## Reproduce the headline results (<15 min)

```bash
# 1. Environment (Python 3.11)
uv venv --python 3.11 && source .venv/bin/activate
make install

# 2. Secrets
cp .env.example .env      # add GROQ_API_KEY (free, no card: https://console.groq.com)
                          # embeddings run locally — no key needed

# 3. Data  (Kaggle API, or drop twcs.csv into ./data/ yourself)
make data

# 4. Pipeline + evaluation
make dataset      # filter to Delta, build corpus + eval holdout (~2 min)
make eval         # baselines + agent + LLM-judge -> reports/results.json + tables
```

`make eval` reuses the committed caches (`data/agent_outputs.jsonl`,
`data/judge_scores.jsonl`) and prints in seconds. Deleting them re-runs the
agent + judge from scratch (~50 min, and needs Groq quota — the free tier caps
at 200k tokens/day/model, which is why the eval runs on a 100-example subsample;
see `data/eval_subset.json` and REPORT.md §top).

**Headline:** intent classification **0.83 acc / 0.79 macro-F1** (vs 0.47 kNN);
routing **0.59 acc, 0.38 false-auto** — *worse than a one-line risk rule*, the
main finding; reply quality **4.13/5** LLM-judge (vs 3.73 nearest-neighbour),
judge↔human pooled κ **0.49**.

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
