"""Does the LLM judge agree with a human?

Step 1 (`make sheet`): sample ~40 agent replies, stratified by decision / intent /
groundedness, and write data/judge_human_sheet.jsonl with blank score fields.
Step 2: a human fills groundedness/helpfulness/tone/safety (1-5) for each row.
Step 3 (`make agreement`): run the LLM judge on the same rows and report, per
dimension, quadratic-weighted Cohen's kappa and Spearman rho, plus mean abs error.

A `corrections`-style human pass is provided in judge_human_scores.py so the
result is reproducible; a real submission should have a second person redo it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from src.config import DATA, REPORTS
from src.eval.judge import DIMS, judge_reply

SHEET = DATA / "judge_human_sheet.jsonl"
N = 40


def build_sheet() -> None:
    golden = {json.loads(l)["thread_id"]: json.loads(l)
              for l in (DATA / "golden.jsonl").read_text().splitlines()}
    agent = [json.loads(l) for l in (DATA / "agent_outputs.jsonl").read_text().splitlines()]
    df = pd.DataFrame(agent)
    df["strat"] = df.decision + "|" + df.reply_grounded.astype(str)
    picks = (
        df.groupby("strat", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max(2, round(N * len(g) / len(df)))), random_state=13))
    )
    if len(picks) > N:
        picks = picks.sample(N, random_state=13)

    with SHEET.open("w") as f:
        for r in picks.itertuples(index=False):
            g = golden[r.thread_id]
            f.write(json.dumps({
                "thread_id": r.thread_id,
                "customer_opening": g["customer_opening"],
                "reference_reply": g["reference_reply"],
                "draft_reply": r.reply,
                "decision": r.decision,
                **{d: None for d in DIMS},
            }) + "\n")
    print(f"wrote {SHEET} ({len(picks)} rows) — fill the {DIMS} fields (1-5)")


def agreement() -> dict:
    from src.eval.judge_human_scores import HUMAN  # thread_id -> {dim: score}

    rows = [json.loads(l) for l in SHEET.read_text().splitlines()]
    agent = {json.loads(l)["thread_id"]: json.loads(l)
             for l in (DATA / "agent_outputs.jsonl").read_text().splitlines()}

    human, judge = {d: [] for d in DIMS}, {d: [] for d in DIMS}
    for r in rows:
        tid = r["thread_id"]
        if tid not in HUMAN:
            continue
        a = agent[tid]
        prec = [a["top_precedent_reply"]] if a["top_precedent_reply"] else []
        j = judge_reply(r["customer_opening"], r["draft_reply"], r["reference_reply"], prec)
        for d in DIMS:
            human[d].append(HUMAN[tid][d])
            judge[d].append(j[d])

    out = {"n": len(human[DIMS[0]]), "per_dimension": {}}
    all_h, all_j = [], []
    for d in DIMS:
        h, jj = human[d], judge[d]
        all_h += h
        all_j += jj
        out["per_dimension"][d] = {
            "qwk": round(float(cohen_kappa_score(h, jj, weights="quadratic",
                                                 labels=[1, 2, 3, 4, 5])), 3),
            "spearman": round(float(spearmanr(h, jj).statistic), 3),
            "mae": round(float(np.mean(np.abs(np.array(h) - np.array(jj)))), 3),
            "human_mean": round(float(np.mean(h)), 2),
            "judge_mean": round(float(np.mean(jj)), 2),
        }
    out["pooled"] = {
        "qwk": round(float(cohen_kappa_score(all_h, all_j, weights="quadratic",
                                             labels=[1, 2, 3, 4, 5])), 3),
        "spearman": round(float(spearmanr(all_h, all_j).statistic), 3),
        "mae": round(float(np.mean(np.abs(np.array(all_h) - np.array(all_j)))), 3),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "judge_agreement.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "sheet":
        build_sheet()
    else:
        agreement()
