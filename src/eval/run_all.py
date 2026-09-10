"""Phase 6: one command -> all metrics for all three tasks, vs baselines.

Writes reports/results.json and prints tables. Assumes:
  - data/golden.jsonl              (Phase 4)
  - data/agent_outputs.jsonl       (src.eval.run_agent)
The LLM judge runs over every draft once and is cached to data/judge_scores.jsonl.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import DATA, JUDGE_MODEL, REPORTS
from src.eval import baselines as bl
from src.eval.judge import judge_reply, reference_similarity
from src.eval.metrics import classification_metrics, routing_metrics

JUDGE_CACHE = DATA / "judge_scores.jsonl"


def _load():
    golden = pd.DataFrame(json.loads(l) for l in (DATA / "golden.jsonl").read_text().splitlines())
    agent = pd.DataFrame(json.loads(l) for l in (DATA / "agent_outputs.jsonl").read_text().splitlines())
    sub_path = DATA / "eval_subset.json"
    if sub_path.exists():
        keep = set(json.loads(sub_path.read_text()))
        golden = golden[golden.thread_id.isin(keep)].reset_index(drop=True)
        agent = agent[agent.thread_id.isin(keep)].reset_index(drop=True)
    df = golden.merge(agent, on="thread_id", suffixes=("", "_a"))
    return golden, df


def classification_block(golden, df) -> dict:
    texts = golden.customer_opening.tolist()
    y = golden.intent.tolist()
    return {
        "trivial_majority": classification_metrics(y, bl.majority_class(y)),
        "simple_knn_loo": classification_metrics(y, bl.knn_loo(texts, y)),
        "agent_llm": classification_metrics(df.intent.tolist(), df.intent_pred.tolist()),
    }


def routing_block(golden, df) -> dict:
    y = golden.route.tolist()
    gold_intent = golden.intent.tolist()
    pred_intent = df.intent_pred.tolist()
    out = {
        "trivial_always_escalate": routing_metrics(y, ["escalate"] * len(y)),
        "trivial_always_auto": routing_metrics(y, ["auto"] * len(y)),
        "simple_risk_rule_gold_intent": routing_metrics(y, bl.route_risk_rule(gold_intent)),
        "simple_risk_rule_pred_intent": routing_metrics(y, bl.route_risk_rule(pred_intent)),
        "agent_router": routing_metrics(df.route.tolist(), df.decision.tolist()),
    }
    # slice: easy only
    easy = df[df.difficulty == "easy"]
    out["agent_router_easy_only"] = routing_metrics(easy.route.tolist(), easy.decision.tolist())
    return out


BASELINE_JUDGE_MODEL = "qwen/qwen3.6-27b"  # spread the per-model daily token cap


def _judge_all(df) -> pd.DataFrame:
    # agent replies judged on all rows; baseline replies on a fixed 40-row sample
    base_ids = set(df.sample(min(40, len(df)), random_state=13).thread_id)
    done = set()
    if JUDGE_CACHE.exists():
        done = {(json.loads(l)["thread_id"], json.loads(l)["variant"])
                for l in JUDGE_CACHE.read_text().splitlines()}
    with JUDGE_CACHE.open("a") as f:
        for r in tqdm(df.itertuples(index=False), total=len(df), desc="judge"):
            prec = [r.top_precedent_reply] if r.top_precedent_reply else []
            variants = [("agent", r.reply)]
            if r.thread_id in base_ids:
                variants += [
                    ("trivial_canned", bl.CANNED_REPLY),
                    ("simple_nn_reply", r.top_precedent_reply or bl.CANNED_REPLY),
                ]
            for variant, reply in variants:
                if (r.thread_id, variant) in done:
                    continue
                jm = JUDGE_MODEL if variant == "agent" else BASELINE_JUDGE_MODEL
                s = judge_reply(r.customer_opening, reply, r.reference_reply, prec, model=jm)
                f.write(json.dumps({"thread_id": r.thread_id, "variant": variant, **s}) + "\n")
    return pd.DataFrame(json.loads(l) for l in JUDGE_CACHE.read_text().splitlines())


def reply_block(golden, df) -> dict:
    j = _judge_all(df)
    sim = reference_similarity(df.reply.tolist(), df.reference_reply.tolist())
    out = {}
    for variant in ("trivial_canned", "simple_nn_reply", "agent"):
        v = j[j.variant == variant]
        out[variant] = {
            "n": len(v),
            **{d: round(float(v[d].mean()), 2)
               for d in ("groundedness", "helpfulness", "tone", "safety", "overall")},
            "hallucination_rate": round(float(v["hallucination"].mean()), 3),
        }
    out["agent"]["reply_vs_reference_cosine"] = round(float(np.mean(sim)), 3)
    # agent replies only make it to a customer when decision == auto
    auto = df[df.decision == "auto"]
    ja = j[(j.variant == "agent") & (j.thread_id.isin(auto.thread_id))]
    out["agent_auto_only"] = {
        "n": len(ja),
        **{d: round(float(ja[d].mean()), 2)
           for d in ("groundedness", "helpfulness", "tone", "safety", "overall")},
        "hallucination_rate": round(float(ja["hallucination"].mean()), 3),
    }
    return out


def main() -> None:
    golden, df = _load()
    results = {
        "n_golden": len(golden),
        "n_agent_outputs": len(df),
        "classification": classification_block(golden, df),
        "routing": routing_block(golden, df),
        "reply_quality": reply_block(golden, df),
    }
    try:
        results["judge_agreement"] = json.loads((REPORTS / "judge_agreement.json").read_text())
    except FileNotFoundError:
        results["judge_agreement"] = None

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.json").write_text(json.dumps(results, indent=2))
    _print_tables(results)
    print(f"\nfull results -> {REPORTS / 'results.json'}")


def _print_tables(r: dict) -> None:
    print("\n=== CLASSIFICATION (accuracy / macro-F1) ===")
    for k, v in r["classification"].items():
        print(f"  {k:24s}  acc {v['accuracy']:.3f}   macroF1 {v['macro_f1']:.3f}")
    print("\n=== ROUTING (acc / escalate-recall / false-auto / over-escalation) ===")
    for k, v in r["routing"].items():
        print(f"  {k:32s}  acc {v['accuracy']:.3f}  rec {v['escalate_recall']:.3f}  "
              f"false-auto {v['false_auto_rate']:.3f}  over-esc {v['over_escalation_rate']:.3f}")
    print("\n=== REPLY QUALITY (LLM judge, 1-5) ===")
    for k, v in r["reply_quality"].items():
        print(f"  {k:20s}  overall {v['overall']:.2f}  ground {v['groundedness']:.2f}  "
              f"help {v['helpfulness']:.2f}  tone {v['tone']:.2f}  safe {v['safety']:.2f}  "
              f"halluc {v['hallucination_rate']:.3f}")
    ja = r.get("judge_agreement")
    if ja:
        print(f"\n=== JUDGE vs HUMAN (n={ja['n']}) pooled QWK {ja['pooled']['qwk']}  "
              f"Spearman {ja['pooled']['spearman']}  MAE {ja['pooled']['mae']}")


if __name__ == "__main__":
    main()
