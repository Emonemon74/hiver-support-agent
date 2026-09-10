"""Replay the routing decision offline with the current rule layer (no LLM calls).

Uses cached per-example signals (predicted intent, top precedent similarity,
draft.grounded) from data/agent_outputs.jsonl, and keeps the LLM upgrade-only
escalations that were recorded in the original run (trigger == 'E-LLM').

Prints and writes reports/routing_v2.json:
  - agent router v2 on the 100-example eval subsample (tuned on these, optimistic)
  - rule layer v2 with GOLD intent on all 199 (semi-independent check that the
    rules themselves aren't just memorising the subsample)
"""
from __future__ import annotations

import json

from src.agent.route import route_rules
from src.config import DATA, REPORTS
from src.eval.metrics import routing_metrics


def _rows():
    golden = {j["thread_id"]: j for j in
              (json.loads(l) for l in (DATA / "golden.jsonl").read_text().splitlines())}
    agent = {j["thread_id"]: j for j in
             (json.loads(l) for l in (DATA / "agent_outputs.jsonl").read_text().splitlines())}
    return golden, agent


def main() -> None:
    golden, agent = _rows()
    subset = set(json.loads((DATA / "eval_subset.json").read_text()))

    # --- agent router v2 on the eval subsample ---
    y_true, y_v2, y_v1 = [], [], []
    flips = []
    for tid in subset:
        g, a = golden[tid], agent[tid]
        d, trig, _ = route_rules(
            g["customer_opening"], a["intent_pred"], a["intent_confidence"],
            a["top_precedent_score"], a["reply_grounded"],
        )
        if d == "auto" and a["trigger"] == "E-LLM":
            d, trig = "escalate", "E-LLM"      # keep the recorded LLM upgrade
        y_true.append(g["route"]); y_v2.append(d); y_v1.append(a["decision"])
        if d != a["decision"]:
            flips.append((a["decision"] + "->" + d, g["route"], trig,
                          g["customer_opening"][:80]))

    v1 = routing_metrics(y_true, y_v1)
    v2 = routing_metrics(y_true, y_v2)

    # --- rule layer v2 with GOLD intent, all 199 (rules-only sanity check) ---
    gt, gp = [], []
    for tid, g in golden.items():
        d, _, _ = route_rules(g["customer_opening"], g["intent"], 0.99, 0.6, True)
        gt.append(g["route"]); gp.append(d)
    gold_intent_all = routing_metrics(gt, gp)

    out = {
        "agent_router_v1_subsample": v1,
        "agent_router_v2_subsample": v2,
        "rule_v2_gold_intent_all199": gold_intent_all,
        "n_flips": len(flips),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "routing_v2.json").write_text(json.dumps(out, indent=2))

    def line(name, m):
        print(f"  {name:34s} acc {m['accuracy']:.3f}  esc-rec {m['escalate_recall']:.3f}  "
              f"false-auto {m['false_auto_rate']:.3f}  over-esc {m['over_escalation_rate']:.3f}  "
              f"F1 {m['escalate_f1']:.3f}")

    print("ROUTING — before vs after the fix\n")
    line("v1 agent router (subsample)", v1)
    line("v2 agent router (subsample)", v2)
    line("v2 rules w/ gold intent (n=199)", gold_intent_all)
    print(f"\n{len(flips)} decisions changed on the subsample:")
    for f in flips:
        print(f"  {f[0]:18s} gold={f[1]:9s} {f[2]:11s} {f[3]}")


if __name__ == "__main__":
    main()
