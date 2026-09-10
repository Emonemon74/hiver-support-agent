"""Recompute the routing decision for every cached agent output with the current
router (src/agent/route.py) and refresh both data/agent_outputs.jsonl and the
routing metrics.

The agent run (intent + retrieval + draft) is expensive and cached; routing is
cheap, so this lets the router be iterated without re-running the pipeline. It
re-invokes the real `route()` (rules + the LLM upgrade-only check, which is
disk-cached in src.llm), then writes:

  data/agent_outputs.jsonl   decision / trigger / reason / signals refreshed
  reports/routing_v2.json    agent router on the 100-subsample + rule layer with
                             GOLD intent over all 199 (generalisation check)

Run: python -m src.eval.rerun_routing
"""
from __future__ import annotations

import json

from src.agent.classify import IntentPred
from src.agent.draft import Draft
from src.agent.retrieve import Precedent
from src.agent.route import route, route_rules
from src.config import DATA, REPORTS
from src.eval.metrics import routing_metrics

AGENT = DATA / "agent_outputs.jsonl"


def main() -> None:
    golden = {j["thread_id"]: j for j in
              (json.loads(l) for l in (DATA / "golden.jsonl").read_text().splitlines())}
    rows = [json.loads(l) for l in AGENT.read_text().splitlines()]

    y_true, y_pred = [], []
    out_rows = []
    for a in rows:
        g = golden[a["thread_id"]]
        intent = IntentPred(a["intent_pred"], a["intent_confidence"])
        prec = [Precedent(opening="", reply=a["top_precedent_reply"],
                          score=a["top_precedent_score"], is_handoff=False)]
        draft = Draft(reply=a["reply"], grounded=a["reply_grounded"],
                      used_precedent=[], notes="")
        r = route(g["customer_opening"], intent, prec, draft)
        a.update(decision=r.decision, trigger=r.trigger, reason=r.reason, signals=r.signals)
        out_rows.append(a)
        y_true.append(g["route"]); y_pred.append(r.decision)

    with AGENT.open("w") as f:
        for a in out_rows:
            f.write(json.dumps(a) + "\n")

    agent_router = routing_metrics(y_true, y_pred)

    # rule layer only, GOLD intent, all 199 — checks the rules generalise beyond
    # the tuned 100-example subsample
    gt, gp = [], []
    for g in golden.values():
        d, _, _ = route_rules(g["customer_opening"], g["intent"], 0.99, 0.6, True)
        gt.append(g["route"]); gp.append(d)
    rule_gold_all = routing_metrics(gt, gp)

    res = {"agent_router_subsample_n": len(y_true),
           "agent_router_subsample": agent_router,
           "rule_layer_gold_intent_all199": rule_gold_all}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "routing_v2.json").write_text(json.dumps(res, indent=2))

    def line(name, m):
        print(f"  {name:32s} acc {m['accuracy']:.3f}  esc-rec {m['escalate_recall']:.3f}  "
              f"false-auto {m['false_auto_rate']:.3f}  over-esc {m['over_escalation_rate']:.3f}  "
              f"F1 {m['escalate_f1']:.3f}")

    print("ROUTING (current router)\n")
    line(f"agent router (n={len(y_true)})", agent_router)
    line("rule layer / gold intent (n=199)", rule_gold_all)


if __name__ == "__main__":
    main()
