"""Scoring functions for classification and routing."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from src.intents import LABELS


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    macro_f1 = float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))
    per_intent = {
        lab: round(float(f), 3)
        for lab, f in zip(
            LABELS, f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
        )
    }
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    return {"accuracy": round(acc, 3), "macro_f1": round(macro_f1, 3),
            "per_intent_f1": per_intent, "labels": LABELS, "confusion": cm}


def routing_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """'escalate' is the positive class. false-auto = a should-escalate that was
    auto'd (the dangerous error)."""
    tp = sum(t == "escalate" and p == "escalate" for t, p in zip(y_true, y_pred))
    fp = sum(t == "auto" and p == "escalate" for t, p in zip(y_true, y_pred))
    fn = sum(t == "escalate" and p == "auto" for t, p in zip(y_true, y_pred))
    tn = sum(t == "auto" and p == "auto" for t, p in zip(y_true, y_pred))
    n_esc = tp + fn
    n_auto = tn + fp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / n_esc if n_esc else 0.0
    return {
        "accuracy": round((tp + tn) / len(y_true), 3),
        "escalate_precision": round(prec, 3),
        "escalate_recall": round(rec, 3),
        "escalate_f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
        "false_auto_rate": round(fn / n_esc, 3) if n_esc else 0.0,   # missed escalations
        "over_escalation_rate": round(fp / n_auto, 3) if n_auto else 0.0,
        "kappa": round(float(cohen_kappa_score(y_true, y_pred)), 3),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
