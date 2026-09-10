"""Skeleton smoke tests. Real pipeline tests are added per phase."""
import importlib


def test_config_imports():
    cfg = importlib.import_module("src.config")
    assert cfg.SEED == 13
    assert cfg.BRAND == "Delta"


def test_intent_taxonomy_consistent():
    from src.intents import INTENTS, LABELS, RISK_CLASS

    assert set(LABELS) == set(INTENTS) == set(RISK_CLASS)
    assert "other" in LABELS
    assert all(v in {"low", "medium", "high"} for v in RISK_CLASS.values())


def test_modules_import():
    for m in ("src.get_data", "src.threads", "src.clean", "src.llm", "src.embed"):
        importlib.import_module(m)


def test_golden_set_valid():
    import json
    from pathlib import Path

    from src.intents import LABELS

    p = Path("data/golden.jsonl")
    if not p.exists():
        return  # golden set not built yet
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert 150 <= len(rows) <= 250
    ids = {r["thread_id"] for r in rows}
    assert len(ids) == len(rows), "duplicate thread_ids"
    for r in rows:
        assert r["intent"] in LABELS
        assert r["route"] in {"auto", "escalate"}
        assert r["difficulty"] in {"easy", "ambiguous"}
        assert len(r["route_reason"]) > 5
        assert r["reference_reply"]
