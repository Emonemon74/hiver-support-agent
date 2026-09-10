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
