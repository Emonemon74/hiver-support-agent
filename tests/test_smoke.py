"""Skeleton smoke tests. Real pipeline tests are added per phase."""
import importlib


def test_config_imports():
    cfg = importlib.import_module("src.config")
    assert cfg.EMBED_MODEL == "text-embedding-3-small"
    assert cfg.SEED == 13


def test_get_data_module_imports():
    importlib.import_module("src.get_data")
