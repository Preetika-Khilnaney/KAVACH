"""Shared test isolation: no test may write to the real forensic ledger
(data/processed/blockchain.db) or the real results store."""

import pytest

from src.storage import blockchain_ledger, results_store


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(blockchain_ledger, "DB_PATH", tmp_path / "test_chain.db")
    monkeypatch.setattr(results_store, "DB_PATH", tmp_path / "test_results_default.db")
    yield


@pytest.fixture(autouse=True)
def _isolated_benchmark_cache(tmp_path, monkeypatch):
    from src.benchmark import evaluate
    from src.orchestrator import model_registry

    monkeypatch.setattr(evaluate, "DISK_CACHE", tmp_path / "benchmark_cache.json")
    monkeypatch.setattr(model_registry, "ACTIVE_MODEL_OVERRIDE", tmp_path / "active_model.json")
    evaluate._cache.update(key=None, result=None)
    yield


@pytest.fixture(autouse=True)
def _isolated_auth(tmp_path, monkeypatch):
    from src.auth import tiered

    monkeypatch.setattr(tiered, "AUTH_DIR", tmp_path / "auth")
    tiered.reset_state()
    yield
    tiered.reset_state()
