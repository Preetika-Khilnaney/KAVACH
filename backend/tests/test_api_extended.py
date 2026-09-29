"""API contract checks for the endpoints added in the enhancement pass."""

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.benchmark import evaluate
from src.orchestrator import model_registry
from src.storage import results_store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "CHECKPOINT_PATH", tmp_path / "none.pt")
    model_registry._cache.update(path=None, mtime=None, model=None, trained=False)
    with TestClient(main.app) as c:
        yield c


def _seed():
    state = {"nodes": [{"id": "network", "type": "network", "label": "window"}, {"id": "host_0", "type": "host", "label": "10.0.0.5"}], "edges": []}
    for i, p in enumerate([0.2, 0.9]):
        results_store.save_prediction(f"cap-abc-{i}", "cap.pcap", {"flow_count": 1}, state, p, "Reconnaissance", 0.8, True,
                                      [{"feature": "flow_count", "attribution": 0.5}], extra={"scoring_method": "packet_heuristic"})


def test_benchmark_without_windows_cache_is_structured(client, monkeypatch):
    monkeypatch.setattr(evaluate, "load_all_windows", lambda: {})
    r = client.get("/benchmark?refresh=true").json()
    assert r["status"] == "windows_cache_missing" and r["netjepa"] is None


def test_feedback_is_persisted_and_chained(client):
    _seed()
    r = client.post("/feedback", json={"targetId": "alert-cap-abc-1", "verdict": "false-positive", "note": "scanner", "analystId": "a@x"}).json()
    assert r["ok"] and r["block"]["index"] > 0
    assert client.get("/feedback/alert-cap-abc-1").json()[0]["verdict"] == "false_positive"
    assert client.get("/feedback/stats").json()["false_positive"] == 1
    alerts = client.get("/alerts").json()
    assert alerts[0]["analystVerdict"] == "false-positive"
    assert client.get("/blockchain/verify").json()["valid"]


def test_risk_trend_timeline_filters_and_audit(client):
    _seed()
    risk = client.get("/risk-score").json()
    assert risk["trend"] == "up" and risk["delta"] == 70
    assert [p["riskScore"] for p in client.get("/timeline").json()["points"]] == [20, 90]
    assert [f["id"] for f in client.get("/flows?risk_level=critical").json()] == ["cap-abc-1"]
    assert [f["id"] for f in client.get("/flows?sort_by=infiltration_probability&sort_dir=asc").json()] == ["cap-abc-0", "cap-abc-1"]
    steps = client.get("/flows/cap-abc-1/audit").json()["steps"]
    assert [s["stageName"] for s in steps][:5] == ["Raw Flow", "Feature Extraction", "Graph State", "Forecast", "Attack Stage Mapping"]
    chain = client.get("/attack-chain/cap-abc-1").json()
    assert chain["current_stage"] == "Reconnaissance" and chain["remediation"]["current"]


def test_login_remediation_and_tamper_demo(client):
    r = client.post("/auth/login", json={"username": "l1.analyst", "password": "kavach2026"}).json()
    assert r["user"]["analystId"] == "l1.analyst@kavach.local"
    assert client.post("/auth/login", json={"username": "l1.analyst", "password": "x"}).status_code == 401
    acts = client.get("/remediation/Command and Control?dst_ip=6.6.6.6").json()
    assert "6.6.6.6" in acts[0]["command"]
    _seed()
    client.post("/remediation/log", json={"action": "block_ip", "target": "6.6.6.6"}, headers={"Authorization": f"Bearer {r['token']}"})
    for _ in range(3):
        client.post("/feedback", json={"targetId": "cap-abc-1", "verdict": "confirmed"})
    r = client.post("/blockchain/tamper-demo").json()
    assert r["simulated"] and not r["verification"]["valid"]
    assert client.get("/blockchain/verify").json()["valid"]
