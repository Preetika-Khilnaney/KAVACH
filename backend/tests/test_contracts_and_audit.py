"""Smart contracts (consensus block, escalation) and the AI feedback audit."""

import time

from src.blockchain import contracts, feedback_audit
from src.storage import blockchain_ledger as bl
from src.storage import results_store


def _c2(ledger, wid, ip):
    ledger.log_prediction(window_id=wid, source_file="x", infiltration_probability=0.95, attack_stage="Command and Control",
                          confidence=0.9, feature_vector=[1.0], nodes=[{"id": "h", "label": ip, "type": "host"}])


def test_consensus_needs_three_independent_nodes(tmp_path):
    local = bl.get_ledger()
    _c2(local, "w1", "6.6.6.6")
    assert contracts.evaluate_consensus(time.time()) == []  # one node is not enough
    for peer in ("base-b", "base-c"):
        p = bl.Ledger(tmp_path / f"{peer}.db", node_id=peer)
        _c2(p, "pw", "6.6.6.6")
        assert local.import_bundle(p.export_delta())["accepted"]
    fired = contracts.evaluate_consensus(time.time())
    assert len(fired) == 1 and fired[0]["data"]["subject"] == "6.6.6.6"
    assert sorted(fired[0]["data"]["agreeing_nodes"]) == ["base-b", "base-c", "kavach-node-1"]
    assert contracts.evaluate_consensus(time.time()) == []  # idempotent


def test_escalation_when_unacknowledged(monkeypatch):
    L = bl.get_ledger()
    t0 = time.time() - 600
    for wid, stage, dt in (("r1", "Reconnaissance", 0), ("ia1", "Initial Access", 120)):
        with monkeypatch.context() as mp:
            mp.setattr(bl.time, "time", lambda dt=dt: t0 + dt)
            L.log_prediction(window_id=wid, source_file="cap", infiltration_probability=0.8, attack_stage=stage, confidence=0.9, feature_vector=[1.0])
    fired = contracts.evaluate_escalation(time.time())
    assert len(fired) == 1 and fired[0]["data"]["new_priority"] == "CRITICAL"


def test_feedback_audit_flags_subnet_shielding_and_db_edits():
    L = bl.get_ledger()
    for i in range(3):
        fb = results_store.save_feedback(f"alert-w{i}", "alert", "false_positive", "noise", "insider@kavach",
                                         {"window_id": f"w{i}", "infiltration_probability": 0.95, "host_ips": [f"10.0.1.{i + 5}"], "attack_stage": "Reconnaissance"})
        blk = L.log_analyst_action(analyst_id="insider@kavach", action="false_positive", target_id=f"alert-w{i}", target_type="alert", justification="noise", feedback_id=fb["id"])
        results_store.set_feedback_block(fb["id"], blk["index"])
    r = feedback_audit.analyze()
    kinds = {f["kind"] for f in r["flags"]}
    assert "pattern_anomaly" in kinds and "consistency_anomaly" in kinds
    assert r["excluded_feedback"] == 3
    # an edit made directly in results.db is caught against the chain
    conn = results_store._connect()
    conn.execute("UPDATE feedback SET verdict = 'confirmed' WHERE id = 1")
    conn.commit()
    assert any(f["kind"] == "audit_record_altered" for f in feedback_audit.analyze()["flags"])
