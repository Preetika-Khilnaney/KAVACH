"""Forensic ledger: hash chaining, tamper/deletion detection, risk-tiered
chaining, Merkle batches, air-gapped sync."""

import sqlite3

from src.storage import blockchain_ledger as bl


def _pred(ledger, wid, p, stage="Reconnaissance", hosts=None):
    nodes = [{"id": f"h{i}", "label": ip, "type": "host"} for i, ip in enumerate(hosts or [])]
    return ledger.log_prediction(window_id=wid, source_file="cap.pcap", infiltration_probability=p, attack_stage=stage,
                                 confidence=0.9, feature_vector=[0.1, 0.2], top_features=[{"feature": "f", "attribution": 1.0}],
                                 nodes=nodes, model_sha256="abc")


def test_genesis_and_valid_chain(tmp_path):
    L = bl.Ledger(tmp_path / "c.db", node_id="n1")
    for i in range(4):
        _pred(L, str(i), 0.8)
    v = L.verify()
    assert v["valid"] and v["total_blocks"] == 5
    assert L.get_block(0)["block_type"] == bl.GENESIS


def test_modification_is_detected(tmp_path):
    L = bl.Ledger(tmp_path / "c.db", node_id="n1")
    for i in range(4):
        _pred(L, str(i), 0.8)
    conn = sqlite3.connect(L.db_path)
    conn.execute("UPDATE blocks SET data_json = replace(data_json, '0.8', '0.05') WHERE idx = 2")
    conn.commit()
    v = L.verify()
    assert not v["valid"] and v["first_bad_index"] == 2
    assert any(i["kind"] == "content_modified" for i in v["issues"])


def test_deletion_is_detected(tmp_path):
    L = bl.Ledger(tmp_path / "c.db", node_id="n1")
    for i in range(4):
        _pred(L, str(i), 0.8)
    conn = sqlite3.connect(L.db_path)
    conn.execute("DELETE FROM blocks WHERE idx = 2")
    conn.commit()
    v = L.verify()
    assert not v["valid"]
    kinds = {i["kind"] for i in v["issues"]}
    assert "missing_blocks" in kinds and "broken_link" in kinds


def test_simulated_tamper_leaves_real_chain_intact(tmp_path):
    L = bl.Ledger(tmp_path / "c.db", node_id="n1")
    for i in range(4):
        _pred(L, str(i), 0.8)
    r = bl.simulate_tamper(L, mode="modify")
    assert r["simulated"] and not r["verification"]["valid"]
    assert L.verify()["valid"]


def test_low_risk_windows_are_merkle_batched(tmp_path, monkeypatch):
    L = bl.Ledger(tmp_path / "c.db", node_id="n1")
    L.batch_size = 3
    out = [_pred(L, str(i), 0.05, stage="Benign") for i in range(3)]
    assert out[0]["block"] is None and out[2]["batch_block"] is not None
    batch = out[2]["batch_block"]["data"]
    assert batch["count"] == 3 and batch["merkle_root"] == bl.merkle_root(batch["item_hashes"])
    assert L.pending_low_risk() == 0


def test_find_blocks_for_window_includes_batches(tmp_path):
    L = bl.Ledger(tmp_path / "c.db", node_id="n1")
    L.batch_size = 2
    _pred(L, "w-a", 0.05, stage="Benign")
    _pred(L, "w-b", 0.05, stage="Benign")
    assert [b["block_type"] for b in L.find_blocks_for_window("w-a")] == [bl.PREDICTION_BATCH]


def test_sync_bundle_roundtrip_and_fork_detection(tmp_path):
    a = bl.Ledger(tmp_path / "a.db", node_id="base-a")
    b = bl.Ledger(tmp_path / "b.db", node_id="base-b")
    _pred(a, "1", 0.95, stage="Command and Control", hosts=["6.6.6.6"])
    bundle = a.export_delta()
    r = b.import_bundle(bundle)
    assert r["accepted"] and r["blocks_imported"] == 2
    assert b.latest_of_type(bl.SYNC)["data"]["peer_node_id"] == "base-a"
    # a tampered bundle fails verification
    bad = a.export_delta()
    bad["blocks"][1]["data_json"] = bad["blocks"][1]["data_json"].replace("0.95", "0.01")
    assert not bl.Ledger(tmp_path / "c.db", node_id="base-c").import_bundle(bad)["accepted"]


def test_model_update_only_recorded_on_change(tmp_path):
    L = bl.Ledger(tmp_path / "c.db", node_id="n1")
    assert L.record_model_if_changed({"sha256": "aaa"}) is not None
    assert L.record_model_if_changed({"sha256": "aaa"}) is None
    blk = L.record_model_if_changed({"sha256": "bbb"})
    assert blk["data"]["previous_weights_sha256"] == "aaa"
