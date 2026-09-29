"""Tamper-evident forensic ledger (roadmap Workstream 4; see the
"Why Blockchain Is Genuinely Necessary" and "How AI and Blockchain Make
Each Other Stronger" design notes).

A lightweight, local, SQLite-backed hash chain. Every block stores:

    index | timestamp | block_type | node_id | data (canonical JSON) | prev_hash | hash

where hash = SHA-256 over all of those fields. Because each block embeds
the previous block's hash:

* modifying any stored block breaks that block's own hash;
* deleting a block leaves a gap in the index sequence AND breaks the next
  block's prev_hash link -- deletion is detectable, which signed logs /
  append-only databases can't offer;
* rewriting history means recomputing every later block, and every peer
  node that holds a copy (see export_delta/import_bundle) still has the
  original hashes.

The chain lives in its own database (data/processed/blockchain.db), isolated
from results.db, so the operational store can be rebuilt without touching the
evidence trail.

What goes on chain is *hashes plus the decision-relevant fields*, never raw
PCAPs, full feature vectors or weights (those stay in local storage and can
be verified against the on-chain hash later -- see evidence_for_window in
src/api/main.py).

AI -> blockchain synergy implemented here:
* Risk-tiered chaining (log_prediction): low-risk windows are batched 100 at
  a time into one PREDICTION_BATCH block with a Merkle root; medium-risk
  windows get a standard block; high-risk ones get a full block with feature /
  graph / attribution hashes; critical ones additionally get an ALERT block.
* Every prediction block carries the SHA-256 of the model weights that
  produced it, so model provenance is traceable per prediction.

Production note: this is a single-writer local chain for a prototype. A real
multi-base deployment would run a permissioned ledger (e.g. Hyperledger
Fabric) with one peer per site and per-node signing keys; the sync bundle
format below mirrors that shape for air-gapped, sneakernet synchronisation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from src.config import load_config

DB_PATH = Path("data/processed/blockchain.db")

GENESIS = "GENESIS"
PREDICTION = "PREDICTION"
PREDICTION_BATCH = "PREDICTION_BATCH"
ALERT = "ALERT"
ANALYST_ACTION = "ANALYST_ACTION"
MODEL_UPDATE = "MODEL_UPDATE"
TRAINING_MANIFEST = "TRAINING_MANIFEST"
REMEDIATION = "REMEDIATION"
CONTRACT_EXECUTION = "CONTRACT_EXECUTION"
ESCALATION = "ESCALATION"
FEEDBACK_AUDIT = "FEEDBACK_AUDIT"
SYNC = "SYNC"
ACCESS = "ACCESS"

BLOCK_TYPE_LABELS = {
    GENESIS: "Genesis",
    PREDICTION: "Prediction",
    PREDICTION_BATCH: "Prediction Batch",
    ALERT: "Alert",
    ANALYST_ACTION: "Analyst Action",
    MODEL_UPDATE: "Model Update",
    TRAINING_MANIFEST: "Training Manifest",
    REMEDIATION: "Remediation",
    CONTRACT_EXECUTION: "Smart Contract",
    ESCALATION: "Escalation",
    FEEDBACK_AUDIT: "Feedback Audit",
    SYNC: "Node Sync",
    ACCESS: "Access Event",
}

ZERO_HASH = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS blocks (
    idx INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    block_type TEXT NOT NULL,
    node_id TEXT NOT NULL,
    data_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_blocks_type ON blocks(block_type);
CREATE TABLE IF NOT EXISTS peer_blocks (
    node_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    block_type TEXT NOT NULL,
    origin_node_id TEXT NOT NULL,
    data_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL,
    imported_at REAL NOT NULL,
    PRIMARY KEY (node_id, idx)
);
CREATE TABLE IF NOT EXISTS low_risk_buffer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_hash TEXT NOT NULL,
    data_json TEXT NOT NULL
);
"""


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def canonical_json(value: Any) -> str:
    """Deterministic serialisation -- the exact string that gets hashed and
    stored, so verification never depends on re-serialising."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default, allow_nan=False)


def sha256_hex(value: Any) -> str:
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    if not isinstance(value, str):
        value = canonical_json(_finite(value))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _finite(value):
    """NaN/inf aren't valid canonical JSON; clamp recursively."""
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if hasattr(value, "item"):
        return _finite(value.item())
    return value


def block_hash(idx: int, timestamp: float, block_type: str, node_id: str, data_json: str, prev_hash: str) -> str:
    payload = f"{idx}|{timestamp!r}|{block_type}|{node_id}|{prev_hash}|{data_json}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def merkle_root(hashes: list[str]) -> str:
    if not hashes:
        return ZERO_HASH
    level = list(hashes)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256((level[i] + level[i + 1]).encode()).hexdigest() for i in range(0, len(level), 2)]
    return level[0]


def _cfg() -> dict:
    return load_config().get("blockchain", {}) or {}


def risk_tier(probability: float, stage: str | None) -> str:
    """AI-driven chaining policy (synergy doc 2.1)."""
    cfg = _cfg()
    low, high, critical = cfg.get("low_risk_threshold", 0.3), cfg.get("high_risk_threshold", 0.7), cfg.get("critical_threshold", 0.9)
    late_stage = stage in ("Command and Control", "Exfiltration")
    if probability >= critical and late_stage:
        return "critical"
    if probability >= high:
        return "high"
    if probability >= low:
        return "medium"
    return "low"


def prediction_core(*, window_id, source_file, infiltration_probability, attack_stage, confidence, model_sha256,
                    scoring_method, feature_vector, calibrated_probability, tier) -> dict:
    """The decision-relevant fields chained for every prediction. Shared by
    log_prediction and evidence verification, so a stored prediction can be
    re-derived from results.db and checked against the chain."""
    core = {
        "window_id": str(window_id),
        "source_file": source_file,
        "infiltration_probability": round(float(infiltration_probability), 6),
        "attack_stage": attack_stage,
        "confidence": round(float(confidence), 6),
        "model_sha256": model_sha256,
        "scoring_method": scoring_method,
        "feature_vector_sha256": sha256_hex(feature_vector) if feature_vector is not None else None,
        "tier": tier,
    }
    if calibrated_probability is not None:
        core["calibrated_probability"] = round(float(calibrated_probability), 6)
    return core


def batch_item_hash(core: dict) -> str:
    return sha256_hex(canonical_json(_finite(core)))


class Ledger:
    def __init__(self, db_path: str | Path = DB_PATH, node_id: str | None = None):
        self.db_path = Path(db_path)
        self.node_id = node_id or _cfg().get("node_id", "kavach-node-1")
        self.batch_size = int(_cfg().get("low_risk_batch_size", 100))
        self._lock = threading.RLock()
        with self._connect() as conn:
            if conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 0:
                self._append(conn, GENESIS, {"message": "KAVACH forensic ledger genesis", "node_id": self.node_id})

    # ── plumbing ───────────────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(_SCHEMA)
        return conn

    def _append(self, conn: sqlite3.Connection, block_type: str, data: dict, timestamp: float | None = None) -> dict:
        with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                tip = conn.execute("SELECT idx, hash FROM blocks ORDER BY idx DESC LIMIT 1").fetchone()
                idx = (tip["idx"] + 1) if tip else 0
                prev = tip["hash"] if tip else ZERO_HASH
                ts = float(timestamp if timestamp is not None else time.time())
                data_json = canonical_json(_finite(data))
                h = block_hash(idx, ts, block_type, self.node_id, data_json, prev)
                conn.execute(
                    "INSERT INTO blocks (idx, timestamp, block_type, node_id, data_json, prev_hash, hash) VALUES (?,?,?,?,?,?,?)",
                    (idx, ts, block_type, self.node_id, data_json, prev, h),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return {"index": idx, "timestamp": ts, "block_type": block_type, "node_id": self.node_id, "data": json.loads(data_json), "prev_hash": prev, "hash": h}

    def append(self, block_type: str, data: dict) -> dict:
        conn = self._connect()
        try:
            return self._append(conn, block_type, data)
        finally:
            conn.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        return {
            "index": row["idx"],
            "timestamp": row["timestamp"],
            "block_type": row["block_type"],
            "type_label": BLOCK_TYPE_LABELS.get(row["block_type"], row["block_type"]),
            "node_id": row["node_id"],
            "data": json.loads(row["data_json"]),
            "prev_hash": row["prev_hash"],
            "hash": row["hash"],
        }

    # ── reads ──────────────────────────────────────────────────────────────
    def count(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        finally:
            conn.close()

    def get_chain(self, limit: int = 50, block_type: str | None = None, before: int | None = None) -> list[dict]:
        conn = self._connect()
        try:
            q, args = "SELECT * FROM blocks", []
            clauses = []
            if block_type:
                clauses.append("block_type = ?")
                args.append(block_type)
            if before is not None:
                clauses.append("idx < ?")
                args.append(before)
            if clauses:
                q += " WHERE " + " AND ".join(clauses)
            q += " ORDER BY idx DESC LIMIT ?"
            args.append(limit)
            return [self._row(r) for r in conn.execute(q, args).fetchall()]
        finally:
            conn.close()

    def get_block(self, idx: int) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM blocks WHERE idx = ?", (idx,)).fetchone()
            return self._row(row) if row else None
        finally:
            conn.close()

    def iter_blocks(self, block_types: Iterable[str] | None = None, since: float | None = None) -> list[dict]:
        conn = self._connect()
        try:
            q, args, clauses = "SELECT * FROM blocks", [], []
            if block_types:
                types = list(block_types)
                clauses.append(f"block_type IN ({','.join('?' * len(types))})")
                args += types
            if since is not None:
                clauses.append("timestamp >= ?")
                args.append(since)
            if clauses:
                q += " WHERE " + " AND ".join(clauses)
            q += " ORDER BY idx ASC"
            return [self._row(r) for r in conn.execute(q, args).fetchall()]
        finally:
            conn.close()

    def type_counts(self) -> dict:
        conn = self._connect()
        try:
            return {r[0]: r[1] for r in conn.execute("SELECT block_type, COUNT(*) FROM blocks GROUP BY block_type").fetchall()}
        finally:
            conn.close()

    def find_blocks_for_window(self, window_id: str) -> list[dict]:
        """Every block that references this window: its prediction (or the
        Merkle batch containing it), alert, escalations, analyst actions."""
        wid = str(window_id)
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM blocks WHERE data_json LIKE ? ORDER BY idx ASC", (f"%{json.dumps(wid)[1:-1]}%",)).fetchall()
        finally:
            conn.close()
        out = []
        for r in rows:
            b = self._row(r)
            d = b["data"]
            if (d.get("window_id") == wid or d.get("subject") == wid or d.get("target_id") in (wid, f"alert-{wid}")
                    or wid in (d.get("window_ids") or [])):
                out.append(b)
        return out

    # ── verification ───────────────────────────────────────────────────────
    @staticmethod
    def _verify_rows(rows: list, start_prev: str | None = None, start_idx: int | None = None) -> dict:
        issues = []
        prev_hash = start_prev
        expected_idx = start_idx
        for r in rows:
            idx = r["idx"]
            if expected_idx is not None and idx != expected_idx:
                issues.append({"index": expected_idx, "kind": "missing_blocks",
                               "detail": f"blocks {expected_idx}..{idx - 1} are missing (deleted) -- index sequence jumps from {expected_idx - 1} to {idx}"})
            if prev_hash is not None and r["prev_hash"] != prev_hash:
                issues.append({"index": idx, "kind": "broken_link",
                               "detail": f"block {idx} points to previous hash {r['prev_hash'][:16]}... but the preceding block's hash is {prev_hash[:16]}..."})
            origin = r["origin_node_id"] if "origin_node_id" in r.keys() else r["node_id"]
            recomputed = block_hash(idx, r["timestamp"], r["block_type"], origin, r["data_json"], r["prev_hash"])
            if recomputed != r["hash"]:
                issues.append({"index": idx, "kind": "content_modified",
                               "detail": f"block {idx} content does not match its stored hash (stored {r['hash'][:16]}..., recomputed {recomputed[:16]}...)"})
            prev_hash = r["hash"]
            expected_idx = idx + 1
        return {"issues": issues, "tip_hash": prev_hash}

    def verify(self) -> dict:
        """Walk genesis -> tip, recomputing every hash and link. Returns
        {valid, message, total_blocks, first_bad_index, issues, tip_hash}."""
        t0 = time.time()
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM blocks ORDER BY idx ASC").fetchall()
        finally:
            conn.close()
        if not rows:
            return {"valid": False, "message": "Chain is empty -- genesis block missing.", "total_blocks": 0, "issues": [], "first_bad_index": None}
        issues = []
        if rows[0]["idx"] != 0 or rows[0]["block_type"] != GENESIS:
            issues.append({"index": 0, "kind": "missing_genesis", "detail": "genesis block has been removed or replaced"})
        result = self._verify_rows(rows, start_prev=ZERO_HASH if rows[0]["idx"] == 0 else None, start_idx=0)
        issues += result["issues"]
        valid = not issues
        first_bad = min((i["index"] for i in issues), default=None)
        msg = (f"Chain intact: {len(rows)} blocks verified from genesis to tip, no tampering detected."
               if valid else f"TAMPERING DETECTED: {len(issues)} integrity violation(s), first at block {first_bad}. {issues[0]['detail']}")
        return {"valid": valid, "message": msg, "total_blocks": len(rows), "first_bad_index": first_bad,
                "issues": issues[:50], "tip_hash": rows[-1]["hash"], "tip_index": rows[-1]["idx"],
                "node_id": self.node_id, "verified_in_ms": round((time.time() - t0) * 1000, 1)}

    # ── domain logging (synergy doc) ───────────────────────────────────────
    def log_prediction(self, *, window_id: str, source_file: str | None, infiltration_probability: float,
                       attack_stage: str, confidence: float, feature_vector: list | None = None,
                       top_features: list | None = None, nodes: list | None = None, edges: list | None = None,
                       model_sha256: str | None = None, scoring_method: str | None = None,
                       calibrated_probability: float | None = None, predicted_stages: list | None = None,
                       remediation: list | None = None) -> dict:
        """Risk-tiered chaining. Returns {"tier", "block" (or None if
        buffered), "batch_block" (when a buffer flush happened)}."""
        tier = risk_tier(infiltration_probability, attack_stage)
        core = prediction_core(window_id=window_id, source_file=source_file, infiltration_probability=infiltration_probability,
                               attack_stage=attack_stage, confidence=confidence, model_sha256=model_sha256,
                               scoring_method=scoring_method, feature_vector=feature_vector,
                               calibrated_probability=calibrated_probability, tier=tier)

        if tier == "low":
            batch_block = self._buffer_low_risk(core)
            return {"tier": tier, "block": None, "batch_block": batch_block}

        if tier in ("high", "critical"):
            core.update({
                "top_features_sha256": sha256_hex(top_features or []),
                "graph_sha256": sha256_hex({"nodes": nodes or [], "edges": edges or []}),
                "host_ips": sorted({n.get("label") for n in (nodes or []) if n.get("type") == "host"}),
                "top_features": [f.get("feature") for f in (top_features or [])[:5]],
                "predicted_stages": predicted_stages,
                "remediation_actions": [a.get("action") for a in (remediation or [])][:5],
            })
        block = self.append(PREDICTION, core)
        return {"tier": tier, "block": block, "batch_block": None}

    def _buffer_low_risk(self, core: dict) -> dict | None:
        item_json = canonical_json(_finite(core))
        conn = self._connect()
        try:
            with self._lock:
                conn.execute("INSERT INTO low_risk_buffer (item_hash, data_json) VALUES (?, ?)", (batch_item_hash(core), item_json))
                n = conn.execute("SELECT COUNT(*) FROM low_risk_buffer").fetchone()[0]
            if n >= self.batch_size:
                return self._flush(conn)
            return None
        finally:
            conn.close()

    def _flush(self, conn: sqlite3.Connection) -> dict | None:
        with self._lock:
            rows = conn.execute("SELECT id, item_hash, data_json FROM low_risk_buffer ORDER BY id ASC").fetchall()
            if not rows:
                return None
            items = [json.loads(r["data_json"]) for r in rows]
            hashes = [r["item_hash"] for r in rows]
            data = {
                "count": len(items),
                "merkle_root": merkle_root(hashes),
                "item_hashes": hashes,
                "window_ids": [i["window_id"] for i in items],
                "source_files": sorted({i.get("source_file") or "" for i in items}),
                "max_infiltration_probability": max(i["infiltration_probability"] for i in items),
                "model_sha256": sorted({i.get("model_sha256") or "" for i in items}),
            }
            block = self._append(conn, PREDICTION_BATCH, data)
            conn.execute("DELETE FROM low_risk_buffer WHERE id <= ?", (rows[-1]["id"],))
            return block

    def flush_low_risk(self) -> dict | None:
        """Chain whatever low-risk windows are buffered (end of a capture)."""
        conn = self._connect()
        try:
            return self._flush(conn)
        finally:
            conn.close()

    def pending_low_risk(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM low_risk_buffer").fetchone()[0]
        finally:
            conn.close()

    def log_alert(self, *, alert_id: str, window_id: str, severity: str, attack_stage: str, tactic: str | None,
                  technique: str | None, host_ips: list[str], infiltration_probability: float, confidence: float,
                  top_features: list | None = None, model_sha256: str | None = None) -> dict:
        return self.append(ALERT, {
            "alert_id": alert_id, "window_id": str(window_id), "severity": severity, "attack_stage": attack_stage,
            "mitre_tactic": tactic, "mitre_technique": technique, "host_ips": sorted(host_ips or []),
            "infiltration_probability": round(float(infiltration_probability), 6), "confidence": round(float(confidence), 6),
            "top_features_sha256": sha256_hex(top_features or []), "model_sha256": model_sha256,
        })

    def log_analyst_action(self, *, analyst_id: str, action: str, target_id: str, target_type: str,
                           justification: str | None, feedback_id: int | None = None, extra: dict | None = None) -> dict:
        data = {"analyst_id": analyst_id, "action": action, "target_id": target_id, "target_type": target_type,
                "justification": justification or "", "feedback_id": feedback_id}
        if extra:
            data.update(extra)
        return self.append(ANALYST_ACTION, data)

    def log_training_manifest(self, manifest: dict) -> dict:
        return self.append(TRAINING_MANIFEST, manifest)

    def log_remediation(self, *, action: str, target: str, authorized_by: str, window_id: str | None = None,
                        command: str | None = None, automated: bool = False, extra: dict | None = None) -> dict:
        data = {"action": action, "target": target, "authorized_by": authorized_by,
                "window_id": window_id, "command": command, "automated": automated}
        if extra:
            data.update(extra)
        return self.append(REMEDIATION, data)

    def log_access(self, *, event: str, username: str, **extra) -> dict:
        """Login success / failure / lockout / logout (classification-tiered auth)."""
        return self.append(ACCESS, {"event": event, "username": username, **extra})

    def latest_of_type(self, block_type: str) -> dict | None:
        rows = self.get_chain(limit=1, block_type=block_type)
        return rows[0] if rows else None

    def record_model_if_changed(self, meta: dict) -> dict | None:
        """MODEL_UPDATE block whenever the served weights hash differs from
        the last one on chain (synergy doc 1.1 model provenance)."""
        sha = meta.get("sha256")
        if not sha:
            return None
        last = self.latest_of_type(MODEL_UPDATE)
        if last and last["data"].get("new_weights_sha256") == sha:
            return None
        metrics = meta.get("validation_metrics") or {}
        return self.append(MODEL_UPDATE, {
            "previous_weights_sha256": last["data"].get("new_weights_sha256") if last else None,
            "new_weights_sha256": sha,
            "checkpoint_path": meta.get("path"),
            "training_manifest_hash": meta.get("training_manifest_hash"),
            "training_metrics": metrics,
            "training_metrics_sha256": sha256_hex(metrics),
            "heads_validated": meta.get("heads_validated"),
            "temperature": meta.get("temperature"),
            "deployed_at": time.time(),
        })

    # ── multi-node / air-gapped sync ───────────────────────────────────────
    def export_delta(self, since_index: int = 0) -> dict:
        """A self-contained bundle of this node's blocks from since_index to
        tip -- write it to secure removable media and import on another node."""
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM blocks WHERE idx >= ? ORDER BY idx ASC", (since_index,)).fetchall()
            genesis = conn.execute("SELECT hash FROM blocks WHERE idx = 0").fetchone()
        finally:
            conn.close()
        blocks = [{"index": r["idx"], "timestamp": r["timestamp"], "block_type": r["block_type"], "node_id": r["node_id"],
                   "data_json": r["data_json"], "prev_hash": r["prev_hash"], "hash": r["hash"]} for r in rows]
        return {"format": "kavach-chain-bundle/1", "node_id": self.node_id, "genesis_hash": genesis["hash"] if genesis else None,
                "since_index": since_index, "tip_hash": blocks[-1]["hash"] if blocks else None,
                "exported_at": time.time(), "blocks": blocks}

    def import_bundle(self, bundle: dict) -> dict:
        """Verify a peer's bundle (hash chain + continuity with what we
        already hold for that peer), store it, and append a SYNC receipt block
        to our own chain -- the receipt is the "Base B acknowledged at 14:02"
        record from the multi-site dispute scenario."""
        peer = bundle.get("node_id")
        blocks = bundle.get("blocks") or []
        if not peer or peer == self.node_id:
            return {"accepted": False, "reason": "bundle has no node_id or is from this node"}
        if not blocks:
            return {"accepted": False, "reason": "bundle contains no blocks"}

        conn = self._connect()
        try:
            last = conn.execute("SELECT idx, hash FROM peer_blocks WHERE node_id = ? ORDER BY idx DESC LIMIT 1", (peer,)).fetchone()
            first = blocks[0]
            rows = [{"idx": b["index"], "timestamp": b["timestamp"], "block_type": b["block_type"], "origin_node_id": b["node_id"],
                     "node_id": b["node_id"], "data_json": b["data_json"], "prev_hash": b["prev_hash"], "hash": b["hash"]} for b in blocks]

            class _R(dict):
                def keys(self):  # sqlite3.Row-compatible for _verify_rows
                    return super().keys()

            rows_r = [_R(r) for r in rows]
            conflicts = []
            if first["index"] == 0:
                start_prev, start_idx = ZERO_HASH, 0
            elif last is not None and first["index"] <= last["idx"] + 1:
                prev_row = conn.execute("SELECT hash FROM peer_blocks WHERE node_id = ? AND idx = ?", (peer, first["index"] - 1)).fetchone()
                start_prev, start_idx = (prev_row["hash"] if prev_row else None), first["index"]
            else:
                return {"accepted": False, "reason": f"gap: we hold {peer} up to block {last['idx'] if last else -1}, bundle starts at {first['index']}"}

            check = self._verify_rows(rows_r, start_prev=start_prev, start_idx=start_idx)
            if check["issues"]:
                return {"accepted": False, "reason": "peer bundle failed hash-chain verification", "issues": check["issues"][:20]}

            imported = 0
            for r in rows:
                existing = conn.execute("SELECT hash FROM peer_blocks WHERE node_id = ? AND idx = ?", (peer, r["idx"])).fetchone()
                if existing:
                    if existing["hash"] != r["hash"]:
                        conflicts.append({"index": r["idx"], "held_hash": existing["hash"], "bundle_hash": r["hash"]})
                    continue
                conn.execute(
                    "INSERT INTO peer_blocks (node_id, idx, timestamp, block_type, origin_node_id, data_json, prev_hash, hash, imported_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (peer, r["idx"], r["timestamp"], r["block_type"], r["origin_node_id"], r["data_json"], r["prev_hash"], r["hash"], time.time()),
                )
                imported += 1
        finally:
            conn.close()

        receipt = self.append(SYNC, {"peer_node_id": peer, "blocks_imported": imported, "bundle_range": [blocks[0]["index"], blocks[-1]["index"]],
                                     "peer_tip_hash": blocks[-1]["hash"], "peer_genesis_hash": bundle.get("genesis_hash"),
                                     "conflicts": conflicts, "acknowledged_at": time.time()})
        return {"accepted": not conflicts, "peer_node_id": peer, "blocks_imported": imported, "conflicts": conflicts,
                "reason": "fork detected: peer rewrote blocks we already hold" if conflicts else "ok", "receipt_block": receipt["index"]}

    def peer_nodes(self) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT node_id, COUNT(*) AS n, MAX(idx) AS tip, MAX(imported_at) AS last_sync FROM peer_blocks GROUP BY node_id").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def peer_blocks(self, block_types: Iterable[str] | None = None, since: float | None = None) -> list[dict]:
        conn = self._connect()
        try:
            q, args, clauses = "SELECT * FROM peer_blocks", [], []
            if block_types:
                types = list(block_types)
                clauses.append(f"block_type IN ({','.join('?' * len(types))})")
                args += types
            if since is not None:
                clauses.append("timestamp >= ?")
                args.append(since)
            if clauses:
                q += " WHERE " + " AND ".join(clauses)
            return [{"node_id": r["node_id"], "index": r["idx"], "timestamp": r["timestamp"], "block_type": r["block_type"],
                     "data": json.loads(r["data_json"]), "hash": r["hash"]} for r in conn.execute(q, args).fetchall()]
        finally:
            conn.close()


# ── tamper simulation (safe: always on a throw-away copy) ──────────────────
def simulate_tamper(ledger: Ledger, index: int | None = None, mode: str = "modify") -> dict:
    """Copies the chain to a temp file, tampers with the copy the way an
    attacker with root would (modify a block's data, or delete a block), and
    runs verification on the copy. The real chain is never touched."""
    tmpdir = Path(tempfile.mkdtemp(prefix="kavach_tamper_"))
    try:
        copy_path = tmpdir / "chain.db"
        src = sqlite3.connect(ledger.db_path)
        dst = sqlite3.connect(copy_path)
        src.backup(dst)
        src.close()
        total = dst.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        if total < 3:
            dst.close()
            return {"simulated": False, "reason": "need at least 3 blocks to demonstrate tampering"}
        if index is None or not (0 < index < total):
            # pick the most recent alert/prediction if possible -- the
            # evidence an intruder would most want gone
            row = dst.execute("SELECT idx FROM blocks WHERE block_type IN ('ALERT','PREDICTION') AND idx < ? ORDER BY idx DESC LIMIT 1", (total - 1,)).fetchone()
            index = row[0] if row else total // 2
        before = dst.execute("SELECT data_json FROM blocks WHERE idx = ?", (index,)).fetchone()[0]
        if mode == "delete":
            dst.execute("DELETE FROM blocks WHERE idx = ?", (index,))
            action = f"deleted block {index} (as an intruder erasing the record of their detection)"
        else:
            data = json.loads(before)
            if "infiltration_probability" in data:
                data["infiltration_probability"] = 0.05
                data["attack_stage"] = "Benign"
            else:
                data["tampered"] = True
            dst.execute("UPDATE blocks SET data_json = ? WHERE idx = ?", (canonical_json(data), index))
            action = f"rewrote block {index} (e.g. '95% infiltration' -> '5% Benign')"
        dst.commit()
        dst.close()
        tampered = Ledger(copy_path, node_id=ledger.node_id)
        result = tampered.verify()
        return {"simulated": True, "mode": mode, "tampered_index": index, "action": action, "verification": result,
                "real_chain_untouched": True}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


_ledgers: dict[str, Ledger] = {}


def get_ledger() -> Ledger:
    """Process-wide ledger for the current DB_PATH (tests can repoint
    DB_PATH and get a fresh ledger)."""
    key = str(DB_PATH.resolve())
    if key not in _ledgers:
        _ledgers[key] = Ledger(DB_PATH)
    return _ledgers[key]
