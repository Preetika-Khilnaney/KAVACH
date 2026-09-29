"""Results store (architecture doc): persists predictions, attack-stage
mappings, and explainability outputs so the frontend has something to
query beyond a single live event stream that fires once and disappears --
backs the Operations dashboard, Flows list, and Alerts views.

SQLite on local disk, per the architecture doc's own suggestion ("local
filesystem + SQLite/DuckDB is sufficient given the fully offline
requirement") -- stdlib only, no new dependency, persists across server
restarts (unlike the in-memory JOBS dict in src/api/main.py).

One row per feature window processed by the pipeline (src/orchestrator/pipeline.py
calls save_prediction after computing attack_mapping + explainability for
each window) -- not one row per raw flow. A "window" here is what the
frontend's Flow list ends up displaying; see the API layer for how the
field names get adapted (there's no real per-flow src/dst IP for
CSV-derived windows -- same caveat as everywhere else in this project).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path("data/processed/results.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_id TEXT NOT NULL,
    source_file TEXT,
    created_at REAL NOT NULL,
    flow_count INTEGER,
    packet_count INTEGER,
    infiltration_probability REAL,
    attack_stage TEXT,
    confidence REAL,
    trained INTEGER,
    label TEXT,
    top_features_json TEXT,
    nodes_json TEXT,
    edges_json TEXT,
    window_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_infiltration ON predictions(infiltration_probability);
CREATE INDEX IF NOT EXISTS idx_predictions_window ON predictions(window_id);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    verdict TEXT NOT NULL,
    note TEXT,
    analyst_id TEXT NOT NULL DEFAULT 'analyst@kavach.demo',
    created_at REAL NOT NULL,
    window_id TEXT,
    prediction_probability REAL,
    prediction_confidence REAL,
    attack_stage TEXT,
    model_sha256 TEXT,
    host_ips_json TEXT,
    block_index INTEGER
);
CREATE INDEX IF NOT EXISTS idx_feedback_target ON feedback(target_id);
"""

# Columns added after the first release -- applied with ALTER TABLE on
# existing databases (see _migrate), so a results.db created by an older
# version keeps working.
_EXTRA_PREDICTION_COLUMNS = {
    "raw_probability": "REAL",
    "calibrated_probability": "REAL",
    "scoring_method": "TEXT",
    "infiltration_curve_json": "TEXT",
    "stage_distribution_json": "TEXT",
    "predicted_stages_json": "TEXT",
    "model_sha256": "TEXT",
    "attribution_method": "TEXT",
    "block_index": "INTEGER",
    "chain_tier": "TEXT",
    "cross_node_json": "TEXT",
    "concepts_json": "TEXT",
    "forecast_json": "TEXT",
}

VERDICTS = ("confirmed", "false_positive", "needs_review", "escalated")


def _json_default(obj):
    """Window dicts can carry numpy scalar types (from pandas aggregation
    in src/features/extract.py) that json.dumps doesn't know how to
    serialize natively -- coerce anything with a numpy-style .item() to a
    plain Python scalar, and tuples (e.g. packet_edges) to lists."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _dumps(value) -> str:
    return json.dumps(value, default=_json_default)


_migrated: set[str] = set()


def _migrate(conn: sqlite3.Connection) -> None:
    key = str(DB_PATH.resolve())
    if key in _migrated:
        return
    have = {r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    for col, typ in _EXTRA_PREDICTION_COLUMNS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")
    conn.commit()
    _migrated.add(key)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    # WAL (roadmap Task 1.3): the pipeline writes predictions while the
    # dashboard reads them concurrently; the default rollback journal makes
    # readers and the writer block each other ("database is locked").
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def save_prediction(
    window_id: str,
    source_file: str | None,
    window: dict,
    state: dict,
    infiltration_probability: float,
    attack_stage: str,
    confidence: float,
    trained: bool,
    top_features: list,
    extra: dict | None = None,
) -> int:
    """Persist one window's prediction. `extra` carries the optional
    newer columns (raw/calibrated probability, rollout curve, predicted
    stages, model hash, ledger block index...) -- see _EXTRA_PREDICTION_COLUMNS.
    Returns the new row id."""
    extra = extra or {}
    extra_cols, extra_vals = [], []
    for key, value in extra.items():
        col = key if key in _EXTRA_PREDICTION_COLUMNS else f"{key}_json"
        if col not in _EXTRA_PREDICTION_COLUMNS:
            continue
        extra_cols.append(col)
        extra_vals.append(_dumps(value) if col.endswith("_json") else value)
    conn = _connect()
    try:
        cols = ["window_id", "source_file", "created_at", "flow_count", "packet_count", "infiltration_probability",
                "attack_stage", "confidence", "trained", "label", "top_features_json", "nodes_json", "edges_json", "window_json", *extra_cols]
        cur = conn.execute(
            f"INSERT INTO predictions ({', '.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            (
                window_id,
                source_file,
                time.time(),
                window.get("flow_count"),
                window.get("packet_count"),
                infiltration_probability,
                attack_stage,
                confidence,
                int(trained),
                window.get("flow_dominant_label"),
                _dumps(top_features),
                _dumps(state.get("nodes", [])),
                _dumps(state.get("edges", [])),
                _dumps(window),
                *extra_vals,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_prediction(row_id: int, **fields) -> None:
    """Set optional columns after the fact (e.g. the ledger block index,
    which is only known once the block has been appended)."""
    cols, vals = [], []
    for key, value in fields.items():
        col = key if key in _EXTRA_PREDICTION_COLUMNS else f"{key}_json"
        if col in _EXTRA_PREDICTION_COLUMNS:
            cols.append(f"{col} = ?")
            vals.append(_dumps(value) if col.endswith("_json") else value)
    if not cols:
        return
    conn = _connect()
    try:
        conn.execute(f"UPDATE predictions SET {', '.join(cols)} WHERE id = ?", (*vals, row_id))
        conn.commit()
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["top_features"] = json.loads(d.pop("top_features_json") or "[]")
    d["nodes"] = json.loads(d.pop("nodes_json") or "[]")
    d["edges"] = json.loads(d.pop("edges_json") or "[]")
    d["window"] = json.loads(d.pop("window_json") or "{}")
    d["trained"] = bool(d["trained"])
    for col in [c for c in list(d) if c.endswith("_json")]:
        raw = d.pop(col)
        d[col[:-5]] = json.loads(raw) if raw else None
    return d


def recent_predictions(limit: int = 100) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_prediction(window_id: str) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM predictions WHERE window_id = ? ORDER BY created_at DESC LIMIT 1", (window_id,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def alerts(min_probability: float = 0.5, limit: int = 100) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE infiltration_probability >= ? ORDER BY created_at DESC LIMIT ?",
            (min_probability, limit),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def summary_stats() -> dict:
    """Aggregate stats for the Operations dashboard: latest prediction
    (current risk score / active stage), per-stage counts (kill chain
    tracker), total predictions stored."""
    conn = _connect()
    try:
        total = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
        latest_row = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 1").fetchone()
        stage_rows = conn.execute("SELECT attack_stage, COUNT(*) AS n FROM predictions GROUP BY attack_stage").fetchall()
        return {
            "total_predictions": total,
            "latest": _row_to_dict(latest_row) if latest_row else None,
            "stage_counts": {r["attack_stage"]: r["n"] for r in stage_rows},
        }
    finally:
        conn.close()


RISK_BUCKETS = {
    # same cut-points as src/api/adapters._risk_level (score = prob * 100)
    "low": (0.0, 0.30),
    "medium": (0.30, 0.60),
    "high": (0.60, 0.80),
    "critical": (0.80, 1.01),
}
_SORTABLE = {"created_at", "infiltration_probability", "confidence", "attack_stage", "window_id", "source_file", "flow_count", "packet_count"}


def query_predictions(
    limit: int = 100,
    offset: int = 0,
    risk_level: list[str] | str | None = None,
    min_risk: float | None = None,
    max_risk: float | None = None,
    search: str | None = None,
    attack_stage: str | None = None,
    source_file: str | None = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
) -> list[dict]:
    """Server-side filtering for GET /flows (roadmap Task 2.5). Every
    filter is a bound parameter; sort_by is whitelisted."""
    clauses, args = [], []
    if risk_level:
        levels = [risk_level] if isinstance(risk_level, str) else list(risk_level)
        levels = [l for lvl in levels for l in lvl.split(",") if l in RISK_BUCKETS]
        if levels:
            clauses.append("(" + " OR ".join("(infiltration_probability >= ? AND infiltration_probability < ?)" for _ in levels) + ")")
            for lvl in levels:
                args += list(RISK_BUCKETS[lvl])
    if min_risk is not None:
        clauses.append("infiltration_probability >= ?")
        args.append(min_risk / 100 if min_risk > 1 else min_risk)
    if max_risk is not None:
        clauses.append("infiltration_probability <= ?")
        args.append(max_risk / 100 if max_risk > 1 else max_risk)
    if search:
        clauses.append("(source_file LIKE ? OR window_id LIKE ? OR nodes_json LIKE ? OR attack_stage LIKE ?)")
        like = f"%{search}%"
        args += [like, like, like, like]
    if attack_stage:
        clauses.append("attack_stage = ?")
        args.append(attack_stage)
    if source_file:
        clauses.append("source_file = ?")
        args.append(source_file)
    col = sort_by if sort_by in _SORTABLE else "created_at"
    direction = "ASC" if str(sort_dir).lower() == "asc" else "DESC"
    q = "SELECT * FROM predictions"
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += f" ORDER BY {col} {direction}, id {direction} LIMIT ? OFFSET ?"
    conn = _connect()
    try:
        return [_row_to_dict(r) for r in conn.execute(q, (*args, limit, offset)).fetchall()]
    finally:
        conn.close()


def latest_predictions(n: int = 2) -> list[dict]:
    """Newest first -- GET /risk-score compares the two most recent."""
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC, id DESC LIMIT ?", (n,)).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def timeline(limit: int = 200, source_file: str | None = None) -> list[dict]:
    """Oldest-first slice of the most recent `limit` predictions (lightweight
    columns only) for the infiltration timeline chart."""
    conn = _connect()
    try:
        q = ("SELECT id, window_id, source_file, created_at, infiltration_probability, raw_probability, calibrated_probability, "
             "attack_stage, confidence, scoring_method FROM predictions")
        args: list = []
        if source_file:
            q += " WHERE source_file = ?"
            args.append(source_file)
        q += " ORDER BY created_at DESC, id DESC LIMIT ?"
        args.append(limit)
        rows = [dict(r) for r in conn.execute(q, args).fetchall()]
        return list(reversed(rows))
    finally:
        conn.close()


def predictions_for_source(source_file: str | None, limit: int = 5000) -> list[dict]:
    """Every stored window of one capture, in processing order."""
    conn = _connect()
    try:
        if source_file is None:
            rows = conn.execute("SELECT * FROM predictions WHERE source_file IS NULL ORDER BY created_at ASC, id ASC LIMIT ?", (limit,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM predictions WHERE source_file = ? ORDER BY created_at ASC, id ASC LIMIT ?", (source_file, limit)).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def neighbors(window_id: str, before: int = 5, after: int = 5) -> list[dict]:
    """Predictions around one window from the same capture (for the flow
    detail risk sparkline)."""
    target = get_prediction(window_id)
    if target is None:
        return []
    conn = _connect()
    try:
        prev_rows = conn.execute(
            "SELECT id, window_id, created_at, infiltration_probability, attack_stage FROM predictions "
            "WHERE source_file IS ? AND id < ? ORDER BY id DESC LIMIT ?", (target["source_file"], target["id"], before)).fetchall()
        next_rows = conn.execute(
            "SELECT id, window_id, created_at, infiltration_probability, attack_stage FROM predictions "
            "WHERE source_file IS ? AND id > ? ORDER BY id ASC LIMIT ?", (target["source_file"], target["id"], after)).fetchall()
        return [dict(r) for r in reversed(prev_rows)] + [
            {k: target[k] for k in ("id", "window_id", "created_at", "infiltration_probability", "attack_stage")}
        ] + [dict(r) for r in next_rows]
    finally:
        conn.close()


def previous_rows(row: dict, n: int) -> list[dict]:
    """The n windows processed immediately before `row` in the same capture
    run (oldest first) -- the temporal context the model saw."""
    if n <= 0:
        return []
    prefix = row["window_id"].rsplit("-", 1)[0] + "-" if row["window_id"].count("-") >= 2 else None
    conn = _connect()
    try:
        if prefix:
            rows = conn.execute("SELECT * FROM predictions WHERE id < ? AND window_id LIKE ? ORDER BY id DESC LIMIT ?",
                                (row["id"], prefix + "%", n)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM predictions WHERE id < ? AND source_file IS ? ORDER BY id DESC LIMIT ?",
                                (row["id"], row.get("source_file"), n)).fetchall()
        return [_row_to_dict(r) for r in reversed(rows)]
    finally:
        conn.close()


def source_files() -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT source_file, COUNT(*) AS windows, MIN(created_at) AS first_at, MAX(created_at) AS last_at, "
            "MAX(infiltration_probability) AS max_probability FROM predictions GROUP BY source_file ORDER BY last_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def replay_predictions(source_file: str | None):
    """Generator for replay mode (roadmap Task 8.3): yields
    (prediction, seconds_since_previous) in original processing order."""
    prev_ts = None
    for row in predictions_for_source(source_file):
        gap = 0.0 if prev_ts is None else max(0.0, row["created_at"] - prev_ts)
        prev_ts = row["created_at"]
        yield row, gap


# ── Analyst feedback (roadmap Task 2.3) ──────────────────────────────────────

def save_feedback(
    target_id: str,
    target_type: str,
    verdict: str,
    note: str | None = None,
    analyst_id: str = "analyst@kavach.demo",
    context: dict | None = None,
) -> dict:
    verdict = verdict.replace("-", "_")
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    context = context or {}
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO feedback (target_id, target_type, verdict, note, analyst_id, created_at, window_id, prediction_probability, "
            "prediction_confidence, attack_stage, model_sha256, host_ips_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (target_id, target_type, verdict, note, analyst_id, time.time(), context.get("window_id"), context.get("infiltration_probability"),
             context.get("confidence"), context.get("attack_stage"), context.get("model_sha256"), _dumps(context.get("host_ips") or [])),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM feedback WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _feedback_row(row)
    finally:
        conn.close()


def set_feedback_block(feedback_id: int, block_index: int) -> None:
    conn = _connect()
    try:
        conn.execute("UPDATE feedback SET block_index = ? WHERE id = ?", (block_index, feedback_id))
        conn.commit()
    finally:
        conn.close()


def _feedback_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["host_ips"] = json.loads(d.pop("host_ips_json") or "[]")
    return d


def get_feedback(target_id: str) -> list[dict]:
    """All verdicts for one target, newest first (the history is kept --
    the latest entry is the current verdict)."""
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM feedback WHERE target_id = ? ORDER BY created_at DESC, id DESC", (target_id,)).fetchall()
        return [_feedback_row(r) for r in rows]
    finally:
        conn.close()


def all_feedback(since: float | None = None, limit: int = 100000) -> list[dict]:
    conn = _connect()
    try:
        if since is None:
            rows = conn.execute("SELECT * FROM feedback ORDER BY created_at ASC, id ASC LIMIT ?", (limit,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM feedback WHERE created_at >= ? ORDER BY created_at ASC, id ASC LIMIT ?", (since, limit)).fetchall()
        return [_feedback_row(r) for r in rows]
    finally:
        conn.close()


def latest_verdicts() -> dict[str, dict]:
    """target_id -> most recent feedback row."""
    out: dict[str, dict] = {}
    for fb in all_feedback():
        out[fb["target_id"]] = fb
    return out


def feedback_stats(min_probability: float = 0.5) -> dict:
    latest = latest_verdicts()
    counts = {v: 0 for v in VERDICTS}
    for fb in latest.values():
        counts[fb["verdict"]] = counts.get(fb["verdict"], 0) + 1
    conn = _connect()
    try:
        alert_windows = {r[0] for r in conn.execute("SELECT window_id FROM predictions WHERE infiltration_probability >= ?", (min_probability,)).fetchall()}
    finally:
        conn.close()
    reviewed_windows = {t[len("alert-"):] if t.startswith("alert-") else t for t in latest}
    n_alerts = len(alert_windows)
    reviewed = len(alert_windows & reviewed_windows)
    return {
        "confirmed": counts.get("confirmed", 0),
        "false_positive": counts.get("false_positive", 0),
        "needs_review": counts.get("needs_review", 0),
        "escalated": counts.get("escalated", 0),
        "total_feedback": len(latest),
        "total_alerts": n_alerts,
        "pending": max(0, n_alerts - reviewed),
    }


def clear() -> None:
    """Wipes all stored predictions and feedback. Mainly for tests."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM predictions")
        conn.execute("DELETE FROM feedback")
        conn.commit()
    finally:
        conn.close()
