"""Adapts src.storage.results_store rows into the shapes the frontend's
TypeScript types expect (frontend/src/api/types.ts) -- Flow, Alert,
KillChainStage, RiskScoreResponse, AuditTrail, TimelinePoint.

Read before trusting every field: a stored row is one WINDOW (an aggregate
over 100 flow records or a 5-second packet slice -- see
src/features/extract.py), not a literal single network flow, and
MachineLearningCVE CSV windows carry no real src/dst IP, port, or protocol.
Fields we have real data for are real; fields we don't get an honest
placeholder (0, "TCP") rather than a fabricated realistic-looking number.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from src.scoring.attack_stage import STAGE_META, STAGE_NAMES

_STAGE_TACTIC = {m["stage"]: m["tactic"] for m in STAGE_META}
_STAGE_TECHNIQUE = {m["stage"]: m["technique"] for m in STAGE_META}
TREND_THRESHOLD = 5  # risk points


def _finite(value) -> float:
    """NaN/inf aren't valid JSON -- clamp to 0.0 at the API boundary."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _risk_level(score: int) -> str:
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 30:
        return "medium"
    return "low"


def _severity(score: int) -> str:
    if score >= 70:
        return "critical"
    if score >= 40:
        return "warning"
    return "info"


def _iso(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def verdict_to_frontend(verdict: str | None) -> str | None:
    if verdict is None:
        return None
    return {"false_positive": "false-positive"}.get(verdict, verdict)


def _feedback_fields(fb: dict | None) -> dict:
    if not fb:
        return {}
    return {"analystVerdict": verdict_to_frontend(fb["verdict"]), "verdictNote": fb.get("note") or None,
            "verdictAt": _iso(fb["created_at"]), "verdictBy": fb.get("analyst_id"), "verdictBlockIndex": fb.get("block_index")}


def _host_ips(row: dict) -> list[str]:
    return [n["label"] for n in row.get("nodes", []) if n.get("type") == "host"]


def row_to_flow(row: dict, feedback: dict | None = None) -> dict:
    hosts = _host_ips(row)
    src_ip = hosts[0] if hosts else "N/A"
    dst_ip = hosts[1] if len(hosts) > 1 else src_ip
    risk_score = round(_finite(row["infiltration_probability"]) * 100)
    features = row.get("top_features", []) or []
    max_abs = max((abs(_finite(f.get("attribution"))) for f in features), default=0.0) or 1.0
    out = {
        "id": row["window_id"],
        "srcIP": src_ip,
        "dstIP": dst_ip,
        "srcPort": 0,  # not available -- see module docstring
        "dstPort": 0,
        "protocol": "TCP",
        "flags": "",
        "bytes": 0,  # window values are z-scored; a normalized "bytes" would read as nonsense
        "duration": 0,
        "iatMean": 0,
        "iatStd": 0,
        "riskScore": risk_score,
        "riskLevel": _risk_level(risk_score),
        "timestamp": _iso(row["created_at"]),
        "features": [
            {"name": f["feature"], "value": _finite(f["attribution"]), "contribution": abs(_finite(f["attribution"])) / max_abs,
             "method": f.get("method")}
            for f in features
        ],
        # extended, real fields (optional in the TS type)
        "attackStage": row.get("attack_stage"),
        "confidence": _finite(row.get("confidence")),
        "rawProbability": None if row.get("raw_probability") is None else _finite(row["raw_probability"]),
        "calibratedProbability": None if row.get("calibrated_probability") is None else _finite(row["calibrated_probability"]),
        "scoringMethod": row.get("scoring_method"),
        "attributionMethod": row.get("attribution_method"),
        "sourceFile": row.get("source_file"),
        "hostIPs": hosts,
        "flowCount": row.get("flow_count"),
        "packetCount": row.get("packet_count"),
        "infiltrationCurve": [_finite(p) for p in (row.get("infiltration_curve") or [])],
        "predictedStages": row.get("predicted_stages") or [],
        "stageDistribution": row.get("stage_distribution"),
        "modelSha256": row.get("model_sha256"),
        "blockIndex": row.get("block_index"),
        "chainTier": row.get("chain_tier"),
        "groundTruthLabel": row.get("label"),
        "concepts": row.get("concepts"),
        "crossNode": row.get("cross_node"),
        "forecast": row.get("forecast"),
    }
    out.update(_feedback_fields(feedback))
    return out


def row_to_alert(row: dict, feedback: dict | None = None) -> dict:
    risk_score = round(_finite(row["infiltration_probability"]) * 100)
    stage = row["attack_stage"]
    method = row.get("scoring_method")
    out = {
        "id": f"alert-{row['window_id']}",
        "severity": _severity(risk_score),
        "title": f"{stage} activity detected" if stage != "Benign" else "Elevated infiltration probability",
        "description": (
            f"Window {row['window_id']} from {row.get('source_file') or 'unknown source'} scored "
            f"{_finite(row['infiltration_probability']):.0%} infiltration probability"
            + (" (calibrated)" if row.get("calibrated_probability") is not None else "")
            + f", mapped to MITRE stage '{stage}' ({_STAGE_TACTIC.get(stage, 'n/a')}) at {_finite(row['confidence']):.0%} confidence"
            + (" [packet heuristic]" if method == "packet_heuristic" else "") + "."
        ),
        "timestamp": _iso(row["created_at"]),
        "flowId": row["window_id"],
        "killChainStage": stage,
        "mitreTactic": _STAGE_TACTIC.get(stage),
        "mitreTechnique": _STAGE_TECHNIQUE.get(stage),
        "hostIPs": _host_ips(row),
        "riskScore": risk_score,
        "scoringMethod": method,
        "blockIndex": row.get("block_index"),
        "modelSha256": row.get("model_sha256"),
        "topFeatures": [
            {
                "featureName": f["feature"],
                "value": _finite(f.get("value", f["attribution"])),
                "contribution": _finite(f["attribution"]),
                "direction": "positive" if _finite(f["attribution"]) >= 0 else "negative",
            }
            for f in row.get("top_features", []) or []
        ],
    }
    fb = feedback
    out.update(_feedback_fields(fb))
    return out


def kill_chain_state(stats: dict) -> list[dict]:
    """One entry per MITRE stage. probability = share of stored windows in
    that stage; active = the latest window's stage; predicted = stages the
    latest window's NetJEPA rollout forecasts (dashed in the UI)."""
    latest = stats.get("latest")
    latest_stage = latest["attack_stage"] if latest else None
    predicted = {p["stage"]: p for p in ((latest or {}).get("predicted_stages") or []) if p["stage"] != latest_stage}
    stage_counts = stats.get("stage_counts", {})
    total = sum(stage_counts.values()) or 1
    order = list(STAGE_NAMES)
    active_idx = order.index(latest_stage) if latest_stage in order else -1
    result = []
    for i, name in enumerate(order):
        count = stage_counts.get(name, 0)
        is_active = name == latest_stage
        entry = {
            "name": name,
            "probability": round(count / total, 3),
            "isActive": is_active,
            "isPredicted": name in predicted and not is_active,
            "isComplete": (not is_active) and count > 0 and (active_idx < 0 or i < active_idx),
            "observedCount": count,
            "tactic": _STAGE_TACTIC.get(name),
            "technique": _STAGE_TECHNIQUE.get(name),
        }
        if name in predicted:
            entry["predictedProbability"] = round(float(predicted[name]["probability"]), 3)
            entry["predictedStepsAhead"] = predicted[name]["step"]
        result.append(entry)
    return result


def risk_score_response(stats: dict, previous: dict | None = None) -> dict:
    """previous = the second-most-recent prediction (Task 2.6 trend/delta)."""
    latest = stats.get("latest")
    if latest is None:
        return {"score": 0, "trend": "stable", "delta": 0, "activeStage": "none",
                "explanation": "No predictions recorded yet -- upload a capture on the Ingestion page."}

    score = round(_finite(latest["infiltration_probability"]) * 100)
    delta = 0
    if previous is not None:
        delta = score - round(_finite(previous["infiltration_probability"]) * 100)
    trend = "up" if delta > TREND_THRESHOLD else "down" if delta < -TREND_THRESHOLD else "stable"
    method = latest.get("scoring_method")
    explanation = (
        f"Most recent window ({latest['window_id']}) scored {_finite(latest['infiltration_probability']):.0%} infiltration probability"
        + (f" (raw model output {_finite(latest['raw_probability']):.0%}, temperature-calibrated)" if latest.get("calibrated_probability") is not None and latest.get("raw_probability") is not None else "")
        + f", mapped to '{latest['attack_stage']}' at {_finite(latest['confidence']):.0%} confidence."
        + (f" {'Up' if delta > 0 else 'Down'} {abs(delta)} points from the previous window." if delta else "")
        + (" Scored by the packet heuristic (packet-only capture)." if method == "packet_heuristic" else "")
        + ("" if latest["trained"] else " Model is untrained -- this number is not meaningful yet.")
    )
    return {"score": score, "trend": trend, "delta": delta, "activeStage": latest["attack_stage"], "explanation": explanation,
            "confidence": _finite(latest.get("confidence")), "rawProbability": latest.get("raw_probability"),
            "calibratedProbability": latest.get("calibrated_probability"), "scoringMethod": method,
            "windowId": latest["window_id"], "forecast": [_finite(p) for p in (latest.get("infiltration_curve") or [])]}


def timeline_points(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        score = round(_finite(r["infiltration_probability"]) * 100)
        out.append({"timestamp": _iso(r["created_at"]), "riskScore": score, "windowId": r["window_id"], "attackStage": r["attack_stage"],
                    "sourceFile": r.get("source_file"), "confidence": {"low": score, "high": score},
                    "rawScore": None if r.get("raw_probability") is None else round(_finite(r["raw_probability"]) * 100),
                    "scoringMethod": r.get("scoring_method")})
    return out


def audit_trail(row: dict, blocks: list[dict], meta: dict, feedback: list[dict]) -> dict:
    """Real 5-stage decision chain for one window (Task 2.4), plus the
    ledger evidence and analyst actions for it."""
    base = row["created_at"]
    window = row.get("window") or {}
    hosts = _host_ips(row)
    edges = row.get("edges") or []
    flow_edges = [e for e in edges if e.get("type") == "flow"]
    steps = [
        {"stageName": "Raw Flow",
         "summary": f"Ingested {row.get('flow_count') or 0} flow records and {row.get('packet_count') or 0} packets from {row.get('source_file') or 'an unknown source'}.",
         "detail": {"sourceFile": row.get("source_file"), "flowCount": row.get("flow_count"), "packetCount": row.get("packet_count"),
                    "windowStart": window.get("window_start"), "windowEnd": window.get("window_end"), "groundTruthLabel": row.get("label")}},
        {"stageName": "Feature Extraction",
         "summary": "Extracted the 23 canonical flow/packet features for this window (z-scored across the capture).",
         "detail": {"canonicalFeatures": 23, "window": {k: v for k, v in window.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}}},
        {"stageName": "Graph State",
         "summary": f"Built a heterogeneous graph with {len(row.get('nodes') or [])} nodes ({len(hosts)} hosts) and {len(edges)} edges ({len(flow_edges)} host-to-host flows).",
         "detail": {"nodes": len(row.get("nodes") or []), "hosts": hosts[:20], "edges": len(edges), "hostFlows": len(flow_edges)}},
        {"stageName": "Forecast",
         "summary": (f"NetJEPA encoded the window into a {meta.get('embedding_dim', 128)}-d latent state and rolled it out {len(row.get('infiltration_curve') or []) or 5} steps."
                     + ("" if row.get("trained") else " (untrained model)")),
         "detail": {"embeddingDim": meta.get("embedding_dim", 128), "rolloutSteps": len(row.get("infiltration_curve") or []),
                    "forecastCurve": row.get("infiltration_curve"), "predictedStages": row.get("predicted_stages"),
                    "trained": row.get("trained"), "modelSha256": row.get("model_sha256")}},
        {"stageName": "Attack Stage Mapping",
         "summary": (f"Scored {_finite(row['infiltration_probability']):.0%} infiltration probability and mapped to {row['attack_stage']} "
                     f"({_STAGE_TACTIC.get(row['attack_stage'], 'n/a')}) at {_finite(row['confidence']):.0%} confidence via "
                     + ("the calibrated NetJEPA heads." if row.get("scoring_method") == "netjepa" else "the packet heuristic.")),
         "detail": {"infiltrationProbability": row["infiltration_probability"], "rawProbability": row.get("raw_probability"),
                    "calibratedProbability": row.get("calibrated_probability"), "attackStage": row["attack_stage"],
                    "confidence": row["confidence"], "stageDistribution": row.get("stage_distribution"), "scoringMethod": row.get("scoring_method"),
                    "topFeatures": [{"name": f["feature"], "value": _finite(f.get("value", 0)), "contribution": _finite(f["attribution"])}
                                    for f in (row.get("top_features") or [])[:8]],
                    "attributionMethod": row.get("attribution_method")}},
    ]
    alert_blocks = [b for b in blocks if b["block_type"] == "ALERT"]
    pred_blocks = [b for b in blocks if b["block_type"] in ("PREDICTION", "PREDICTION_BATCH")]
    if alert_blocks or row["infiltration_probability"] >= 0.5:
        steps.append({"stageName": "Alert",
                      "summary": (f"Alert chained on the forensic ledger at block #{alert_blocks[0]['index']}." if alert_blocks
                                  else "Crossed the alert threshold (not chained: below the high-risk tier)."),
                      "detail": {"alertBlock": alert_blocks[0]["index"] if alert_blocks else None,
                                 "alertHash": alert_blocks[0]["hash"] if alert_blocks else None}})
    steps.append({"stageName": "Blockchain",
                  "summary": (f"Evidence chained: {', '.join(b['type_label'] + ' #' + str(b['index']) for b in blocks[:6])}." if blocks
                              else "Low-risk window: awaiting its batched Merkle block (or recorded before the ledger existed)."),
                  "detail": {"blocks": [{"index": b["index"], "type": b["block_type"], "hash": b["hash"], "prevHash": b["prev_hash"],
                                         "timestamp": _iso(b["timestamp"])} for b in blocks],
                             "chainTier": row.get("chain_tier"), "predictionBlocks": [b["index"] for b in pred_blocks]}})
    for fb in feedback:
        steps.append({"stageName": "Analyst Action",
                      "summary": f"{fb['analyst_id']} marked this {fb['target_type']} as {verdict_to_frontend(fb['verdict'])}"
                                 + (f": \"{fb['note']}\"" if fb.get("note") else "") + (f" (block #{fb['block_index']})" if fb.get("block_index") is not None else ""),
                      "detail": fb, "_ts": fb["created_at"]})
    for i, s in enumerate(steps):
        ts = s.pop("_ts", None)
        s["stepIndex"] = i
        s["timestamp"] = _iso(ts if ts is not None else base + i * 0.001)
    return {"flowId": row["window_id"], "alertId": f"alert-{row['window_id']}", "steps": steps}
