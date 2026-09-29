"""Smart contracts: pre-agreed rules that fire on AI predictions plus
blockchain consensus (deep-dive doc "Smart Contracts", synergy doc 1.5/2.2).

Contracts are deterministic functions over the verified chain (local blocks
+ hash-verified peer blocks). Each execution is itself chained as a
CONTRACT_EXECUTION / ESCALATION block, and each rule fires at most once per
subject, so re-evaluating is idempotent.

1. consensus_ip_block -- if >= N independent nodes classify the same IP as
   Command and Control with calibrated probability >= 0.85 and confidence
   >= 0.8 inside 24h, emit a firewall rule for every node. A single
   compromised node can neither trigger it alone nor suppress it.
2. auto_escalation -- Reconnaissance followed by Initial Access within 10
   minutes, with no analyst acknowledgement within 5 minutes, escalates to
   CRITICAL and notifies the next command level.
3. model_rollback -- after a model update, once enough of the new model's
   alerts have analyst verdicts, if its false-positive rate exceeds 2x the
   previous model's, pin the previous weights (identified by the hash on the
   MODEL_UPDATE block) and chain the rollback.

Actions are *recorded*, not executed against real firewalls: the generated
rule is the artefact an operator (or an enforcement agent) applies.
"""

from __future__ import annotations

import time

from src.config import load_config
from src.orchestrator import model_registry
from src.storage import results_store
from src.storage.blockchain_ledger import (ALERT, ANALYST_ACTION, CONTRACT_EXECUTION, ESCALATION, MODEL_UPDATE, PREDICTION,
                                           get_ledger)

CONTRACTS = [
    {"id": "consensus_ip_block", "name": "Consensus-based IP block",
     "rule": "IF >= {consensus_min_nodes} independent nodes classify the same IP as Command and Control with probability >= {consensus_min_probability} "
             "and confidence >= {consensus_min_confidence} within {consensus_window_hours}h THEN generate + distribute a firewall block rule, chain it, alert all SOCs."},
    {"id": "auto_escalation", "name": "Kill-chain progression escalation",
     "rule": "IF Reconnaissance -> Initial Access within {escalation_progression_minutes} min AND no analyst acknowledgement within "
             "{escalation_ack_minutes} min THEN escalate to CRITICAL, broadcast to next command level, chain it."},
    {"id": "model_rollback", "name": "Model regression rollback",
     "rule": "IF after a model update the false-positive rate on its next {rollback_eval_predictions} reviewed alerts exceeds "
             "{rollback_fpr_multiplier}x the previous model's THEN revert to the previous weights hash and chain the rollback."},
]


def _cfg() -> dict:
    defaults = {"consensus_min_nodes": 3, "consensus_min_confidence": 0.8, "consensus_min_probability": 0.85, "consensus_window_hours": 24,
                "escalation_progression_minutes": 10, "escalation_ack_minutes": 5, "rollback_eval_predictions": 100, "rollback_fpr_multiplier": 2.0}
    defaults.update(((load_config().get("blockchain", {}) or {}).get("contracts", {}) or {}))
    return defaults


def definitions() -> list[dict]:
    cfg = _cfg()
    return [{**c, "rule": c["rule"].format(**cfg)} for c in CONTRACTS]


def _executions(rule: str) -> list[dict]:
    return [b for b in get_ledger().iter_blocks([CONTRACT_EXECUTION, ESCALATION]) if b["data"].get("contract") == rule]


# ── 1. consensus IP block ──────────────────────────────────────────────────
def evaluate_consensus(now: float) -> list[dict]:
    cfg = _cfg()
    ledger = get_ledger()
    since = now - cfg["consensus_window_hours"] * 3600
    votes: dict[str, dict[str, dict]] = {}

    def consider(node: str, block: dict):
        d = block["data"]
        if d.get("attack_stage") != "Command and Control":
            return
        if (d.get("confidence") or 0) < cfg["consensus_min_confidence"] or (d.get("infiltration_probability") or 0) < cfg["consensus_min_probability"]:
            return
        for ip in d.get("host_ips") or []:
            votes.setdefault(ip, {})
            prev = votes[ip].get(node)
            if prev is None or block["timestamp"] > prev["timestamp"]:
                votes[ip][node] = {"node_id": node, "block_index": block["index"], "block_hash": block["hash"], "timestamp": block["timestamp"],
                                   "probability": d.get("infiltration_probability"), "confidence": d.get("confidence")}

    for b in ledger.iter_blocks([ALERT, PREDICTION], since=since):
        consider(ledger.node_id, b)
    for b in ledger.peer_blocks([ALERT, PREDICTION], since=since):
        consider(b["node_id"], b)

    already = {b["data"].get("subject") for b in _executions("consensus_ip_block") if b["timestamp"] >= since}
    fired = []
    for ip, by_node in votes.items():
        if len(by_node) < cfg["consensus_min_nodes"] or ip in already:
            continue
        block = ledger.append(CONTRACT_EXECUTION, {
            "contract": "consensus_ip_block", "subject": ip, "action": "block_ip",
            "firewall_rule": f"iptables -I FORWARD -s {ip} -j DROP && iptables -I FORWARD -d {ip} -j DROP",
            "agreeing_nodes": sorted(by_node), "evidence": list(by_node.values()), "distributed_to": "all nodes (via ledger sync)",
            "notified": "SOC analysts at every agreeing site"})
        fired.append(block)
    return fired


def consensus_status(now: float | None = None) -> list[dict]:
    """Per-IP vote tally (for the dashboard), even below threshold."""
    cfg = _cfg()
    now = now or time.time()
    ledger = get_ledger()
    since = now - cfg["consensus_window_hours"] * 3600
    tally: dict[str, set] = {}
    for node, blocks in ((ledger.node_id, ledger.iter_blocks([ALERT, PREDICTION], since=since)),):
        for b in blocks:
            if b["data"].get("attack_stage") == "Command and Control" and (b["data"].get("confidence") or 0) >= cfg["consensus_min_confidence"]:
                for ip in b["data"].get("host_ips") or []:
                    tally.setdefault(ip, set()).add(node)
    for b in ledger.peer_blocks([ALERT, PREDICTION], since=since):
        if b["data"].get("attack_stage") == "Command and Control" and (b["data"].get("confidence") or 0) >= cfg["consensus_min_confidence"]:
            for ip in b["data"].get("host_ips") or []:
                tally.setdefault(ip, set()).add(b["node_id"])
    return sorted(({"ip": ip, "nodes": sorted(n), "votes": len(n), "required": cfg["consensus_min_nodes"]} for ip, n in tally.items()),
                  key=lambda r: -r["votes"])


# ── 2. automatic escalation ────────────────────────────────────────────────
def evaluate_escalation(now: float) -> list[dict]:
    cfg = _cfg()
    ledger = get_ledger()
    prog = cfg["escalation_progression_minutes"] * 60
    ack = cfg["escalation_ack_minutes"] * 60
    preds = [b for b in ledger.iter_blocks([PREDICTION], since=now - 24 * 3600)]
    acks = ledger.iter_blocks([ANALYST_ACTION], since=now - 24 * 3600)
    escalated = {b["data"].get("subject") for b in _executions("auto_escalation")}
    fired = []
    recon = [b for b in preds if b["data"].get("attack_stage") == "Reconnaissance"]
    for ia in (b for b in preds if b["data"].get("attack_stage") == "Initial Access"):
        subject = ia["data"].get("window_id")
        if subject in escalated or now - ia["timestamp"] < ack:
            continue
        prior = [r for r in recon if 0 <= ia["timestamp"] - r["timestamp"] <= prog and (
            r["data"].get("source_file") == ia["data"].get("source_file") or set(r["data"].get("host_ips") or []) & set(ia["data"].get("host_ips") or []))]
        if not prior:
            continue
        targets = {subject, f"alert-{subject}"}
        acknowledged = any(a["data"].get("target_id") in targets and 0 <= a["timestamp"] - ia["timestamp"] <= ack for a in acks)
        if acknowledged:
            continue
        block = ledger.append(ESCALATION, {
            "contract": "auto_escalation", "subject": subject, "new_priority": "CRITICAL",
            "reason": f"Reconnaissance (block #{prior[-1]['index']}) progressed to Initial Access (block #{ia['index']}) within "
                      f"{(ia['timestamp'] - prior[-1]['timestamp']) / 60:.1f} min with no analyst acknowledgement in {cfg['escalation_ack_minutes']} min",
            "recon_block": prior[-1]["index"], "initial_access_block": ia["index"], "broadcast_to": "next-level command",
            "host_ips": ia["data"].get("host_ips")})
        fired.append(block)
    return fired


# ── 3. regression rollback ─────────────────────────────────────────────────
def _fpr_for_model(sha: str | None, feedback: list[dict]) -> tuple[float | None, int]:
    reviewed = [f for f in feedback if f.get("model_sha256") == sha and f["verdict"] in ("confirmed", "false_positive")]
    if not reviewed:
        return None, 0
    return sum(f["verdict"] == "false_positive" for f in reviewed) / len(reviewed), len(reviewed)


def rollback_status() -> dict:
    cfg = _cfg()
    updates = get_ledger().iter_blocks([MODEL_UPDATE])
    if not updates:
        return {"state": "no_model_recorded"}
    cur = updates[-1]["data"]
    new_sha, prev_sha = cur.get("new_weights_sha256"), cur.get("previous_weights_sha256")
    latest = results_store.latest_verdicts()
    feedback = list(latest.values())
    new_fpr, n_new = _fpr_for_model(new_sha, feedback)
    prev_fpr, n_prev = _fpr_for_model(prev_sha, feedback)
    state = "monitoring"
    if prev_sha is None:
        state = "no_previous_model"
    elif n_new >= cfg["rollback_eval_predictions"] and new_fpr is not None:
        state = "regression_detected" if new_fpr > cfg["rollback_fpr_multiplier"] * max(prev_fpr or 0.0, 0.01) else "passed"
    return {"state": state, "current_weights_sha256": new_sha, "previous_weights_sha256": prev_sha, "model_update_block": updates[-1]["index"],
            "reviewed_alerts_current": n_new, "required": cfg["rollback_eval_predictions"], "fpr_current": new_fpr,
            "fpr_previous": prev_fpr, "reviewed_alerts_previous": n_prev, "multiplier": cfg["rollback_fpr_multiplier"]}


def evaluate_rollback(now: float) -> list[dict]:
    status = rollback_status()
    if status.get("state") != "regression_detected":
        return []
    prev_sha = status["previous_weights_sha256"]
    if any(b["data"].get("subject") == status["current_weights_sha256"] for b in _executions("model_rollback")):
        return []
    target = next((c for c in model_registry.available_checkpoints() if c["sha256"] == prev_sha and c.get("compatible", True)), None)
    ledger = get_ledger()
    if target is None:
        return [ledger.append(CONTRACT_EXECUTION, {"contract": "model_rollback", "subject": status["current_weights_sha256"], "action": "rollback_blocked",
                                                   "reason": f"previous weights {prev_sha[:16]}... not found on disk or trained on an older feature schema", **status})]
    block = ledger.append(CONTRACT_EXECUTION, {"contract": "model_rollback", "subject": status["current_weights_sha256"], "action": "rollback",
                                               "rolled_back_to": target["path"], "rolled_back_to_sha256": prev_sha, **status})
    model_registry.set_active_checkpoint(target["path"])
    model_registry.get_models()  # reload -> chains a MODEL_UPDATE back to prev_sha
    return [block]


def evaluate_all(now: float | None = None, include_rollback: bool = True) -> dict:
    now = now or time.time()
    fired = {"consensus_ip_block": evaluate_consensus(now), "auto_escalation": evaluate_escalation(now)}
    if include_rollback:
        fired["model_rollback"] = evaluate_rollback(now)
    return {"evaluated_at": now, "fired": {k: [b["index"] for b in v] for k, v in fired.items()},
            "fired_blocks": [b for v in fired.values() for b in v]}


def recent_executions(limit: int = 50) -> list[dict]:
    blocks = get_ledger().iter_blocks([CONTRACT_EXECUTION, ESCALATION])
    return list(reversed(blocks))[:limit]
