"""Cross-node threat intelligence (synergy doc 2.3).

Blocks from other KAVACH nodes arrive through air-gapped sync bundles
(Ledger.import_bundle) and are hash-verified before they are stored. This
module lets the local AI use them as extra evidence: when the current
window involves an IP that other nodes flagged, their verified assessments
are fused with the local probability (noisy-OR), so a moderate local score
about an IP two other bases already saw scanning becomes an escalation.

Because peer evidence only ever comes from verified chains, a compromised
node can't inject intelligence that isn't in its own hash chain -- and the
consensus contract (contracts.py) still needs several independent nodes
before any automated action fires.
"""

from __future__ import annotations

import time

from src.storage.blockchain_ledger import ALERT, PREDICTION, get_ledger


def _ips(block: dict) -> set[str]:
    return set(block["data"].get("host_ips") or [])


def cross_node_assessment(host_ips: list[str] | None, local_probability: float, window_hours: float = 24.0) -> dict:
    ips = set(host_ips or [])
    result = {"local_probability": float(local_probability), "combined_probability": float(local_probability),
              "evidence": [], "peer_nodes": [], "escalate": False}
    if not ips:
        return result
    since = time.time() - window_hours * 3600
    evidence = []
    for b in get_ledger().peer_blocks([ALERT, PREDICTION], since=since):
        shared = ips & _ips(b)
        if not shared:
            continue
        d = b["data"]
        evidence.append({"node_id": b["data"].get("origin_node_id") or b["node_id"], "block_index": b["index"], "block_type": b["block_type"],
                         "shared_ips": sorted(shared), "attack_stage": d.get("attack_stage"),
                         "infiltration_probability": d.get("infiltration_probability"), "confidence": d.get("confidence"),
                         "timestamp": b["timestamp"], "block_hash": b["hash"]})
    if not evidence:
        return result
    # strongest report per peer node, so one chatty node can't dominate
    best: dict[str, dict] = {}
    for e in evidence:
        score = (e["infiltration_probability"] or 0) * (e["confidence"] or 0)
        if e["node_id"] not in best or score > (best[e["node_id"]]["infiltration_probability"] or 0) * (best[e["node_id"]]["confidence"] or 0):
            best[e["node_id"]] = e
    miss = 1.0 - float(local_probability)
    for e in best.values():
        miss *= 1.0 - 0.9 * (e["infiltration_probability"] or 0) * (e["confidence"] or 0)
    combined = 1.0 - miss
    result.update({"combined_probability": combined, "evidence": sorted(best.values(), key=lambda e: -e["timestamp"]),
                   "peer_nodes": sorted(best), "escalate": combined >= 0.85 and local_probability < 0.7})
    return result
