"""Attack-chain reconstruction (roadmap Task 6.1).

Turns isolated per-window predictions into one narrative: where the attack
started, how it progressed, where it is now, where the model expects it to
go next, and what to do about it.

* Backward walk: from the requested window, walk back through the same
  capture's stored predictions while risk stays elevated (small benign gaps
  tolerated); if the window has host IPs, only windows sharing a host count.
  The earliest such window is the chain origin.
* Observed chain: consecutive windows with the same stage are merged into a
  step (time span, peak probability, hosts, top features, narrative).
* Predicted chain: the stored K-step NetJEPA rollout stages (see
  pipeline._score_window). For heuristic-scored windows there is no rollout,
  so the next kill-chain stage is shown as a clearly-labelled prior.
* Remediation for the current and the next predicted stage.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.scoring.attack_stage import STAGE_META, STAGE_NAMES
from src.scoring.remediation import remediation_for_stage
from src.storage import results_store

ELEVATED = 0.3
MAX_GAP = 2
KILL_CHAIN = STAGE_NAMES[1:]
_META = {m["stage"]: m for m in STAGE_META}

_TEMPLATES = {
    "Reconnaissance": "{hosts} probed the network{feat}, mapping exposed services.",
    "Initial Access": "Attempted entry against {hosts}{feat} -- consistent with brute-force or exploit traffic.",
    "Lateral Movement": "Activity spread between internal hosts ({hosts}){feat}.",
    "Command and Control": "{hosts} showed command-and-control behaviour{feat}.",
    "Exfiltration": "Outbound transfer behaviour from {hosts}{feat} suggests data leaving the network.",
    "Benign": "Traffic involving {hosts} looked normal{feat}.",
}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _hosts(row: dict) -> list[str]:
    return [n["label"] for n in row.get("nodes") or [] if n.get("type") == "host"]


def _feature_phrase(features: list[dict]) -> str:
    names = [f["feature"] for f in features[:2] if f.get("feature")]
    return f" (driven by {', '.join(names)})" if names else ""


def _narrate(stage: str, hosts: list[str], features: list[dict]) -> str:
    who = ", ".join(hosts[:3]) + (f" and {len(hosts) - 3} more" if len(hosts) > 3 else "") if hosts else "hosts in this capture"
    return _TEMPLATES.get(stage, "{hosts}: {feat}").format(hosts=who, feat=_feature_phrase(features))


def reconstruct(window_id: str) -> dict | None:
    target = results_store.get_prediction(window_id)
    if target is None:
        return None
    rows = results_store.predictions_for_source(target.get("source_file"))
    # same pipeline run only (window ids are "<capture>-<run>-<window>")
    run_prefix = target["window_id"].rsplit("-", 1)[0] + "-" if target["window_id"].count("-") >= 2 else None
    if run_prefix:
        rows = [r for r in rows if r["window_id"].startswith(run_prefix)] or rows
    pos = next((i for i, r in enumerate(rows) if r["id"] == target["id"]), len(rows) - 1)
    target_hosts = set(_hosts(target))

    def related(r):
        return not target_hosts or not _hosts(r) or bool(target_hosts & set(_hosts(r)))

    origin, gap = pos, 0
    for i in range(pos - 1, -1, -1):
        r = rows[i]
        if r["infiltration_probability"] >= ELEVATED and related(r) and r["attack_stage"] != "Benign":
            origin, gap = i, 0
        else:
            gap += 1
            if gap > MAX_GAP:
                break
    chain_rows = [r for r in rows[origin : pos + 1] if r["infiltration_probability"] >= ELEVATED and r["attack_stage"] != "Benign" and related(r)] or [target]
    if chain_rows[-1]["id"] != target["id"]:
        chain_rows.append(target)

    # one step per stage, in order of first appearance (windows of a stage
    # that recur later are folded into its step, so an oscillating
    # detector doesn't produce a 50-step "progression")
    observed, by_stage = [], {}
    for r in chain_rows:
        stage = r["attack_stage"]
        if stage in by_stage:
            s = by_stage[stage]
            s["window_ids"].append(r["window_id"])
            s["end_timestamp"] = _iso(r["created_at"])
            s["peak_probability"] = max(s["peak_probability"], r["infiltration_probability"])
            s["confidence"] = max(s["confidence"], r["confidence"])
            s["affected_hosts"] = sorted(set(s["affected_hosts"]) | set(_hosts(r)))
            if r["infiltration_probability"] >= s["_best"]:
                s["_best"], s["top_features"] = r["infiltration_probability"], (r.get("top_features") or [])[:5]
            continue
        meta = _META.get(stage, {})
        s = {
            "kind": "observed", "stage": stage, "tactic": meta.get("tactic"), "technique": meta.get("technique"),
            "timestamp": _iso(r["created_at"]), "end_timestamp": _iso(r["created_at"]), "window_ids": [r["window_id"]],
            "peak_probability": r["infiltration_probability"], "confidence": r["confidence"], "_best": r["infiltration_probability"],
            "top_features": (r.get("top_features") or [])[:5], "affected_hosts": sorted(_hosts(r)),
            "scoring_method": r.get("scoring_method"), "block_index": r.get("block_index"),
        }
        by_stage[stage] = s
        observed.append(s)
    # make sure the current stage is the last observed step
    cur = by_stage.get(target["attack_stage"])
    if cur is not None and observed[-1] is not cur:
        observed.remove(cur)
        observed.append(cur)
    for s in observed:
        s.pop("_best", None)
        s["window_count"] = len(s["window_ids"])
        s["window_ids"] = s["window_ids"][-20:]
        s["narrative"] = _narrate(s["stage"], s["affected_hosts"], s["top_features"])

    current = target["attack_stage"]
    predicted = []
    stored = target.get("predicted_stages") or []
    if stored:
        seen = current
        for p in stored:
            if p["stage"] == seen:
                continue
            meta = _META.get(p["stage"], {})
            predicted.append({"kind": "predicted", "stage": p["stage"], "tactic": meta.get("tactic"), "technique": meta.get("technique"),
                              "steps_ahead": p["step"], "probability": p["probability"], "infiltration_probability": p.get("infiltration_probability"),
                              "source": "netjepa_rollout",
                              "narrative": f"NetJEPA's {p['step']}-step rollout predicts {p['stage']} ({p['probability']:.0%} stage probability)."})
            seen = p["stage"]
        if not predicted:
            last = stored[-1]
            predicted.append({"kind": "predicted", "stage": last["stage"], "steps_ahead": last["step"], "probability": last["probability"],
                              "infiltration_probability": last.get("infiltration_probability"), "source": "netjepa_rollout",
                              "narrative": f"The rollout expects the current {last['stage']} activity to persist for the next {last['step']} windows."})
    elif current in KILL_CHAIN and KILL_CHAIN.index(current) < len(KILL_CHAIN) - 1:
        nxt = KILL_CHAIN[KILL_CHAIN.index(current) + 1]
        meta = _META.get(nxt, {})
        predicted.append({"kind": "predicted", "stage": nxt, "tactic": meta.get("tactic"), "technique": meta.get("technique"), "steps_ahead": None,
                          "probability": None, "source": "kill_chain_prior",
                          "narrative": f"No model rollout for this window (heuristic scoring); {nxt} is the next stage of the kill chain."})

    hosts = sorted({h for s in observed for h in s["affected_hosts"]})
    remediation = {"current_stage": current, "current": remediation_for_stage(current, {"host_ips": hosts}) if current in KILL_CHAIN else []}
    nxt_stage = predicted[0]["stage"] if predicted else None
    if nxt_stage and nxt_stage in KILL_CHAIN and nxt_stage != current:
        remediation["next_stage"] = nxt_stage
        remediation["next"] = remediation_for_stage(nxt_stage, {"host_ips": hosts})

    origin_step = observed[0]
    parts = [f"Chain origin: {origin_step['stage']} at {origin_step['timestamp'][11:19]} UTC"
             + (f" involving {', '.join(origin_step['affected_hosts'][:3])}" if origin_step["affected_hosts"] else "") + "."]
    if len(observed) > 1:
        parts.append("Progression: " + " -> ".join(s["stage"] for s in observed) + ".")
    parts.append(f"Currently {current} at {target['infiltration_probability']:.0%} infiltration probability ({target['confidence']:.0%} stage confidence).")
    if predicted:
        p = predicted[0]
        if p["source"] == "netjepa_rollout":
            parts.append(f"Forecast: {p['stage']} within {p['steps_ahead']} window(s) ({p['probability']:.0%}).")
        else:
            parts.append(f"Next kill-chain stage to guard against: {p['stage']}.")
    return {"window_id": window_id, "source_file": target.get("source_file"), "current_stage": current,
            "infiltration_probability": target["infiltration_probability"], "observed": observed, "predicted": predicted,
            "remediation": remediation, "affected_hosts": hosts, "narrative": " ".join(parts),
            "chain_length_windows": len(chain_rows)}
