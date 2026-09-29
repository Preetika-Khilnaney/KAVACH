"""The AI watches the audit trail (synergy doc 1.3 + 2.4).

Analyst verdicts feed model retraining, which makes them an attack surface:
an insider who marks real attacks as false positives for months can slowly
teach the model to ignore them. Every verdict is chained as an
ANALYST_ACTION block, and this module audits that trail before any of it is
trusted for retraining:

* chain consistency -- each feedback row in results.db must match the block
  it was chained in; a row edited or inserted after the fact is flagged
  (audit_record_altered / not_on_chain);
* volume anomaly -- an analyst dismissing far more than their own history or
  the team norm;
* pattern anomaly -- an analyst dismissing every alert from one /24 subnet,
  or every alert of one kill-chain stage;
* consistency anomaly -- dismissing a prediction the calibrated model scored
  >= 90% (statistically unusual, request a second opinion);
* response-time anomaly -- an analyst far slower than the team median;
* post-model-update spike -- a jump in dismissals right after a model update
  (a regression signal for the rollback contract).

The audit result itself is chained as a FEEDBACK_AUDIT block, so the
meta-monitoring is tamper-evident too. verified_training_feedback() returns
only feedback that passed, which is what retraining should consume.
"""

from __future__ import annotations

import ipaddress
import statistics
import time
from collections import defaultdict

from src.config import load_config
from src.storage import results_store
from src.storage.blockchain_ledger import ANALYST_ACTION, FEEDBACK_AUDIT, MODEL_UPDATE, get_ledger

DISMISS = "false_positive"


def _cfg() -> dict:
    return (load_config().get("blockchain", {}) or {}).get("feedback_audit", {}) or {}


def _subnet(ip: str) -> str | None:
    try:
        return str(ipaddress.ip_network(f"{ip}/24", strict=False))
    except ValueError:
        return None


def _chain_consistency(feedback: list[dict]) -> tuple[list[dict], set[int]]:
    ledger = get_ledger()
    blocks = {b["index"]: b for b in ledger.iter_blocks([ANALYST_ACTION])}
    by_feedback_id = {b["data"].get("feedback_id"): b for b in blocks.values() if b["data"].get("feedback_id") is not None}
    flags, bad = [], set()
    for fb in feedback:
        block = blocks.get(fb.get("block_index")) or by_feedback_id.get(fb["id"])
        if block is None:
            flags.append({"kind": "not_on_chain", "severity": "high", "analyst_id": fb["analyst_id"], "feedback_id": fb["id"],
                          "detail": f"feedback #{fb['id']} ({fb['verdict']} on {fb['target_id']}) has no matching ANALYST_ACTION block -- inserted outside the audited path"})
            bad.add(fb["id"])
            continue
        d = block["data"]
        if d.get("action") != fb["verdict"] or d.get("target_id") != fb["target_id"] or d.get("analyst_id") != fb["analyst_id"]:
            flags.append({"kind": "audit_record_altered", "severity": "critical", "analyst_id": d.get("analyst_id"), "feedback_id": fb["id"],
                          "block_index": block["index"],
                          "detail": f"results.db says {fb['analyst_id']} -> {fb['verdict']} on {fb['target_id']}, but block #{block['index']} recorded "
                                    f"{d.get('analyst_id')} -> {d.get('action')} on {d.get('target_id')}. The database record was modified after chaining."})
            bad.add(fb["id"])
    return flags, bad


def analyze(now: float | None = None, record: bool = False) -> dict:
    cfg = _cfg()
    vol_mult = float(cfg.get("volume_multiplier", 3.0))
    min_actions = int(cfg.get("min_actions", 5))
    hi_conf = float(cfg.get("high_confidence_dismissal", 0.9))
    now = now or time.time()
    week = now - 7 * 86400

    feedback = results_store.all_feedback()
    flags, untrusted = _chain_consistency(feedback)

    per: dict[str, list[dict]] = defaultdict(list)
    for fb in feedback:
        per[fb["analyst_id"]].append(fb)

    # team-level dismissal rate (for analysts with little own history)
    team_rates = [sum(f["verdict"] == DISMISS for f in fbs) / len(fbs) for fbs in per.values() if len(fbs) >= min_actions]
    team_rate = statistics.median(team_rates) if team_rates else None

    preds = {p["window_id"]: p for p in results_store.recent_predictions(limit=100000)}
    response_times: dict[str, list[float]] = defaultdict(list)
    analysts = []
    for analyst, fbs in per.items():
        dismissals = [f for f in fbs if f["verdict"] == DISMISS]
        recent = [f for f in dismissals if f["created_at"] >= week]
        older = [f for f in dismissals if f["created_at"] < week]
        first_ts = min(f["created_at"] for f in fbs)
        weeks_of_history = max(1.0, (week - first_ts) / (7 * 86400)) if older else None
        baseline_weekly = (len(older) / weeks_of_history) if weeks_of_history else None
        rate = len(dismissals) / len(fbs)

        if baseline_weekly is not None and len(recent) >= min_actions and len(recent) >= vol_mult * max(baseline_weekly, 1.0):
            flags.append({"kind": "volume_anomaly", "severity": "high", "analyst_id": analyst,
                          "detail": f"{analyst} dismissed {len(recent)} alerts this week -- {len(recent) / max(baseline_weekly, 1.0):.1f}x their usual weekly rate. Review before using for retraining."})
            untrusted.update(f["id"] for f in recent)
        elif team_rate is not None and len(fbs) >= min_actions and rate >= min(0.95, vol_mult * max(team_rate, 0.1)) and len(dismissals) >= min_actions:
            flags.append({"kind": "volume_anomaly", "severity": "medium", "analyst_id": analyst,
                          "detail": f"{analyst} dismisses {rate:.0%} of reviewed alerts vs a team median of {team_rate:.0%}."})

        # pattern: every dismissal from one subnet / one stage
        by_subnet: dict[str, list] = defaultdict(list)
        by_stage: dict[str, list] = defaultdict(list)
        for f in fbs:
            for ip in f.get("host_ips") or []:
                sn = _subnet(ip)
                if sn:
                    by_subnet[sn].append(f)
            if f.get("attack_stage"):
                by_stage[f["attack_stage"]].append(f)
        for sn, items in by_subnet.items():
            uniq = {f["id"]: f for f in items}.values()
            if len(uniq) >= 3 and all(f["verdict"] == DISMISS for f in uniq):
                flags.append({"kind": "pattern_anomaly", "severity": "high", "analyst_id": analyst, "subnet": sn,
                              "detail": f"{analyst} dismissed every one of {len(uniq)} alerts involving subnet {sn}. Verify manually before retraining (possible insider shielding attacker infrastructure, or an undocumented exception)."})
                untrusted.update(f["id"] for f in uniq)
        for stage, items in by_stage.items():
            if stage != "Benign" and len(items) >= 4 and all(f["verdict"] == DISMISS for f in items):
                flags.append({"kind": "pattern_anomaly", "severity": "medium", "analyst_id": analyst, "stage": stage,
                              "detail": f"{analyst} dismissed all {len(items)} '{stage}' alerts they reviewed."})

        for f in dismissals:
            p = f.get("prediction_probability")
            if p is not None and p >= hi_conf:
                flags.append({"kind": "consistency_anomaly", "severity": "medium", "analyst_id": analyst, "feedback_id": f["id"], "target_id": f["target_id"],
                              "detail": f"{analyst} marked {f['target_id']} as a false positive although the calibrated model scored it {p:.0%}. Request a second opinion."})
                untrusted.add(f["id"])

        for f in fbs:
            wid = f.get("window_id") or f["target_id"].replace("alert-", "")
            if wid in preds:
                response_times[analyst].append(max(0.0, f["created_at"] - preds[wid]["created_at"]))
        analysts.append({"analyst_id": analyst, "actions": len(fbs), "dismissals": len(dismissals), "dismissal_rate": round(rate, 3),
                         "dismissals_this_week": len(recent), "baseline_weekly_dismissals": None if baseline_weekly is None else round(baseline_weekly, 2),
                         "median_response_seconds": round(statistics.median(response_times[analyst]), 1) if response_times[analyst] else None})

    medians = [a["median_response_seconds"] for a in analysts if a["median_response_seconds"] is not None]
    if len(medians) >= 2:
        team_median = statistics.median(medians)
        for a in analysts:
            if a["median_response_seconds"] and team_median and a["median_response_seconds"] > 3 * team_median and a["median_response_seconds"] > 300:
                flags.append({"kind": "response_time_anomaly", "severity": "low", "analyst_id": a["analyst_id"],
                              "detail": f"{a['analyst_id']} takes {a['median_response_seconds'] / 60:.0f} min to respond vs a team median of {team_median / 60:.0f} min."})

    # dismissal spike after the latest model update -> regression signal
    updates = get_ledger().iter_blocks([MODEL_UPDATE])
    if len(updates) >= 2:
        t_upd = updates[-1]["timestamp"]
        before = [f for f in feedback if f["created_at"] < t_upd]
        after = [f for f in feedback if f["created_at"] >= t_upd]
        if len(before) >= min_actions and len(after) >= min_actions:
            rb = sum(f["verdict"] == DISMISS for f in before) / len(before)
            ra = sum(f["verdict"] == DISMISS for f in after) / len(after)
            if ra >= 2 * max(rb, 0.05):
                flags.append({"kind": "post_update_dismissal_spike", "severity": "high", "analyst_id": None, "block_index": updates[-1]["index"],
                              "detail": f"Dismissal rate rose from {rb:.0%} to {ra:.0%} after model update block #{updates[-1]['index']} -- the new model may have a higher false-positive rate."})

    trusted = [f for f in feedback if f["id"] not in untrusted]
    result = {"generated_at": now, "total_feedback": len(feedback), "trusted_feedback": len(trusted), "excluded_feedback": len(untrusted),
              "analysts": sorted(analysts, key=lambda a: -a["actions"]), "flags": flags,
              "status": "clean" if not flags else ("critical" if any(f["severity"] == "critical" for f in flags) else "review")}
    if record:
        block = get_ledger().append(FEEDBACK_AUDIT, {"status": result["status"], "flag_count": len(flags), "flags": [
            {k: v for k, v in f.items() if k in ("kind", "severity", "analyst_id", "subnet", "stage", "feedback_id", "block_index")} for f in flags][:50],
            "trusted_feedback": len(trusted), "excluded_feedback": len(untrusted)})
        result["block_index"] = block["index"]
    return result


def verified_training_feedback() -> list[dict]:
    """Feedback that passed the audit -- what a retraining job may use."""
    audit = analyze()
    excluded = {f.get("feedback_id") for f in audit["flags"] if f.get("feedback_id") is not None}
    _, bad = _chain_consistency(results_store.all_feedback())
    excluded |= bad
    flagged_analysts = {f["analyst_id"] for f in audit["flags"] if f["kind"] in ("volume_anomaly", "pattern_anomaly") and f["severity"] == "high"}
    return [f for f in results_store.all_feedback() if f["id"] not in excluded and f["analyst_id"] not in flagged_analysts]
