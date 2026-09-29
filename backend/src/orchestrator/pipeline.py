"""Orchestrator: runs the pipeline stages in order and yields events.

Stages (one event each, per window):
- ingestion: src.ingestion.flow_parser / packet_parser, one event per batch.
- feature_extraction: src.features.extract, one event per window.
- state_representation / forward_pass / rollout / attack_mapping /
  explainability: graph build -> NetJEPA -> scoring heads -> attribution.

Scoring (see _score_window):
* NetJEPA path -- when the served checkpoint's heads are *validated*
  (src/training/train_heads.py beat the baseline on held-out data) and the
  window is in the model's training distribution (it carries flow-level
  features, like the CIC-IDS-2017 flow windows the model was trained on),
  the infiltration probability is the real head's output, temperature-
  calibrated, and the K-step rollout gives a real forecast curve and
  predicted future stages.
* packet_heuristic path -- packet-only PCAP windows have none of the
  flow-level features the model was trained on (every flow_* input would be
  0), so scoring them with the network would be out-of-distribution. They
  get a transparent rule-based reading over real per-window packet signals
  (scan flag, retransmits, TTL / payload variance). Every payload carries
  `scoring_method` so the UI can say which one it is.

Every scored window is persisted to results.db AND chained on the forensic
ledger (risk-tiered: src/storage/blockchain_ledger.py), with the SHA-256 of
the model weights that produced it. High-risk windows also get an ALERT
block (and critical ones an ESCALATION block); smart contracts are
evaluated when a capture finishes.

Every payload from state_representation onward carries `trained: bool`.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

import asyncio
import pandas as pd
import torch

from src.config import load_config
from src.explainability.model_forward import infil_logit
from src.explainability.shap_explainer import explain_prediction, heuristic_attribution
from src.features import extract
from src.graph.state_builder import CANONICAL_FEATURE_NAMES, build_graph_state
from src.ingestion import flow_parser, packet_parser
from src.models.netjepa import graph_batch_from_state
from src.orchestrator import model_registry
from src.scoring.attack_stage import STAGE_META, STAGE_NAMES
from src.scoring.calibration import calibrate_probability
from src.scoring.remediation import remediation_for_stage
from src.storage import results_store

log = logging.getLogger(__name__)

_PCAP_SUFFIXES = {".pcap", ".pcapng", ".cap"}

# Default cap on how many packets a *live* pipeline run parses out of a raw
# capture (the full captures are 10-13 GB). Pass max_packets=None for a
# real offline/batch pass over the whole file.
_DEFAULT_LIVE_PACKET_CAP = 50_000

# Same idea for windows. Pass max_windows=None for every window.
_DEFAULT_LIVE_WINDOW_CAP = 500

# Integrated Gradients does a few dozen forward passes per call -- only
# spend that on windows that look interesting; the rest get the fast
# heuristic attribution (never an empty explanation).
_EXPLAIN_INFILTRATION_THRESHOLD = 0.3

_STAGE_TACTIC = {m["stage"]: m for m in STAGE_META}

_FLOW_FEATURES = [n for n in CANONICAL_FEATURE_NAMES if n.startswith("flow_")]


def _csv_rows_per_window_default() -> int | None:
    return load_config().get("windowing", {}).get("csv_rows_per_window")


# ── Packet-only heuristic reading ────────────────────────────────────────────
def _packet_heuristic_reading(window: dict, rollout_steps: int) -> tuple[float, list[float], str, float]:
    """Transparent rule-based reading for packet-only windows (see module
    docstring): composite of real, per-capture z-scored packet signals."""
    signals = [
        window.get("packet_port_scan_flagged"),
        window.get("packet_retransmit_count"),
        window.get("packet_ttl_variance"),
        window.get("packet_payload_size_std"),
    ]
    present = [float(s) for s in signals if s is not None and not (isinstance(s, float) and math.isnan(s))]
    composite = sum(present) / len(present) if present else 0.0

    score = 1 / (1 + math.exp(-2.5 * (composite + 0.15)))
    curve = [score] * rollout_steps

    scan = float(window.get("packet_port_scan_flagged") or 0.0)
    payload_std = float(window.get("packet_payload_size_std") or 0.0)
    retransmit = float(window.get("packet_retransmit_count") or 0.0)

    if score < 0.25:
        stage = "Benign"
    elif scan >= 0.5:
        stage = "Reconnaissance"
    elif payload_std >= 0.5 or retransmit >= 0.5:
        stage = "Exfiltration"
    elif score >= 0.5:
        stage = "Command and Control"
    else:
        stage = "Lateral Movement"

    confidence = 0.55 + min(abs(composite), 1.5) / 1.5 * 0.4
    return score, curve, stage, confidence


# Backwards-compatible alias (older code/tests import this name).
_calibrated_infiltration_reading = _packet_heuristic_reading


def _in_model_distribution(window: dict, meta: dict | None = None) -> bool:
    """v2 models (feature schema 2) were trained on flow-only CSV windows AND
    joint flow+packet PCAP windows, so any window qualifies. v1 models only
    saw flow-level features."""
    if (meta or {}).get("feature_schema_version", 1) >= 2:
        return True
    return any(window.get(n) not in (None, 0, 0.0) for n in _FLOW_FEATURES)


def _stage_distribution(stage_clf, z: torch.Tensor, temperature: float | None) -> list[float]:
    with torch.no_grad():
        return torch.softmax(stage_clf(z)[0] / (temperature or 1.0), dim=-1).tolist()


def _score_window(window: dict, state: dict, embedding: torch.Tensor, rollout: list[torch.Tensor], models, meta: dict) -> dict:
    """Returns the scoring dict used by attack_mapping + persistence."""
    model, infil_head, stage_clf, trained = models
    rollout_steps = len(rollout)
    use_model = trained and meta.get("heads_validated") and _in_model_distribution(window, meta)
    if use_model:
        T, Ts = meta.get("temperature"), meta.get("stage_temperature")
        with torch.no_grad():
            raw = torch.sigmoid(infil_logit(infil_head, embedding)).item()
            curve_raw = [torch.sigmoid(infil_logit(infil_head, z)).item() for z in rollout]
        calibrated = calibrate_probability(raw, T)
        dist = _stage_distribution(stage_clf, embedding, Ts)
        idx = int(max(range(len(dist)), key=dist.__getitem__))
        predicted = []
        for k, z in enumerate(rollout, start=1):
            d = _stage_distribution(stage_clf, z, Ts)
            j = int(max(range(len(d)), key=d.__getitem__))
            predicted.append({"step": k, "stage": STAGE_NAMES[j], "probability": d[j],
                              "infiltration_probability": calibrate_probability(curve_raw[k - 1], T)})
        out = {
            "scoring_method": "netjepa",
            "infiltration_probability": calibrated,
            "raw_probability": raw,
            "calibrated_probability": calibrated,
            "infiltration_curve": [calibrate_probability(p, T) for p in curve_raw],
            "attack_stage": STAGE_NAMES[idx],
            "confidence": dist[idx],
            "stage_distribution": dict(zip(STAGE_NAMES, dist)),
            "predicted_stages": predicted,
        }
        if getattr(model.predictor, "probabilistic", False):
            out.update(_monte_carlo_forecast(model, infil_head, stage_clf, embedding, len(rollout), T, Ts, idx))
        return out
    score, curve, stage, confidence = _packet_heuristic_reading(window, rollout_steps)
    reason = ("untrained model" if not trained else
              "model heads not validated (run python -m src.training.train_heads)" if not meta.get("heads_validated") else
              "packet-only window: outside the flow-feature distribution NetJEPA was trained on")
    return {
        "scoring_method": "packet_heuristic",
        "scoring_reason": reason,
        "infiltration_probability": score,
        "raw_probability": score,
        "calibrated_probability": None,
        "infiltration_curve": curve,
        "attack_stage": stage,
        "confidence": confidence,
        "stage_distribution": None,
        "predicted_stages": [],
    }


_MC_SAMPLES = 64


def _monte_carlo_forecast(model, infil_head, stage_clf, z, steps, T, Ts, current_idx) -> dict:
    """Sample trajectories from the learned transition distribution
    p(z_{t+1} | z_t) and read each with the heads:
    * infiltration_band: 10th/90th percentile of the calibrated infiltration
      probability at each future step (uncertainty band);
    * infiltration_within_k: P(some window in the next k is scored an attack
      at the decision threshold) across sampled futures;
    * progression_probability: fraction of sampled futures in which the
      predicted MITRE stage moves LATER in the kill chain than now;
    * predicted_stage_frequency: how often each stage is predicted per step."""
    import numpy as np

    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        traj = model.predictor.sample_rollout(z, steps, n_samples=_MC_SAMPLES, generator=gen)  # (S, K, D)
        flat = traj.reshape(-1, traj.shape[-1])
        logits = infil_logit(infil_head, flat).reshape(_MC_SAMPLES, steps)
        probs = torch.sigmoid(logits / (T or 1.0)).numpy()
        stages = torch.softmax(stage_clf(flat) / (Ts or 1.0), -1).argmax(-1).reshape(_MC_SAMPLES, steps).numpy()
    thr = model_registry.get_model_meta().get("decision_threshold") or 0.5
    hit = probs >= thr
    within = [float(hit[:, : k + 1].any(axis=1).mean()) for k in range(steps)]
    later = (stages > current_idx) & (stages > 0) if current_idx > 0 else stages > 0
    freq = []
    for k in range(steps):
        counts = np.bincount(stages[:, k], minlength=len(STAGE_NAMES)) / _MC_SAMPLES
        freq.append({STAGE_NAMES[i]: float(c) for i, c in enumerate(counts) if c > 0})
    return {
        "infiltration_band": {"p10": np.percentile(probs, 10, axis=0).tolist(), "p90": np.percentile(probs, 90, axis=0).tolist(),
                              "mean": probs.mean(axis=0).tolist()},
        "infiltration_within_k": within,
        "progression_probability": float(later.any(axis=1).mean()),
        "predicted_stage_frequency": freq,
        "forecast_samples": _MC_SAMPLES,
    }


def _concepts(embedding: torch.Tensor, scoring: dict) -> dict | None:
    if scoring["scoring_method"] != "netjepa":
        return None
    try:
        from src.explainability.explain import load_concept_bottleneck

        cb = load_concept_bottleneck()
        if cb is None:
            return None
        pred = cb.predict(embedding)
        return {"stage": pred["stage"], "stage_probability": pred["stage_probability"], "explanation": pred["explanation"],
                "active": [c for c in pred["concepts"] if c["probability"] >= 0.5]}
    except Exception:
        log.exception("concept bottleneck failed")
        return None


def _chain(window_id, source_file, window, state, scoring, top_features, remediation, meta, cross_node) -> dict:
    """Ledger writes for one window. Never raises (the evidence trail must
    not take the detection pipeline down)."""
    try:
        from src.storage import blockchain_ledger as bl

        ledger = bl.get_ledger()
        host_ips = sorted({n["label"] for n in state.get("nodes", []) if n.get("type") == "host"})
        res = ledger.log_prediction(
            window_id=window_id, source_file=source_file, infiltration_probability=scoring["infiltration_probability"],
            attack_stage=scoring["attack_stage"], confidence=scoring["confidence"], feature_vector=state["feature_vector"],
            top_features=top_features, nodes=state.get("nodes"), edges=state.get("edges"), model_sha256=meta.get("sha256"),
            scoring_method=scoring["scoring_method"], calibrated_probability=scoring.get("calibrated_probability"),
            predicted_stages=[p["stage"] for p in scoring.get("predicted_stages") or []], remediation=remediation)
        out = {"tier": res["tier"], "block_index": res["block"]["index"] if res["block"] else None,
               "block_hash": res["block"]["hash"] if res["block"] else None,
               "batched": res["block"] is None, "batch_block_index": res["batch_block"]["index"] if res["batch_block"] else None}
        if res["tier"] in ("high", "critical") and scoring["attack_stage"] != "Benign":
            meta_stage = _STAGE_TACTIC.get(scoring["attack_stage"], {})
            alert = ledger.log_alert(
                alert_id=f"alert-{window_id}", window_id=window_id,
                severity="critical" if res["tier"] == "critical" or scoring["infiltration_probability"] >= 0.7 else "warning",
                attack_stage=scoring["attack_stage"], tactic=meta_stage.get("tactic"), technique=meta_stage.get("technique"),
                host_ips=host_ips, infiltration_probability=scoring["infiltration_probability"], confidence=scoring["confidence"],
                top_features=top_features, model_sha256=meta.get("sha256"))
            out["alert_block_index"] = alert["index"]
        if res["tier"] == "critical" or (cross_node and cross_node.get("escalate")):
            esc = ledger.append(bl.ESCALATION, {
                "contract": "critical_tier_escalation", "subject": window_id, "new_priority": "CRITICAL",
                "reason": ("critical-tier prediction (>= 90% and late kill-chain stage)" if res["tier"] == "critical"
                           else f"cross-node intelligence raised the combined probability to {cross_node['combined_probability']:.0%}"),
                "host_ips": host_ips, "evidence_block": out.get("alert_block_index") or out.get("block_index")})
            out["escalation_block_index"] = esc["index"]
        return out
    except Exception:
        log.exception("ledger write failed for window %s", window_id)
        return {"error": "ledger write failed"}


def _capture_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in _PCAP_SUFFIXES:
        return "pcap"
    raise ValueError(f"Unrecognized capture file type: {path!r} (expected .csv or .pcap/.pcapng)")


def _csv_ingestion_batches(path: str):
    total = 0
    for chunk in flow_parser.iter_flow_csv(path):
        n = len(chunk)
        total += n
        if n and "timestamp" in chunk.columns:
            time_range = [chunk["timestamp"].min().isoformat(), chunk["timestamp"].max().isoformat()]
        else:
            time_range = [None, None]
        payload = {"kind": "csv", "source": path, "records_ingested": n, "records_total": total, "time_range": time_range}
        yield chunk, payload, "in_progress"
    yield None, {"kind": "csv", "source": path, "records_ingested": 0, "records_total": total, "time_range": [None, None]}, "complete"


def _pcap_ingestion_batches(path: str, max_packets: int | None):
    total = 0
    for batch in packet_parser.iter_packets(path, max_packets=max_packets):
        n = len(batch)
        total += n
        timestamps = [r["timestamp"] for r in batch]
        time_range = [
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(min(timestamps))),
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(timestamps))),
        ] if timestamps else [None, None]
        payload = {"kind": "pcap", "source": path, "records_ingested": n, "records_total": total, "time_range": time_range}
        yield batch, payload, "in_progress"
    yield None, {"kind": "pcap", "source": path, "records_ingested": 0, "records_total": total, "time_range": [None, None]}, "complete"


async def _run_ingestion(capture_path: str | None, window_id: str, max_packets: int | None):
    if capture_path is None:
        yield None, {
            "stage": "ingestion",
            "window_id": window_id,
            "timestamp": time.time(),
            "payload": {"records_ingested": 0, "source": "placeholder"},
            "status": "complete",
        }
        return

    kind = _capture_kind(capture_path)
    batches = _csv_ingestion_batches(capture_path) if kind == "csv" else _pcap_ingestion_batches(capture_path, max_packets)
    for chunk_data, payload, status in batches:
        yield chunk_data, {
            "stage": "ingestion",
            "window_id": window_id,
            "timestamp": time.time(),
            "payload": payload,
            "status": status,
        }
        await asyncio.sleep(0)


def _placeholder_model_stage_events(window_id: str, status: str):
    for stage, payload in (
        ("state_representation", {"nodes": [], "edges": []}),
        ("forward_pass", {"embedding_dim": 0, "trained": False}),
        ("rollout", {"k_steps": 0, "trained": False}),
        ("attack_mapping", {"infiltration_probability": 0.0, "attack_stage": "none", "trained": False}),
        ("explainability", {"top_features": [], "trained": False}),
    ):
        yield {"stage": stage, "window_id": window_id, "timestamp": time.time(), "payload": payload, "status": status}


def _run_model_stages(window: dict, window_id: str, status: str, source_file: str | None = None, history: list | None = None,
                      persist: bool = True):
    """state_representation -> forward_pass -> rollout -> attack_mapping ->
    explainability for ONE feature window, then persistence + chaining.

    `history` is a list the caller keeps across windows of one capture: the
    previous windows' graphs, so the GRU sees the same temporal context
    (meta["context_len"] windows) the model was trained with."""
    models = model_registry.get_models()
    model, infil_head, stage_clf, trained = models
    meta = model_registry.get_model_meta()
    rollout_steps = load_config().get("model", {}).get("rollout_steps", 5)

    state = build_graph_state(window)
    yield {
        "stage": "state_representation",
        "window_id": window_id,
        "timestamp": time.time(),
        "payload": {"nodes": state["nodes"], "edges": state["edges"]},
        "status": status,
    }

    graph_batch = graph_batch_from_state(state)
    context_len = int(meta.get("context_len") or 1)
    prior = list(history[-(context_len - 1):]) if (history is not None and context_len > 1) else []
    with torch.no_grad():
        _, _, z_context = model([[*prior, graph_batch]])
    embedding = z_context[0]
    if history is not None:
        history.append(graph_batch)
        del history[: max(0, len(history) - 8)]
    yield {
        "stage": "forward_pass",
        "window_id": window_id,
        "timestamp": time.time(),
        "payload": {"embedding_dim": embedding.shape[-1], "trained": trained, "model_sha256": meta.get("sha256"),
                    "context_windows": len(prior) + 1},
        "status": status,
    }

    rollout = model.predictor.rollout(embedding, steps=rollout_steps)
    yield {
        "stage": "rollout",
        "window_id": window_id,
        "timestamp": time.time(),
        "payload": {"k_steps": rollout_steps, "trained": trained},
        "status": status,
    }

    scoring = _score_window(window, state, embedding, rollout, models, meta)
    yield {
        "stage": "attack_mapping",
        "window_id": window_id,
        "timestamp": time.time(),
        "payload": {
            "infiltration_probability": scoring["infiltration_probability"],
            "raw_probability": scoring["raw_probability"],
            "calibrated_probability": scoring["calibrated_probability"],
            "infiltration_curve": scoring["infiltration_curve"],
            "attack_stage": scoring["attack_stage"],
            "confidence": scoring["confidence"],
            "stage_distribution": scoring["stage_distribution"],
            "predicted_stages": scoring["predicted_stages"],
            "scoring_method": scoring["scoring_method"],
            "infiltration_band": scoring.get("infiltration_band"),
            "infiltration_within_k": scoring.get("infiltration_within_k"),
            "progression_probability": scoring.get("progression_probability"),
            "predicted_stage_frequency": scoring.get("predicted_stage_frequency"),
            "trained": trained,
        },
        "status": status,
    }

    top_features: list = []
    attribution_method = None
    if trained:
        if scoring["scoring_method"] == "netjepa" and scoring["infiltration_probability"] >= _EXPLAIN_INFILTRATION_THRESHOLD:
            try:
                top_features = explain_prediction(model, infil_head, state=state, n_features=10)
                attribution_method = top_features[0]["method"] if top_features else None
            except Exception:
                log.exception("integrated gradients failed; using heuristic attribution")
        if not top_features:
            top_features = heuristic_attribution(state["feature_vector"], n_features=10)
            attribution_method = "heuristic"

    host_ips = sorted({n["label"] for n in state["nodes"] if n.get("type") == "host"})
    remediation = remediation_for_stage(scoring["attack_stage"], {"host_ips": host_ips}) if scoring["attack_stage"] not in ("Benign", "none") else []
    concepts = _concepts(embedding, scoring)
    cross_node = None
    if host_ips:
        try:
            from src.blockchain.intelligence import cross_node_assessment

            cross_node = cross_node_assessment(host_ips, scoring["infiltration_probability"])
            if not cross_node["evidence"]:
                cross_node = None
        except Exception:
            log.exception("cross-node assessment failed")

    if not persist:
        yield {"stage": "explainability", "window_id": window_id, "timestamp": time.time(),
               "payload": {"top_features": top_features, "attribution_method": attribution_method, "trained": trained,
                           "remediation": remediation[:3], "concepts": concepts, "cross_node": cross_node, "ledger": None,
                           "host_ips": host_ips},
               "status": status}
        return

    row_id = results_store.save_prediction(
        window_id=window_id,
        source_file=source_file,
        window=window,
        state=state,
        infiltration_probability=scoring["infiltration_probability"],
        attack_stage=scoring["attack_stage"],
        confidence=scoring["confidence"],
        trained=trained,
        top_features=top_features,
        extra={
            "raw_probability": scoring["raw_probability"],
            "calibrated_probability": scoring["calibrated_probability"],
            "scoring_method": scoring["scoring_method"],
            "infiltration_curve": scoring["infiltration_curve"],
            "stage_distribution": scoring["stage_distribution"],
            "predicted_stages": scoring["predicted_stages"],
            "model_sha256": meta.get("sha256"),
            "attribution_method": attribution_method,
            "cross_node": cross_node,
            "concepts": concepts,
            "forecast": {k: scoring.get(k) for k in ("infiltration_band", "infiltration_within_k", "progression_probability",
                                                     "predicted_stage_frequency", "forecast_samples") if scoring.get(k) is not None} or None,
        },
    )
    ledger_info = _chain(window_id, source_file, window, state, scoring, top_features, remediation, meta, cross_node) if trained else None
    if ledger_info and not ledger_info.get("error"):
        results_store.update_prediction(row_id, block_index=ledger_info.get("block_index"), chain_tier=ledger_info.get("tier"))

    yield {
        "stage": "explainability",
        "window_id": window_id,
        "timestamp": time.time(),
        "payload": {"top_features": top_features, "attribution_method": attribution_method, "trained": trained,
                    "remediation": remediation[:3], "concepts": concepts, "cross_node": cross_node, "ledger": ledger_info},
        "status": status,
    }


def _finalize_capture() -> dict | None:
    """End of capture: chain any buffered low-risk windows and evaluate the
    smart contracts over the new evidence."""
    try:
        from src.blockchain import contracts
        from src.storage import blockchain_ledger as bl

        batch = bl.get_ledger().flush_low_risk()
        fired = contracts.evaluate_all(include_rollback=True)
        return {"batch_block": batch["index"] if batch else None, "contracts_fired": fired["fired"]}
    except Exception:
        log.exception("capture finalisation failed")
        return None


async def _run_feature_extraction_and_model_stages(kind: str | None, flow_chunks: list, packet_batches: list, window_id: str,
                                                   source_file: str | None = None, max_windows: int | None = _DEFAULT_LIVE_WINDOW_CAP,
                                                   csv_rows_per_window: int | None = None, run_tag: str | None = None,
                                                   persist: bool = True):
    window_seconds = load_config().get("windowing", {}).get("window_seconds", 5)

    if kind is None:
        yield {"stage": "feature_extraction", "window_id": window_id, "timestamp": time.time(), "payload": {"feature_vector": []}, "status": "complete"}
        for event in _placeholder_model_stage_events(window_id, "complete"):
            yield event
        return

    flow_df = pd.concat(flow_chunks, ignore_index=True) if flow_chunks else None
    packet_records = packet_batches if packet_batches else None

    if kind == "pcap":
        # PCAPs: BOTH traffic levels from the same packets on the same time
        # base -- bidirectional flow records (flow level) + packet statistics
        # (packet level) per window (src/features/joint_windows.py), the
        # same construction the v2 model was trained on.
        from src.features.joint_windows import iter_joint_windows, normalize_windows

        raw = list(iter_joint_windows(packet_records or [], window_seconds=window_seconds))
        windows, _ = normalize_windows(raw)
    else:
        # CSV captures: fixed row-count windows (same as the training cache, see
        # src/training/build_windows_cache.py) so live windows match what the
        # model was trained on -- time windows over flow_parser's synthesized
        # timestamps produce far sparser, differently-shaped windows. Files
        # with real timestamps (CIC-IDS-2018, CTU-13) could use time windows;
        # row windows keep behaviour consistent across formats.
        rows = csv_rows_per_window
        windows = list(extract.iter_feature_windows(flow_df=flow_df, packet_records=packet_records, window_seconds=window_seconds, rows_per_window=rows))
    if max_windows is not None and len(windows) > max_windows:
        windows = windows[:max_windows]
    if not windows:
        yield {"stage": "feature_extraction", "window_id": window_id, "timestamp": time.time(),
               "payload": {"feature_vector": [], "reason": "not enough records to fill a window"}, "status": "complete"}
        for event in _placeholder_model_stage_events(window_id, "complete"):
            yield event
        return

    history: list = []
    for i, window in enumerate(windows):
        status = "in_progress" if i < len(windows) - 1 else "complete"
        raw_id = str(window.get("window_id", window_id))
        # unique across captures/runs, so /flows/{id} never collides
        w_id = f"{run_tag}-{raw_id}" if run_tag else raw_id

        yield {"stage": "feature_extraction", "window_id": w_id, "timestamp": time.time(),
               "payload": {"feature_vector": window, "window_index": i, "window_count": len(windows)}, "status": status}
        await asyncio.sleep(0)

        for event in _run_model_stages(window, w_id, status, source_file=source_file, history=history, persist=persist):
            yield event
            await asyncio.sleep(0)

    if persist:
        _finalize_capture()


def _run_tag(source_file: str | None) -> str:
    stem = Path(source_file).stem if source_file else "capture"
    stem = "".join(c if c.isalnum() else "_" for c in stem)[:24].strip("_") or "capture"
    return f"{stem}-{uuid.uuid4().hex[:6]}"


async def run_pipeline(
    capture_path: str | None = None,
    max_packets: int | None = _DEFAULT_LIVE_PACKET_CAP,
    max_windows: int | None = _DEFAULT_LIVE_WINDOW_CAP,
    csv_rows_per_window: int | None | str = "config",
    persist: bool = True,
) -> AsyncGenerator[dict, None]:
    """persist=False: analyse only (no results.db rows, no ledger blocks) --
    used by the CLI's `predict` command."""
    window_id = "w0"
    source_file = Path(capture_path).name if capture_path is not None else None
    if csv_rows_per_window == "config":
        csv_rows_per_window = _csv_rows_per_window_default()

    kind = _capture_kind(capture_path) if capture_path is not None else None
    flow_chunks: list = []
    packet_batches: list = []

    async for chunk_data, event in _run_ingestion(capture_path, window_id, max_packets):
        yield event
        if chunk_data is not None:
            if kind == "csv":
                flow_chunks.append(chunk_data)
            elif kind == "pcap":
                packet_batches.extend(chunk_data)

    async for event in _run_feature_extraction_and_model_stages(
        kind, flow_chunks, packet_batches, window_id, source_file=source_file, max_windows=max_windows,
        csv_rows_per_window=csv_rows_per_window, run_tag=_run_tag(source_file) if capture_path else None, persist=persist,
    ):
        yield event
