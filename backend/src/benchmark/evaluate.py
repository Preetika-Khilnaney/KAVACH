"""Benchmark / evaluation (architecture doc; roadmap Tasks 1.2 + 7.3;
evaluation doc "validation pyramid").

Everything is evaluated on the same held-out validation windows
src.training.train uses (last 20% of each file -- temporal, no leakage):

Layer 1 -- outputs
  binary        logistic-regression baseline vs NetJEPA (F1/precision/recall/FPR/AUC)
  multiclass    6-class MITRE stage head: macro-F1, per-class F1, confusion matrix
  calibration   ECE before/after temperature scaling + reliability bins
  predictive    AUC of the score at window t for "attack present at t+k"
Layer 2 -- representations
  embed_std, linear probes per layer (from the interpretability report),
  explanation consistency (Jaccard of top-5 IG features within an attack type)
Layer 3 -- mechanisms
  causal tracing averaged per attack stage, tuned-lens progression, SAE stats
Layer 4 -- trust
  adversarial robustness (PGD), concept-bottleneck accuracy

A clean environment with no windows cache returns a structured "not built
yet" response instead of crashing (Task 1.2). Results are cached in memory
and on disk (keyed by the model weights hash) because the full suite takes a
couple of minutes; `python -m src.benchmark.evaluate` precomputes it.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.graph.state_builder import build_graph_state
from src.models.netjepa import graph_batch_from_state
from src.orchestrator import model_registry
from src.scoring.attack_stage import STAGE_NAMES
from src.scoring.calibration import expected_calibration_error
from src.training.dataset import build_examples, load_all_windows
from src.training.labels import infiltration_label, stage_label

log = logging.getLogger(__name__)

DISK_CACHE = Path("data/processed/benchmark_cache.json")
INTERP_REPORT = Path("data/processed/interpretability/interpretability_report.json")
_cache: dict = {"key": None, "result": None}


def _feature_matrix(examples: list[tuple[list[dict], dict]]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for _, target in examples:
        X.append(build_graph_state(target)["feature_vector"])
        y.append(infiltration_label(target))
    return np.array(X, dtype=float), np.array(y, dtype=int)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    out = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if scores is not None and len(set(y_true.tolist())) > 1:
        out["auc"] = float(roc_auc_score(y_true, scores))
    return out


def _evaluate_baseline(train_examples, val_examples) -> tuple[dict, np.ndarray]:
    X_train, y_train = _feature_matrix(train_examples)
    X_val, y_val = _feature_matrix(val_examples)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_val)[:, 1]
    return _binary_metrics(y_val, clf.predict(X_val), scores), scores


@torch.no_grad()
def _netjepa_outputs(val_examples, batch_size: int = 256) -> dict | None:
    model, infil_head, stage_clf, trained = model_registry.get_models()
    if not trained:
        return None
    meta = model_registry.get_model_meta()
    raw, zs, stage_logits = [], [], []
    for i in range(0, len(val_examples), batch_size):
        chunk = val_examples[i : i + batch_size]
        _, _, z = model([[graph_batch_from_state(build_graph_state(t))] for _, t in chunk])
        raw.append(infil_head(z).squeeze(-1).numpy())
        stage_logits.append(stage_clf(z).numpy())
        zs.append(z.numpy())
    raw = np.concatenate(raw)
    T = meta.get("temperature") or 1.0
    eps = 1e-6
    logits = np.log(np.clip(raw, eps, 1 - eps) / np.clip(1 - raw, eps, 1))
    return {"raw": raw, "calibrated": 1 / (1 + np.exp(-logits / T)), "stage_logits": np.concatenate(stage_logits),
            "Z": np.concatenate(zs), "meta": meta}


def _multiclass(stage_true: np.ndarray, stage_logits: np.ndarray) -> dict:
    m = stage_true >= 0
    y, pred = stage_true[m], stage_logits[m].argmax(1)
    labels = list(range(len(STAGE_NAMES)))
    per_class = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    support = np.bincount(y, minlength=len(STAGE_NAMES))
    present = [i for i in labels if support[i] > 0]
    return {
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()) if len(y) else None,
        "per_class": [{"stage": STAGE_NAMES[i], "f1": float(per_class[i]), "support": int(support[i])} for i in labels],
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "labels": STAGE_NAMES,
        "note": "Macro-F1 over stages present in the validation split. DoS/DDoS windows are excluded (MITRE 'Impact' is not one of the 6 classes).",
    }


def _predictive_auc(windows_by_file: dict, val_fraction: float, max_k: int = 3) -> dict:
    """AUC of NetJEPA's current-window score for 'attack present k windows
    later' -- genuine forecasting, not same-window detection."""
    model, infil_head, _, trained = model_registry.get_models()
    if not trained:
        return {}
    meta = model_registry.get_model_meta()
    T = meta.get("temperature") or 1.0
    out = {}
    scores_by_file = {}
    with torch.no_grad():
        for name, windows in windows_by_file.items():
            n = len(windows)
            split = int(n * (1 - val_fraction))
            val = windows[split:]
            if len(val) <= max_k:
                continue
            probs = []
            for i in range(0, len(val), 256):
                _, _, z = model([[graph_batch_from_state(build_graph_state(w))] for w in val[i : i + 256]])
                probs.append(infil_head(z).squeeze(-1).numpy())
            scores_by_file[name] = (np.concatenate(probs), np.array([infiltration_label(w) for w in val]))
    for k in range(0, max_k + 1):
        s, y = [], []
        for probs, labels in scores_by_file.values():
            s.append(probs[: len(probs) - k] if k else probs)
            y.append(labels[k:])
        s, y = np.concatenate(s), np.concatenate(y)
        out[f"t+{k}"] = float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else None
    return {"auc": out, "note": "t+0 is same-window detection; t+k scores window t against the label of window t+k."}


def _explanation_consistency(val_examples, per_label: int = 12) -> dict:
    from src.explainability.shap_explainer import explain_prediction
    from src.training.embeddings import primary_label

    model, infil_head, _, trained = model_registry.get_models()
    if not trained:
        return {"available": False}
    by_label = defaultdict(list)
    for _, t in val_examples:
        lab = primary_label(t)
        if lab != "BENIGN":
            by_label[lab].append(t)
    rng = random.Random(0)
    results = []
    for lab, ws in sorted(by_label.items()):
        if len(ws) < 3:
            continue
        sample = rng.sample(ws, min(per_label, len(ws)))
        tops = [frozenset(f["feature"] for f in explain_prediction(model, infil_head, window=w, n_features=5, n_steps=16)) for w in sample]
        pairs = [(a, b) for i, a in enumerate(tops) for b in tops[i + 1 :]]
        jac = [len(a & b) / len(a | b) for a, b in pairs if a | b]
        common = defaultdict(int)
        for t in tops:
            for f in t:
                common[f] += 1
        results.append({"attack_label": lab.replace("�", "-"), "windows": len(sample), "mean_jaccard": float(np.mean(jac)) if jac else None,
                        "most_common_features": [f for f, _ in sorted(common.items(), key=lambda kv: -kv[1])[:5]]})
    vals = [r["mean_jaccard"] for r in results if r["mean_jaccard"] is not None]
    return {"available": True, "per_attack": results, "overall_mean_jaccard": float(np.mean(vals)) if vals else None,
            "note": "Jaccard similarity of top-5 Integrated-Gradients features between windows of the same attack type (>0.7 = stable reasoning, <0.3 = unstable)."}


def _causal_by_stage(val_examples, per_stage: int = 8) -> dict:
    from src.explainability import causal_tracing

    model, infil_head, _, trained = model_registry.get_models()
    if not trained:
        return {"available": False}
    meta = model_registry.get_model_meta()
    groups = defaultdict(list)
    for _, t in val_examples:
        s = stage_label(t)
        if s is not None and s > 0:
            groups[STAGE_NAMES[s]].append(t)
    rng = random.Random(0)
    out = {}
    for stage, ws in groups.items():
        imp = defaultdict(list)
        for w in rng.sample(ws, min(per_stage, len(ws))):
            r = causal_tracing.trace(model, infil_head, build_graph_state(w), temperature=meta.get("temperature"))
            for layer in r["layers"]:
                if layer["causal_importance"] is not None:
                    imp[layer["layer"]].append(layer["causal_importance"])
        if imp:
            out[stage] = {k: float(np.mean(v)) for k, v in imp.items()}
    return {"available": bool(out), "mean_causal_importance": out,
            "note": "Mean fraction of the attack-vs-benign probability gap closed by patching each layer with a benign reference (encoder layers: current score; predictor layers: t+1 forecast)."}


def _robustness(val_examples, train_examples=None, max_windows: int = 30) -> dict:
    from src.benchmark.adversarial import robustness_report

    model, infil_head, _, trained = model_registry.get_models()
    if not trained:
        return {"available": False}
    meta = model_registry.get_model_meta()
    attacks = [build_graph_state(t) for _, t in val_examples if infiltration_label(t) == 1]
    random.Random(0).shuffle(attacks)
    mask = None
    if train_examples:
        X = np.array([build_graph_state(t)["feature_vector"] for _, t in train_examples[::10]])
        mask = torch.tensor((X.std(0) > 1e-6).astype(np.float32))
    return robustness_report(model, infil_head, attacks, meta.get("decision_threshold", 0.5), meta.get("temperature"),
                             max_windows=max_windows, feature_mask=mask)


def _run_benchmark_uncached(val_fraction: float = 0.2, full: bool = True) -> dict:
    t0 = time.time()
    windows_by_file = load_all_windows()
    if not windows_by_file or not any(windows_by_file.values()):
        return {
            "status": "windows_cache_missing",
            "message": "The training windows cache (data/processed/windows/*.pkl) hasn't been built yet. Run: python -m src.training.build_windows_cache",
            "val_size": 0, "train_size": 0, "baseline": None, "netjepa": None, "results": [],
        }
    train_examples, val_examples = build_examples(windows_by_file, context_len=1, val_fraction=val_fraction)
    if not train_examples or not val_examples:
        return {"status": "insufficient_data", "message": "Not enough windows to form train/validation splits.", "val_size": len(val_examples),
                "train_size": len(train_examples), "baseline": None, "netjepa": None, "results": []}

    baseline_metrics, _ = _evaluate_baseline(train_examples, val_examples)
    y_val = np.array([int(infiltration_label(t)) for _, t in val_examples])
    stage_val = np.array([(-1 if stage_label(t) is None else stage_label(t)) for _, t in val_examples])
    out = {"status": "ok", "val_size": len(val_examples), "train_size": len(train_examples),
           "baseline": {"model": "Logistic Regression (baseline)", **baseline_metrics}}

    nj = _netjepa_outputs(val_examples)
    if nj is None:
        netjepa = {"model": "NetJEPA", "f1": None, "precision": None, "recall": None, "fpr": None,
                   "note": "no trained checkpoint yet (data/processed/netjepa_weights.pt) -- run python -m src.training.train"}
        out.update(netjepa=netjepa, results=[out["baseline"], netjepa])
        return out

    meta = nj["meta"]
    validated = meta.get("heads_validated")
    thr = meta.get("decision_threshold", 0.5) if validated else 0.5
    scores = nj["calibrated"] if validated else nj["raw"]
    netjepa = {"model": "NetJEPA", **_binary_metrics(y_val, (scores >= thr).astype(int), scores), "decision_threshold": thr,
               "heads_validated": bool(validated), "model_sha256": meta.get("sha256")}
    if not validated:
        netjepa["note"] = "Checkpoint heads are not validated (jointly-trained head is saturated). Run python -m src.training.train_heads."
    out["netjepa"] = netjepa
    out["results"] = [out["baseline"], netjepa]
    out["multiclass"] = _multiclass(stage_val, nj["stage_logits"])
    raw_ece = expected_calibration_error(nj["raw"], y_val)
    cal_ece = expected_calibration_error(nj["calibrated"], y_val)
    out["calibration"] = {"temperature": meta.get("temperature"), "ece_raw": raw_ece["ece"], "ece_calibrated": cal_ece["ece"],
                          "reliability_raw": raw_ece["bins"], "reliability_calibrated": cal_ece["bins"]}
    out["representation"] = {"embed_std": float(nj["Z"].std(0).mean()),
                             "collapse_check": "healthy (> 0.5)" if nj["Z"].std(0).mean() > 0.5 else "possible representation collapse"}
    out["predictive"] = _predictive_auc(windows_by_file, val_fraction)
    if INTERP_REPORT.exists():
        rep = json.loads(INTERP_REPORT.read_text())
        if rep.get("model_sha256") == meta.get("sha256"):
            out["interpretability"] = {k: rep.get(k) for k in ("probes", "sae", "concept_bottleneck", "tuned_lens")}
        else:
            out["interpretability"] = {"stale": True, "note": "interpretability artefacts were trained on a different checkpoint -- rerun src.training.train_interpretability"}
    out["cross_dataset"] = {"available": False, "note": "UNSW-NB15 is not present in data/raw; cross-dataset (zero-day) evaluation needs it."}
    if full:
        for key, fn in (("explanation_consistency", lambda: _explanation_consistency(val_examples)),
                        ("causal_by_stage", lambda: _causal_by_stage(val_examples)),
                        ("robustness", lambda: _robustness(val_examples, train_examples))):
            try:
                out[key] = fn()
            except Exception as exc:  # one slow/fragile section must not sink the benchmark
                log.exception("benchmark section %s failed", key)
                out[key] = {"available": False, "error": str(exc)}
    out["computed_in_seconds"] = round(time.time() - t0, 1)
    out["computed_at"] = time.time()
    return out


def _run_benchmark_v2(val_fraction: float = 0.2, full: bool = True) -> dict:
    """Validation for NetJEPA v2: joint flow+packet windows, temporal context,
    probabilistic transitions, supervised forecasting; plus the temporal
    logistic-regression baseline, modality ablation and the leave-one-attack-
    out / context studies (src/benchmark/generalization.py)."""
    from src.benchmark.evaluate_v2 import baselines, model_outputs, summarize
    from src.benchmark.generalization import load_results
    from src.training.dataset_v2 import build_examples_v2, load_sources
    from src.training.train_v2 import GraphCache, feature_vector

    t0 = time.time()
    model, infil_head, stage_clf, trained = model_registry.get_models()
    meta = model_registry.get_model_meta()
    sources = load_sources()
    if not sources:
        return {"status": "windows_cache_missing", "message": "No training windows found (data/processed/windows_v2, data/processed/pcap_windows). "
                "Run python -m src.training.build_windows_cache and python -m src.training.build_pcap_windows <capture>.",
                "val_size": 0, "train_size": 0, "baseline": None, "netjepa": None, "results": []}
    L, H = int(meta.get("context_len") or 3), int(meta.get("horizon") or 3)
    train, val = build_examples_v2(sources, L, H, val_fraction)
    cache = GraphCache(sources)
    T, thr = meta.get("temperature"), meta.get("decision_threshold") or 0.5
    out = model_outputs(model, infil_head, stage_clf, val, lambda ex, j: cache.get(ex, j), L, H)
    rep = summarize(out, val, T, thr, H)
    base = baselines(train, val, feature_vector, context_len=L)
    y0 = np.array([ex.infil[0] for ex in val])
    raw = 1 / (1 + np.exp(-out["cur_logit"]))
    cal = 1 / (1 + np.exp(-out["cur_logit"] / (T or 1.0)))
    raw_e, cal_e = expected_calibration_error(raw, y0), expected_calibration_error(cal, y0)

    # modality ablation on joint (PCAP) validation windows
    ablation = {}
    pcap_val = [ex for ex in val if ex.source == "pcap"]
    if pcap_val and len({ex.infil[0] for ex in pcap_val}) > 1:
        yp = np.array([ex.infil[0] for ex in pcap_val])
        for cond, drop in (("flow + packet (both levels)", None), ("flow level only", "packet"), ("packet level only", "flow")):
            o = model_outputs(model, infil_head, stage_clf, pcap_val, lambda ex, j, d=drop: cache.get(ex, j, d), L, H)
            p = 1 / (1 + np.exp(-o["cur_logit"] / (T or 1.0)))
            ablation[cond] = _binary_metrics(yp, (p >= thr).astype(int), p)
    nj = {"model": "NetJEPA v2", **rep["detection"], "decision_threshold": thr, "heads_validated": bool(meta.get("heads_validated")),
          "model_sha256": meta.get("sha256")}
    bl = {"model": "Logistic Regression (baseline)", **base["logreg_current_window"]}
    tb = {"model": "Logistic Regression (temporal context)", **base["logreg_temporal_context"]}
    result = {
        "status": "ok", "model_version": 2, "val_size": len(val), "train_size": len(train), "context_len": L, "horizon": H,
        "baseline": bl, "temporal_baseline": tb, "netjepa": nj, "results": [bl, tb, nj],
        "detection_by_source": {"netjepa": rep["detection_by_source"], "baseline": bl.get("by_source"), "temporal_baseline": tb.get("by_source")},
        "forecast": rep["forecast"], "stage_forecast": rep.get("stage_forecast"), "transition_model": rep["transition_model"],
        "predictive": {"auc": {"t+0": rep["detection"].get("auc"), **{k: v["auc_forecast_head"] for k, v in rep["forecast"].items()}},
                       "note": "t+0 = detection on the current window; t+k = forecast head on the predictor's k-step rollout vs the label of window t+k."},
        "multiclass": {"macro_f1": rep.get("stage_now", {}).get("macro_f1"), "accuracy": rep.get("stage_now", {}).get("accuracy"),
                       "confusion_matrix": rep.get("stage_now", {}).get("confusion_matrix"), "labels": STAGE_NAMES,
                       "per_class": _per_class(rep.get("stage_now", {}).get("confusion_matrix")),
                       "note": "Macro-F1 over stages present in the validation split. DoS/DDoS windows are excluded (MITRE 'Impact' is not one of the 6 classes)."},
        "calibration": {"temperature": T, "ece_raw": raw_e["ece"], "ece_calibrated": cal_e["ece"], "reliability_raw": raw_e["bins"], "reliability_calibrated": cal_e["bins"]},
        "representation": {"embed_std": rep["embed_std"], "collapse_check": "healthy (> 0.5)" if rep["embed_std"] > 0.5 else "possible representation collapse"},
        "modality_ablation": ablation,
        "generalization": load_results(),
        "cross_dataset": {"available": False, "note": "Parsers for CSE-CIC-IDS-2018 and CTU-13 are included (src/ingestion/flow_parser.py); "
                          "those datasets are not in data/raw yet, so no cross-dataset score is reported. Leave-one-attack-out results above test unseen attack families."},
    }
    if INTERP_REPORT.exists():
        rep_i = json.loads(INTERP_REPORT.read_text())
        result["interpretability"] = ({k: rep_i.get(k) for k in ("probes", "sae", "concept_bottleneck", "tuned_lens")}
                                      if rep_i.get("model_sha256") == meta.get("sha256") else
                                      {"stale": True, "note": "interpretability artefacts were trained on a different checkpoint -- rerun src.training.train_interpretability"})
    if full:
        tr1, va1 = build_examples(load_all_windows(), context_len=1, val_fraction=val_fraction)
        for key, fn in (("explanation_consistency", lambda: _explanation_consistency(va1)),
                        ("causal_by_stage", lambda: _causal_by_stage_v2(val, cache, L, thr)),
                        ("robustness", lambda: _robustness_v2(val, train, cache, L))):
            try:
                result[key] = fn()
            except Exception as exc:
                log.exception("benchmark section %s failed", key)
                result[key] = {"available": False, "error": str(exc)}
    result["computed_in_seconds"] = round(time.time() - t0, 1)
    result["computed_at"] = time.time()
    return result


def _with_history(ex, cache, L):
    """(state of the current window, graphs of the L-1 windows before it) --
    the exact temporal context the deployed model scores with."""
    return build_graph_state(ex.context[-1]), [cache.get(ex, j) for j in range(L - 1)]


def _causal_by_stage_v2(val, cache, L, thr, per_stage: int = 8) -> dict:
    from src.explainability import causal_tracing

    model, infil_head, _, trained = model_registry.get_models()
    if not trained:
        return {"available": False}
    meta = model_registry.get_model_meta()
    groups = defaultdict(list)
    for ex in val:
        if ex.stage[0] > 0:
            groups[STAGE_NAMES[ex.stage[0]]].append(ex)
    rng = random.Random(0)
    out = {}
    for stage, exs in groups.items():
        imp = defaultdict(list)
        for ex in rng.sample(exs, min(per_stage, len(exs))):
            state, hist = _with_history(ex, cache, L)
            r = causal_tracing.trace(model, infil_head, state, temperature=meta.get("temperature"), history=hist)
            for layer in r["layers"]:
                if layer["causal_importance"] is not None:
                    imp[layer["layer"]].append(layer["causal_importance"])
        if imp:
            out[stage] = {k: float(np.mean(v)) for k, v in imp.items()}
    return {"available": bool(out), "mean_causal_importance": out,
            "note": "Mean fraction of the attack-vs-benign probability gap closed by patching each layer of the current window with a benign "
                    "reference, history windows held fixed (encoder layers: current score; predictor layers: t+1 forecast)."}


def _robustness_v2(val, train, cache, L, max_windows: int = 30) -> dict:
    from src.benchmark.adversarial import robustness_report

    model, infil_head, _, trained = model_registry.get_models()
    if not trained:
        return {"available": False}
    meta = model_registry.get_model_meta()
    attacks = [ex for ex in val if ex.infil[0] == 1]
    random.Random(0).shuffle(attacks)
    items = [_with_history(ex, cache, L) for ex in attacks[: max_windows * 4]]
    X = np.array([build_graph_state(ex.context[-1])["feature_vector"] for ex in train[::10]])
    mask = torch.tensor((X.std(0) > 1e-6).astype(np.float32))
    rep = robustness_report(model, infil_head, items, meta.get("decision_threshold", 0.5), meta.get("temperature"),
                            max_windows=max_windows, feature_mask=mask)
    rep["note"] = "PGD on the current window's features with its temporal context (previous windows) held fixed."
    return rep


def _per_class(cm) -> list:
    if not cm:
        return []
    cm = np.array(cm)
    out = []
    for i, name in enumerate(STAGE_NAMES):
        tp, support, predicted = cm[i, i], cm[i].sum(), cm[:, i].sum()
        p = tp / predicted if predicted else 0.0
        r = tp / support if support else 0.0
        out.append({"stage": name, "f1": float(2 * p * r / (p + r)) if (p + r) else 0.0, "support": int(support)})
    return out


def _cache_key() -> str | None:
    path = model_registry.resolve_checkpoint_path()
    if path is None:
        return None
    return f"{path}:{path.stat().st_mtime}"


def run_benchmark(val_fraction: float = 0.2, full: bool = True, refresh: bool = False) -> dict:
    """Cached on the served checkpoint (memory + disk by weights hash)."""
    key = f"{_cache_key()}:{val_fraction}:{full}"
    if not refresh and key == _cache["key"] and _cache["result"] is not None:
        return _cache["result"]
    trained = model_registry.get_models()[3]
    sha = model_registry.get_model_meta().get("sha256") if trained else None
    if sha and not refresh and DISK_CACHE.exists():
        try:
            disk = json.loads(DISK_CACHE.read_text())
            entry = disk.get(f"{sha}:{val_fraction}:{full}")
            if entry:
                _cache.update(key=key, result=entry)
                return entry
        except Exception:
            pass
    v2 = trained and (model_registry.get_model_meta().get("feature_schema_version") or 1) >= 2
    result = _run_benchmark_v2(val_fraction=val_fraction, full=full) if v2 else _run_benchmark_uncached(val_fraction=val_fraction, full=full)
    _cache.update(key=key, result=result)
    if sha and result.get("status") == "ok":
        try:
            disk = json.loads(DISK_CACHE.read_text()) if DISK_CACHE.exists() else {}
            disk[f"{sha}:{val_fraction}:{full}"] = result
            DISK_CACHE.write_text(json.dumps(disk, default=float))
        except Exception:
            log.exception("could not write benchmark disk cache")
    return result


if __name__ == "__main__":
    import sys

    r = run_benchmark(refresh="--refresh" in sys.argv)
    print(json.dumps({k: r.get(k) for k in ("status", "val_size", "baseline", "netjepa", "calibration", "predictive", "robustness", "computed_in_seconds")},
                     indent=1, default=float)[:4000])
