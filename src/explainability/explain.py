"""Unified deep-explainability service behind GET /explain/{window_id}
(roadmap Task 5.5). Methods:

  ig          Integrated Gradients (stored from the pipeline, or computed now)
  heuristic   fast z-score attribution (Task 8.4)
  sae         top sparse-autoencoder features with their labels
  causal      activation patching vs a benign reference window
  logit_lens  per-layer belief progression (tuned lens)
  probes      layer x concept probe matrix (pre-computed)
  concepts    concept-bottleneck reasoning (supports analyst overrides)
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.explainability import causal_tracing, logit_lens, probing
from src.explainability.shap_explainer import explain_prediction, heuristic_attribution
from src.explainability.sparse_autoencoder import load_sae
from src.graph.state_builder import CANONICAL_FEATURE_NAMES, build_graph_state
from src.models.concept_bottleneck import ConceptBottleneck
from src.models.netjepa import graph_batch_from_state
from src.orchestrator import model_registry

CB_PATH = Path("data/processed/interpretability/concept_bottleneck.pt")
ALL_METHODS = ["ig", "heuristic", "sae", "causal", "logit_lens", "probes", "concepts"]
_cb_cache: dict = {}


def load_concept_bottleneck() -> ConceptBottleneck | None:
    if not CB_PATH.exists():
        return None
    mtime = CB_PATH.stat().st_mtime
    if _cb_cache.get("mtime") != mtime:
        ckpt = torch.load(CB_PATH, map_location="cpu", weights_only=False)
        cb = ConceptBottleneck(ckpt["embedding_dim"])
        cb.load_state_dict(ckpt["state_dict"])
        cb.eval()
        _cb_cache.update(mtime=mtime, value=cb)
    return _cb_cache["value"]


def _window_from_row(row: dict) -> dict:
    window = dict(row.get("window") or {})
    # packet_edges round-trips through JSON as lists; state_builder expects pairs
    if window.get("packet_edges"):
        window["packet_edges"] = [tuple(e) for e in window["packet_edges"]]
    return window


def explain_window(row: dict, methods: list[str], overrides: dict | None = None) -> dict:
    model, infil_head, stage_clf, trained = model_registry.get_models()
    meta = model_registry.get_model_meta()
    window = _window_from_row(row)
    state = build_graph_state(window)
    L = int(meta.get("context_len") or 1)
    from src.storage import results_store

    history = [graph_batch_from_state(build_graph_state(_window_from_row(r))) for r in results_store.previous_rows(row, L - 1)] if L > 1 else []
    with torch.no_grad():
        _, _, z = model([[*history, graph_batch_from_state(state)]])
    methods = ALL_METHODS if "all" in methods else [m for m in methods if m in ALL_METHODS]
    out: dict = {"window_id": row["window_id"], "trained": trained, "model_sha256": meta.get("sha256"),
                 "scoring_method": row.get("scoring_method"), "infiltration_probability": row["infiltration_probability"],
                 "attack_stage": row["attack_stage"], "methods": methods}
    if row.get("scoring_method") == "packet_heuristic":
        out["caveat"] = ("This window was scored by the packet heuristic (packet-only capture, outside the flow-feature distribution NetJEPA "
                         "was trained on). Model-internal explanations below describe what NetJEPA computes for it, not the score shown.")

    if "ig" in methods:
        stored = row.get("top_features") or []
        if stored and stored[0].get("method") == "integrated_gradients":
            out["ig"] = {"source": "stored", "features": stored}
        elif trained:
            out["ig"] = {"source": "computed", "features": explain_prediction(model, infil_head, state=state, n_features=10, history=history)}
        else:
            out["ig"] = {"source": "unavailable", "features": []}
    if "heuristic" in methods:
        out["heuristic"] = heuristic_attribution(state["feature_vector"], n_features=10, update=False)
    if "sae" in methods:
        sae, labels = load_sae()
        if sae is None:
            out["sae"] = {"available": False, "reason": "run python -m src.training.train_interpretability"}
        else:
            feats = []
            for fid, act in sae.top_features(z[0], k=8):
                lab = labels.get(str(fid), {})
                feats.append({"feature_id": fid, "activation": act, "label": lab.get("label", f"SAE feature #{fid} (unlabelled)"),
                              "dominant_label": lab.get("dominant_label"), "purity": lab.get("purity"), "top_feature": lab.get("top_feature"),
                              "activation_frequency": lab.get("activation_frequency")})
            out["sae"] = {"available": True, "features": feats, "dictionary_size": sae.hidden_dim, "k": sae.k}
    if "causal" in methods:
        out["causal"] = causal_tracing.trace(model, infil_head, state, temperature=meta.get("temperature"), history=history) if trained else {"available": False}
    if "logit_lens" in methods:
        out["logit_lens"] = logit_lens.lens(model, infil_head, stage_clf, state, meta.get("temperature"), meta.get("stage_temperature"), history=history)
    if "probes" in methods:
        out["probes"] = probing.load_probe_report()
    if "concepts" in methods:
        cb = load_concept_bottleneck()
        out["concepts"] = cb.predict(z[0], overrides=overrides) if cb is not None else {"available": False, "reason": "run python -m src.training.train_interpretability"}
    out["context_windows"] = len(history) + 1
    out["feature_vector"] = dict(zip(CANONICAL_FEATURE_NAMES, state["feature_vector"]))
    return out
