"""Activation patching / causal tracing (roadmap Task 5.2, evaluation doc
Eval 3.2).

Integrated Gradients says which *inputs* mattered; patching says which
*internal computations* were causally responsible. For a flagged window:

1. run it through NetJEPA, recording every hook point's activation;
2. run a clean benign window through the *same graph structure*, recording
   the same activations;
3. NetJEPA is a serial stack, so overwriting a *whole* layer with the clean
   activation trivially reproduces the clean output (every layer would score
   1.0). Instead each layer is split into 8 contiguous subspaces (and, for
   the graph layers, into its "network" vs "host" node rows) and each piece
   is patched on its own;
4. causal importance of a layer = the largest share of the flagged-vs-clean
   probability gap that any single piece of it closes. High = the decision
   is concentrated in a small, identifiable part of that layer; low = the
   layer only carries it diffusely. `node_type_effect` shows whether the
   graph layers' effect runs through the window-level node or the host
   (topology) nodes.

Encoder points (HGT, GRU, BatchNorm) are scored on the current-window
probability. Predictor points only affect the *forecast*, so they are
scored on the predictor's t+1 probability instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.explainability import model_forward as mf
from src.scoring.calibration import calibrate_probability

BENIGN_REF_PATH = Path("data/processed/interpretability/benign_reference.json")


def benign_reference_vector(dim: int) -> torch.Tensor:
    if BENIGN_REF_PATH.exists():
        v = json.loads(BENIGN_REF_PATH.read_text())["feature_vector"]
        if len(v) == dim:
            return torch.tensor(v, dtype=torch.float32)
    return torch.zeros(dim)


@torch.no_grad()
def _run(model, infil_head, state, x, temperature, history=None):
    g = mf.graph_with_features(state, x)
    _, _, z = model([[*(history or []), g]])
    cur = torch.sigmoid(mf.infil_logit(infil_head, z)).item()
    fut = torch.sigmoid(mf.infil_logit(infil_head, model.predictor(z))).item()
    return calibrate_probability(cur, temperature), calibrate_probability(fut, temperature)


@torch.no_grad()
def trace(model, infil_head, state: dict, clean_vector: torch.Tensor | None = None, temperature: float | None = None,
          history: list | None = None) -> dict:
    model.eval()
    x_mal = mf.feature_tensor(state)
    x_clean = clean_vector if clean_vector is not None else benign_reference_vector(len(x_mal))

    with mf.capture(model) as mal_store:
        p_mal, f_mal = _run(model, infil_head, state, x_mal, temperature, history)
    with mf.capture(model) as clean_store:
        p_clean, f_clean = _run(model, infil_head, state, x_clean, temperature, history)

    gap_cur, gap_fut = p_mal - p_clean, f_mal - f_clean
    layers = []
    for point in mf.HOOK_POINTS:
        if not clean_store.get(point):
            continue
        # spatial hooks fire once per window of the sequence -> the current window is the last call
        clean_act = clean_store[point][-1] if point in ("hgt_conv1", "hgt_conv2", "spatial_out") else clean_store[point][0]
        is_pred = point in mf.PREDICTOR_POINTS
        gap = gap_fut if is_pred else gap_cur
        base = f_mal if is_pred else p_mal
        dim = (next(iter(clean_act.values())) if isinstance(clean_act, dict) else clean_act[1] if isinstance(clean_act, tuple) else clean_act).shape[-1]
        n_blocks = 8
        size = max(1, dim // n_blocks)
        block_effects = []
        for b in range(n_blocks):
            sl = slice(b * size, dim if b == n_blocks - 1 else (b + 1) * size)
            with mf.patch_partial(model, point, clean_act, sl):
                p_patch, f_patch = _run(model, infil_head, state, x_mal, temperature, history)
            patched = f_patch if is_pred else p_patch
            block_effects.append(None if abs(gap) < 0.02 else (base - patched) / gap)
        node_effect = None
        if isinstance(clean_act, dict) and abs(gap) >= 0.02:
            node_effect = {}
            for nt in clean_act:
                with mf.patch_partial(model, point, clean_act, slice(0, dim), node_type=nt):
                    p_patch, _ = _run(model, infil_head, state, x_mal, temperature, history)
                node_effect[nt] = (base - p_patch) / gap
        valid = [e for e in block_effects if e is not None]
        importance = max(0.0, min(1.0, max(valid))) if valid else None
        layers.append({
            "layer": point,
            "component": "predictor" if is_pred else ("graph_encoder" if point.startswith("hgt") or point == "spatial_out" else "temporal_encoder" if point == "gru" else "normalization"),
            "scored_on": "forecast_t+1" if is_pred else "current", "prediction_before": base,
            "causal_importance": importance, "block_effects": block_effects,
            "top_block": int(max(range(len(valid)), key=lambda i: valid[i])) if valid else None,
            "node_type_effect": node_effect,
        })
    ranked = sorted((l for l in layers if l["causal_importance"] is not None), key=lambda l: -l["causal_importance"])
    note = None
    if abs(gap_cur) < 0.02:
        note = "The window and the benign reference score almost the same (< 2 points apart), so there is no decision to attribute to a layer."
    return {"malicious_probability": p_mal, "clean_probability": p_clean, "malicious_forecast": f_mal, "clean_forecast": f_clean,
            "layers": layers, "critical_layers": [l["layer"] for l in ranked[:2]], "note": note,
            "method": "subspace activation patching against a benign reference window with the same graph structure"}
