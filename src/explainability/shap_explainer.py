"""Explainability: per-feature attribution for a NetJEPA-derived prediction,
tied back to the window's named feature vector
(src.graph.state_builder.CANONICAL_FEATURE_NAMES).

Two methods, and every result is tagged with which one produced it:

* "integrated_gradients" -- Captum IntegratedGradients w.r.t. the window's
  feature vector, relative to a benign reference window (see
  causal_tracing.benign_reference_vector), through the *real* inference path (HeteroData graph ->
  HGT -> GRU -> BatchNorm -> head; see src/explainability/model_forward.py),
  explaining the infiltration head's pre-sigmoid logit. Used for windows
  that look interesting (the pipeline's threshold) because it costs a few
  dozen forward passes.
* "heuristic" (roadmap Task 8.4) -- a fast, backward-pass-free fallback for
  every other window: z-score of each feature against a running baseline of
  previously seen windows, weighted by the feature's magnitude. Cheap enough
  to run on every window, so no window ends up with an empty explanation.

Falls back to gradient x input when Captum isn't installed.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import torch
import torch.nn as nn

from src.graph.state_builder import CANONICAL_FEATURE_NAMES, build_graph_state


def _finite(v: float) -> float:
    return float(v) if math.isfinite(float(v)) else 0.0


def explain_prediction(
    model,
    scoring_head: nn.Module,
    feature_vector=None,
    feature_names: list[str] | None = None,
    target_index: int = 0,
    n_features: int = 10,
    state: dict | None = None,
    window: dict | None = None,
    n_steps: int = 24,
    baseline=None,
    history: list | None = None,
) -> list[dict]:
    """Per-feature attribution scores for one window's prediction.

    Pass `window` (a feature-window dict) or `state` (its build_graph_state
    output) so attribution runs through the window's real graph; a bare
    `feature_vector` is wrapped in a host-less graph. `scoring_head` is an
    InfiltrationHead (explained on its logit) or ATTACKStageClassifier
    (`target_index` = stage class).

    Returns [{"feature", "attribution", "value", "method"}] sorted by
    |attribution|, at most n_features long.
    """
    from src.explainability.model_forward import context_embeddings, infil_logit
    from src.scoring.infiltration import InfiltrationHead

    model.eval()
    scoring_head.eval()
    if state is None:
        if window is not None:
            state = build_graph_state(window)
        elif len(feature_vector) == len(CANONICAL_FEATURE_NAMES):
            state = build_graph_state(dict(zip(CANONICAL_FEATURE_NAMES, [float(v) for v in feature_vector])))
        else:  # non-canonical width (custom models/tests): host-less graph
            from src.graph.state_builder import HAS_PYG, _to_hetero_data

            fv = [float(v) for v in feature_vector]
            state = {"feature_vector": fv, "feature_names": feature_names}
            if HAS_PYG:
                state["graph"] = _to_hetero_data(fv, [], [], {})
    x = torch.tensor(state["feature_vector"], dtype=torch.float32)
    feature_names = feature_names or list(state.get("feature_names") or CANONICAL_FEATURE_NAMES)
    is_infil = isinstance(scoring_head, InfiltrationHead)

    def forward_fn(x_input: torch.Tensor) -> torch.Tensor:
        z = context_embeddings(model, x_input, state, history=history)
        if is_infil:
            return infil_logit(scoring_head, z).unsqueeze(-1)
        return scoring_head(z)

    if baseline is None:
        # Attribute relative to typical benign traffic, not to an all-zeros
        # vector: several canonical features are constant in the training
        # data (e.g. flow_count = 100 rows per window), and a zero baseline
        # would hand them large, meaningless attributions.
        from src.explainability.causal_tracing import benign_reference_vector

        base = benign_reference_vector(len(x))
    else:
        base = torch.as_tensor(baseline, dtype=torch.float32)
    method = "integrated_gradients"
    try:
        from captum.attr import IntegratedGradients

        ig = IntegratedGradients(forward_fn)
        attrs = ig.attribute(x.unsqueeze(0), baselines=base.unsqueeze(0), target=0 if is_infil else target_index, n_steps=n_steps)
        attrs = attrs.squeeze(0).detach().numpy()
    except ImportError:
        method = "gradient_x_input"
        xg = x.clone().requires_grad_(True)
        out = forward_fn(xg.unsqueeze(0)).flatten()
        out[0 if is_infil else target_index].backward()
        attrs = (xg.grad * (xg - base)).detach().numpy() if xg.grad is not None else np.zeros(len(x))

    ranked = sorted(
        ({"feature": feature_names[i] if i < len(feature_names) else f"feature_{i}", "attribution": _finite(attrs[i]),
          "value": _finite(x[i].item()), "method": method} for i in range(len(attrs))),
        key=lambda d: abs(d["attribution"]),
        reverse=True,
    )
    return ranked[:n_features]


class RunningBaseline:
    """Welford running mean/variance over every window's feature vector the
    process has seen -- the reference point for heuristic attribution."""

    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim)
        self.m2 = np.zeros(dim)
        self._lock = threading.Lock()

    def update(self, x: np.ndarray) -> None:
        with self._lock:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (x - self.mean)

    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(self.m2 / (self.n - 1))


_BASELINE = RunningBaseline(len(CANONICAL_FEATURE_NAMES))


def heuristic_attribution(feature_vector, feature_names: list[str] | None = None, n_features: int = 10,
                          baseline: RunningBaseline | None = None, update: bool = True) -> list[dict]:
    """Fast attribution (Task 8.4): importance_i = z_i * |x_i| where z_i is
    the feature's deviation from the running baseline. Signed: positive
    means the feature sits above its usual level (pushes risk up)."""
    baseline = baseline or _BASELINE
    x = np.nan_to_num(np.asarray(feature_vector, dtype=float))
    names = feature_names or list(CANONICAL_FEATURE_NAMES)
    if baseline.n >= 2:
        z = (x - baseline.mean) / np.maximum(baseline.std(), 1e-6)
    else:
        z = x  # windows are already z-scored per capture by src.features.extract
    importance = z * np.maximum(np.abs(x), 1.0)
    if update:
        baseline.update(x)
    ranked = sorted(
        ({"feature": names[i], "attribution": _finite(importance[i]), "value": _finite(x[i]), "z_score": _finite(z[i]), "method": "heuristic"}
         for i in range(len(x))),
        key=lambda d: abs(d["attribution"]),
        reverse=True,
    )
    return ranked[:n_features]
