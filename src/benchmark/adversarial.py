"""Adversarial robustness (roadmap Task 7.2, evaluation doc Eval 4.1).

Projected Gradient Descent against the infiltration head, through the real
inference path (model_forward). Features are z-scored per capture, so a
perturbation budget epsilon is measured in standard deviations (sigma):

* min flip epsilon -- binary search for the smallest L-inf budget that lets
  PGD push a correctly detected attack window below the decision threshold;
* robust accuracy at eps = 0.1 / 0.3 / 1.0 sigma;
* minimum features to flip -- the fewest features an attacker must change
  (each by up to 3 sigma) to evade, found greedily by gradient saliency.

Caveat reported alongside the numbers: this is a feature-space attack.
Real traffic can't be edited freely -- a port scan can't lower its
distinct-port count without scanning fewer ports -- so feature-space
robustness is a conservative (attacker-favourable) bound.
"""

from __future__ import annotations

import numpy as np
import torch

from src.explainability import model_forward as mf
from src.graph.state_builder import CANONICAL_FEATURE_NAMES
from src.scoring.calibration import logit as to_logit


def _logit_threshold(threshold: float, temperature: float | None) -> float:
    # calibrated p >= thr  <=>  raw_logit / T >= logit(thr)
    return to_logit(threshold) * (temperature or 1.0)


def pgd(f, x0: torch.Tensor, eps: float, steps: int = 20, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Minimise f (a logit) within an L-inf ball of radius eps around x0."""
    mask = torch.ones_like(x0) if mask is None else mask
    step = eps / 6 if eps > 0 else 0.0
    delta = torch.zeros_like(x0, requires_grad=True)
    for _ in range(steps):
        out = f((x0 + delta * mask).unsqueeze(0))[0]
        grad, = torch.autograd.grad(out, delta)
        with torch.no_grad():
            delta -= step * grad.sign()
            delta.clamp_(-eps, eps)
    return (x0 + delta.detach() * mask)


def evaluate_window(model, infil_head, state: dict, threshold: float, temperature: float | None,
                    eps_grid=(0.1, 0.3, 1.0), max_eps: float = 5.0, feature_mask: torch.Tensor | None = None,
                    history: list | None = None) -> dict | None:
    f = mf.score_fn(model, infil_head, state, "current", history=history)
    x0 = mf.feature_tensor(state)
    thr = _logit_threshold(threshold, temperature)
    with torch.no_grad():
        base = f(x0.unsqueeze(0))[0].item()
    if base < thr:
        return None  # not detected in the first place

    def flips(eps, mask=None):
        mask = feature_mask if mask is None else (mask if feature_mask is None else mask * feature_mask)
        xa = pgd(f, x0, eps, mask=mask)
        with torch.no_grad():
            return f(xa.unsqueeze(0))[0].item() < thr, xa

    robust = {str(e): not flips(e)[0] for e in eps_grid}
    lo, hi = 0.0, max_eps
    if not flips(hi)[0]:
        min_eps = None
    else:
        for _ in range(7):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if flips(mid)[0] else (mid, hi)
        min_eps = hi

    # greedy minimum feature count at a generous 3-sigma per-feature budget
    xg = x0.clone().requires_grad_(True)
    out = f(xg.unsqueeze(0))[0]
    grad, = torch.autograd.grad(out, xg)
    if feature_mask is not None:
        grad = grad * feature_mask
    order = torch.argsort(grad.abs(), descending=True).tolist()
    min_features, features_used = None, []
    for k in (1, 2, 3, 5, 8, 12, len(order)):
        mask = torch.zeros_like(x0)
        mask[order[:k]] = 1.0
        ok, _ = flips(3.0, mask)
        if ok:
            min_features, features_used = k, [CANONICAL_FEATURE_NAMES[i] for i in order[:k]]
            break
    return {"base_logit": base, "min_flip_eps": min_eps, "robust": robust, "min_features_to_flip": min_features, "features_used": features_used}


def robustness_report(model, infil_head, states: list[dict], threshold: float, temperature: float | None, max_windows: int = 40,
                      feature_mask: torch.Tensor | None = None) -> dict:
    """feature_mask: 1 for features an attacker could plausibly move (the
    benchmark excludes features that are constant across the training data,
    e.g. packet-level fields absent from flow CSVs)."""
    results = []
    for st in states:
        # an item is a graph state, or (state, history) for temporal-context models
        st, hist = st if isinstance(st, tuple) else (st, None)
        r = evaluate_window(model, infil_head, st, threshold, temperature, feature_mask=feature_mask, history=hist)
        if r is not None:
            results.append(r)
        if len(results) >= max_windows:
            break
    if not results:
        return {"available": False, "reason": "no correctly detected attack windows to attack"}
    flips = [r["min_flip_eps"] for r in results if r["min_flip_eps"] is not None]
    feats = [r["min_features_to_flip"] for r in results if r["min_features_to_flip"] is not None]
    robust_acc = {e: float(np.mean([r["robust"][e] for r in results])) for e in results[0]["robust"]}
    from collections import Counter

    common = Counter(f for r in results for f in r["features_used"]).most_common(5)
    mean_eps = float(np.mean(flips)) if flips else None
    mean_feats = float(np.mean(feats)) if feats else None
    summary = (f"Across {len(results)} correctly detected attack windows, evading detection needed an L-inf perturbation of "
               f"{mean_eps:.2f} sigma on average" if mean_eps is not None else f"None of {len(results)} detected attack windows could be flipped within 5 sigma")
    if mean_feats is not None:
        summary += f", touching at least {mean_feats:.1f} features at up to 3 sigma each"
    summary += (". Most-targeted features: " + ", ".join(f for f, _ in common) + "." if common else ".")
    summary += " Feature-space attack: in real traffic these features are coupled to the attack itself (fewer ports scanned = less reconnaissance), so this bound favours the attacker."
    if mean_eps is not None and mean_eps < 0.5:
        summary += (" Finding: the decision boundary is close to the detected windows (small flips suffice), so this checkpoint is NOT adversarially robust in feature space. "
                    "Recommended next step: retrain the encoder with noise/adversarial augmentation and re-run this test.")
    return {"available": True, "windows_attacked": len(results), "attackable_features": int(feature_mask.sum().item()) if feature_mask is not None else len(CANONICAL_FEATURE_NAMES), "mean_min_flip_eps": mean_eps,
            "median_min_flip_eps": float(np.median(flips)) if flips else None, "unflippable_within_5_sigma": sum(r["min_flip_eps"] is None for r in results),
            "robust_accuracy": robust_acc, "mean_min_features_to_flip": mean_feats, "most_targeted_features": [f for f, _ in common],
            "summary": summary}
