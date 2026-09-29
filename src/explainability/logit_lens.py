"""Logit / tuned lens (roadmap Task 5.4, evaluation doc Eval 3.3):
"watching the model think".

At each hook point, translate the intermediate representation into the
final embedding space and read it with the real scoring heads, as if
processing stopped there. Intermediate layers don't live in the heads'
input space, so a plain logit lens would read noise; instead we use a tuned
lens -- one affine translator per layer, fitted by ridge regression on
training windows (src/training/train_interpretability.py). The final layer's
translator is the identity, so the last row equals the real prediction.

A final "forecast" row reads the predictor's t+1 embedding with the same
heads.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.explainability import model_forward as mf
from src.scoring.attack_stage import STAGE_NAMES
from src.scoring.calibration import calibrate_probability

LENS_PATH = Path("data/processed/interpretability/tuned_lens.pt")
LAYER_LABELS = {
    "input": "Raw input features",
    "hgt_conv1": "After HGT layer 1",
    "hgt_conv2": "After HGT layer 2",
    "spatial_out": "After spatial projection",
    "gru": "After GRU (temporal)",
    "context_norm": "Final embedding (z_t)",
    "forecast": "Predictor forecast (z_t+1)",
}
_cache: dict = {}


def load_translators() -> dict | None:
    if not LENS_PATH.exists():
        return None
    mtime = LENS_PATH.stat().st_mtime
    if _cache.get("mtime") != mtime:
        _cache.update(mtime=mtime, value=torch.load(LENS_PATH, map_location="cpu", weights_only=False))
    return _cache["value"]


@torch.no_grad()
def lens(model, infil_head, stage_clf, state: dict, temperature: float | None = None, stage_temperature: float | None = None,
         history: list | None = None) -> dict:
    translators = load_translators()
    if translators is None:
        return {"available": False, "reason": "tuned lens not trained -- run python -m src.training.train_interpretability"}
    model.eval()
    x = mf.feature_tensor(state)
    with mf.capture(model) as store:
        _, _, z = model([[*(history or []), mf.graph_with_features(state, x)]])
        z_next = model.predictor(z)
    acts = {"input": x.unsqueeze(0)}
    for point, calls in store.items():
        if calls:
            call = calls[-1] if point in ("hgt_conv1", "hgt_conv2", "spatial_out") else calls[0]
            v = mf._pool(call)
            acts[point] = v if v.dim() == 2 else v.unsqueeze(0)

    rows = []
    for point in ["input", "hgt_conv1", "hgt_conv2", "spatial_out", "gru", "context_norm"]:
        if point not in translators or point not in acts:
            continue
        t = translators[point]
        zl = acts[point] @ t["W"] + t["b"]
        rows.append(_read(point, zl, infil_head, stage_clf, temperature, stage_temperature))
    rows.append(_read("forecast", z_next, infil_head, stage_clf, temperature, stage_temperature))

    # where did the decision flip / firm up?
    for prev, cur in zip(rows, rows[1:]):
        cur["delta_infiltration"] = cur["infiltration_probability"] - prev["infiltration_probability"]
        cur["stage_changed"] = cur["top_stage"] != prev["top_stage"]
    rows[0]["delta_infiltration"], rows[0]["stage_changed"] = 0.0, False
    encoder_rows = [r for r in rows if r["layer"] != "forecast"]
    decisive = max(encoder_rows[1:], key=lambda r: abs(r["delta_infiltration"]), default=None)
    return {"available": True, "layers": rows, "decisive_layer": decisive["layer"] if decisive else None,
            "narrative": _narrative(rows, decisive)}


def _read(point, z, infil_head, stage_clf, temperature, stage_temperature) -> dict:
    p = torch.sigmoid(mf.infil_logit(infil_head, z)).item()
    logits = stage_clf(z)[0] / (stage_temperature or 1.0)
    dist = torch.softmax(logits, -1)
    probs = {STAGE_NAMES[i]: float(v) for i, v in enumerate(dist)}
    top = max(probs, key=probs.get)
    entropy = float(-(dist * dist.clamp(min=1e-9).log()).sum())
    return {"layer": point, "label": LAYER_LABELS.get(point, point), "infiltration_probability": calibrate_probability(p, temperature),
            "stage_distribution": probs, "top_stage": top, "top_stage_probability": probs[top], "entropy": entropy}


def _narrative(rows, decisive) -> str:
    first, last = rows[0], [r for r in rows if r["layer"] != "forecast"][-1]
    s = (f"From raw features alone the model would say {first['infiltration_probability']:.0%} infiltration ({first['top_stage']}). "
         f"By the final embedding it says {last['infiltration_probability']:.0%} ({last['top_stage']}, {last['top_stage_probability']:.0%}).")
    if decisive is not None and abs(decisive["delta_infiltration"]) >= 0.05:
        s += f" The largest shift happens at '{decisive['label']}' ({decisive['delta_infiltration']:+.0%})."
    fc = rows[-1]
    if fc["layer"] == "forecast":
        s += f" The predictor forecasts {fc['infiltration_probability']:.0%} for the next window ({fc['top_stage']})."
    return s
