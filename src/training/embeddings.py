"""Batch-encode cached training windows through a (frozen) NetJEPA, the
same single-window way the live pipeline and the benchmark do. Shared by
train_heads.py, train_interpretability.py and the extended benchmark."""

from __future__ import annotations

import numpy as np
import torch

from src.graph.state_builder import build_graph_state
from src.models.netjepa import graph_batch_from_state
from src.training.labels import infiltration_label, stage_label


def primary_label(window: dict) -> str:
    """Most informative label present in the window: the first non-benign
    label if any, else BENIGN."""
    labels = [l for l in (window.get("flow_labels_present") or []) if l != "BENIGN"]
    return labels[0] if labels else "BENIGN"


@torch.no_grad()
def embed_windows(model, windows: list[dict], batch_size: int = 256, with_forecast: bool = False) -> dict:
    Z, ZP, X, y, s, lab = [], [], [], [], [], []
    for i in range(0, len(windows), batch_size):
        chunk = windows[i : i + batch_size]
        states = [build_graph_state(w) for w in chunk]
        z_pred, _, z = model([[graph_batch_from_state(st)] for st in states])
        Z.append(z.numpy())
        if with_forecast:
            ZP.append(z_pred.numpy())
        X += [st["feature_vector"] for st in states]
        y += [infiltration_label(w) for w in chunk]
        s += [(-1 if stage_label(w) is None else stage_label(w)) for w in chunk]
        lab += [primary_label(w) for w in chunk]
    out = {
        "Z": np.concatenate(Z) if Z else np.zeros((0, 0)),
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "stage": np.asarray(s, dtype=np.int64),
        "label": np.asarray(lab),
    }
    if with_forecast:
        out["Z_forecast"] = np.concatenate(ZP) if ZP else np.zeros((0, 0))
    return out
