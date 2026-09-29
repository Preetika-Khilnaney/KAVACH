"""Shared, differentiable access to NetJEPA's exact inference path.

Explainability (Integrated Gradients), adversarial testing, causal tracing,
the tuned lens and probes all need to (a) vary a window's feature vector
while keeping everything else about the forward pass identical to live
inference, and (b) read or overwrite intermediate activations.

Why this module exists: the previous IG code fed `{"network": x}` into the
model, which takes the SpatialEncoder's *fallback MLP* branch (used only when
there is no edge_index_dict). Live inference always builds a HeteroData graph
and runs the HGT branch, so the old attributions were computed through
weights that are never used for real predictions. Here every call clones the
window's real graph and swaps in the new feature rows, so gradients flow
through the same HGT -> GRU -> BatchNorm -> head path as the pipeline.
"""

from __future__ import annotations

import contextlib
from typing import Callable

import torch

from src.graph.state_builder import build_graph_state
from src.models.netjepa import graph_batch_from_state

# Named hook points, in forward order (evaluation doc Eval 3.2/3.3).
HOOK_POINTS = [
    "hgt_conv1",
    "hgt_conv2",
    "spatial_out",
    "gru",
    "context_norm",
    "predictor_l1",
    "predictor_l2",
    "predictor_l3",
]
ENCODER_POINTS = HOOK_POINTS[:5]
PREDICTOR_POINTS = HOOK_POINTS[5:]


def hook_modules(model) -> dict:
    enc = model.context_encoder
    return {
        "hgt_conv1": getattr(enc.spatial, "conv1", None),
        "hgt_conv2": getattr(enc.spatial, "conv2", None),
        "spatial_out": enc.spatial.out,
        "gru": enc.temporal.gru,
        "context_norm": model.context_norm,
        "predictor_l1": model.predictor.net[1],
        "predictor_l2": model.predictor.net[3],
        "predictor_l3": model.predictor.net[4],
    }


def infil_logit(infil_head, z: torch.Tensor) -> torch.Tensor:
    """Pre-sigmoid infiltration logit (InfiltrationHead ends in Sigmoid)."""
    if z.dim() == 1:
        z = z.unsqueeze(0)
    return infil_head.net[:-1](z).squeeze(-1)


def graph_with_features(state: dict, x_row: torch.Tensor):
    """Clone a window's graph and replace its node features with x_row
    (host nodes reuse the window vector, exactly as state_builder does)."""
    if "graph" not in state:
        return {"network": x_row.unsqueeze(0)}
    g = state["graph"].clone()
    g["network"].x = x_row.unsqueeze(0)
    n_hosts = g["host"].x.shape[0]
    has_hosts = g["network", "has", "host"].edge_index.shape[1] > 0
    g["host"].x = x_row.unsqueeze(0).repeat(n_hosts, 1) if has_hosts else torch.zeros_like(g["host"].x)
    return g


def context_embeddings(model, X: torch.Tensor, state: dict, history: list | None = None) -> torch.Tensor:
    """(B, F) feature rows -> (B, D) z_context, through the real path.
    `history` optionally prepends earlier windows' graph batches (fixed) for
    a temporal context longer than one window."""
    history = history or []
    seqs = [[*history, graph_with_features(state, X[i])] for i in range(X.shape[0])]
    _, _, z_context = model(seqs)
    return z_context


def feature_tensor(state: dict) -> torch.Tensor:
    return torch.tensor(state["feature_vector"], dtype=torch.float32)


def score_fn(model, infil_head, state: dict, target: str = "current", history: list | None = None) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns f(X: (B, F)) -> (B,) infiltration *logits*; target "current"
    scores z_t, "forecast" scores the predictor's z_{t+1}."""

    def f(X: torch.Tensor) -> torch.Tensor:
        z = context_embeddings(model, X, state, history)
        if target == "forecast":
            z = model.predictor(z)
        return infil_logit(infil_head, z)

    return f


def state_for_window(window: dict) -> dict:
    return build_graph_state(window)


def graph_batch(window_or_state: dict):
    state = window_or_state if "feature_vector" in window_or_state else build_graph_state(window_or_state)
    return graph_batch_from_state(state)


def _pool(value) -> torch.Tensor:
    """Reduce any hooked activation to one vector per example, the same way
    the model pools (mean over nodes, then over node types)."""
    if isinstance(value, dict):
        vs = [v.mean(dim=0) for v in value.values() if v is not None and v.numel()]
        return torch.stack(vs).mean(dim=0)
    if isinstance(value, tuple):  # GRU -> (output, h_n): use last layer's h_n
        return value[1][-1].squeeze(0)
    return value


@contextlib.contextmanager
def per_window(model):
    """Hooks below assume one spatial-encoder call per window; the batched
    encoder makes one call per batch, so switch it off while hooked."""
    enc = model.context_encoder
    prev = getattr(enc, "batched", True)
    enc.batched = False
    try:
        yield
    finally:
        enc.batched = prev


@contextlib.contextmanager
def capture(model, points: list[str] | None = None):
    """Record every hooked activation for one forward call. Yields a dict
    name -> list of raw outputs (one per module call; a batch of B
    single-window examples calls spatial/GRU hooks B times)."""
    store: dict[str, list] = {p: [] for p in (points or HOOK_POINTS)}
    handles = []
    for name, module in hook_modules(model).items():
        if module is None or name not in store:
            continue

        def hook(_m, _inp, out, _name=name):
            store[_name].append(out)

        handles.append(module.register_forward_hook(hook))
    try:
        with per_window(model):
            yield store
    finally:
        for h in handles:
            h.remove()


def pooled_activations(store: dict) -> dict:
    """name -> (B, dim) tensor of pooled activations from a capture()."""
    out = {}
    for name, calls in store.items():
        if not calls:
            continue
        if len(calls) == 1 and isinstance(calls[0], torch.Tensor) and calls[0].dim() == 2:
            out[name] = calls[0].detach()  # batch-level module (BatchNorm / predictor)
        else:
            out[name] = torch.stack([_pool(c) for c in calls]).detach()  # per-example modules
    return out


def _detach(value):
    if isinstance(value, dict):
        return {k: v.detach().clone() for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach(v) for v in value)
    return value.detach().clone()


@contextlib.contextmanager
def patch(model, point: str, replacement):
    """Overwrite one hook point's output with `replacement` (activation
    patching). `replacement` must come from a capture() on an input with the
    same graph structure."""
    module = hook_modules(model)[point]
    rep = _detach(replacement)
    handle = module.register_forward_hook(lambda _m, _i, _o: rep)
    try:
        with per_window(model):
            yield
    finally:
        handle.remove()


def _mix(mal, clean, dims: slice, node_type: str | None = None):
    if isinstance(mal, dict):
        out = {}
        for k, v in mal.items():
            if (node_type is None or k == node_type) and k in clean and clean[k].shape == v.shape:
                v = v.clone()
                v[..., dims] = clean[k][..., dims]
            out[k] = v
        return out
    if isinstance(mal, tuple):
        return tuple(_mix(m, c, dims, node_type) for m, c in zip(mal, clean))
    v = mal.clone()
    v[..., dims] = clean[..., dims]
    return v


@contextlib.contextmanager
def patch_partial(model, point: str, replacement, dims: slice, node_type: str | None = None):
    """Overwrite only `dims` of one hook point's output (and optionally only
    one node type's rows) with the clean activation. In a serial network,
    patching a whole layer trivially reproduces the clean output, so causal
    tracing patches subspaces instead."""
    module = hook_modules(model)[point]
    rep = _detach(replacement)

    def hook(_m, _i, out):
        # history windows have different node sets; only patch calls whose
        # structure matches the replacement (the current window)
        if isinstance(out, dict) and isinstance(rep, dict):
            if any(k not in rep or rep[k].shape != v.shape for k, v in out.items()):
                return out
        return _mix(out, rep, dims, node_type)

    handle = module.register_forward_hook(hook)
    try:
        with per_window(model):
            yield
    finally:
        handle.remove()
