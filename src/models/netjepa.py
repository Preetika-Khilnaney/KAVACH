"""NetJEPA world model: HGT spatial encoder -> GRU temporal encoder ->
K-step latent predictor, with an EMA target path for VICReg training.

Rewritten from a teammate's branch (pulled in from an unrelated git
history — see the commit that merged it) to match this project's actual
graph shape: src.graph.state_builder produces "network"/"host" node types
(state_builder.NODE_TYPES/EDGE_TYPES), not the "network"/"flow"/"external"
shape the original assumed, and a fixed-width canonical feature vector
(state_builder.CANONICAL_FEATURE_NAMES) instead of a raw per-flow matrix.


Falls back to a plain MLP over the "network" node's feature vector when
torch-geometric isn't installed — the same shape a network-only window
(no host graph — see state_builder's docstring) already degrades to, so
there's one fallback path instead of two.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.graph.state_builder import EDGE_TYPES, HAS_PYG, NODE_TYPES

if HAS_PYG:
    from torch_geometric.data import Batch
    from torch_geometric.nn import HGTConv
    from torch_geometric.utils import scatter


class SpatialEncoder(nn.Module):
    """HGT over one window's graph snapshot, mean-pooled to a single
    (hidden_dim,) embedding. Falls back to an MLP over just the "network"
    node's feature vector when torch-geometric isn't installed, or when
    no edge_index_dict is given (a network-only window with no hosts)."""

    def __init__(self, feature_dim: int, hidden_dim: int, heads: int = 4, node_types=NODE_TYPES, edge_types=EDGE_TYPES):
        super().__init__()
        self.proj = nn.ModuleDict({nt: nn.Linear(feature_dim, hidden_dim) for nt in node_types})
        self.fallback = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        if HAS_PYG:
            metadata = (list(node_types), list(edge_types))
            self.conv1 = HGTConv(hidden_dim, hidden_dim, metadata, heads=heads)
            self.conv2 = HGTConv(hidden_dim, hidden_dim, metadata, heads=heads)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_dict: dict, edge_index_dict: dict | None = None) -> torch.Tensor:
        if not HAS_PYG or edge_index_dict is None:
            return self.out(self.fallback(x_dict["network"])).mean(dim=0)

        h = {k: torch.relu(self.proj[k](v)) for k, v in x_dict.items()}
        h = self.conv1(h, edge_index_dict)
        h = {k: torch.relu(v) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        pooled = torch.stack([v.mean(dim=0) for v in h.values() if v.numel()]).mean(dim=0)
        return self.out(pooled)

    def forward_batch(self, graphs: list) -> torch.Tensor:
        """Encode many HeteroData graphs in ONE HGT call (PyG batching),
        then pool per graph exactly as forward() does (mean over each node
        type's nodes, then mean over node types). (G, hidden_dim)."""
        batch = Batch.from_data_list(graphs)
        h = {k: torch.relu(self.proj[k](v)) for k, v in batch.x_dict.items()}
        h = self.conv1(h, batch.edge_index_dict)
        h = {k: torch.relu(v) for k, v in h.items()}
        h = self.conv2(h, batch.edge_index_dict)
        per_type = [scatter(v, batch[k].batch, dim=0, dim_size=batch.num_graphs, reduce="mean") for k, v in h.items()]
        return self.out(torch.stack(per_type).mean(dim=0))


class TemporalEncoder(nn.Module):
    """GRU over a sequence of per-window spatial embeddings.
    (S, hidden_dim) -> (hidden_dim,). A length-1 sequence (the common
    inference case: "just encode the current window") still works, the
    GRU just runs one step."""

    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(seq.unsqueeze(0))  # add batch dim: (1, S, D)
        return h[-1].squeeze(0)  # (D,)


class Predictor(nn.Module):
    """Latent transition model p(z_{t+1} | z_t).

    v2 is probabilistic: a diagonal Gaussian with mean `net(z)` and
    log-variance `logvar_head(h)` (h = the last hidden layer), trained with
    Gaussian negative log-likelihood against the target encoder's z_{t+1}.
    forward() returns the mean, so deterministic callers are unchanged;
    distribution() returns (mean, logvar); sample_rollout() draws whole
    K-step trajectories for Monte-Carlo probabilities of progression.
    Checkpoints from v1 (no logvar head) load with probabilistic=False.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, probabilistic: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.probabilistic = probabilistic
        if probabilistic:
            self.logvar_head = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def distribution(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net[:4](z)
        mean = self.net[4](h)
        if not self.probabilistic:
            return mean, torch.zeros_like(mean)
        return mean, self.logvar_head(h).clamp(-8.0, 4.0)

    @torch.no_grad()
    def rollout(self, z: torch.Tensor, steps: int) -> list[torch.Tensor]:
        """Mean trajectory."""
        cur, predictions = z, []
        for _ in range(steps):
            cur = self.forward(cur)
            predictions.append(cur)
        return predictions

    @torch.no_grad()
    def sample_rollout(self, z: torch.Tensor, steps: int, n_samples: int = 64, generator: torch.Generator | None = None) -> torch.Tensor:
        """(n_samples, steps, D) sampled trajectories from p(z_{t+1} | z_t)."""
        cur = z.unsqueeze(0).expand(n_samples, -1) if z.dim() == 1 else z
        out = []
        for _ in range(steps):
            mean, logvar = self.distribution(cur)
            eps = torch.randn(mean.shape, generator=generator, device=mean.device)
            cur = mean + eps * (0.5 * logvar).exp()
            out.append(cur)
        return torch.stack(out, dim=1)


def _x_dict_and_edges(graph_batch) -> tuple[dict, dict | None]:
    """Accept either a HeteroData (from state_builder when PyG is
    installed) or a plain {"network": tensor} fallback dict (when it
    isn't) — the two shapes state_builder.build_graph_state can produce."""
    if hasattr(graph_batch, "x_dict"):
        return graph_batch.x_dict, graph_batch.edge_index_dict
    return graph_batch, None


def graph_batch_from_state(state: dict, device: str = "cpu"):
    """Adapt a src.graph.state_builder.build_graph_state(...) result into
    what this model's spatial encoder needs: the HeteroData directly when
    torch-geometric produced one, else a plain x_dict fallback — moved to
    `device` either way. The shared entry point callers (training,
    inference) should use instead of poking at state_builder's output
    directly."""
    if "graph" in state:
        return state["graph"].to(device)
    return {"network": torch.tensor([state["feature_vector"]], dtype=torch.float32, device=device)}


class _SpatioTemporalEncoder(nn.Module):
    """Bundles the spatial + temporal encoders into one unit so
    context/target paths are each a single object, matching the
    pre-existing skeleton's context_encoder/target_encoder naming."""

    def __init__(self, feature_dim: int, hidden_dim: int, heads: int, gru_layers: int):
        super().__init__()
        self.spatial = SpatialEncoder(feature_dim, hidden_dim, heads=heads)
        self.temporal = TemporalEncoder(hidden_dim, num_layers=gru_layers)

    def forward(self, window_sequence: list) -> torch.Tensor:
        """window_sequence: list of >=1 graph batches (state_builder
        output, oldest first). Returns one (hidden_dim,) embedding —
        one example, no batch dim. NetJEPA.forward loops this over a
        batch of examples and BatchNorms the stacked result; this method
        deliberately doesn't normalize on its own (nothing to normalize
        against with a single example)."""
        spatial_embeds = torch.stack([self.spatial(*_x_dict_and_edges(w)) for w in window_sequence])
        return self.temporal(spatial_embeds)

    def forward_batch(self, sequences: list[list]) -> torch.Tensor:
        """Many window sequences at once: one batched HGT call over every
        graph, then the GRU over each sequence (grouped by length).
        (N, hidden_dim). Falls back to per-sequence forward() for
        non-HeteroData inputs."""
        # per-window path when explainability hooks are attached (they expect
        # one spatial call per window) -- see explainability/model_forward.py
        if not getattr(self, "batched", True) or not HAS_PYG or not all(hasattr(g, "x_dict") for seq in sequences for g in seq):
            return torch.stack([self.forward(seq) for seq in sequences])
        flat = [g for seq in sequences for g in seq]
        emb = self.spatial.forward_batch(flat)
        out = [None] * len(sequences)
        offsets, pos = [], 0
        for seq in sequences:
            offsets.append(pos)
            pos += len(seq)
        by_len: dict[int, list[int]] = {}
        for i, seq in enumerate(sequences):
            by_len.setdefault(len(seq), []).append(i)
        for L, idxs in by_len.items():
            x = torch.stack([emb[offsets[i] : offsets[i] + L] for i in idxs])  # (n, L, D)
            _, h = self.temporal.gru(x)
            for j, i in enumerate(idxs):
                out[i] = h[-1][j]
        return torch.stack(out)


class NetJEPA(nn.Module):
    """Wraps the context encoder, EMA target encoder, predictor, and the
    BatchNorm that keeps the encoder from collapsing (see module
    docstring). Operates on a BATCH of examples, not one at a time —
    BatchNorm needs a real batch (>=2 examples) to normalize against, so
    single-example calls only make sense once the model is in eval() mode
    (running stats from training), e.g. for inference/explainability.

    Usage (training — a real batch of examples):
        context_batches = [[graph_batch_from_state(build_graph_state(w)) for w in ctx] for ctx, _ in batch]
        target_batches = [[graph_batch_from_state(build_graph_state(tgt))] for _, tgt in batch]
        z_pred, z_target, z_context = model(context_batches, target_batches)  # each (N, embedding_dim)
        loss = vicreg_loss(z_pred, z_target)  # src.models.losses

    Usage (inference, single window, model.eval() already called):
        z_pred, _, z_context = model([[graph_batch]])
        future = model.predictor.rollout(z_context[0], steps=5)
    """

    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int = 128,
        predictor_hidden_dim: int = 256,
        heads: int = 4,
        gru_layers: int = 2,
        probabilistic: bool = True,
    ):
        super().__init__()
        self.context_encoder = _SpatioTemporalEncoder(feature_dim, embedding_dim, heads, gru_layers)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = Predictor(embedding_dim, predictor_hidden_dim, probabilistic=probabilistic)
        # affine=False: pure normalization, no learnable scale/shift that
        # could re-introduce collapse (e.g. a scale trained toward 0).
        self.context_norm = nn.BatchNorm1d(embedding_dim, affine=False)
        self.target_norm = nn.BatchNorm1d(embedding_dim, affine=False)

    def forward(self, context_batches: list[list], target_batches: list[list] | None = None):
        """context_batches / target_batches: each a list of N examples,
        every example itself a list of >=1 graph batches (a window
        sequence, oldest first). Returns (z_pred, z_target, z_context),
        each (N, embedding_dim)."""
        z_context = self.context_norm(self.context_encoder.forward_batch(context_batches))
        z_target = None
        if target_batches is not None:
            with torch.no_grad():
                z_target = self.target_norm(self.target_encoder.forward_batch(target_batches))
        z_pred = self.predictor(z_context)
        return z_pred, z_target, z_context

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.996) -> None:
        """EMA update of the target encoder from the context encoder. Call
        this after each optimizer step."""
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1 - momentum)
        # context_norm/target_norm are affine=False (no learnable weight/bias
        # to EMA-update) -- each just tracks its own running mean/var from
        # the values it actually sees during forward, which is correct as-is.
