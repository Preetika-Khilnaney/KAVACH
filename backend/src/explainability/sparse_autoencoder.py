"""Sparse autoencoder over NetJEPA's context embedding (roadmap Task 5.1,
evaluation doc Eval 3.1).

Embedding dimensions are polysemantic -- one dimension responds to many
unrelated patterns. An overcomplete SAE (128 -> 4096 -> 128) with a
TopK activation (only the k largest codes survive per input, Gao et al. 2024
"Scaling and evaluating sparse autoencoders" -- sparsity is guaranteed
rather than hoped for from an L1 penalty, which on these BatchNorm'd
embeddings left ~50% of codes weakly active) re-expresses each embedding as a handful of
sparse features that tend to be monosemantic. After training, each live
feature is auto-labelled from the windows that activate it most strongly
(dominant CIC-IDS-2017 label + the raw feature most correlated with it); the
labels are stored in data/processed/interpretability/sae_labels.json and can
be edited by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

SAE_PATH = Path("data/processed/interpretability/sae_weights.pt")
LABELS_PATH = Path("data/processed/interpretability/sae_labels.json")


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 4096, k: int = 32):
        super().__init__()
        self.input_dim, self.hidden_dim, self.k = input_dim, hidden_dim, k
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encoder(x - self.pre_bias)
        if self.k and self.k < pre.shape[-1]:
            vals, idx = pre.topk(self.k, dim=-1)
            return torch.zeros_like(pre).scatter(-1, idx, torch.relu(vals))
        return torch.relu(pre)

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        return self.decoder(h) + self.pre_bias, h

    @staticmethod
    def loss(x, x_hat, h, l1_coeff: float) -> tuple[torch.Tensor, dict]:
        recon = ((x - x_hat) ** 2).mean()
        sparsity = h.abs().mean()
        return recon + l1_coeff * sparsity, {"mse": float(recon.detach()), "l1": float(sparsity.detach())}

    @torch.no_grad()
    def normalize_decoder(self):
        """Unit-norm decoder columns so L1 can't be dodged by shrinking codes."""
        self.decoder.weight.data /= self.decoder.weight.data.norm(dim=0, keepdim=True).clamp(min=1e-8)

    @torch.no_grad()
    def top_features(self, z: torch.Tensor, k: int = 8) -> list[tuple[int, float]]:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        h = self.encode(z)[0]
        vals, idx = h.topk(min(k, h.numel()))
        return [(int(i), float(v)) for i, v in zip(idx, vals) if v > 0]

    @staticmethod
    @torch.no_grad()
    def top_activating_examples(h_all: torch.Tensor, feature: int, n: int = 10) -> list[int]:
        col = h_all[:, feature]
        vals, idx = col.topk(min(n, len(col)))
        return [int(i) for i, v in zip(idx, vals) if v > 0]


_loaded: dict = {}


def load_sae() -> tuple[SparseAutoencoder | None, dict]:
    """(sae, labels) or (None, {}) if not trained yet. Cached on mtime."""
    if not SAE_PATH.exists():
        return None, {}
    mtime = SAE_PATH.stat().st_mtime
    if _loaded.get("mtime") == mtime:
        return _loaded["sae"], _loaded["labels"]
    ckpt = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
    sae = SparseAutoencoder(ckpt["input_dim"], ckpt["hidden_dim"], ckpt.get("k", 0))
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    labels = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}
    _loaded.update(mtime=mtime, sae=sae, labels=labels)
    return sae, labels
