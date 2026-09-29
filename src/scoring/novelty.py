"""Zero-day novelty score: how far is this window's world-model embedding
from everything the model has seen as benign?

The supervised infiltration head only knows the attack families in its
training labels; the leave-one-attack-out benchmark shows it misses most of
a family it never saw (e.g. PortScan AUC ~0.58 when held out). A world
model gives a second, label-free signal: benign traffic occupies a compact
region of the latent space, and a genuinely new behaviour lands outside it.
The score is the mean Euclidean distance to the k nearest benign reference
embeddings, expressed as a percentile of the benign reference's own
(leave-one-out) scores, so 0.99 means "further out than 99% of benign".

Status: evaluated, NOT deployed. On CIC-IDS-2017 the leave-one-attack-out
runs show this latent-space novelty score is ~chance for unseen families
(PortScan held out: AUC 0.50; raw-feature kNN novelty 0.46), because unseen
attack windows are not low-density relative to benign traffic -- they're
regular, machine-like and often *more* predictable than benign (world-model
surprise AUC 0.28, i.e. inverted). It is reported by train_v2 /
generalization as a diagnostic so this stays measured, not assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

DEFAULT_K = 10
MAX_REFERENCE = 3000


class NoveltyDetector:
    def __init__(self, reference: np.ndarray, k: int = DEFAULT_K, benign_scores: np.ndarray | None = None,
                 threshold_quantile: float = 0.99):
        self.reference = np.asarray(reference, dtype=np.float32)
        self.k = int(min(k, max(1, len(self.reference) - 1)))
        self._ref_t = torch.from_numpy(self.reference)
        if benign_scores is None:
            benign_scores = self._raw(self.reference, exclude_self=True)
        self.benign_scores = np.sort(np.asarray(benign_scores, dtype=np.float32))
        self.threshold_quantile = threshold_quantile
        self.threshold = float(np.quantile(self.benign_scores, threshold_quantile))

    @classmethod
    def fit(cls, benign_embeddings: np.ndarray, k: int = DEFAULT_K, seed: int = 0, **kw) -> "NoveltyDetector":
        Z = np.asarray(benign_embeddings, dtype=np.float32)
        if len(Z) > MAX_REFERENCE:
            Z = Z[np.random.default_rng(seed).choice(len(Z), MAX_REFERENCE, replace=False)]
        return cls(Z, k=k, **kw)

    def _raw(self, Z: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        Zt = torch.as_tensor(np.asarray(Z, dtype=np.float32))
        if Zt.dim() == 1:
            Zt = Zt.unsqueeze(0)
        out = []
        for chunk in Zt.split(1024):
            d = torch.cdist(chunk, self._ref_t)
            kk = self.k + 1 if exclude_self else self.k
            nearest = d.topk(kk, largest=False).values
            if exclude_self:
                nearest = nearest[:, 1:]
            out.append(nearest.mean(dim=1))
        return torch.cat(out).numpy()

    def score(self, Z) -> np.ndarray:
        """Raw kNN distance (higher = more novel)."""
        if isinstance(Z, torch.Tensor):
            Z = Z.detach().cpu().numpy()
        return self._raw(Z)

    def percentile(self, raw: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.benign_scores, np.asarray(raw), side="right") / max(1, len(self.benign_scores))

    def assess(self, z) -> dict:
        raw = float(self.score(z)[0])
        pct = float(self.percentile(np.array([raw]))[0])
        return {"score": raw, "benign_percentile": pct, "threshold": self.threshold,
                "is_novel": raw > self.threshold, "k": self.k, "reference_size": int(len(self.reference))}

    def state(self) -> dict:
        return {"reference": torch.from_numpy(self.reference.astype(np.float16)), "k": self.k,
                "benign_scores": torch.from_numpy(self.benign_scores), "threshold_quantile": self.threshold_quantile}

    @classmethod
    def from_state(cls, st: dict) -> "NoveltyDetector":
        return cls(st["reference"].float().numpy(), k=int(st["k"]), benign_scores=st["benign_scores"].numpy(),
                   threshold_quantile=float(st.get("threshold_quantile", 0.99)))


def sidecar_path(checkpoint: Path, sha256: str) -> Path:
    return Path(checkpoint).with_name(f"novelty_{sha256[:8]}.pt")


def save(det: NoveltyDetector, checkpoint: Path, sha256: str) -> Path:
    p = sidecar_path(checkpoint, sha256)
    torch.save({**det.state(), "model_sha256": sha256}, p)
    return p


def load(checkpoint: Path | None, sha256: str | None) -> NoveltyDetector | None:
    if checkpoint is None or not sha256:
        return None
    p = sidecar_path(checkpoint, sha256)
    if not p.exists():
        return None
    st = torch.load(p, map_location="cpu", weights_only=False)
    if st.get("model_sha256") != sha256:
        return None
    return NoveltyDetector.from_state(st)
