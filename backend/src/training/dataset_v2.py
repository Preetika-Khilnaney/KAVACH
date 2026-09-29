"""Training data for NetJEPA v2: flow-level CSV windows (CIC-IDS-2017
MachineLearningCVE, 100 flows per window) + joint flow+packet PCAP windows
(CIC-IDS-2017 raw captures, 5-second windows, labelled from the published
attack timeline). Both are "sources"; each file is one time-ordered stream.

An example is (context, futures, labels):
  context = windows t-L+1 .. t          (the observed state history)
  futures = windows t+1 .. t+K          (ground-truth next states)
  labels  = infiltration + MITRE stage for t, t+1, ..., t+K
so dynamics are supervised on *labelled transitions* (what the attack
timeline says happens next), not just same-window classification.

Split is temporal per file: an example is validation when its last future
window falls in the file's final `val_fraction` (never trained on).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

from src.features.joint_windows import normalize_windows
from src.training.labels import infiltration_label, stage_label

CSV_WINDOWS_DIR = Path("data/processed/windows_v2")
PCAP_WINDOWS_DIR = Path("data/processed/pcap_windows")


@dataclass
class Example:
    source: str
    file: str
    index: int
    context: list = field(repr=False)
    futures: list = field(repr=False)
    infil: list = field(default_factory=list)   # len K+1: t, t+1..t+K
    stage: list = field(default_factory=list)   # -1 = excluded from stage loss

    @property
    def target(self) -> dict:
        return self.context[-1]


def load_sources(csv_dir: Path = CSV_WINDOWS_DIR, pcap_dir: Path = PCAP_WINDOWS_DIR, include: tuple = ("csv", "pcap")) -> dict[str, tuple[str, list[dict]]]:
    """name -> (source kind, windows in time order). PCAP windows are
    z-scored per capture here (CSV caches are stored already normalised)."""
    out = {}
    if "csv" in include:
        for f in sorted(Path(csv_dir).glob("*.pkl")):
            with f.open("rb") as fh:
                out[f"csv:{f.stem}"] = ("csv", pickle.load(fh))
    if "pcap" in include:
        for f in sorted(Path(pcap_dir).glob("*.pkl")):
            with f.open("rb") as fh:
                windows, _ = normalize_windows(pickle.load(fh))
            out[f"pcap:{f.stem}"] = ("pcap", windows)
    return out


def _stage(w):
    s = stage_label(w)
    return -1 if s is None else s


def build_examples_v2(sources: dict, context_len: int = 3, horizon: int = 3, val_fraction: float = 0.2,
                      exclude_labels: set[str] | None = None, return_dropped: bool = False):
    """exclude_labels: attack labels to drop from TRAINING entirely (every
    example whose context or futures contain them) -- used by the
    leave-one-attack-out generalisation test. Validation keeps them."""
    from src.training.labels import _norm

    exclude_norm = {_norm(l) for l in (exclude_labels or ())}
    train, val, dropped = [], [], []
    for name, (kind, windows) in sources.items():
        n = len(windows)
        split = int(n * (1 - val_fraction))
        for t in range(context_len - 1, n - horizon):
            ctx = windows[t - context_len + 1 : t + 1]
            fut = windows[t + 1 : t + 1 + horizon]
            seq = [ctx[-1], *fut]
            ex = Example(kind, name, t, ctx, fut, [infiltration_label(w) for w in seq], [_stage(w) for w in seq])
            is_val = t + horizon >= split
            if not is_val and exclude_norm:
                present = {_norm(l) for w in (*ctx, *fut) for l in (w.get("flow_labels_present") or [])}
                if present & exclude_norm:
                    dropped.append(ex)
                    continue
            (val if is_val else train).append(ex)
    return (train, val, dropped) if return_dropped else (train, val)
