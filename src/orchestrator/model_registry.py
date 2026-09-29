"""Lazily loads the trained NetJEPA + scoring-head checkpoint, reloading
automatically if the file changes on disk (e.g. src/training/train.py or
src/training/train_heads.py finishes) -- no server restart needed.

Checkpoint lookup (roadmap Task 1.1): instead of one hard-coded filename,
the registry walks CHECKPOINT_CANDIDATES in priority order and uses the
first file that exists. An explicit override written by the blockchain
rollback contract (data/processed/active_model.json -> {"path": ...})
takes precedence over the list, so a regression rollback survives a
server restart.

If no checkpoint exists, falls back to freshly-initialized (untrained)
models so the pipeline can still run end-to-end. Callers get a
`trained: bool` back alongside the models -- always propagate that into
event payloads rather than let an untrained prediction look real.

Every (re)load is also an accountability event: the registry computes the
SHA-256 of the weights file and, when it differs from the last model
recorded on the blockchain ledger, appends a MODEL_UPDATE block (old hash
-> new hash, plus the training-data manifest hash and metrics the training
script stored inside the checkpoint). Every prediction block then carries
the weights hash that produced it -- see src/storage/blockchain_ledger.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch

from src.config import load_config
from src.graph.state_builder import CANONICAL_FEATURE_NAMES
from src.models.netjepa import NetJEPA
from src.scoring.attack_stage import ATTACKStageClassifier
from src.scoring.infiltration import InfiltrationHead

log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
CHECKPOINT_CANDIDATES = [
    PROCESSED_DIR / "netjepa_weights.pt",        # what src/training/train.py + train_heads.py write
    PROCESSED_DIR / "netjepa_real_weights.pth",  # name used by some shared/pre-trained bundles
    Path("netjepa_weights.pt"),                  # repo-root fallback
    Path("netjepa_real_weights.pth"),
]
ACTIVE_MODEL_OVERRIDE = PROCESSED_DIR / "active_model.json"
# Kept for backwards compatibility with callers that imported it directly.
CHECKPOINT_PATH = CHECKPOINT_CANDIDATES[0]
FEATURE_DIM = len(CANONICAL_FEATURE_NAMES)

# Some older/shared checkpoints use different top-level key names.
_KEY_ALIASES = {
    "model": ("model", "netjepa", "model_state_dict", "state_dict"),
    "infil_head": ("infil_head", "infiltration_head", "infil"),
    "stage_clf": ("stage_clf", "stage_classifier", "attack_stage"),
}

_cache: dict = {
    "path": None, "mtime": None, "model": None, "infil_head": None, "stage_clf": None,
    "trained": False, "meta": {},
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


_DEFAULT_PRIMARY = CHECKPOINT_CANDIDATES[0]


def resolve_checkpoint_path() -> Path | None:
    """The checkpoint the registry should load right now, or None.

    If CHECKPOINT_PATH has been re-pointed (tests, or a caller pinning one
    specific file), that path is used exclusively."""
    if CHECKPOINT_PATH != _DEFAULT_PRIMARY:
        return CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None
    if ACTIVE_MODEL_OVERRIDE.exists():
        try:
            override = Path(json.loads(ACTIVE_MODEL_OVERRIDE.read_text())["path"])
            if override.exists() and is_compatible(override):
                return override
        except Exception:  # corrupt override file -> ignore it, fall back to the list
            log.warning("ignoring unreadable %s", ACTIVE_MODEL_OVERRIDE)
    for candidate in CHECKPOINT_CANDIDATES:
        if candidate.exists() and is_compatible(candidate):
            return candidate
    return None


_compat_cache: dict = {}


def is_compatible(path: Path) -> bool:
    """A checkpoint is servable only if it was trained on the current
    feature schema (v1 checkpoints have 23 inputs, v2 has 29)."""
    key = (str(path), path.stat().st_mtime)
    if key not in _compat_cache:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            dim = ck.get("feature_dim")
            if dim is None:  # infer from the first projection layer
                sd = ck.get("model", {})
                w = next((v for k, v in sd.items() if k.endswith("spatial.proj.network.weight")), None)
                dim = w.shape[1] if w is not None else None
            _compat_cache[key] = dim == FEATURE_DIM
        except Exception:
            _compat_cache[key] = False
    return _compat_cache[key]


def set_active_checkpoint(path: str | Path | None) -> None:
    """Pin (or with None, un-pin) which checkpoint the registry serves.
    Used by the rollback smart contract."""
    if path is None:
        if ACTIVE_MODEL_OVERRIDE.exists():
            ACTIVE_MODEL_OVERRIDE.unlink()
        return
    ACTIVE_MODEL_OVERRIDE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_MODEL_OVERRIDE.write_text(json.dumps({"path": str(path)}))


def _pick(checkpoint: dict, role: str):
    for key in _KEY_ALIASES[role]:
        if key in checkpoint:
            return checkpoint[key]
    raise KeyError(f"checkpoint has no {role!r} weights (looked for {_KEY_ALIASES[role]}; has {list(checkpoint)})")


def _record_model_update(meta: dict) -> None:
    """Append a MODEL_UPDATE block when the served weights hash changes.
    Imported lazily so the registry never hard-depends on the ledger."""
    try:
        from src.storage import blockchain_ledger

        blockchain_ledger.get_ledger().record_model_if_changed(meta)
    except Exception:  # the ledger must never take model serving down with it
        log.exception("could not record model update on the ledger")


def get_models():
    """Returns (model, infil_head, stage_clf, trained), all in eval()
    mode. Cheap to call repeatedly -- only rebuilds/reloads when the
    resolved checkpoint path or its mtime changes."""
    path = resolve_checkpoint_path()
    mtime = path.stat().st_mtime if path is not None else None
    if path == _cache["path"] and mtime == _cache["mtime"] and _cache["model"] is not None:
        return _cache["model"], _cache["infil_head"], _cache["stage_clf"], _cache["trained"]

    cfg = load_config().get("model", {})
    embedding_dim = cfg.get("embedding_dim", 128)
    predictor_hidden_dim = cfg.get("predictor_hidden_dim", 256)
    checkpoint = None
    if path is not None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # Trust the checkpoint's own dims over config.yaml's current values.
        embedding_dim = checkpoint.get("embedding_dim", embedding_dim)
        predictor_hidden_dim = checkpoint.get("predictor_hidden_dim", predictor_hidden_dim)

    probabilistic = True
    if checkpoint is not None:
        sd = checkpoint.get("model", {}) if isinstance(checkpoint.get("model"), dict) else {}
        probabilistic = bool(checkpoint.get("probabilistic", "predictor.logvar_head.weight" in sd))
    model = NetJEPA(feature_dim=FEATURE_DIM, embedding_dim=embedding_dim, predictor_hidden_dim=predictor_hidden_dim, probabilistic=probabilistic)
    infil_head = InfiltrationHead(embedding_dim=embedding_dim)
    stage_clf = ATTACKStageClassifier(embedding_dim=embedding_dim)

    trained = False
    meta: dict = {"path": None, "sha256": None, "trained": False}
    if checkpoint is not None:
        model.load_state_dict(_pick(checkpoint, "model"))
        infil_head.load_state_dict(_pick(checkpoint, "infil_head"))
        stage_clf.load_state_dict(_pick(checkpoint, "stage_clf"))
        trained = True
        meta = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "mtime": mtime,
            "trained": True,
            "epoch": checkpoint.get("epoch"),
            "embedding_dim": embedding_dim,
            "predictor_hidden_dim": predictor_hidden_dim,
            # Written by src/training/train_heads.py. heads_validated is only
            # True when the retrained heads beat the baseline on held-out
            # data -- the pipeline uses it to decide whether to trust the
            # network's own scores (see pipeline._score_window).
            "heads_version": checkpoint.get("heads_version", 1),
            "heads_validated": bool(checkpoint.get("heads_validated", False)),
            "temperature": checkpoint.get("temperature"),
            "stage_temperature": checkpoint.get("stage_temperature"),
            "decision_threshold": checkpoint.get("decision_threshold", 0.5),
            "validation_metrics": checkpoint.get("validation_metrics"),
            "training_manifest_hash": checkpoint.get("training_manifest_hash"),
            "parent_sha256": checkpoint.get("parent_sha256"),
            "model_version": checkpoint.get("model_version", 1),
            "probabilistic": probabilistic,
            "context_len": checkpoint.get("context_len", 1),
            "horizon": checkpoint.get("horizon"),
            "feature_schema_version": checkpoint.get("feature_schema_version", 1),
        }

    model.eval()
    infil_head.eval()
    stage_clf.eval()
    meta["num_parameters"] = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in infil_head.parameters()) + sum(p.numel() for p in stage_clf.parameters())

    _cache.update(path=path, mtime=mtime, model=model, infil_head=infil_head, stage_clf=stage_clf, trained=trained, meta=meta)
    if trained:
        _record_model_update(meta)
    return model, infil_head, stage_clf, trained


def get_model_meta() -> dict:
    """Metadata about the currently-served checkpoint (hash, size,
    calibration temperature, validation metrics...)."""
    get_models()
    return dict(_cache["meta"])


def available_checkpoints() -> list[dict]:
    """Every *.pt/*.pth checkpoint on disk under data/processed, with its
    hash -- the rollback contract uses this to find the previous model by
    the weights hash recorded on the blockchain."""
    out = []
    for p in sorted(PROCESSED_DIR.glob("*.pt*")):
        if p.suffix not in (".pt", ".pth") or not p.is_file():
            continue
        out.append({"path": str(p), "sha256": sha256_file(p), "mtime": p.stat().st_mtime, "size_bytes": p.stat().st_size,
                    "compatible": is_compatible(p)})
    return out
