"""Training-data integrity (AI<->blockchain synergy doc, section 1.2).

Before a model is trained, hash every file of the training set (the cached
windows under data/processed/windows) and record the manifest on the ledger.
Later, `verify_training_data()` re-hashes what is on disk and compares it to
the manifest that was chained when the served model was trained -- a data
poisoning edit made after training shows up as a per-file hash mismatch.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from src.storage.blockchain_ledger import TRAINING_MANIFEST, canonical_json, get_ledger

WINDOWS_DIR = Path("data/processed/windows")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(windows_dir: Path = WINDOWS_DIR, extra: dict | None = None) -> dict:
    files = {p.name: {"sha256": _sha256(p), "size_bytes": p.stat().st_size} for p in sorted(windows_dir.glob("*.pkl"))}
    manifest = {"dataset": str(windows_dir), "files": files, "file_count": len(files)}
    if extra:
        manifest.update(extra)
    manifest["manifest_hash"] = hashlib.sha256(canonical_json({"files": files}).encode()).hexdigest()
    manifest["created_at"] = time.time()
    return manifest


def record_manifest(manifest: dict) -> dict:
    return get_ledger().log_training_manifest(manifest)


def verify_training_data(manifest_hash: str | None = None, windows_dir: Path = WINDOWS_DIR) -> dict:
    """Compare the training files on disk with the manifest on the chain.
    Uses the manifest whose hash the served model recorded (manifest_hash),
    else the most recent TRAINING_MANIFEST block."""
    ledger = get_ledger()
    blocks = ledger.iter_blocks([TRAINING_MANIFEST])
    block = None
    if manifest_hash:
        block = next((b for b in reversed(blocks) if b["data"].get("manifest_hash") == manifest_hash), None)
    if block is None and blocks:
        block = blocks[-1]
    if block is None:
        return {"verified": False, "status": "no_manifest", "message": "No training manifest has been recorded on the ledger yet -- run python -m src.training.train_heads."}

    recorded = block["data"]["files"]
    dirs = block["data"].get("dirs")
    if dirs:  # v2 manifest: flow-window cache + "pcap/"-prefixed joint windows
        current = dict(build_manifest(Path(dirs[0]))["files"])
        for extra in dirs[1:]:
            current.update({f"pcap/{k}": v for k, v in build_manifest(Path(extra))["files"].items()})
    else:
        current = build_manifest(Path(block["data"].get("dataset") or windows_dir))["files"]
    altered = [n for n in recorded if n in current and current[n]["sha256"] != recorded[n]["sha256"]]
    missing = [n for n in recorded if n not in current]
    added = [n for n in current if n not in recorded]
    ok = not (altered or missing)
    return {
        "verified": ok,
        "status": "intact" if ok else "tampered",
        "manifest_block": block["index"],
        "manifest_hash": block["data"].get("manifest_hash"),
        "recorded_at": block["timestamp"],
        "files_checked": len(recorded),
        "altered_files": altered,
        "missing_files": missing,
        "new_files": added,
        "message": ("Training data on disk matches the manifest chained at block %d." % block["index"]) if ok else
                   f"Training data changed since block {block['index']}: {len(altered)} altered, {len(missing)} missing -- possible data poisoning.",
    }
