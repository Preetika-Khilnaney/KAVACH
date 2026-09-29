"""Retrain the infiltration + attack-stage heads on a frozen NetJEPA encoder,
calibrate them, and record the result on the forensic ledger.

Why: the jointly-trained infiltration head in the original checkpoint is
saturated -- it outputs ~1.0 for every window (benchmark: recall 1.0,
FPR 1.0), so its gradients (and Integrated Gradients) are all zero. The
encoder underneath is fine: a linear probe on its embeddings separates
attack from benign windows at ~0.80 AUC. The collapse came from training the
head on post-sigmoid outputs with BCE and a very large pos_weight. Here the
heads are trained on *logits* (BCEWithLogits, numerically stable), with
class balancing, early stopping on a calibration split, and temperature
scaling -- the encoder, predictor and EMA target are left untouched.

Splits (no leakage into the benchmark's held-out set):
* validation = the same last-20%-per-file windows src.benchmark.evaluate uses
* calibration = a random 10% of the remaining training windows (early
  stopping, decision threshold, temperature)
* training = the rest

Blockchain integration (synergy doc 1.1/1.2): the training-data manifest is
chained BEFORE training, and the new checkpoint stores its manifest hash,
metrics and parent weights hash; model_registry chains a MODEL_UPDATE block
the first time it serves the new weights.

Run: python -m src.training.train_heads [--epochs 40]
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.orchestrator import model_registry
from src.scoring.attack_stage import ATTACKStageClassifier, STAGE_NAMES
from src.scoring.calibration import TemperatureScaler, expected_calibration_error, multiclass_ece
from src.scoring.infiltration import InfiltrationHead
from src.training import manifest as manifest_mod
from src.training.dataset import build_examples, load_all_windows
from src.training.embeddings import embed_windows

OUT_PATH = Path("data/processed/netjepa_weights.pt")
EMB_CACHE = Path("data/processed/interpretability/embeddings.npz")


def _binary(y, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"f1": float(f1_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0}


def _infil_logits(head: InfiltrationHead, Z: torch.Tensor) -> torch.Tensor:
    return head.net[:-1](Z).squeeze(-1)


def train_infiltration_head(Zt, yt, Zc, yc, embedding_dim, epochs, seed=0):
    torch.manual_seed(seed)
    head = InfiltrationHead(embedding_dim=embedding_dim)
    pos = float(yt.mean())
    pos_weight = torch.tensor(min(10.0, (1 - pos) / max(pos, 1e-6)))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Zt))
        for i in range(0, len(Zt), 256):
            idx = perm[i : i + 256]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(_infil_logits(head, Zt[idx]), yt[idx], pos_weight=pos_weight)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pc = torch.sigmoid(_infil_logits(head, Zc)).numpy()
        f1 = f1_score(yc.numpy(), pc >= 0.5, zero_division=0)
        if f1 > best:
            best, best_state, best_ep = f1, copy.deepcopy(head.state_dict()), ep
    head.load_state_dict(best_state)
    head.eval()
    return head, {"best_epoch": best_ep, "calibration_f1": float(best), "pos_weight": float(pos_weight)}


def train_stage_head(Zt, st, Zc, sc, embedding_dim, epochs, seed=0):
    torch.manual_seed(seed)
    clf = ATTACKStageClassifier(embedding_dim=embedding_dim)
    mt, mc = st >= 0, sc >= 0
    Zs, ys = Zt[mt], st[mt]
    counts = torch.bincount(ys, minlength=len(STAGE_NAMES)).float()
    # sqrt-inverse-frequency: helps the rare stages without letting 35
    # Lateral Movement windows dominate the loss
    weights = (counts.sum() / counts.clamp(min=1)).sqrt()
    weights = weights / weights.mean()
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        clf.train()
        perm = torch.randperm(len(Zs))
        for i in range(0, len(Zs), 256):
            idx = perm[i : i + 256]
            opt.zero_grad()
            F.cross_entropy(clf(Zs[idx]), ys[idx], weight=weights).backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            pred = clf(Zc[mc]).argmax(-1).numpy()
        f1 = f1_score(sc[mc].numpy(), pred, average="macro", zero_division=0)
        if f1 > best:
            best, best_state, best_ep = f1, copy.deepcopy(clf.state_dict()), ep
    clf.load_state_dict(best_state)
    clf.eval()
    return clf, {"best_epoch": best_ep, "calibration_macro_f1": float(best), "class_counts": counts.tolist()}


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--calib-fraction", type=float, default=0.1)
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()
    t0 = time.time()

    model, _, _, trained = model_registry.get_models()
    base_meta = model_registry.get_model_meta()
    if not trained:
        sys.exit("No trained NetJEPA checkpoint found -- run python -m src.training.train first (the encoder is reused).")
    base_path = Path(base_meta["path"])
    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    embedding_dim = base_ckpt.get("embedding_dim", 128)
    print(f"base checkpoint: {base_path} sha256={base_meta['sha256'][:16]}...")

    windows_by_file = load_all_windows()
    if not windows_by_file:
        sys.exit("No cached windows -- run python -m src.training.build_windows_cache first.")
    man = manifest_mod.build_manifest(extra={"purpose": "infiltration/stage head retraining", "base_weights_sha256": base_meta["sha256"]})
    block = manifest_mod.record_manifest(man)
    print(f"training manifest {man['manifest_hash'][:16]}... chained at block {block['index']} ({man['file_count']} files)")

    train_ex, val_ex = build_examples(windows_by_file, context_len=1, val_fraction=args.val_fraction)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train_ex))
    n_cal = int(len(train_ex) * args.calib_fraction)
    cal_idx, tr_idx = set(perm[:n_cal].tolist()), perm[n_cal:]
    train_w = [train_ex[i][1] for i in tr_idx]
    cal_w = [train_ex[i][1] for i in sorted(cal_idx)]
    val_w = [t for _, t in val_ex]
    print(f"windows: train={len(train_w)} calib={len(cal_w)} val={len(val_w)}; encoding (frozen encoder)...")

    E_tr, E_cal, E_val = (embed_windows(model, w, with_forecast=True) for w in (train_w, cal_w, val_w))
    print(f"encoded in {time.time() - t0:.0f}s; embed_std(val)={E_val['Z'].std(0).mean():.3f}")

    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)  # noqa: E731
    Zt, Zc, Zv = T(E_tr["Z"]), T(E_cal["Z"]), T(E_val["Z"])
    infil_head, infil_info = train_infiltration_head(Zt, T(E_tr["y"]), Zc, T(E_cal["y"]), embedding_dim, args.epochs)
    stage_clf, stage_info = train_stage_head(Zt, T(E_tr["stage"], torch.long), Zc, T(E_cal["stage"], torch.long), embedding_dim, args.epochs)

    with torch.no_grad():
        cal_logits = _infil_logits(infil_head, Zc).numpy()
        val_logits = _infil_logits(infil_head, Zv).numpy()
        cal_stage_logits = stage_clf(Zc).numpy()
        val_stage_logits = stage_clf(Zv).numpy()
    temp = TemperatureScaler(binary=True).fit(cal_logits, E_cal["y"])
    mc = E_cal["stage"] >= 0
    stage_temp = TemperatureScaler(binary=False).fit(cal_stage_logits[mc], E_cal["stage"][mc])

    # decision threshold on the *calibrated* probability, chosen on the
    # calibration split (never on validation)
    cal_prob = temp.scale(cal_logits).numpy()
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(E_cal["y"], cal_prob >= t, zero_division=0) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(f1s))])

    yv = E_val["y"].astype(int)
    raw_val = 1 / (1 + np.exp(-val_logits))
    cal_val = temp.scale(val_logits).numpy()
    netjepa = {**_binary(yv, (cal_val >= threshold).astype(int)), "auc": float(roc_auc_score(yv, cal_val)) if len(set(yv)) > 1 else None,
               "ece_raw": expected_calibration_error(raw_val, yv)["ece"], "ece_calibrated": expected_calibration_error(cal_val, yv)["ece"]}
    mv = E_val["stage"] >= 0
    stage_pred = val_stage_logits[mv].argmax(1)
    stage_probs = torch.softmax(torch.as_tensor(val_stage_logits[mv]) / stage_temp.temperature, -1).numpy()
    netjepa.update({"stage_macro_f1": float(f1_score(E_val["stage"][mv], stage_pred, average="macro", zero_division=0)),
                    "stage_accuracy": float((stage_pred == E_val["stage"][mv]).mean()),
                    "stage_ece_calibrated": multiclass_ece(stage_probs, E_val["stage"][mv])})

    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(E_tr["X"], E_tr["y"].astype(int))
    baseline = _binary(yv, lr.predict(E_val["X"]))
    validated = bool(netjepa["auc"] and netjepa["auc"] >= 0.7 and netjepa["fpr"] <= 0.3 and netjepa["f1"] >= 0.5 * max(baseline["f1"], 1e-6))
    print(f"validation (held-out): NetJEPA f1={netjepa['f1']:.3f} p={netjepa['precision']:.3f} r={netjepa['recall']:.3f} fpr={netjepa['fpr']:.3f} "
          f"auc={netjepa['auc']:.3f} ece {netjepa['ece_raw']:.3f}->{netjepa['ece_calibrated']:.3f} | stage macroF1={netjepa['stage_macro_f1']:.3f} "
          f"| baseline f1={baseline['f1']:.3f} | T={temp.temperature:.3f} thr={threshold:.2f} validated={validated}")

    out = Path(args.out)
    if out.exists() and out.resolve() == base_path.resolve():
        backup = out.with_name("netjepa_weights.pre_heads_v2.pt")
        if not backup.exists():
            shutil.copy2(out, backup)
            print(f"original checkpoint backed up to {backup}")

    ckpt = {k: v for k, v in base_ckpt.items() if k not in ("infil_head", "stage_clf")}
    ckpt.update({
        "infil_head": infil_head.state_dict(),
        "stage_clf": stage_clf.state_dict(),
        "heads_version": 2,
        "heads_validated": validated,
        "temperature": temp.temperature,
        "stage_temperature": stage_temp.temperature,
        "decision_threshold": threshold,
        "validation_metrics": {"netjepa": netjepa, "baseline": baseline, "val_size": int(len(yv)),
                               "infiltration_training": infil_info, "stage_training": stage_info},
        "training_manifest_hash": man["manifest_hash"],
        "parent_sha256": base_meta["sha256"],
        "heads_trained_at": time.time(),
    })
    torch.save(ckpt, out)
    EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMB_CACHE, **{f"{split}_{k}": v for split, E in (("train", E_tr), ("calib", E_cal), ("val", E_val)) for k, v in E.items()})
    model_registry.get_models()  # serve + chain the MODEL_UPDATE block now
    print(f"saved {out} (sha256={model_registry.get_model_meta()['sha256'][:16]}...) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
