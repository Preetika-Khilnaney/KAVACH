"""Train NetJEPA v2 end to end.

What's new vs src/training/train.py (v1):
* Input: the v2 schema -- flow-level AND packet-level features plus
  modality flags (src/graph/state_builder.py), from flow-CSV windows and
  joint PCAP windows (src/training/dataset_v2.py). Modality dropout on PCAP
  windows (randomly hide one level) teaches the model to use either level
  alone and both together.
* Transition model: the predictor is a diagonal Gaussian p(z_{t+1} | z_t),
  trained with VICReg (anti-collapse) + Gaussian NLL against the EMA target
  encoder's embedding of the true next window.
* Supervised dynamics: the infiltration and MITRE-stage heads are trained on
  z_t (label of t) AND on the predictor's k-step rollout (labels of t+k,
  k=1..horizon) -- ground-truth transitions from the dataset's attack
  annotations / timeline.
* Temporal context: every example is a sequence of `context_len` windows.
* Heads use BCE-with-logits (the v1 head saturated).
* After training: temperature scaling + decision threshold on a calibration
  split carved from training data, then held-out validation vs logistic
  regression baselines (current-window and temporal-context).
* Chains the training-data manifest before training; stores the manifest
  hash, metrics and parent weights hash in the checkpoint.

Run: python -m src.training.train_v2 [--epochs 8] [--exclude-labels PortScan]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.benchmark.evaluate_v2 import baselines, bce_logits, gaussian_nll, model_outputs, summarize
from src.config import load_config
from src.explainability.model_forward import infil_logit
from src.graph.state_builder import CANONICAL_FEATURE_NAMES, FEATURE_SCHEMA_VERSION, PACKET_LEVEL_FEATURES, FLOW_LEVEL_FEATURES, build_graph_state
from src.models.losses import vicreg_loss
from src.models.netjepa import NetJEPA, graph_batch_from_state
from src.scoring.attack_stage import ATTACKStageClassifier, STAGE_NAMES
from src.scoring.calibration import TemperatureScaler
from src.scoring.infiltration import InfiltrationHead
from src.training.dataset_v2 import build_examples_v2, load_sources

OUT_PATH = Path("data/processed/netjepa_weights.pt")
FEATURE_DIM = len(CANONICAL_FEATURE_NAMES)
_NAMES = list(CANONICAL_FEATURE_NAMES)
_PACKET_IDX = [_NAMES.index(n) for n in PACKET_LEVEL_FEATURES]
_FLOW_IDX = [_NAMES.index(n) for n in FLOW_LEVEL_FEATURES]
_HAS_FLOW, _HAS_PKT = _NAMES.index("has_flow_level"), _NAMES.index("has_packet_level")


class GraphCache:
    """Build each window's graph once (every window appears in several
    examples)."""

    def __init__(self, sources: dict):
        self.graphs = {name: [graph_batch_from_state(build_graph_state(w)) for w in windows] for name, (_, windows) in sources.items()}

    def get(self, ex, j: int, drop: str | None = None):
        idx = ex.index - (len(ex.context) - 1) + j
        g = self.graphs[ex.file][idx]
        if drop is None:
            return g
        g = g.clone()
        for nt in ("network", "host"):
            x = g[nt].x.clone()
            cols = _PACKET_IDX + [_HAS_PKT] if drop == "packet" else _FLOW_IDX + [_HAS_FLOW]
            x[:, cols] = 0.0
            g[nt].x = x
        return g


def feature_vector(w: dict) -> np.ndarray:
    return np.asarray(build_graph_state(w)["feature_vector"], dtype=np.float32)


def run_step(model, infil_head, stage_clf, batch, cache, cfg, pos_weight, stage_weights, rng, train=True):
    L, H = cfg["context_len"], cfg["horizon"]
    drops = []
    for ex in batch:
        d = None
        if train and ex.source == "pcap":
            r = rng.random()
            d = "packet" if r < cfg["drop_packet_p"] else ("flow" if r < cfg["drop_packet_p"] + cfg["drop_flow_p"] else None)
        drops.append(d)
    ctx = [[cache.get(ex, j, d) for j in range(L)] for ex, d in zip(batch, drops)]
    tgt = [[cache.get(ex, j, d) for j in range(1, L + 1)] for ex, d in zip(batch, drops)]
    _, z_tgt, z = model(ctx, tgt)
    mean, logvar = model.predictor.distribution(z)
    vic = vicreg_loss(mean, z_tgt, **cfg["vicreg"])
    nll = gaussian_nll(z_tgt, mean, logvar)

    y = torch.tensor([ex.infil for ex in batch], dtype=torch.float32)  # (B, H+1)
    s = torch.tensor([ex.stage for ex in batch], dtype=torch.long)
    sup = bce_logits(infil_logit(infil_head, z), y[:, 0], pos_weight)
    m0 = s[:, 0] >= 0
    if m0.any():
        sup = sup + F.cross_entropy(stage_clf(z[m0]), s[m0, 0], weight=stage_weights)
    trans = torch.tensor(0.0)
    m = mean
    for k in range(1, H + 1):
        trans = trans + bce_logits(infil_logit(infil_head, m), y[:, k], pos_weight)
        mk = s[:, k] >= 0
        if mk.any():
            trans = trans + F.cross_entropy(stage_clf(m[mk]), s[mk, k], weight=stage_weights)
        if k < H:
            m = model.predictor(m)
    total = vic["loss"] + cfg["nll_coeff"] * nll + sup + cfg["transition_coeff"] * trans / H
    return total, {"vicreg": float(vic["loss"]), "nll": float(nll), "sup": float(sup), "transition": float(trans / H),
                   "embed_std": float(z.std(0).mean())}


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--context-len", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--calib-fraction", type=float, default=0.1)
    ap.add_argument("--exclude-labels", type=str, default="", help="comma-separated attack labels removed from training (generalisation test)")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--sources", type=str, default="csv,pcap")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    ap.add_argument("--no-chain", action="store_true", help="don't record manifest / model on the ledger (experiments)")
    ap.add_argument("--report", type=str, default=None, help="write the validation report JSON here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    t0 = time.time()

    mcfg = load_config().get("model", {})
    emb, hid = mcfg.get("embedding_dim", 128), mcfg.get("predictor_hidden_dim", 256)
    cfg = {"context_len": args.context_len, "horizon": args.horizon, "drop_packet_p": 0.25, "drop_flow_p": 0.1,
           "nll_coeff": 1.0, "transition_coeff": 1.0,
           "vicreg": {k: mcfg.get("vicreg", {}).get(k, v) for k, v in (("sim_coeff", 25.0), ("std_coeff", 25.0), ("cov_coeff", 1.0))}}
    exclude = {l.strip() for l in args.exclude_labels.split(",") if l.strip()}

    sources = load_sources(include=tuple(args.sources.split(",")))
    train_all, val, dropped = build_examples_v2(sources, args.context_len, args.horizon, args.val_fraction,
                                                exclude_labels=exclude or None, return_dropped=True)
    rng.shuffle(train_all)
    n_cal = int(len(train_all) * args.calib_fraction)
    calib, train = train_all[:n_cal], train_all[n_cal:]
    if args.max_train:
        train = train[: args.max_train]
    print(f"sources={len(sources)} train={len(train)} calib={len(calib)} val={len(val)} exclude={sorted(exclude) or '-'}")

    manifest_hash, block_idx = None, None
    if not args.no_chain:
        from src.training import manifest as manifest_mod

        man = manifest_mod.build_manifest(Path("data/processed/windows_v2"), extra={"purpose": "NetJEPA v2 training"})
        pcap_man = manifest_mod.build_manifest(Path("data/processed/pcap_windows"))
        man["files"].update({f"pcap/{k}": v for k, v in pcap_man["files"].items()})
        man["file_count"] = len(man["files"])
        man["manifest_hash"] = hashlib.sha256(json.dumps(man["files"], sort_keys=True).encode()).hexdigest()
        # store the combined manifest so verify_training_data can re-check both dirs
        man["dirs"] = ["data/processed/windows_v2", "data/processed/pcap_windows"]
        block_idx = manifest_mod.record_manifest(man)["index"]
        manifest_hash = man["manifest_hash"]
        print(f"training manifest {manifest_hash[:16]}... chained at block {block_idx}")

    cache = GraphCache(sources)
    print(f"graphs built [{time.time() - t0:.0f}s]")
    model = NetJEPA(FEATURE_DIM, emb, hid, probabilistic=True)
    infil_head, stage_clf = InfiltrationHead(emb), ATTACKStageClassifier(emb)
    params = list(model.context_encoder.parameters()) + list(model.predictor.parameters()) + list(infil_head.parameters()) + list(stage_clf.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * (len(train) // args.batch_size)))

    pos = np.mean([ex.infil[0] for ex in train])
    pos_weight = torch.tensor(float(min(10.0, (1 - pos) / max(pos, 1e-6))))
    counts = np.bincount([s for ex in train for s in ex.stage if s >= 0], minlength=len(STAGE_NAMES)).astype(float)
    sw = np.sqrt(counts.sum() / np.maximum(counts, 1))
    stage_weights = torch.tensor(sw / sw.mean(), dtype=torch.float32)
    print(f"pos_weight={float(pos_weight):.2f} stage counts={counts.astype(int).tolist()}")

    def graph_fn(ex, j):
        return cache.get(ex, j)

    best, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train(); infil_head.train(); stage_clf.train()
        rng.shuffle(train)
        logs = []
        for i in range(0, len(train) - 1, args.batch_size):
            batch = train[i : i + args.batch_size]
            if len(batch) < 2:
                continue
            opt.zero_grad()
            loss, m = run_step(model, infil_head, stage_clf, batch, cache, cfg, pos_weight, stage_weights, rng)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            model.update_target_encoder(mcfg.get("ema_momentum", 0.996))
            logs.append(m)
        infil_head.eval(); stage_clf.eval()
        out = model_outputs(model, infil_head, stage_clf, calib, graph_fn, args.context_len, args.horizon)
        s = summarize(out, calib, None, 0.5, args.horizon)
        score = s["detection"].get("f1", 0) + 0.5 * (s["forecast"]["t+1"]["auc_forecast_head"] or 0)
        avg = {k: round(float(np.mean([l[k] for l in logs])), 3) for k in logs[0]}
        print(f"epoch {epoch}: {avg} | calib F1={s['detection']['f1']:.3f} AUC={s['detection'].get('auc', 0):.3f} "
              f"t+1 AUC={s['forecast']['t+1']['auc_forecast_head']:.3f} [{time.time() - t0:.0f}s]")
        if score > best:
            best = score
            best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(infil_head.state_dict()), copy.deepcopy(stage_clf.state_dict()), epoch)
    model.load_state_dict(best_state[0]); infil_head.load_state_dict(best_state[1]); stage_clf.load_state_dict(best_state[2])
    model.eval(); infil_head.eval(); stage_clf.eval()

    # calibration on the calibration split
    out_c = model_outputs(model, infil_head, stage_clf, calib, graph_fn, args.context_len, args.horizon)
    yc = np.array([ex.infil[0] for ex in calib])
    temp = TemperatureScaler(binary=True).fit(out_c["cur_logit"], yc)
    sc = np.array([ex.stage[0] for ex in calib])
    stemp = TemperatureScaler(binary=False).fit(out_c["stage_now"][sc >= 0], sc[sc >= 0])
    pc = 1 / (1 + np.exp(-out_c["cur_logit"] / temp.temperature))
    from sklearn.metrics import f1_score

    grid = np.linspace(0.05, 0.95, 91)
    threshold = float(grid[int(np.argmax([f1_score(yc, pc >= t, zero_division=0) for t in grid]))])

    # zero-day novelty reference: benign *training* embeddings, scored
    # against held-aside benign calibration windows for an honest threshold
    novelty = fit_novelty(model, infil_head, stage_clf, train, out_c, calib, graph_fn, args, seed=args.seed)

    # held-out validation
    out_v = model_outputs(model, infil_head, stage_clf, val, graph_fn, args.context_len, args.horizon)
    report = summarize(out_v, val, temp.temperature, threshold, args.horizon)
    report["baselines"] = baselines(train, val, feature_vector, context_len=args.context_len)
    report.update({"temperature": temp.temperature, "stage_temperature": stemp.temperature, "decision_threshold": threshold,
                   "best_epoch": best_state[3], "train_size": len(train), "val_size": len(val), "calib_size": len(calib),
                   "excluded_labels": sorted(exclude), "context_len": args.context_len, "horizon": args.horizon,
                   "seconds": round(time.time() - t0, 1)})
    if exclude:
        # every window of the held-out family is unseen: evaluate on the
        # validation split plus the training-period windows that were removed
        out_d = model_outputs(model, infil_head, stage_clf, dropped, graph_fn, args.context_len, args.horizon) if dropped else None
        pool = val + dropped
        cat = {k: np.concatenate([out_v[k], out_d[k]]) if out_d is not None else out_v[k] for k in ("cur_logit", "Z", "nll")}
        report["held_out_attack"] = held_out_attack_report(cat, pool, exclude, temp.temperature, threshold, report["baselines"], train,
                                                           feature_vector, novelty=novelty)
    report["novelty"] = novelty_report(novelty, out_v, val, temp.temperature, threshold)
    nj, bl = report["detection"], report["baselines"]["logreg_current_window"]
    validated = bool((nj.get("auc") or 0) >= 0.75 and nj["f1"] >= bl["f1"] - 0.02 and nj["fpr"] <= 0.2)
    report["heads_validated"] = validated
    print(json.dumps({"detection": {k: nj.get(k) for k in ("f1", "precision", "recall", "fpr", "auc")},
                      "baseline": {k: bl.get(k) for k in ("f1", "precision", "recall", "fpr", "auc")},
                      "temporal_baseline_f1": report["baselines"]["logreg_temporal_context"]["f1"],
                      "forecast": report["forecast"], "transition_model": report["transition_model"], "validated": validated}, indent=1, default=float))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=1, default=float))

    if args.out:
        out = Path(args.out)
        parent_sha = None
        if out.exists():
            from src.orchestrator.model_registry import sha256_file

            parent_sha = sha256_file(out)
            backup = out.with_name(f"netjepa_weights.before_v2.{parent_sha[:8]}.pt")
            if not backup.exists():
                shutil.copy2(out, backup)
        torch.save({
            "model": model.state_dict(), "infil_head": infil_head.state_dict(), "stage_clf": stage_clf.state_dict(),
            "feature_dim": FEATURE_DIM, "feature_names": _NAMES, "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "embedding_dim": emb, "predictor_hidden_dim": hid, "probabilistic": True, "model_version": 2,
            "context_len": args.context_len, "horizon": args.horizon, "epoch": best_state[3],
            "heads_version": 3, "heads_validated": validated, "temperature": temp.temperature,
            "stage_temperature": stemp.temperature, "decision_threshold": threshold,
            "validation_metrics": {"netjepa": {**{k: nj.get(k) for k in ("f1", "precision", "recall", "fpr", "auc")},
                                               "forecast": report["forecast"], "stage_now": report.get("stage_now", {}).get("macro_f1")},
                                   "baseline": {k: bl.get(k) for k in ("f1", "precision", "recall", "fpr", "auc")},
                                   "temporal_baseline": {k: report["baselines"]["logreg_temporal_context"].get(k) for k in ("f1", "precision", "recall", "fpr", "auc")},
                                   "val_size": len(val)},
            "training_manifest_hash": manifest_hash, "parent_sha256": parent_sha, "trained_at": time.time(),
        }, out)
        print(f"saved {out} [{time.time() - t0:.0f}s]")
    return report


def fit_novelty(model, infil_head, stage_clf, train, out_c, calib, graph_fn, args, seed=0, n_ref=3000):
    from src.scoring.novelty import NoveltyDetector

    rng = np.random.default_rng(seed)
    benign = [ex for ex in train if ex.infil[0] == 0]
    if len(benign) > n_ref:
        benign = [benign[i] for i in rng.choice(len(benign), n_ref, replace=False)]
    Zref = model_outputs(model, infil_head, stage_clf, benign, graph_fn, args.context_len, args.horizon)["Z"]
    det = NoveltyDetector(Zref, benign_scores=np.array([0.0]))
    yc = np.array([ex.infil[0] for ex in calib])
    cal_scores = det.score(out_c["Z"][yc == 0])
    return NoveltyDetector(Zref, benign_scores=cal_scores)


def novelty_report(det, out_v, val, T, thr) -> dict:
    from sklearn.metrics import roc_auc_score

    y = np.array([ex.infil[0] for ex in val])
    s = det.score(out_v["Z"])
    p = 1 / (1 + np.exp(-out_v["cur_logit"] / (T or 1.0)))
    fused = (p >= thr) | (s > det.threshold)
    res = {"auc_all_attacks": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None,
           "fpr_on_benign": float((s[y == 0] > det.threshold).mean()), "recall": float((s[y == 1] > det.threshold).mean()),
           "threshold_quantile": det.threshold_quantile, "reference_size": int(len(det.reference)),
           "fused_detector": {"recall": float(fused[y == 1].mean()), "fpr": float(fused[y == 0].mean())}}
    return res


def held_out_attack_report(out_v, val, exclude, T, thr, base, train, feature_fn, novelty=None) -> dict:
    """Detection of attack families the model NEVER saw in training, vs
    benign validation windows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    p = 1 / (1 + np.exp(-out_v["cur_logit"] / (T or 1.0)))
    from src.training.labels import _norm

    excl = {_norm(l) for l in exclude}
    labs = [{_norm(l) for l in (ex.context[-1].get("flow_labels_present") or [])} for ex in val]
    is_held = np.array([bool(l & excl) for l in labs])
    is_benign = np.array([ex.infil[0] == 0 for ex in val])
    res = {"held_out_windows": int(is_held.sum()), "benign_windows": int(is_benign.sum())}
    if is_held.sum() == 0:
        return res
    m = is_held | is_benign
    y = is_held[m].astype(int)
    res["netjepa"] = {"recall_at_threshold": float((p[is_held] >= thr).mean()), "fpr_on_benign": float((p[is_benign] >= thr).mean()),
                      "auc_vs_benign": float(roc_auc_score(y, p[m]))}
    Xtr = np.array([feature_fn(ex.context[-1]) for ex in train])
    ytr = np.array([ex.infil[0] for ex in train])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    pb = clf.predict_proba(np.array([feature_fn(ex.context[-1]) for ex in val]))[:, 1]
    res["logreg"] = {"recall_at_threshold": float((pb[is_held] >= 0.5).mean()), "fpr_on_benign": float((pb[is_benign] >= 0.5).mean()),
                     "auc_vs_benign": float(roc_auc_score(y, pb[m]))}
    if novelty is not None and "Z" in out_v:
        s = novelty.score(out_v["Z"])
        res["netjepa_novelty"] = {"recall_at_threshold": float((s[is_held] > novelty.threshold).mean()),
                                  "fpr_on_benign": float((s[is_benign] > novelty.threshold).mean()),
                                  "auc_vs_benign": float(roc_auc_score(y, s[m]))}
        fused = (p >= thr) | (s > novelty.threshold)
        res["netjepa_fused"] = {"recall_at_threshold": float(fused[is_held].mean()), "fpr_on_benign": float(fused[is_benign].mean()),
                                "note": "flag if supervised head >= threshold OR novelty > benign p99"}
        # same kNN novelty in raw feature space (diagnostic: does the learned
        # latent keep the novelty signal the raw features carry?)
        from src.scoring.novelty import NoveltyDetector

        ben_tr = [ex for ex in train if ex.infil[0] == 0]
        idx = np.random.default_rng(0).choice(len(ben_tr), min(3000, len(ben_tr)), replace=False)
        Ftr = np.array([feature_fn(ben_tr[i].context[-1]) for i in idx], dtype=np.float32)
        Fv = np.array([feature_fn(ex.context[-1]) for ex in val], dtype=np.float32)
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        fdet = NoveltyDetector((Ftr - mu) / sd)
        fs = fdet.score((Fv - mu) / sd)
        res["feature_space_novelty"] = {"auc_vs_benign": float(roc_auc_score(y, fs[m]))}
    if "nll" in out_v:
        res["world_model_surprise"] = {"auc_vs_benign": float(roc_auc_score(y, out_v["nll"][m]))}
    return res


if __name__ == "__main__":
    main()
