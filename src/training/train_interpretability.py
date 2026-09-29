"""Train every interpretability artefact on top of the served NetJEPA
checkpoint (roadmap Workstream 5 + Task 8.1):

* sparse autoencoder over z_context + auto-labelled features  (Task 5.1)
* tuned-lens translators per hook point                      (Task 5.4)
* linear probes per layer x concept                          (Task 5.3)
* concept bottleneck (embedding -> 10 concepts -> stage)     (Task 8.1)
* a benign reference window (clean baseline for causal tracing, Task 5.2)

Everything lands in data/processed/interpretability/ together with
interpretability_report.json (metrics for the benchmark page).

Run after train_heads: python -m src.training.train_interpretability
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from src.explainability import model_forward as mf
from src.explainability.sparse_autoencoder import LABELS_PATH, SAE_PATH, SparseAutoencoder
from src.graph.state_builder import CANONICAL_FEATURE_NAMES, build_graph_state
from src.models.concept_bottleneck import CONCEPT_NAMES, ConceptBottleneck, concept_labels
from src.models.netjepa import graph_batch_from_state
from src.orchestrator import model_registry
from src.scoring.attack_stage import STAGE_NAMES
from src.training.dataset import load_all_windows
from src.training.embeddings import embed_windows, primary_label
from src.training.labels import infiltration_label, stage_label
from src.training.train_heads import EMB_CACHE

OUT = Path("data/processed/interpretability")
LENS_PATH = OUT / "tuned_lens.pt"
CB_PATH = OUT / "concept_bottleneck.pt"
PROBES_PATH = OUT / "probes.json"
BENIGN_REF_PATH = OUT / "benign_reference.json"
REPORT_PATH = OUT / "interpretability_report.json"
LENS_POINTS = ["input", "hgt_conv1", "hgt_conv2", "spatial_out", "gru", "context_norm"]


def _load_embeddings(model) -> dict:
    meta = model_registry.get_model_meta()
    if (meta.get("feature_schema_version") or 1) >= 2:
        return _embeddings_v2(model, meta)
    if EMB_CACHE.exists():
        d = np.load(EMB_CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    print("embedding cache missing -- encoding windows (run train_heads first to share it)")
    from src.training.dataset import build_examples

    train_ex, val_ex = build_examples(load_all_windows(), context_len=1)
    out = {}
    for split, ex in (("train", train_ex), ("val", val_ex)):
        for k, v in embed_windows(model, [t for _, t in ex]).items():
            out[f"{split}_{k}"] = v
    return out


@torch.no_grad()
def _embeddings_v2(model, meta) -> dict:
    """Embeddings with the deployed temporal context over BOTH sources
    (flow CSV windows + joint PCAP windows), same splits as training."""
    from src.training.dataset_v2 import build_examples_v2, load_sources
    from src.training.train_v2 import GraphCache

    L, H = int(meta.get("context_len") or 3), int(meta.get("horizon") or 3)
    sources = load_sources()
    train, val = build_examples_v2(sources, L, H, 0.2)
    cache = GraphCache(sources)
    out = {}
    for split, exs in (("train", train), ("val", val)):
        Z, X, Y, S, LAB = [], [], [], [], []
        for i in range(0, len(exs), 256):
            chunk = exs[i : i + 256]
            _, _, z = model([[cache.get(ex, j) for j in range(L)] for ex in chunk])
            Z.append(z.numpy())
            X += [build_graph_state(ex.context[-1])["feature_vector"] for ex in chunk]
            Y += [ex.infil[0] for ex in chunk]
            S += [ex.stage[0] for ex in chunk]
            LAB += [primary_label(ex.context[-1]) for ex in chunk]
        out.update({f"{split}_Z": np.concatenate(Z), f"{split}_X": np.asarray(X, dtype=np.float32), f"{split}_y": np.asarray(Y, dtype=np.float32),
                    f"{split}_stage": np.asarray(S), f"{split}_label": np.asarray(LAB)})
    return out


# ── SAE ─────────────────────────────────────────────────────────────────────
def train_sae(Z: np.ndarray, X: np.ndarray, labels: np.ndarray, Zv: np.ndarray, hidden: int, epochs: int, l1: float, k: int = 32) -> dict:
    torch.manual_seed(0)
    Zt = torch.tensor(Z, dtype=torch.float32)
    sae = SparseAutoencoder(Zt.shape[1], hidden, k)
    sae.decoder.weight.data = sae.encoder.weight.data.T.clone()  # tied init
    sae.normalize_decoder()
    sae.pre_bias.data = Zt.mean(0)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(Zt))
        for i in range(0, len(Zt), 256):
            x = Zt[perm[i : i + 256]]
            x_hat, h = sae(x)
            loss, _ = SparseAutoencoder.loss(x, x_hat, h, l1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()
    sae.eval()
    with torch.no_grad():
        Zvt = torch.tensor(Zv, dtype=torch.float32)
        xv_hat, hv = sae(Zvt)
        mse = float(((Zvt - xv_hat) ** 2).mean())
        rel = float(((Zvt - xv_hat) ** 2).sum() / ((Zvt - Zvt.mean(0)) ** 2).sum())
        _, h_all = sae(Zt)
    active_frac = float((hv > 0).float().mean())
    alive = (h_all > 0).any(0)

    # auto-label each alive feature from its top-activating training windows
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-6)
    labels_out = {}
    for f in torch.nonzero(alive).flatten().tolist():
        col = h_all[:, f]
        freq = float((col > 0).float().mean())
        top = SparseAutoencoder.top_activating_examples(h_all, f, 20)
        if len(top) < 3:
            continue
        counts = Counter(labels[top].tolist())
        dom, n = counts.most_common(1)[0]
        a = col.numpy()
        corr = (Xc * ((a - a.mean()) / (a.std() + 1e-6))[:, None]).mean(0)
        j = int(np.nanargmax(np.abs(corr)))
        direction = "high" if corr[j] > 0 else "low"
        pretty = dom.replace("�", "-").replace("�", "-")
        labels_out[str(f)] = {
            "label": f"{pretty} pattern ({CANONICAL_FEATURE_NAMES[j]} {direction})" if dom != "BENIGN" else f"Benign traffic ({CANONICAL_FEATURE_NAMES[j]} {direction})",
            "dominant_label": pretty, "purity": round(n / len(top), 3), "top_feature": CANONICAL_FEATURE_NAMES[j],
            "feature_correlation": round(float(corr[j]), 3), "direction": direction, "activation_frequency": round(freq, 4),
            "top_examples": top[:10], "auto_labelled": True,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sae.state_dict(), "input_dim": sae.input_dim, "hidden_dim": sae.hidden_dim, "k": k, "l1_coeff": l1}, SAE_PATH)
    LABELS_PATH.write_text(json.dumps(labels_out, indent=1))
    attack_features = sum(1 for v in labels_out.values() if v["dominant_label"] != "BENIGN" and v["purity"] >= 0.7)
    return {"hidden_dim": hidden, "alive_features": int(alive.sum()), "labelled_features": len(labels_out),
            "attack_specific_features": attack_features, "val_mse": mse, "val_fraction_variance_unexplained": rel,
            "avg_fraction_active": active_frac, "avg_active_features": active_frac * hidden, "l1_coeff": l1, "top_k": k}


# ── activations for lens / probes ───────────────────────────────────────────
@torch.no_grad()
def collect_activations(model, sequences: list[list[dict]], batch_size: int = 128) -> dict:
    """sequences: list of window lists (oldest first). Returns per hook point
    an (N, dim) array: spatial layers use the LAST window of each sequence."""
    out: dict[str, list] = {p: [] for p in ["input", *mf.HOOK_POINTS]}
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i : i + batch_size]
        states = [[build_graph_state(w) for w in seq] for seq in chunk]
        with mf.capture(model) as store:
            model([[graph_batch_from_state(s) for s in seq] for seq in states])
        L = len(chunk[0])
        for name, calls in store.items():
            if not calls:
                continue
            if len(calls) == 1 and isinstance(calls[0], torch.Tensor) and calls[0].dim() == 2:
                out[name].append(calls[0].numpy())
            elif name in ("hgt_conv1", "hgt_conv2", "spatial_out"):
                pooled = [mf._pool(c).numpy() for c in calls]
                out[name].append(np.stack(pooled[L - 1 :: L]))
            else:
                out[name].append(np.stack([mf._pool(c).numpy() for c in calls]))
        out["input"].append(np.array([s[-1]["feature_vector"] for s in states], dtype=np.float32))
    return {k: np.concatenate(v) for k, v in out.items() if v}


def train_tuned_lens(model, infil_head, stage_clf, acts_tr: dict, acts_va: dict, y_va, s_va, temperature) -> dict:
    translators, report = {}, []
    for point in LENS_POINTS:
        if point not in acts_tr:
            continue
        A, target = acts_tr[point], acts_tr["context_norm"]
        if point == "context_norm":
            W, b = np.eye(A.shape[1], dtype=np.float32), np.zeros(A.shape[1], dtype=np.float32)
        else:
            r = Ridge(alpha=1.0).fit(A, target)
            W, b = r.coef_.T.astype(np.float32), r.intercept_.astype(np.float32)
        translators[point] = {"W": torch.tensor(W), "b": torch.tensor(b)}
        zv = torch.tensor(acts_va[point] @ W + b)
        with torch.no_grad():
            p = torch.sigmoid(mf.infil_logit(infil_head, zv) / (temperature or 1.0)).numpy()
            st = stage_clf(zv).argmax(-1).numpy()
        m = s_va >= 0
        report.append({"layer": point, "infiltration_auc": float(roc_auc_score(y_va, p)) if len(set(y_va)) > 1 else None,
                       "stage_accuracy": float((st[m] == s_va[m]).mean()) if m.any() else None,
                       "stage_macro_f1": float(f1_score(s_va[m], st[m], average="macro", zero_division=0)) if m.any() else None})
    torch.save(translators, LENS_PATH)
    return {"layers": report}


# ── probes ──────────────────────────────────────────────────────────────────
def _concept_targets(seqs: list[list[dict]], futures: list[dict]) -> dict:
    def ratio(w):
        return float(w.get("flow_attack_ratio") or 0.0)

    return {
        "is_scanning": np.array([int(primary_label(s[-1]) == "PortScan" or "PortScan" in (s[-1].get("flow_labels_present") or [])) for s in seqs]),
        "mitre_stage": np.array([(-1 if stage_label(s[-1]) is None else stage_label(s[-1])) for s in seqs]),
        "is_escalating": np.array([int(ratio(s[-1]) > ratio(s[0]) + 1e-9) for s in seqs]),
        "attack_in_3_windows": np.array([int(infiltration_label(f)) for f in futures]),
    }


def run_probes(acts_tr, acts_va, tg_tr, tg_va) -> dict:
    layers = ["input", "hgt_conv1", "hgt_conv2", "gru", "context_norm", "predictor_l3"]
    matrix = []
    for concept in tg_tr:
        ytr, yva = tg_tr[concept], tg_va[concept]
        mtr, mva = ytr >= 0, yva >= 0
        if len(set(ytr[mtr])) < 2 or len(set(yva[mva])) < 2:
            continue
        row = {"concept": concept, "classes": int(len(set(ytr[mtr]))), "val_positive_rate": float((yva[mva] > 0).mean()), "layers": {}}
        for layer in layers:
            if layer not in acts_tr:
                continue
            A, B = acts_tr[layer][mtr], acts_va[layer][mva]
            mu, sd = A.mean(0), A.std(0) + 1e-6
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit((A - mu) / sd, ytr[mtr])
            pred = clf.predict((B - mu) / sd)
            row["layers"][layer] = {"balanced_accuracy": round(float(balanced_accuracy_score(yva[mva], pred)), 4),
                                    "accuracy": round(float((pred == yva[mva]).mean()), 4)}
        best = max(row["layers"].items(), key=lambda kv: kv[1]["balanced_accuracy"])
        row["best_layer"] = best[0]
        matrix.append(row)
    return {"layers": layers, "concepts": matrix, "chance_balanced_accuracy": 0.5,
            "note": "Balanced accuracy of a linear probe trained on 80%-style temporal train windows and scored on held-out windows; 0.5 = chance for binary concepts (1/6 for mitre_stage)."}


# ── concept bottleneck ──────────────────────────────────────────────────────
def train_concept_bottleneck(Z, X, S, Zv, Xv, Sv, epochs: int) -> dict:
    torch.manual_seed(0)
    cb = ConceptBottleneck(Z.shape[1])
    Zt, Ct, St = torch.tensor(Z), concept_labels(torch.tensor(X)), torch.tensor(S)
    pos = Ct.mean(0).clamp(1e-3, 1 - 1e-3)
    pw = ((1 - pos) / pos).clamp(max=20)
    opt = torch.optim.Adam(cb.parameters(), lr=1e-3, weight_decay=1e-4)
    m = St >= 0
    counts = torch.bincount(St[m], minlength=len(STAGE_NAMES)).float()
    w = (counts.sum() / counts.clamp(min=1)).sqrt()
    w = w / w.mean()
    for ep in range(epochs):
        perm = torch.randperm(len(Zt))
        for i in range(0, len(Zt), 256):
            idx = perm[i : i + 256]
            c_logit = cb.concept_predictor(Zt[idx])
            loss = F.binary_cross_entropy_with_logits(c_logit, Ct[idx], pos_weight=pw)
            mi = m[idx]
            if mi.any():
                # classifier sees the *predicted* concepts, detached so it
                # can't bend concept meanings to help the stage loss
                logits = cb.classifier(torch.sigmoid(c_logit[mi]).detach())
                loss = loss + F.cross_entropy(logits, St[idx][mi], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
    cb.eval()
    with torch.no_grad():
        cv, logits = cb(torch.tensor(Zv))
    Cv = concept_labels(torch.tensor(Xv)).numpy()
    concept_auc = {}
    for j, name in enumerate(CONCEPT_NAMES):
        concept_auc[name] = float(roc_auc_score(Cv[:, j], cv[:, j].numpy())) if len(set(Cv[:, j])) > 1 else None
    mv = Sv >= 0
    pred = logits.argmax(-1).numpy()
    torch.save({"state_dict": cb.state_dict(), "embedding_dim": Z.shape[1], "concepts": CONCEPT_NAMES}, CB_PATH)
    return {"concept_auc": concept_auc, "stage_macro_f1": float(f1_score(Sv[mv], pred[mv], average="macro", zero_division=0)),
            "stage_accuracy": float((pred[mv] == Sv[mv]).mean()),
            "classifier_weights": {STAGE_NAMES[i]: {CONCEPT_NAMES[j]: round(float(cb.classifier.weight[i, j]), 3) for j in range(len(CONCEPT_NAMES))}
                                   for i in range(len(STAGE_NAMES))}}


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-hidden", type=int, default=4096)
    ap.add_argument("--sae-epochs", type=int, default=25)
    ap.add_argument("--sae-l1", type=float, default=0.0)
    ap.add_argument("--sae-k", type=int, default=32)
    ap.add_argument("--probe-train", type=int, default=4000)
    ap.add_argument("--probe-val", type=int, default=1500)
    ap.add_argument("--cb-epochs", type=int, default=25)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    model, infil_head, stage_clf, trained = model_registry.get_models()
    meta = model_registry.get_model_meta()
    if not trained:
        sys.exit("no trained checkpoint -- run src.training.train (and train_heads) first")
    E = _load_embeddings(model)
    Z, X, S, Y, LAB = E["train_Z"], E["train_X"], E["train_stage"], E["train_y"], E["train_label"]
    Zv, Xv, Sv, Yv = E["val_Z"], E["val_X"], E["val_stage"], E["val_y"]
    report = {"model_sha256": meta["sha256"], "generated_at": time.time()}

    print("training sparse autoencoder...")
    report["sae"] = train_sae(Z, X, LAB, Zv, args.sae_hidden, args.sae_epochs, args.sae_l1, args.sae_k)
    print(f"  {report['sae']}  [{time.time() - t0:.0f}s]")

    print("training concept bottleneck...")
    report["concept_bottleneck"] = train_concept_bottleneck(Z, X, S, Zv, Xv, Sv, args.cb_epochs)
    print(f"  stage macroF1={report['concept_bottleneck']['stage_macro_f1']:.3f} [{time.time() - t0:.0f}s]")

    # temporal sequences for probes + lens (context of 3 windows, target +3)
    from src.training.dataset_v2 import load_sources

    windows_by_file = {k: v[1] for k, v in load_sources().items()} if (meta.get("feature_schema_version") or 1) >= 2 else load_all_windows()
    tr_seq, tr_fut, va_seq, va_fut = [], [], [], []
    for windows in windows_by_file.values():
        n = len(windows)
        split = int(n * 0.8)
        for i in range(2, n - 3):
            seq, fut = windows[i - 2 : i + 1], windows[i + 3]
            (tr_seq if i + 3 < split else va_seq).append(seq)
            (tr_fut if i + 3 < split else va_fut).append(fut)
    rng = np.random.default_rng(0)
    ti = rng.choice(len(tr_seq), min(args.probe_train, len(tr_seq)), replace=False)
    vi = rng.choice(len(va_seq), min(args.probe_val, len(va_seq)), replace=False)
    tr_seq, tr_fut = [tr_seq[i] for i in ti], [tr_fut[i] for i in ti]
    va_seq, va_fut = [va_seq[i] for i in vi], [va_fut[i] for i in vi]

    print(f"collecting activations ({len(tr_seq)} + {len(va_seq)} sequences)...")
    acts_tr, acts_va = collect_activations(model, tr_seq), collect_activations(model, va_seq)
    print(f"  done [{time.time() - t0:.0f}s]")
    tg_tr, tg_va = _concept_targets(tr_seq, tr_fut), _concept_targets(va_seq, va_fut)
    report["probes"] = run_probes(acts_tr, acts_va, tg_tr, tg_va)
    PROBES_PATH.write_text(json.dumps(report["probes"], indent=1))
    for row in report["probes"]["concepts"]:
        print(f"  probe {row['concept']:<20} " + " ".join(f"{k}={v['balanced_accuracy']:.2f}" for k, v in row["layers"].items()))

    # tuned lens on single-window activations (same shape as live inference)
    print("fitting tuned lens...")
    # same temporal context as live inference (context_len windows)
    L = int(meta.get("context_len") or 1)
    single_tr = acts_tr if L >= 3 else collect_activations(model, [s[-L:] for s in tr_seq])
    single_va = acts_va if L >= 3 else collect_activations(model, [s[-L:] for s in va_seq])
    yv_single = np.array([infiltration_label(s[-1]) for s in va_seq])
    sv_single = np.array([(-1 if stage_label(s[-1]) is None else stage_label(s[-1])) for s in va_seq])
    report["tuned_lens"] = train_tuned_lens(model, infil_head, stage_clf, single_tr, single_va, yv_single, sv_single, meta.get("temperature"))
    for r in report["tuned_lens"]["layers"]:
        print(f"  lens {r['layer']:<14} auc={r['infiltration_auc']} stage_f1={r['stage_macro_f1']}")

    # benign reference: the benign training window the model is most sure about
    with torch.no_grad():
        p = torch.sigmoid(mf.infil_logit(infil_head, torch.tensor(Z))).numpy()
    benign = np.where((Y == 0))[0]
    centroid = Z[benign].mean(0)
    cand = benign[np.argsort(p[benign])[:200]]
    best = int(cand[np.argmin(((Z[cand] - centroid) ** 2).sum(1))])
    BENIGN_REF_PATH.write_text(json.dumps({"feature_vector": X[best].tolist(), "feature_names": list(CANONICAL_FEATURE_NAMES),
                                           "infiltration_probability": float(p[best]), "label": str(LAB[best]),
                                           "note": "Most-confidently-benign training window closest to the benign centroid; clean baseline for activation patching."}))
    report["benign_reference"] = {"infiltration_probability": float(p[best])}
    REPORT_PATH.write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {OUT}/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
