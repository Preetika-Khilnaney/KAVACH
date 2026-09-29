"""Evaluation for NetJEPA v2 (shared by training, the benchmark API, the
leave-one-attack-out study and the CLI).

Everything is computed on held-out examples (temporal split) with the model
in eval mode:

* detection at t      -- infiltration head on z_t (F1/precision/recall/FPR/AUC), per source
* forecasting t+k     -- infiltration head on the predictor's k-step mean
                         rollout, scored against the label of window t+k
* stage mapping       -- stage head on z_t and on the k-step forecast
* probabilistic fit   -- Gaussian NLL of the true next embedding and the
                         coverage of the predicted 90% interval
* baselines           -- logistic regression on the current window's
                         features, and a *temporal* logistic regression on
                         the concatenated context windows (so "temporal
                         modelling helps" is tested against a baseline that
                         also sees history, not just a single window)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.explainability.model_forward import infil_logit
from src.scoring.attack_stage import STAGE_NAMES


def binary_metrics(y, pred, scores=None) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"f1": float(f1_score(y, pred, zero_division=0)), "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)), "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
           "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "n": int(len(y)), "positives": int(y.sum())}
    if scores is not None and len(set(y.tolist())) > 1:
        out["auc"] = float(roc_auc_score(y, scores))
    return out


@torch.no_grad()
def model_outputs(model, infil_head, stage_clf, examples, graph_fn, context_len: int, horizon: int, batch_size: int = 128) -> dict:
    """Runs the model over examples. graph_fn(example, i) -> graph of the
    i-th window of the example's (context + futures) sequence."""
    model.eval()
    cur_logit, fut_logit, stage_now, stage_fut, nll, cover = [], [], [], [], [], []
    Z = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        ctx = [[graph_fn(ex, j) for j in range(context_len)] for ex in chunk]
        tgt = [[graph_fn(ex, j) for j in range(1, context_len + 1)] for ex in chunk]
        _, z_tgt, z = model(ctx, tgt)
        Z.append(z.numpy())
        cur_logit.append(infil_logit(infil_head, z).numpy())
        stage_now.append(stage_clf(z).numpy())
        mean, logvar = model.predictor.distribution(z)
        var = logvar.exp()
        nll.append((0.5 * ((z_tgt - mean) ** 2 / var + logvar + np.log(2 * np.pi))).mean(dim=1).numpy())
        cover.append(((z_tgt - mean).abs() <= 1.645 * var.sqrt()).float().mean(dim=1).numpy())
        m = z
        fl, fs = [], []
        for _ in range(horizon):
            m = model.predictor(m)
            fl.append(infil_logit(infil_head, m).numpy())
            fs.append(stage_clf(m).numpy())
        fut_logit.append(np.stack(fl, 1))
        stage_fut.append(np.stack(fs, 1))
    return {"cur_logit": np.concatenate(cur_logit), "fut_logit": np.concatenate(fut_logit), "stage_now": np.concatenate(stage_now),
            "stage_fut": np.concatenate(stage_fut), "nll": np.concatenate(nll), "coverage90": np.concatenate(cover), "Z": np.concatenate(Z)}


def _sig(x, T=1.0):
    return 1.0 / (1.0 + np.exp(-np.asarray(x) / (T or 1.0)))


def summarize(outputs: dict, examples, temperature: float | None, threshold: float, horizon: int) -> dict:
    y0 = np.array([ex.infil[0] for ex in examples])
    s0 = np.array([ex.stage[0] for ex in examples])
    p0 = _sig(outputs["cur_logit"], temperature)
    res = {"detection": binary_metrics(y0, p0 >= threshold, p0)}
    by_src = {}
    for src in sorted({ex.source for ex in examples}):
        m = np.array([ex.source == src for ex in examples])
        if m.sum() and len(set(y0[m].tolist())) > 1:
            by_src[src] = binary_metrics(y0[m], p0[m] >= threshold, p0[m])
    res["detection_by_source"] = by_src
    fk = {}
    for k in range(1, horizon + 1):
        yk = np.array([ex.infil[k] for ex in examples])
        pk_forecast = _sig(outputs["fut_logit"][:, k - 1], temperature)
        entry = {"auc_forecast_head": float(roc_auc_score(yk, pk_forecast)) if len(set(yk.tolist())) > 1 else None,
                 "auc_current_score": float(roc_auc_score(yk, p0)) if len(set(yk.tolist())) > 1 else None}
        # transitions that matter most: benign now, attack later
        onset = (y0 == 0)
        if onset.sum() and len(set(yk[onset].tolist())) > 1:
            entry["auc_onset_forecast_head"] = float(roc_auc_score(yk[onset], pk_forecast[onset]))
            entry["onset_positives"] = int(yk[onset].sum())
        fk[f"t+{k}"] = entry
    res["forecast"] = fk
    m = s0 >= 0
    if m.any():
        pred = outputs["stage_now"][m].argmax(1)
        present = sorted(set(s0[m].tolist()))
        res["stage_now"] = {"macro_f1": float(f1_score(s0[m], pred, labels=present, average="macro", zero_division=0)),
                            "accuracy": float((pred == s0[m]).mean()),
                            "confusion_matrix": confusion_matrix(s0[m], pred, labels=list(range(len(STAGE_NAMES)))).tolist()}
    sf = {}
    for k in range(1, horizon + 1):
        sk = np.array([ex.stage[k] for ex in examples])
        mk = sk >= 0
        if mk.any():
            pred = outputs["stage_fut"][mk, k - 1].argmax(1)
            present = sorted(set(sk[mk].tolist()))
            sf[f"t+{k}"] = {"macro_f1": float(f1_score(sk[mk], pred, labels=present, average="macro", zero_division=0)),
                            "accuracy": float((pred == sk[mk]).mean())}
    res["stage_forecast"] = sf
    res["transition_model"] = {"mean_gaussian_nll_per_dim": float(outputs["nll"].mean()),
                               "coverage_of_90pct_interval": float(outputs["coverage90"].mean()),
                               "note": "Coverage near 0.90 means the predicted uncertainty of P(S_t+1|S_t) is well calibrated."}
    res["embed_std"] = float(outputs["Z"].std(0).mean())
    return res


def baselines(train_examples, val_examples, feature_fn, threshold: float = 0.5, context_len: int = 3) -> dict:
    """Logistic regression on (a) the current window only and (b) the
    concatenated context windows (temporal baseline). Both class-balanced,
    trained on the same training examples."""
    def feats(exs, temporal):
        rows = []
        for ex in exs:
            if temporal:
                rows.append(np.concatenate([feature_fn(w) for w in ex.context[-context_len:]]))
            else:
                rows.append(np.asarray(feature_fn(ex.context[-1])))
        return np.array(rows)

    ytr = np.array([ex.infil[0] for ex in train_examples])
    yva = np.array([ex.infil[0] for ex in val_examples])
    out = {}
    for name, temporal in (("logreg_current_window", False), ("logreg_temporal_context", True)):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(feats(train_examples, temporal), ytr)
        p = clf.predict_proba(feats(val_examples, temporal))[:, 1]
        out[name] = binary_metrics(yva, p >= 0.5, p)
        by_src = {}
        for src in sorted({ex.source for ex in val_examples}):
            m = np.array([ex.source == src for ex in val_examples])
            if len(set(yva[m].tolist())) > 1:
                by_src[src] = binary_metrics(yva[m], p[m] >= 0.5, p[m])
        out[name]["by_source"] = by_src
        if not temporal:
            fk = {}
            for k in (1, 2, 3):
                yk = np.array([ex.infil[k] for ex in val_examples if len(ex.infil) > k])
                if len(yk) == len(p) and len(set(yk.tolist())) > 1:
                    fk[f"t+{k}"] = float(roc_auc_score(yk, p))
            out[name]["forecast_auc_current_score"] = fk
    return out


def gaussian_nll(target, mean, logvar):
    return 0.5 * ((target - mean) ** 2 * torch.exp(-logvar) + logvar).mean()


def bce_logits(logits, y, pos_weight):
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
