"""Confidence calibration (roadmap Task 7.1, evaluation doc Eval 1.2).

When NetJEPA says "82% infiltration probability", an analyst should be able
to take that at face value. Temperature scaling (Guo et al., 2017) learns a
single scalar T on held-out data and rescales logits: p = sigmoid(logit / T)
for the binary infiltration head, softmax(logits / T) for the 6-class stage
head. It changes confidence, never the ranking/argmax, so F1/AUC at a fixed
threshold ordering are unaffected.

Expected Calibration Error (ECE) bins predictions into 10 confidence
buckets and reports the weighted gap between average confidence and actual
accuracy -- lower is better; < 0.05 is "trustworthy as a probability".
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0, binary: bool = True):
        self.temperature = float(temperature)
        self.binary = binary

    def fit(self, logits, labels, max_iter: int = 200) -> "TemperatureScaler":
        """logits: (N,) for binary, (N, C) for multiclass. labels: (N,)."""
        logits = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
        labels = torch.as_tensor(np.asarray(labels))
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            t = log_t.exp()
            if self.binary:
                loss = F.binary_cross_entropy_with_logits(logits / t, labels.float())
            else:
                loss = F.cross_entropy(logits / t, labels.long())
            loss.backward()
            return loss

        opt.step(closure)
        t = float(log_t.exp().item())
        self.temperature = t if math.isfinite(t) and t > 0 else 1.0
        return self

    def scale(self, logits):
        """Calibrated probabilities for new logits."""
        logits = torch.as_tensor(logits, dtype=torch.float32)
        if self.binary:
            return torch.sigmoid(logits / self.temperature)
        return torch.softmax(logits / self.temperature, dim=-1)


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1 - eps)
    return math.log(p / (1 - p))


def calibrate_probability(p: float, temperature: float | None) -> float:
    """Apply binary temperature scaling to an already-sigmoided probability."""
    if not temperature or temperature == 1.0:
        return float(p)
    return 1.0 / (1.0 + math.exp(-logit(p) / temperature))


def expected_calibration_error(probs, labels, n_bins: int = 10) -> dict:
    """Binary ECE on P(attack) vs the 0/1 label, plus the reliability
    diagram bins for the frontend."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece, n = [], 0.0, max(1, len(probs))
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & ((probs < hi) if hi < 1.0 else (probs <= hi))
        count = int(mask.sum())
        if count:
            conf = float(probs[mask].mean())
            acc = float(labels[mask].mean())
            ece += count / n * abs(conf - acc)
        else:
            conf = acc = None
        bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": count, "avg_confidence": conf, "observed_frequency": acc})
    return {"ece": float(ece), "bins": bins}


def multiclass_ece(prob_matrix, labels, n_bins: int = 10) -> float:
    prob_matrix = np.asarray(prob_matrix, dtype=float)
    labels = np.asarray(labels)
    conf = prob_matrix.max(axis=1)
    correct = (prob_matrix.argmax(axis=1) == labels).astype(float)
    return expected_calibration_error(conf, correct, n_bins)["ece"]
