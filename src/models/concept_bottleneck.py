"""Concept bottleneck (roadmap Task 8.1, evaluation doc Eval 4.2).

An alternative, fully transparent prediction path that runs alongside the
standard heads:

    z (128-d NetJEPA embedding) -> concept predictor -> 10 human-readable
    concept probabilities -> ONE linear layer -> 6 MITRE stages

The final decision is a plain weighted sum of named concepts, so an analyst
can read exactly why a stage was chosen, and can override a concept ("that
'periodic beaconing' is our health-check cron job") and see how the stage
changes (`predict(..., overrides=...)`).

Concept labels are derived from the canonical features themselves (windows
are z-scored per capture by src.features.extract), so no manual labelling is
needed. They are deliberately simple and documented in CONCEPTS below.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.graph.state_builder import CANONICAL_FEATURE_NAMES
from src.scoring.attack_stage import STAGE_NAMES

_IDX = {n: i for i, n in enumerate(CANONICAL_FEATURE_NAMES)}

# name, description, rule over the (z-scored) canonical feature vector
CONCEPTS = [
    ("port_scanning", "Many distinct destination ports with short connections (or packet scan signature)",
     lambda x: ((x[_IDX["flow_dst_port_nunique"]] > 0.5) & (x[_IDX["flow_duration_mean"]] < -0.3)) | (x[_IDX["packet_port_scan_flagged"]] > 0.5)),
    ("syn_flag_anomaly", "Unusual TCP flag combination across the window (SYN floods / stealth scans)",
     lambda x: x[_IDX["flow_tcp_flags_union"]].abs() > 1.0),
    ("periodic_beaconing", "Regular, low-variance inter-arrival times with small transfers (C2 heartbeat rhythm)",
     lambda x: (x[_IDX["flow_iat_var_mean"]] < -0.2) & (x[_IDX["flow_duration_mean"]] < -0.2) & (x[_IDX["flow_bytes_mean"]] < 0.0)),
    ("lateral_fan_out", "One window touching unusually many destination hosts",
     lambda x: (x[_IDX["flow_dst_ip_nunique"]] > 1.0) | (x[_IDX["packet_dst_ip_nunique"]] > 1.0)),
    ("high_volume_transfer", "Byte / packet volume far above the capture's norm (possible exfiltration)",
     lambda x: (x[_IDX["flow_bytes_sum"]] > 1.0) | (x[_IDX["flow_packets_sum"]] > 1.0)),
    ("brute_force_burst", "Repeated connections to the same service with bursty timing (credential guessing / flooding)",
     lambda x: (x[_IDX["flow_iat_var_mean"]] > 0.5) & (x[_IDX["flow_dst_port_nunique"]] < -0.3)),
    ("long_lived_sessions", "Sessions lasting far longer than usual (tunnels, slow-rate DoS, persistent channels)",
     lambda x: x[_IDX["flow_duration_mean"]] > 1.0),
    ("abnormal_ttl", "TTL variance far from normal (spoofing / mixed-origin traffic)",
     lambda x: x[_IDX["packet_ttl_variance"]].abs() > 1.0),
    ("retransmission_storm", "Unusually many TCP retransmissions",
     lambda x: x[_IDX["packet_retransmit_count"]] > 1.0),
    ("payload_irregularity", "Payload sizes far more varied than usual (tunnelling / staging)",
     lambda x: x[_IDX["packet_payload_size_std"]] > 1.0),
]
CONCEPT_NAMES = [c[0] for c in CONCEPTS]
CONCEPT_DESCRIPTIONS = {c[0]: c[1] for c in CONCEPTS}


def concept_labels(X: torch.Tensor) -> torch.Tensor:
    """(N, F) z-scored canonical features -> (N, n_concepts) {0,1}."""
    X = torch.as_tensor(X, dtype=torch.float32)
    cols = [rule(X.T).float() for _, _, rule in CONCEPTS]
    return torch.stack(cols, dim=1)


class ConceptBottleneck(nn.Module):
    def __init__(self, embedding_dim: int = 128, n_concepts: int = len(CONCEPTS), n_stages: int = len(STAGE_NAMES)):
        super().__init__()
        self.concept_predictor = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, n_concepts))
        # no hidden layer: the stage decision is a transparent weighted sum
        self.classifier = nn.Linear(n_concepts, n_stages)

    def concepts(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return torch.sigmoid(self.concept_predictor(z))

    def forward(self, z: torch.Tensor):
        c = self.concepts(z)
        return c, self.classifier(c)

    @torch.no_grad()
    def predict(self, z: torch.Tensor, overrides: dict[str, float] | None = None) -> dict:
        """Human-readable prediction. overrides: concept_name -> value in
        [0, 1] set by an analyst (human-in-the-loop correction)."""
        self.eval()
        c = self.concepts(z)[0].clone()
        applied = {}
        for name, value in (overrides or {}).items():
            if name in CONCEPT_NAMES:
                c[CONCEPT_NAMES.index(name)] = float(value)
                applied[name] = float(value)
        logits = self.classifier(c.unsqueeze(0))[0]
        probs = torch.softmax(logits, dim=-1)
        idx = int(probs.argmax())
        w = self.classifier.weight[idx]
        contributions = sorted(
            ({"concept": n, "description": CONCEPT_DESCRIPTIONS[n], "probability": float(c[i]), "weight": float(w[i]),
              "contribution": float(c[i] * w[i])} for i, n in enumerate(CONCEPT_NAMES)),
            key=lambda d: -abs(d["contribution"]))
        active = [d for d in contributions if d["probability"] >= 0.5]
        reason = " + ".join(f"{d['concept'].replace('_', ' ')} ({d['probability']:.2f})" for d in (active or contributions[:3])[:4])
        return {
            "stage": STAGE_NAMES[idx], "stage_probability": float(probs[idx]),
            "distribution": {STAGE_NAMES[i]: float(p) for i, p in enumerate(probs)},
            "concepts": [{"concept": n, "description": CONCEPT_DESCRIPTIONS[n], "probability": float(c[i])} for i, n in enumerate(CONCEPT_NAMES)],
            "contributions": contributions, "overrides_applied": applied, "bias": float(self.classifier.bias[idx]),
            "explanation": f"Classified as {STAGE_NAMES[idx]} because: {reason} -> weighted sum -> {STAGE_NAMES[idx]} ({float(probs[idx]):.0%}).",
        }
