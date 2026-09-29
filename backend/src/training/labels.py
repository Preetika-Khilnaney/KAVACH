"""Derive training labels from a feature window's flow labels.

Uses `flow_labels_present` (every distinct label seen in the window), not
`flow_dominant_label` (the majority vote) -- with 100 rows per window (see
build_windows_cache.py), a rare attack like Infiltration (35 windows out
of 288k rows) never wins a majority vote and `flow_dominant_label` would
silently always say "BENIGN" for it. Presence, not dominance, is what
matters for training.
"""

from __future__ import annotations

from src.scoring.attack_stage import STAGE_NAMES

# Maps a MachineLearningCVE label to one of this project's 6 MITRE stage
# classes (Benign + the 5 the architecture doc scopes: Reconnaissance,
# Initial Access, Lateral Movement, Command and Control, Exfiltration).
# DoS Hulk/GoldenEye/slowloris/Slowhttptest, Heartbleed, and DDoS are
# deliberately NOT mapped: they're closer to MITRE's "Impact" tactic,
# which isn't one of this project's 6 classes. Forcing them into the
# nearest available bucket would just teach the model a wrong mapping, so
# windows whose only non-benign labels are these are excluded from the
# attack-stage loss (see stage_label below) -- they still count for the
# infiltration head, since "something anomalous is happening" is still
# true regardless of which stage it maps to.
STAGE_MAP = {
    "PortScan": "Reconnaissance",
    "FTP-Patator": "Initial Access",
    "SSH-Patator": "Initial Access",
    "Web Attack � Brute Force": "Initial Access",
    "Web Attack � XSS": "Initial Access",
    "Web Attack � Sql Injection": "Initial Access",
    "Infiltration": "Lateral Movement",
    "Bot": "Command and Control",
    # PCAP timeline labels (src/training/attack_timeline.py)
    "Heartbleed": "Initial Access",  # T1190 exploit of a public-facing service
    # CSE-CIC-IDS-2018 spellings
    "FTP-BruteForce": "Initial Access",
    "SSH-Bruteforce": "Initial Access",
    "Brute Force -Web": "Initial Access",
    "Brute Force -XSS": "Initial Access",
    "SQL Injection": "Initial Access",
    "Infilteration": "Lateral Movement",
    # CTU-13
    "Botnet": "Command and Control",
}


def _norm(label: str) -> str:
    """Encoding-proof label key: CIC-IDS-2017's web-attack labels contain a
    stray 0x96 byte that decodes differently under utf-8 vs latin-1."""
    import re

    return re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()


_STAGE_BY_NORM = {_norm(k): v for k, v in STAGE_MAP.items()}


def stage_for_label(label: str) -> str | None:
    return _STAGE_BY_NORM.get(_norm(label))


def infiltration_label(window: dict) -> float:
    """1.0 if any row in the window was non-benign, else 0.0.

    Uses the labels actually present (robust), not flow_attack_ratio: the v1
    windows cache z-scored that ratio, so "> 0" meant "above the file mean"."""
    present = window.get("flow_labels_present")
    if present is not None:
        return 1.0 if any(l != "BENIGN" for l in present) else 0.0
    return 1.0 if (window.get("flow_attack_ratio") or 0.0) > 0 else 0.0


def stage_label(window: dict) -> int | None:
    """MITRE stage class index for this window's target, or None if it
    should be excluded from the attack-stage loss (see module docstring)."""
    labels_present = window.get("flow_labels_present") or []
    non_benign = [l for l in labels_present if l != "BENIGN"]
    if not non_benign:
        return STAGE_NAMES.index("Benign")
    for lbl in non_benign:
        stage = stage_for_label(lbl)
        if stage is not None:
            return STAGE_NAMES.index(stage)
    return None
