"""Probing classifiers (roadmap Task 5.3, evaluation doc Eval 2.2).

Linear probes are trained per (layer x concept) by
src/training/train_interpretability.py (run_probes) and saved to
probes.json; this module serves that matrix and interprets it -- does the
model build simple concepts early and temporal/forecasting concepts in the
GRU / predictor, as the architecture intends?
"""

from __future__ import annotations

import json
from pathlib import Path

from src.training.train_interpretability import run_probes  # noqa: F401  (re-exported: the trainer)

PROBES_PATH = Path("data/processed/interpretability/probes.json")
CONCEPT_LABELS = {
    "is_scanning": "Is any host scanning? (PortScan present)",
    "mitre_stage": "MITRE ATT&CK stage (6-class)",
    "is_escalating": "Is attack activity increasing vs 2 windows ago?",
    "attack_in_3_windows": "Will an attack be present 3 windows ahead?",
}


def load_probe_report() -> dict:
    if not PROBES_PATH.exists():
        return {"available": False, "reason": "probes not trained -- run python -m src.training.train_interpretability"}
    report = json.loads(PROBES_PATH.read_text())
    findings = []
    for row in report["concepts"]:
        row["label"] = CONCEPT_LABELS.get(row["concept"], row["concept"])
        layers = row["layers"]
        before = max(layers.get(l, {}).get("balanced_accuracy", 0) for l in ("input", "hgt_conv1", "hgt_conv2"))
        after = layers.get("gru", {}).get("balanced_accuracy", 0)
        if after - before >= 0.05:
            findings.append(f"'{row['label']}' becomes linearly decodable in the GRU (+{after - before:.2f} balanced accuracy over the graph layers) -- the temporal encoder is adding it.")
        elif row["best_layer"] in ("input", "hgt_conv1"):
            findings.append(f"'{row['label']}' is already decodable from {row['best_layer']} ({layers[row['best_layer']]['balanced_accuracy']:.2f}); later layers preserve rather than create it.")
    report["available"] = True
    report["findings"] = findings
    return report
