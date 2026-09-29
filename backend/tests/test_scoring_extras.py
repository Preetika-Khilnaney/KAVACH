"""Calibration, remediation playbooks, concept labels."""

import numpy as np
import torch

from src.models.concept_bottleneck import CONCEPT_NAMES, ConceptBottleneck, concept_labels
from src.scoring.calibration import TemperatureScaler, calibrate_probability, expected_calibration_error
from src.scoring.remediation import playbook_stages, remediation_for_stage


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    logits = (y * 2 - 1) * 1.0 * 4 + rng.normal(0, 4, 2000)  # overconfident logits
    raw = 1 / (1 + np.exp(-logits))
    ts = TemperatureScaler().fit(logits, y)
    cal = ts.scale(logits).numpy()
    assert ts.temperature > 1.0
    assert expected_calibration_error(cal, y)["ece"] < expected_calibration_error(raw, y)["ece"]
    assert abs(calibrate_probability(0.5, 2.0) - 0.5) < 1e-9


def test_remediation_uses_real_ips_and_covers_every_stage():
    assert set(playbook_stages()) == {"Reconnaissance", "Initial Access", "Lateral Movement", "Command and Control", "Exfiltration"}
    acts = remediation_for_stage("Reconnaissance", {"src_ip": "10.0.1.1"})
    assert acts[0]["priority"] == "immediate" and "iptables" in acts[0]["command"] and "10.0.1.1" in acts[0]["command"]
    assert all(a["mitigation"].startswith("M1") for a in acts)


def test_concept_bottleneck_shapes_and_overrides():
    from src.graph.state_builder import CANONICAL_FEATURE_NAMES

    X = torch.zeros(4, len(CANONICAL_FEATURE_NAMES))
    assert concept_labels(X).shape == (4, len(CONCEPT_NAMES))
    cb = ConceptBottleneck(16)
    out = cb.predict(torch.randn(16), overrides={"port_scanning": 1.0})
    assert out["overrides_applied"] == {"port_scanning": 1.0}
    assert abs(sum(out["distribution"].values()) - 1) < 1e-5 and "Classified as" in out["explanation"]
