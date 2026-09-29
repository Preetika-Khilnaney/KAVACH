"""v2 components: fast PCAP reader, joint flow+packet windows, attack
timeline labels, batched encoder equivalence, probabilistic predictor,
supervised-transition examples, CLI."""

import torch
from scapy.all import IP, TCP, UDP, Ether, Raw, wrpcap

from src.features.joint_windows import iter_joint_windows, normalize_windows
from src.graph.state_builder import CANONICAL_FEATURE_NAMES, build_graph_state
from src.ingestion import fastpcap, packet_parser
from src.models.netjepa import NetJEPA, graph_batch_from_state
from src.training.attack_timeline import make_labeler


def _capture(tmp_path, t0=1499262000.0):
    pkts = []
    for i in range(30):  # scanner: one source, many ports
        p = Ether() / IP(src="172.16.0.1", dst="192.168.10.50", ttl=60) / TCP(sport=40000, dport=1000 + i, flags="S")
        p.time = t0 + i * 0.1
        pkts.append(p)
    for i in range(6):  # a normal bidirectional exchange
        a = Ether() / IP(src="192.168.10.5", dst="8.8.8.8") / UDP(sport=5353, dport=53) / Raw(b"x" * 20)
        b = Ether() / IP(src="8.8.8.8", dst="192.168.10.5") / UDP(sport=53, dport=5353) / Raw(b"y" * 60)
        a.time, b.time = t0 + 6 + i, t0 + 6.2 + i
        pkts += [a, b]
    path = tmp_path / "Wednesday-test.pcap"
    wrpcap(str(path), pkts)
    return str(path)


def test_fastpcap_matches_scapy_fields(tmp_path):
    path = _capture(tmp_path)
    fast = list(fastpcap.iter_packet_records(path))
    assert len(fast) == 42
    assert fast[0]["src_ip"] == "172.16.0.1" and fast[0]["dst_port"] == 1000 and fast[0]["tcp_flags"] == 0x02
    assert fast[-1]["protocol"] == "UDP" and fast[-1]["payload_size"] == 60
    assert any(r["port_scan_signature"] for r in fast[:30])
    batches = list(packet_parser.iter_packets(path, batch_size=10))
    assert sum(len(b) for b in batches) == 42  # packet_parser now uses the fast reader


def test_joint_windows_have_both_levels_and_timeline_labels(tmp_path):
    path = _capture(tmp_path)
    labeler = make_labeler("Wednesday")  # t0 = 2017-07-05 10:40 ADT? -> inside DoS Hulk only if in [10:43, 11:00]
    wins = list(iter_joint_windows(fastpcap.iter_packet_records(path), window_seconds=5, labeler=labeler))
    assert len(wins) >= 2
    w0 = wins[0]
    assert w0["has_flow_level"] == 1.0 and w0["has_packet_level"] == 1.0
    assert w0["flow_count"] == 30 and w0["flow_dst_port_nunique"] == 30  # 30 one-packet flows
    assert w0["packet_port_scan_flagged"] is True
    udp = [w for w in wins if w["flow_udp_fraction"] > 0][0]
    assert udp["flow_bwd_fwd_byte_ratio"] > 1.0  # DNS answers bigger than questions
    norm, stats = normalize_windows(wins)
    assert "flow_bytes_sum" in stats and norm[0]["flow_count"] == 30  # counts stay raw (log-scaled later)


def test_attack_timeline_labels_only_attacker_victim_in_interval():
    lab = make_labeler("Wednesday")
    hulk_ts = 1499262180.0 + 600  # 2017-07-05 10:53 ADT (13:53 UTC) -- inside DoS Hulk
    assert lab({"timestamp": hulk_ts, "src_ip": "172.16.0.1", "dst_ip": "192.168.10.50"}) == "DoS Hulk"
    assert lab({"timestamp": hulk_ts, "src_ip": "192.168.10.5", "dst_ip": "192.168.10.50"}) is None
    assert lab({"timestamp": hulk_ts - 86400, "src_ip": "172.16.0.1", "dst_ip": "192.168.10.50"}) is None


def test_feature_schema_v2_has_both_levels_and_modality_flags():
    names = list(CANONICAL_FEATURE_NAMES)
    assert {"flow_bwd_fwd_byte_ratio", "flow_udp_fraction", "packet_ttl_variance", "has_flow_level", "has_packet_level"} <= set(names)
    st = build_graph_state({"flow_count": 100, "flow_bytes_sum": 1.0})
    assert st["feature_vector"][names.index("has_flow_level")] == 1.0
    assert st["feature_vector"][names.index("has_packet_level")] == 0.0


def test_batched_encoder_matches_single_path_and_predictor_is_gaussian():
    torch.manual_seed(0)
    m = NetJEPA(len(CANONICAL_FEATURE_NAMES), 16, 32).eval()
    ws = [{"flow_count": 10 + i, "flow_bytes_sum": float(i), "packet_count": 5, "packet_edges": [("a", "b"), ("b", "c")] if i % 2 else []} for i in range(5)]
    seqs = [[graph_batch_from_state(build_graph_state(w)) for w in ws[i : i + 2]] for i in range(4)]
    with torch.no_grad():
        a = m.context_encoder.forward_batch(seqs)
        b = torch.stack([m.context_encoder(s) for s in seqs])
    assert torch.allclose(a, b, atol=1e-4)
    z = torch.randn(3, 16)
    mean, logvar = m.predictor.distribution(z)
    assert mean.shape == logvar.shape == (3, 16)
    assert torch.allclose(mean, m.predictor(z))
    samples = m.predictor.sample_rollout(z[0], steps=4, n_samples=8)
    assert samples.shape == (8, 4, 16) and samples.std(0).mean() > 0


def test_examples_carry_future_labels_for_transition_supervision():
    from src.training.dataset_v2 import build_examples_v2

    ws = [{"flow_count": 100, "flow_labels_present": ["BENIGN"] if i < 6 else ["BENIGN", "PortScan"]} for i in range(10)]
    train, val = build_examples_v2({"csv:x": ("csv", ws)}, context_len=3, horizon=3, val_fraction=0.2)
    ex = next(e for e in train + val if e.index == 4)
    assert ex.infil == [0.0, 0.0, 1.0, 1.0]  # t=4 benign; t+2, t+3 are PortScan
    assert ex.stage[2] == 1  # Reconnaissance at t+2
    tr2, _, dropped = build_examples_v2({"csv:x": ("csv", ws)}, 3, 3, 0.2, exclude_labels={"PortScan"}, return_dropped=True)
    assert all("PortScan" not in l for e in tr2 for w in (*e.context, *e.futures) for l in w["flow_labels_present"])
    assert dropped


def test_cli_predict_runs_offline(tmp_path, capsys):
    from src.cli import main

    path = _capture(tmp_path)
    assert main(["predict", path, "--all", "--json", str(tmp_path / "out.json")]) == 0
    out = capsys.readouterr().out
    assert "Infiltration probability timeline" in out and "Stage annotations" in out
    assert (tmp_path / "out.json").exists()


def test_novelty_detector_percentiles():
    import numpy as np

    from src.scoring.novelty import NoveltyDetector

    rng = np.random.default_rng(0)
    ref = rng.normal(size=(500, 8)).astype("float32")
    det = NoveltyDetector.fit(ref, k=5)
    near, far = det.assess(rng.normal(size=8)), det.assess(np.full(8, 10.0))
    assert far["is_novel"] and not near["is_novel"]
    assert far["benign_percentile"] == 1.0 and near["benign_percentile"] < 1.0
    det2 = NoveltyDetector.from_state(det.state())
    assert abs(det2.threshold - det.threshold) < 1e-3
