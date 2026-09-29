"""Joint flow-level + packet-level windows from a packet stream.

One pass over a capture (any size -- constant memory) produces, for every
`window_seconds` window, BOTH feature levels the challenge requires, from
the same packets and therefore on the same real time base:

* packet level: TTL mean/variance, TCP window size, fragment rate,
  payload-size mean/std, retransmissions, port-scan signature, host counts;
* flow level: packets are grouped into bidirectional flows (5-tuple, the
  first packet's sender is the initiator) with an IPFIX-style active
  timeout equal to the window length, so each window's flow records are
  exactly the flows active in it: count, distinct ports/hosts, TCP-flag
  union, bytes/packets, duration, inter-arrival mean/std/max, forward vs
  backward byte and packet ratios, protocol mix.

The output keys match src.features.extract (flow_* / packet_*), so the same
graph builder and model consume windows from a CSV, a PCAP, or both.

`labeler(record) -> str | None` optionally tags packets with a ground-truth
attack name (see src/training/attack_timeline.py); each window then carries
flow_labels_present / flow_attack_ratio exactly like CSV-derived windows.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Iterator

FLOW_PROTO_TCP, FLOW_PROTO_UDP = "TCP", "UDP"


class _Online:
    __slots__ = ("n", "mean", "m2", "max")

    def __init__(self):
        self.n, self.mean, self.m2, self.max = 0, 0.0, 0.0, 0.0

    def add(self, x: float):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        if x > self.max:
            self.max = x

    @property
    def var(self):
        return self.m2 / self.n if self.n else 0.0

    @property
    def std(self):
        return math.sqrt(self.var)


class _FlowSlice:
    __slots__ = ("initiator", "responder", "sport", "dport", "proto", "first", "last", "fwd_pkts", "bwd_pkts",
                 "fwd_bytes", "bwd_bytes", "flags", "iat")

    def __init__(self, rec):
        self.initiator, self.responder = rec["src_ip"], rec["dst_ip"]
        self.sport, self.dport, self.proto = rec["src_port"], rec["dst_port"], rec["protocol"]
        self.first = self.last = rec["timestamp"]
        self.fwd_pkts = self.bwd_pkts = self.fwd_bytes = self.bwd_bytes = self.flags = 0
        self.iat = _Online()


class _Window:
    def __init__(self, start: float):
        self.start = start
        self.n = 0
        self.ttl = _Online()
        self.win = _Online()
        self.payload = _Online()
        self.frag = 0
        self.retrans = 0
        self.scan = False
        self.src_ips: set = set()
        self.dst_ips: set = set()
        self.edges: dict = {}
        self.flows: dict = {}
        self.labels: dict = {}
        self.labelled_pkts = 0


class JointWindower:
    def __init__(self, window_seconds: float = 5.0, max_edges: int = 64, labeler: Callable[[dict], str | None] | None = None):
        self.ws = float(window_seconds)
        self.max_edges = max_edges
        self.labeler = labeler
        self.cur: _Window | None = None

    def _key(self, rec):
        a = (rec["src_ip"], rec["src_port"])
        b = (rec["dst_ip"], rec["dst_port"])
        return (min(a, b, key=str), max(a, b, key=str), rec["protocol"])

    def add(self, rec: dict) -> dict | None:
        """Feed one packet record (time-ordered). Returns a finished window
        dict when rec starts a new window, else None. Empty windows between
        packets are skipped."""
        ts = rec["timestamp"]
        start = math.floor(ts / self.ws) * self.ws
        done = None
        if self.cur is None or start != self.cur.start:
            if self.cur is not None and start > self.cur.start:
                done = self._finish(self.cur)
            if self.cur is None or start > self.cur.start:
                self.cur = _Window(start)
        w = self.cur
        w.n += 1
        w.ttl.add(rec["ttl"])
        if rec.get("tcp_window_size") is not None:
            w.win.add(rec["tcp_window_size"])
        w.payload.add(rec["payload_size"])
        w.frag += bool(rec["fragment_flag"])
        w.retrans += bool(rec["is_retransmission"])
        w.scan = w.scan or bool(rec["port_scan_signature"])
        w.src_ips.add(rec["src_ip"])
        w.dst_ips.add(rec["dst_ip"])
        e = (rec["src_ip"], rec["dst_ip"])
        w.edges[e] = w.edges.get(e, 0) + 1
        if self.labeler is not None:
            lab = self.labeler(rec)
            if lab:
                w.labels[lab] = w.labels.get(lab, 0) + 1
                w.labelled_pkts += 1

        k = self._key(rec)
        fl = w.flows.get(k)
        if fl is None:
            fl = w.flows[k] = _FlowSlice(rec)
        else:
            fl.iat.add((ts - fl.last) * 1e6)  # microseconds, like CICFlowMeter
            fl.last = ts
        size = rec.get("packet_length") or rec["payload_size"]
        if rec["src_ip"] == fl.initiator and rec["src_port"] == fl.sport:
            fl.fwd_pkts += 1
            fl.fwd_bytes += size
        else:
            fl.bwd_pkts += 1
            fl.bwd_bytes += size
        if rec.get("tcp_flags"):
            fl.flags |= rec["tcp_flags"]
        return done

    def flush(self) -> dict | None:
        if self.cur is None:
            return None
        out, self.cur = self._finish(self.cur), None
        return out

    def _finish(self, w: _Window) -> dict:
        flows = list(w.flows.values())
        n_f = len(flows)
        bytes_ = [f.fwd_bytes + f.bwd_bytes for f in flows]
        pkts = [f.fwd_pkts + f.bwd_pkts for f in flows]
        fwd_b, bwd_b = sum(f.fwd_bytes for f in flows), sum(f.bwd_bytes for f in flows)
        fwd_p, bwd_p = sum(f.fwd_pkts for f in flows), sum(f.bwd_pkts for f in flows)
        flags = 0
        for f in flows:
            flags |= f.flags
        edges = sorted(w.edges.items(), key=lambda kv: -kv[1])[: self.max_edges]
        attack_pkts = w.labelled_pkts
        win = {
            "window_id": str(int(w.start)),
            "window_start": w.start,
            "window_end": w.start + self.ws,
            "flow_count": n_f,
            "flow_src_ip_nunique": len({f.initiator for f in flows}),
            "flow_dst_ip_nunique": len({f.responder for f in flows}),
            "flow_dst_port_nunique": len({f.dport for f in flows if f.dport is not None}),
            "flow_tcp_flags_union": flags,
            "flow_bytes_sum": float(sum(bytes_)),
            "flow_bytes_mean": float(sum(bytes_) / n_f) if n_f else 0.0,
            "flow_packets_sum": float(sum(pkts)),
            "flow_duration_mean": float(sum(f.last - f.first for f in flows) / n_f) if n_f else 0.0,
            "flow_iat_mean_mean": float(sum(f.iat.mean for f in flows) / n_f) if n_f else 0.0,
            "flow_iat_var_mean": float(sum(f.iat.std for f in flows) / n_f) if n_f else 0.0,
            "flow_iat_max_max": float(max((f.iat.max for f in flows), default=0.0)),
            "flow_bwd_fwd_byte_ratio": float(bwd_b / fwd_b) if fwd_b else 0.0,
            "flow_bwd_fwd_packet_ratio": float(bwd_p / fwd_p) if fwd_p else 0.0,
            "flow_tcp_fraction": sum(f.proto == FLOW_PROTO_TCP for f in flows) / n_f if n_f else 0.0,
            "flow_udp_fraction": sum(f.proto == FLOW_PROTO_UDP for f in flows) / n_f if n_f else 0.0,
            "packet_count": w.n,
            "packet_ttl_mean": w.ttl.mean,
            "packet_ttl_variance": w.ttl.var,
            "packet_tcp_window_mean": w.win.mean,
            "packet_fragment_rate": w.frag / w.n if w.n else 0.0,
            "packet_payload_size_mean": w.payload.mean,
            "packet_payload_size_std": w.payload.std,
            "packet_retransmit_count": float(w.retrans),
            "packet_port_scan_flagged": bool(w.scan),
            "packet_src_ip_nunique": len(w.src_ips),
            "packet_dst_ip_nunique": len(w.dst_ips),
            "packet_edges": [e for e, _ in edges],
            "has_flow_level": 1.0,
            "has_packet_level": 1.0,
        }
        if self.labeler is not None:
            present = sorted(w.labels) or []
            win["flow_labels_present"] = (["BENIGN"] if attack_pkts < w.n else []) + present
            win["flow_attack_ratio"] = attack_pkts / w.n if w.n else 0.0
            win["flow_dominant_label"] = max(w.labels, key=w.labels.get) if w.labels and attack_pkts * 2 >= w.n else "BENIGN"
            win["attack_packets"] = attack_pkts
        return win


def iter_joint_windows(records: Iterable[dict], window_seconds: float = 5.0, max_edges: int = 64,
                       labeler: Callable[[dict], str | None] | None = None) -> Iterator[dict]:
    """Records must be time-ordered (they are, straight out of a capture)."""
    jw = JointWindower(window_seconds, max_edges, labeler)
    for rec in records:
        w = jw.add(rec)
        if w is not None:
            yield w
    last = jw.flush()
    if last is not None:
        yield last


# Fields that are NOT z-scored per capture (identifiers, flags, labels).
_SKIP = {"window_id", "window_start", "window_end", "packet_edges", "packet_port_scan_flagged", "has_flow_level",
         "has_packet_level", "flow_labels_present", "flow_attack_ratio", "flow_dominant_label", "attack_packets",
         "flow_count", "packet_count", "flow_tcp_fraction", "flow_udp_fraction"}


def normalize_windows(windows: list[dict], stats: dict | None = None) -> tuple[list[dict], dict]:
    """Per-capture z-scoring of numeric features (same convention as
    src.features.extract). Returns (windows, stats); pass `stats` to reuse
    a reference normalisation."""
    import numpy as np

    if not windows:
        return windows, stats or {}
    keys = [k for k, v in windows[0].items() if k not in _SKIP and isinstance(v, (int, float)) and not isinstance(v, bool)]
    if stats is None:
        stats = {}
        for k in keys:
            vals = np.array([float(w.get(k) or 0.0) for w in windows])
            stats[k] = (float(vals.mean()), float(vals.std()))
    out = []
    for w in windows:
        w = dict(w)
        for k, (mu, sd) in stats.items():
            if k in w:
                w[k] = 0.0 if not sd else (float(w[k] or 0.0) - mu) / sd
        out.append(w)
    return out, stats
