"""Fast, dependency-free streaming reader for pcap and pcapng captures.

scapy dissects every layer of every packet in Python, which manages a few
thousand packets per second. The CIC-IDS-2017 captures hold tens of
millions of packets (10-13 GB), so this module parses only what KAVACH
needs (Ethernet/raw/Linux-SLL -> IPv4 -> TCP/UDP headers) with `struct`,
about 50-100x faster. It yields the same per-packet record shape as
src.ingestion.packet_parser, so both paths feed the same feature code.

Supported: classic pcap (both byte orders, micro/nanosecond), pcapng
(SHB/IDB/EPB/SPB/legacy PB, per-interface link type and timestamp
resolution). IPv6 and non-IP frames are skipped (counted, not parsed).
"""

from __future__ import annotations

import struct
from collections import deque
from typing import Iterator

LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101
LINKTYPE_LINUX_SLL = 113
LINKTYPE_RAW_ALT = 12

_PCAP_MAGICS = {
    b"\xd4\xc3\xb2\xa1": ("<", 1e-6),
    b"\xa1\xb2\xc3\xd4": (">", 1e-6),
    b"\x4d\x3c\xb2\xa1": ("<", 1e-9),
    b"\xa1\xb2\x3c\x4d": (">", 1e-9),
}
_PCAPNG_SHB = 0x0A0D0D0A

# Port-scan heuristic (same parameters as packet_parser)
PORT_SCAN_DISTINCT_TARGETS = 20
PORT_SCAN_WINDOW_SECONDS = 5.0


def _iter_frames(path: str) -> Iterator[tuple[float, int, bytes]]:
    """Yield (timestamp, linktype, frame_bytes)."""
    with open(path, "rb") as f:
        head = f.read(4)
        if head in _PCAP_MAGICS:
            endian, res = _PCAP_MAGICS[head]
            rest = f.read(20)
            linktype = struct.unpack(endian + "HHiIII", rest)[5] & 0x0FFFFFFF
            hdr = struct.Struct(endian + "IIII")
            while True:
                h = f.read(16)
                if len(h) < 16:
                    return
                sec, frac, incl, _orig = hdr.unpack(h)
                data = f.read(incl)
                if len(data) < incl:
                    return
                yield sec + frac * res, linktype, data
        elif head and struct.unpack("<I", head)[0] == _PCAPNG_SHB:
            yield from _iter_pcapng(f, head)
        else:
            raise ValueError(f"{path!r} is not a pcap/pcapng capture")


def _iter_pcapng(f, first4: bytes) -> Iterator[tuple[float, int, bytes]]:
    endian = "<"
    interfaces: list[tuple[int, float]] = []  # (linktype, ts_resolution)
    block_type = struct.unpack("<I", first4)[0]
    pending_type = block_type
    while True:
        if pending_type is None:
            raw = f.read(4)
            if len(raw) < 4:
                return
            block_type = struct.unpack(endian + "I", raw)[0]
        else:
            block_type, pending_type = pending_type, None
        raw_len = f.read(4)
        if len(raw_len) < 4:
            return
        if block_type == _PCAPNG_SHB:
            bom = f.read(4)
            endian = "<" if bom == b"\x4d\x3c\x2b\x1a" else ">"
            total = struct.unpack(endian + "I", raw_len)[0]
            f.read(total - 12)  # rest of SHB incl. trailing length
            interfaces = []
            continue
        total = struct.unpack(endian + "I", raw_len)[0]
        body = f.read(total - 8)
        if len(body) < total - 8:
            return
        if block_type == 1:  # Interface Description Block
            linktype = struct.unpack(endian + "H", body[:2])[0]
            res = 1e-6
            opts = body[8:-4]
            i = 0
            while i + 4 <= len(opts):
                code, olen = struct.unpack(endian + "HH", opts[i : i + 4])
                if code == 0:
                    break
                if code == 9 and olen >= 1:  # if_tsresol
                    v = opts[i + 4]
                    res = 2.0 ** -(v & 0x7F) if v & 0x80 else 10.0 ** -v
                i += 4 + ((olen + 3) & ~3)
            interfaces.append((linktype, res))
        elif block_type == 6:  # Enhanced Packet Block
            iface, ts_hi, ts_lo, cap_len = struct.unpack(endian + "IIII", body[:16])
            linktype, res = interfaces[iface] if iface < len(interfaces) else (LINKTYPE_ETHERNET, 1e-6)
            yield ((ts_hi << 32) | ts_lo) * res, linktype, body[20 : 20 + cap_len]
        elif block_type == 2:  # obsolete Packet Block
            iface, _drops, ts_hi, ts_lo, cap_len = struct.unpack(endian + "HHIII", body[:16])
            linktype, res = interfaces[iface] if iface < len(interfaces) else (LINKTYPE_ETHERNET, 1e-6)
            yield ((ts_hi << 32) | ts_lo) * res, linktype, body[20 : 20 + cap_len]
        elif block_type == 3:  # Simple Packet Block (no timestamp)
            linktype, _ = interfaces[0] if interfaces else (LINKTYPE_ETHERNET, 1e-6)
            yield 0.0, linktype, body[4:-4]


def _ip_offset(linktype: int, frame: bytes) -> int | None:
    if linktype == LINKTYPE_ETHERNET:
        if len(frame) < 14:
            return None
        etype = (frame[12] << 8) | frame[13]
        off = 14
        while etype in (0x8100, 0x88A8) and len(frame) >= off + 4:  # VLAN tags
            etype = (frame[off + 2] << 8) | frame[off + 3]
            off += 4
        return off if etype == 0x0800 else None
    if linktype in (LINKTYPE_RAW, LINKTYPE_RAW_ALT):
        return 0 if frame and (frame[0] >> 4) == 4 else None
    if linktype == LINKTYPE_LINUX_SLL:
        return 16 if len(frame) >= 16 and ((frame[14] << 8) | frame[15]) == 0x0800 else None
    return None


def _ip(b: bytes) -> str:
    return f"{b[0]}.{b[1]}.{b[2]}.{b[3]}"


class PacketDecoder:
    """Stateful decoder: retransmission and port-scan heuristics need a
    little per-flow / per-source memory, bounded here so a 13 GB capture
    doesn't grow without limit."""

    def __init__(self, max_seqs_per_flow: int = 64, flow_idle_seconds: float = 120.0):
        self.flow_seqs: dict = {}
        self.scan: dict = {}
        self.max_seqs = max_seqs_per_flow
        self.flow_idle = flow_idle_seconds
        self._last_gc = 0.0
        self.skipped = 0

    def _gc(self, now: float) -> None:
        if now - self._last_gc < 60:
            return
        self._last_gc = now
        dead = [k for k, v in self.flow_seqs.items() if now - v[2] > self.flow_idle]
        for k in dead:
            del self.flow_seqs[k]
        dead = [k for k, v in self.scan.items() if not v[0] or now - v[0][-1][0] > PORT_SCAN_WINDOW_SECONDS]
        for k in dead:
            del self.scan[k]

    def decode(self, ts: float, linktype: int, frame: bytes) -> dict | None:
        off = _ip_offset(linktype, frame)
        if off is None or len(frame) < off + 20:
            self.skipped += 1
            return None
        vihl = frame[off]
        ihl = (vihl & 0x0F) * 4
        total_len = (frame[off + 2] << 8) | frame[off + 3]
        flags_frag = (frame[off + 6] << 8) | frame[off + 7]
        ttl = frame[off + 8]
        proto = frame[off + 9]
        src, dst = _ip(frame[off + 12 : off + 16]), _ip(frame[off + 16 : off + 20])
        l4 = off + ihl
        fragment = bool(flags_frag & 0x2000) or (flags_frag & 0x1FFF) > 0
        sport = dport = tcp_flags = tcp_window = None
        proto_name = "OTHER"
        payload = max(0, total_len - ihl)
        retrans = False
        if proto == 6 and len(frame) >= l4 + 20:
            proto_name = "TCP"
            sport = (frame[l4] << 8) | frame[l4 + 1]
            dport = (frame[l4 + 2] << 8) | frame[l4 + 3]
            seq = struct.unpack(">I", frame[l4 + 4 : l4 + 8])[0]
            doff = (frame[l4 + 12] >> 4) * 4
            tcp_flags = frame[l4 + 13] | ((frame[l4 + 12] & 0x01) << 8)
            tcp_window = (frame[l4 + 14] << 8) | frame[l4 + 15]
            payload = max(0, total_len - ihl - doff)
            key = (src, dst, sport, dport)
            st = self.flow_seqs.get(key)
            if st is None:
                st = self.flow_seqs[key] = [set(), deque(), ts]
            st[2] = ts
            if payload > 0:
                if seq in st[0]:
                    retrans = True
                else:
                    st[0].add(seq)
                    st[1].append(seq)
                    if len(st[1]) > self.max_seqs:
                        st[0].discard(st[1].popleft())
        elif proto == 17 and len(frame) >= l4 + 8:
            proto_name = "UDP"
            sport = (frame[l4] << 8) | frame[l4 + 1]
            dport = (frame[l4 + 2] << 8) | frame[l4 + 3]
            payload = max(0, total_len - ihl - 8)
        elif proto == 1:
            proto_name = "ICMP"

        scan_flag = False
        if dport is not None:
            s = self.scan.get(src)
            if s is None:
                s = self.scan[src] = (deque(), {})
            recent, counts = s
            tgt = (dst, dport)
            recent.append((ts, tgt))
            counts[tgt] = counts.get(tgt, 0) + 1
            while recent and ts - recent[0][0] > PORT_SCAN_WINDOW_SECONDS:
                _, old = recent.popleft()
                c = counts[old] - 1
                if c:
                    counts[old] = c
                else:
                    del counts[old]
            scan_flag = len(counts) >= PORT_SCAN_DISTINCT_TARGETS
        self._gc(ts)
        return {
            "timestamp": ts, "src_ip": src, "dst_ip": dst, "src_port": sport, "dst_port": dport,
            "protocol": proto_name, "ttl": ttl, "tcp_window_size": tcp_window, "tcp_flags": tcp_flags,
            "fragment_flag": fragment, "payload_size": payload, "packet_length": total_len,
            "is_retransmission": retrans, "port_scan_signature": scan_flag,
        }


def iter_packet_records(path: str, max_packets: int | None = None) -> Iterator[dict]:
    dec = PacketDecoder()
    n = 0
    for ts, lt, frame in _iter_frames(path):
        n += 1
        rec = dec.decode(ts, lt, frame)
        if rec is not None:
            yield rec
        if max_packets is not None and n >= max_packets:
            return


def first_timestamp(path: str) -> float:
    for ts, _lt, _f in _iter_frames(path):
        return ts
    raise ValueError(f"No packets in {path!r}")
