"""Load flow-level records from public flow datasets and normalize them into
the schema declared in configs/config.yaml under features.flow_level.

Supported formats (auto-detected from the header row, see detect_format):
  * "cicids2017_ml"   CIC-IDS-2017 MachineLearningCVE CSVs (no IPs/timestamps -- caveat below)
  * "cicids2017_full" CIC-IDS-2017 GeneratedLabelledFlows / CICFlowMeter CSVs (Flow ID,
                      Source IP, Timestamp, ...): real IPs, ports, protocol and timestamps
  * "cicids2018"      CSE-CIC-IDS-2018 CSVs (Dst Port, Protocol, Timestamp, Tot Fwd Pkts,
                      TotLen Fwd Pkts, FIN Flag Cnt, ...; some days also Src IP/Dst IP)
  * "ctu13"           CTU-13 bidirectional NetFlow (.binetflow: StartTime, Dur, Proto,
                      SrcAddr, Sport, Dir, DstAddr, Dport, State, TotPkts, TotBytes,
                      SrcBytes, Label) -- botnet flows are labelled "Botnet"

Whenever a file carries real timestamps / IPs / protocol they are used as-is;
the synthetic-timestamp fallback below only applies to MachineLearningCVE.

Dataset caveat (read before touching timestamps/IPs downstream):
The MachineLearningCVE release of CIC-IDS-2017 is the ML-ready export of
CICFlowMeter's output — it drops Flow ID, Source IP, Source Port,
Destination IP, Protocol and Timestamp, keeping only aggregated per-flow
statistics and a Label. That means:
  - `src_ip`, `dst_ip`, `src_port` are NOT recoverable from this file and are
    always NaN in the returned DataFrame. Only `dst_port` survives.
  - There is no `protocol` column either; it's simply absent from the
    returned DataFrame (don't index it).
  - There is no per-flow timestamp. Rows are written in capture order per
    file, so we synthesize a monotonically increasing `timestamp` by
    anchoring each file to a start time and walking forward by each row's
    own Flow Duration.

    That anchor is the file's real weakness: by default it's a *guess* (per
    the CIC-IDS-2017 paper, the five days run Monday-Friday, 2017-07-03
    through 2017-07-07, "around 09:00 local"). Pass `pcap_path` (or
    `pcap_dir` to `load_flow_dir`) when you have the matching raw PCAP —
    then the anchor becomes that capture's *real* first-packet timestamp
    instead of a guess, which is strictly better. It does NOT make every
    row's timestamp exactly right, though: we still don't know each flow's
    true individual start time, only the file's real start, so interior
    rows are still a walked approximation (and flows that genuinely
    overlapped in real time get serialized here, since the source data
    gives no way to tell). Good enough for windowed aggregation; still not
    a substitute for real per-flow timestamps.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

# Capture week per the CIC-IDS-2017 dataset documentation.
_DAY_START = {
    "monday": datetime(2017, 7, 3, 9, 0, 0),
    "tuesday": datetime(2017, 7, 4, 9, 0, 0),
    "wednesday": datetime(2017, 7, 5, 9, 0, 0),
    "thursday": datetime(2017, 7, 6, 9, 0, 0),
    "friday": datetime(2017, 7, 7, 9, 0, 0),
}

# raw CSV column -> normalized flow_level field (see configs/config.yaml).
# Columns not listed here (e.g. the many CICFlowMeter stats not in the
# config's feature list) are kept, just snake_cased, so nothing is dropped.
_COLUMN_ALIASES = {
    # CIC-IDS-2017 (both releases)
    "destination_port": "dst_port",
    "source_port": "src_port",
    "source_ip": "src_ip",
    "src_ip": "src_ip",
    "destination_ip": "dst_ip",
    "dst_ip": "dst_ip",
    "protocol": "protocol",
    "flow_duration": "duration_us",
    "flow_iat_mean": "iat_mean",
    "flow_iat_std": "iat_var",
    "flow_iat_max": "iat_max",
    "total_fwd_packets": "fwd_packets",
    "total_backward_packets": "bwd_packets",
    "total_length_of_fwd_packets": "fwd_bytes",
    "total_length_of_bwd_packets": "bwd_bytes",
    "label": "label",
    # CSE-CIC-IDS-2018 spellings
    "dst_port": "dst_port",
    "src_port": "src_port",
    "tot_fwd_pkts": "fwd_packets",
    "tot_bwd_pkts": "bwd_packets",
    "totlen_fwd_pkts": "fwd_bytes",
    "totlen_bwd_pkts": "bwd_bytes",
}

# TCP flag *count* columns in this dataset (how many packets in the flow set
# that flag), not a single bitmask. We fold "flag appeared at least once"
# into a synthetic per-flow bitmask so it's comparable to packet_parser's
# single-packet tcp_flags field. Bit positions match a raw TCP flags byte.
_FLAG_COLUMN_BITS = {
    "fin_flag_count": 0x01,
    "syn_flag_count": 0x02,
    "rst_flag_count": 0x04,
    "psh_flag_count": 0x08,
    "ack_flag_count": 0x10,
    "urg_flag_count": 0x20,
    "cwe_flag_count": 0x80,
    "ece_flag_count": 0x40,
    # CSE-CIC-IDS-2018 spellings
    "fin_flag_cnt": 0x01,
    "syn_flag_cnt": 0x02,
    "rst_flag_cnt": 0x04,
    "psh_flag_cnt": 0x08,
    "ack_flag_cnt": 0x10,
    "urg_flag_cnt": 0x20,
    "cwr_flag_count": 0x80,
    "ece_flag_cnt": 0x40,
}

_IANA_PROTO = {6: "TCP", 17: "UDP", 1: "ICMP", 0: "OTHER"}
_CTU_STATE_FLAGS = {"F": 0x01, "S": 0x02, "R": 0x04, "P": 0x08, "A": 0x10, "U": 0x20, "E": 0x40, "C": 0x80}


def detect_format(path: str) -> str:
    """Identify the flow dataset from its header row."""
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        header = f.readline()
    cols = {_snake_case(c) for c in header.split(",")}
    if {"starttime", "srcaddr", "dstaddr", "totbytes"} <= cols:
        return "ctu13"
    if {"tot_fwd_pkts", "totlen_fwd_pkts"} & cols:
        return "cicids2018"
    if {"flow_id", "source_ip", "timestamp"} <= cols:
        return "cicids2017_full"
    return "cicids2017_ml"


def _parse_timestamps(series: pd.Series) -> pd.Series | None:
    """Real per-flow timestamps when the file has them. Tries the known
    dataset formats explicitly (fast, unambiguous), then pandas inference."""
    s = series.astype(str).str.strip()
    for fmt in ("%d/%m/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S.%f", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.95:
            return parsed
    parsed = pd.to_datetime(s, errors="coerce")
    return parsed if parsed.notna().mean() > 0.95 else None


def _normalize_ctu13(chunk: pd.DataFrame, source_file: str) -> pd.DataFrame:
    df = pd.DataFrame()
    df["timestamp"] = pd.to_datetime(chunk["StartTime"].astype(str).str.strip(), format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    df["duration"] = pd.to_numeric(chunk["Dur"], errors="coerce").fillna(0.0)
    df["protocol"] = chunk["Proto"].astype(str).str.upper().str.strip()
    df["src_ip"], df["dst_ip"] = chunk["SrcAddr"].astype(str), chunk["DstAddr"].astype(str)

    def _port(v):
        try:
            return int(str(v), 0)
        except (TypeError, ValueError):
            return pd.NA

    df["src_port"] = chunk["Sport"].map(_port)
    df["dst_port"] = chunk["Dport"].map(_port)
    df["packets"] = pd.to_numeric(chunk["TotPkts"], errors="coerce").fillna(0)
    df["bytes"] = pd.to_numeric(chunk["TotBytes"], errors="coerce").fillna(0)
    df["fwd_bytes"] = pd.to_numeric(chunk["SrcBytes"], errors="coerce").fillna(0)
    df["bwd_bytes"] = (df["bytes"] - df["fwd_bytes"]).clip(lower=0)

    def _flags(state):
        bits = 0
        for ch in str(state):
            bits |= _CTU_STATE_FLAGS.get(ch, 0)
        return bits

    df["tcp_flags"] = [(_flags(st) if p == "TCP" else 0) for st, p in zip(chunk["State"], df["protocol"])]
    raw = chunk["Label"].astype(str)
    # CTU-13 labels look like "flow=From-Botnet-V42-TCP-...", "flow=From-Normal-...",
    # "flow=Background-...". Botnet traffic is command-and-control / bot activity.
    df["label"] = raw.map(lambda l: "Botnet" if "Botnet" in l else "BENIGN")
    df["ctu_label"] = raw
    df["source_day"] = "ctu13"
    df["source_file"] = source_file
    return df



def _snake_case(col: str) -> str:
    col = col.strip()
    col = re.sub(r"[^\w]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col.lower()


def _day_start_for(csv_path: str, pcap_path: str | None = None) -> tuple[str, datetime]:
    name = Path(csv_path).stem.lower()
    day = next((d for d in _DAY_START if name.startswith(d)), "unknown")
    guessed_start = _DAY_START.get(day, datetime(2017, 7, 3, 9, 0, 0))

    if pcap_path is None:
        return day, guessed_start

    # Lazy import: keeps flow_parser usable (CSV-only) without scapy
    # installed when no real PCAP anchor is requested.
    from src.ingestion.fastpcap import first_timestamp

    try:
        epoch = first_timestamp(pcap_path)
    except (OSError, ValueError):
        return day, guessed_start  # fall back to the guess rather than fail the whole load
    real_start = datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None)
    return day, real_start


def _guess_pcap_path(csv_path: str, pcap_dir: str) -> Path | None:
    """CIC-IDS-2017 naming: '<Day>-WorkingHours.pcap_ISCX.csv' pairs with
    '<Day>-WorkingHours.pcap' (note the differing case on some files,
    e.g. Wednesday-workingHours). Returns None if no match exists on disk."""
    stem = Path(csv_path).stem  # "Monday-WorkingHours.pcap_ISCX"
    base = stem.split(".pcap")[0]  # "Monday-WorkingHours"
    for candidate in Path(pcap_dir).glob("*.pcap*"):
        if candidate.stem.lower() == base.lower() or candidate.name.lower().startswith(base.lower() + "."):
            return candidate
    return None


def _normalize_chunk(chunk: pd.DataFrame, source_day: str, day_start: datetime, source_file: str, start_offset: float = 0.0) -> pd.DataFrame:
    chunk = chunk.rename(columns={c: _snake_case(c) for c in chunk.columns})
    chunk = chunk.rename(columns=_COLUMN_ALIASES)
    chunk = chunk.loc[:, ~chunk.columns.duplicated()]

    flag_cols = [c for c in _FLAG_COLUMN_BITS if c in chunk.columns]
    if flag_cols:
        bitmask = pd.Series(0, index=chunk.index, dtype="int64")
        for col in flag_cols:
            bitmask |= (chunk[col].fillna(0).astype("int64") > 0) * _FLAG_COLUMN_BITS[col]
        chunk["tcp_flags"] = bitmask

    real_ts = _parse_timestamps(chunk["timestamp"]) if "timestamp" in chunk.columns else None
    if "duration_us" in chunk.columns:
        duration_s = pd.to_numeric(chunk["duration_us"], errors="coerce").fillna(0).clip(lower=0) / 1_000_000.0
        chunk["duration"] = duration_s
        if real_ts is not None:
            chunk["timestamp"] = real_ts
        else:
            # Synthetic timestamp: walk forward through the file in row order,
            # each flow starting where the previous one's duration left off.
            # See the module docstring for why this can't be a real timestamp.
            elapsed = duration_s.cumsum().shift(fill_value=0.0) + start_offset
            chunk["timestamp"] = [day_start + timedelta(seconds=float(s)) for s in elapsed]
    elif real_ts is not None:
        chunk["timestamp"] = real_ts
    if "protocol" in chunk.columns:
        proto = pd.to_numeric(chunk["protocol"], errors="coerce")
        chunk["protocol"] = [(_IANA_PROTO.get(int(p), "OTHER") if pd.notna(p) else str(raw).upper())
                             for p, raw in zip(proto, chunk["protocol"])]

    if {"fwd_bytes", "bwd_bytes"} <= set(chunk.columns):
        chunk["bytes"] = chunk["fwd_bytes"].fillna(0) + chunk["bwd_bytes"].fillna(0)
    if {"fwd_packets", "bwd_packets"} <= set(chunk.columns):
        chunk["packets"] = chunk["fwd_packets"].fillna(0) + chunk["bwd_packets"].fillna(0)

    if "label" in chunk.columns:
        chunk["label"] = chunk["label"].astype(str).str.strip()

    for missing in ("src_ip", "dst_ip", "src_port"):
        if missing not in chunk.columns:
            chunk[missing] = pd.NA
    for numeric in ("fwd_bytes", "bwd_bytes", "fwd_packets", "bwd_packets", "iat_mean", "iat_var", "iat_max", "dst_port"):
        if numeric in chunk.columns:
            chunk[numeric] = pd.to_numeric(chunk[numeric], errors="coerce")

    chunk["source_day"] = source_day
    chunk["source_file"] = source_file
    return chunk


def iter_flow_csv(path: str, chunksize: int = 50_000, pcap_path: str | None = None) -> Iterator[pd.DataFrame]:
    """Stream-normalize a MachineLearningCVE CSV in chunks.

    Yields one normalized DataFrame per `chunksize` rows so a caller (the
    orchestrator) can emit a stage:ingestion progress event per batch
    instead of blocking until the whole file is loaded.

    Pass `pcap_path` (the matching raw capture, e.g.
    data/raw/PCAP/Monday-WorkingHours.pcap) to anchor the synthesized
    `timestamp` column to that capture's real start time instead of the
    generic per-weekday guess — see the module docstring for what this
    does and doesn't fix.
    """
    source_file = Path(path).name
    fmt = detect_format(path)
    if fmt == "ctu13":
        for raw_chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, skipinitialspace=True):
            yield _normalize_ctu13(raw_chunk, source_file)
        return
    source_day, day_start = _day_start_for(path, pcap_path=pcap_path)
    offset = 0.0
    for raw_chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, encoding="latin-1"):
        out = _normalize_chunk(raw_chunk, source_day, day_start, source_file, start_offset=offset)
        if "duration" in out.columns:
            offset += float(out["duration"].sum())
        yield out


def load_flow_csv(path: str, pcap_path: str | None = None) -> pd.DataFrame:
    """Load and normalize a full MachineLearningCVE CSV into one DataFrame.
    See iter_flow_csv for what `pcap_path` does."""
    chunks = list(iter_flow_csv(path, chunksize=200_000, pcap_path=pcap_path))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def load_flow_dir(dir_path: str, pcap_dir: str | None = None) -> pd.DataFrame:
    """Load every *.csv in a MachineLearningCVE-style directory and concat
    them. If `pcap_dir` is given, each CSV is anchored to its matching raw
    PCAP's real start time when one exists there (e.g. Monday/Wednesday in
    this project); CSVs without a matching PCAP fall back to the guessed
    anchor, same as always."""
    frames = []
    for csv_path in sorted(Path(dir_path).glob("*.csv")):
        pcap_path = _guess_pcap_path(str(csv_path), pcap_dir) if pcap_dir else None
        frames.append(load_flow_csv(str(csv_path), pcap_path=str(pcap_path) if pcap_path else None))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
