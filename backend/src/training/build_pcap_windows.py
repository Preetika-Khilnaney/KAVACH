"""Build joint flow+packet training windows from raw CIC-IDS-2017 captures.

Streams a capture once with the fast reader (src/ingestion/fastpcap.py),
builds 5-second joint windows (src/features/joint_windows.py), labels them
from the published attack timeline (src/training/attack_timeline.py), and
saves raw (un-normalised) windows to data/processed/pcap_windows/<stem>.pkl
(normalisation is applied per capture at load time).

Run (can take 20-40 min per 10+ GB capture):
    python -m src.training.build_pcap_windows data/raw/PCAP/Wednesday-workingHours.pcap
Optional: --max-packets N for a quick trial, --progress-every N.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

from src.features.joint_windows import iter_joint_windows
from src.ingestion.fastpcap import iter_packet_records
from src.training.attack_timeline import day_for, make_labeler

OUT_DIR = Path("data/processed/pcap_windows")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap")
    ap.add_argument("--max-packets", type=int, default=None)
    ap.add_argument("--window-seconds", type=float, default=5.0)
    ap.add_argument("--progress-every", type=int, default=1_000_000)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    day = day_for(args.pcap)
    labeler = make_labeler(day) if day else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.pcap).stem + (f".first{args.max_packets}" if args.max_packets else "")
    t0 = time.time()
    n = [0]

    def counted():
        for rec in iter_packet_records(args.pcap, max_packets=args.max_packets):
            n[0] += 1
            if n[0] % args.progress_every == 0:
                print(f"{n[0]:,} packets  {time.time() - t0:.0f}s  ({n[0] / (time.time() - t0):,.0f} pkt/s)")
            yield rec

    windows = list(iter_joint_windows(counted(), window_seconds=args.window_seconds, labeler=labeler))
    with (out_dir / f"{stem}.pkl").open("wb") as f:
        pickle.dump(windows, f)
    attacks = {}
    for w in windows:
        for lab in w.get("flow_labels_present", []):
            if lab != "BENIGN":
                attacks[lab] = attacks.get(lab, 0) + 1
    summary = {"pcap": args.pcap, "day": day, "packets": n[0], "windows": len(windows), "attack_windows_by_label": attacks,
               "seconds": round(time.time() - t0, 1), "first_window": windows[0]["window_start"] if windows else None,
               "last_window": windows[-1]["window_start"] if windows else None}
    (out_dir / f"{stem}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
