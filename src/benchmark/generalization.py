"""Generalisation to unseen attacks + temporal-context ablation.

1. Leave-one-attack-family-out (LOAO): for each family, retrain NetJEPA v2
   from scratch with every example containing that family REMOVED from
   training (context and futures), then measure how well the model flags
   the family's windows against benign validation windows -- recall at the
   deployed threshold, false-positive rate, AUC. The logistic-regression
   baseline is retrained without the family too. This is the honest test of
   "generalise to unseen attack patterns, not memorise signatures".

2. Context ablation: the same training recipe with context_len=1 (no
   temporal history) vs the deployed context_len=3, on the same validation
   windows -- does temporal dynamics learning actually help?

Results go to data/processed/generalization/*.json; GET /benchmark reads them.

Run: python -m src.benchmark.generalization [--epochs 4] [--families PortScan,Bot]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

OUT_DIR = Path("data/processed/generalization")

FAMILIES = {
    "PortScan (Reconnaissance)": ["PortScan"],
    "Botnet C2 (Command and Control)": ["Bot"],
    "Brute force (FTP/SSH-Patator)": ["FTP-Patator", "SSH-Patator"],
    "Web attacks (Initial Access)": ["Web Attack � Brute Force", "Web Attack � XSS", "Web Attack � Sql Injection",
                                     "Web Attack \x96 Brute Force", "Web Attack \x96 XSS", "Web Attack \x96 Sql Injection"],
    "DDoS": ["DDoS"],
}


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--families", type=str, default="")
    ap.add_argument("--skip-context", action="store_true")
    args = ap.parse_args()
    from src.training.train_v2 import main as train

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wanted = [f.strip() for f in args.families.split(",") if f.strip()]
    for name, labels in FAMILIES.items():
        if wanted and not any(w.lower() in name.lower() for w in wanted):
            continue
        slug = name.split(" ")[0].lower()
        t0 = time.time()
        print(f"=== LOAO: holding out {name} ===")
        report = train(["--epochs", str(args.epochs), "--exclude-labels", ",".join(labels), "--no-chain", "--out", "",
                        "--report", str(OUT_DIR / f"loao_{slug}.full.json")])
        summary = {"family": name, "labels": labels, "epochs": args.epochs, "held_out": report.get("held_out_attack"),
                   "overall_detection": {k: report["detection"].get(k) for k in ("f1", "auc", "fpr")}, "seconds": round(time.time() - t0)}
        (OUT_DIR / f"loao_{slug}.json").write_text(json.dumps(summary, indent=1, default=float))
        print(json.dumps(summary, indent=1, default=float))

    if not args.skip_context:
        for L in (1, 3):
            print(f"=== context ablation: context_len={L} ===")
            report = train(["--epochs", str(args.epochs), "--context-len", str(L), "--no-chain", "--out", "",
                            "--report", str(OUT_DIR / f"context_{L}.full.json")])
            summary = {"context_len": L, "epochs": args.epochs, "detection": report["detection"], "forecast": report["forecast"],
                       "baselines": {k: {m: v.get(m) for m in ("f1", "auc", "fpr")} for k, v in report["baselines"].items()}}
            (OUT_DIR / f"context_{L}.json").write_text(json.dumps(summary, indent=1, default=float))


def load_results() -> dict:
    out = {"loao": [], "context": {}}
    if not OUT_DIR.exists():
        return out
    for f in sorted(OUT_DIR.glob("loao_*.json")):
        if not f.name.endswith(".full.json"):
            out["loao"].append(json.loads(f.read_text()))
    for f in sorted(OUT_DIR.glob("context_*.json")):
        if not f.name.endswith(".full.json"):
            d = json.loads(f.read_text())
            out["context"][str(d["context_len"])] = d
    return out


if __name__ == "__main__":
    main()
