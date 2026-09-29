"""KAVACH command-line interface -- runs fully offline.

    python -m src.cli predict <capture.pcap|.csv|.binetflow> [--max-windows N] [--json out.json] [--store]
    python -m src.cli benchmark [--refresh]
    python -m src.cli train [train_v2 options...]
    python -m src.cli build-pcap-windows <capture.pcap> [--max-packets N]
    python -m src.cli verify-chain
    python -m src.cli info

`predict` ingests a PCAP (joint flow + packet features) or a flow CSV
(CIC-IDS-2017/2018, CTU-13), runs NetJEPA inference with K-step forward
simulation, and prints the infiltration-probability timeline, the flagged
windows with their MITRE ATT&CK stage, forecast and top contributing
features. By default nothing is written to the dashboard database or the
ledger; pass --store to persist like an upload through the web UI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_BARS = " ▁▂▃▄▅▆▇█"


def _spark(values: list[float]) -> str:
    return "".join(_BARS[min(len(_BARS) - 1, int(v * (len(_BARS) - 1) + 0.5))] for v in values)


def _pct(v) -> str:
    return "  -  " if v is None else f"{v * 100:4.0f}%"


def cmd_predict(args) -> int:
    from src.orchestrator import model_registry
    from src.orchestrator.pipeline import run_pipeline

    path = Path(args.capture)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    meta = model_registry.get_model_meta()
    thr = meta.get("decision_threshold") or 0.5
    print(f"KAVACH predict · model {(meta.get('sha256') or 'untrained')[:12]} · schema v{meta.get('feature_schema_version', 1)} · "
          f"context {meta.get('context_len', 1)} windows · threshold {thr:.2f}")

    windows: dict[str, dict] = {}

    async def run():
        async for ev in run_pipeline(str(path), max_packets=args.max_packets, max_windows=args.max_windows, persist=args.store):
            if ev["stage"] == "ingestion" and ev["status"] == "complete":
                print(f"ingested {ev['payload'].get('records_total', 0):,} records ({ev['payload'].get('kind', '?')})")
            if ev["stage"] in ("attack_mapping", "explainability") and ev["payload"].get("trained") is not None:
                windows.setdefault(ev["window_id"], {}).update({ev["stage"]: ev["payload"]})

    asyncio.run(run())
    rows = [(wid, w["attack_mapping"], w.get("explainability", {})) for wid, w in windows.items() if "attack_mapping" in w]
    if not rows:
        print("no windows produced (capture too small?)")
        return 1

    probs = [am["infiltration_probability"] for _, am, _ in rows]
    print(f"\nInfiltration probability timeline ({len(rows)} windows):")
    for i in range(0, len(probs), 80):
        print("  " + _spark(probs[i : i + 80]))
    flagged = [(wid, am, ex) for wid, am, ex in rows if am["infiltration_probability"] >= thr and am["attack_stage"] != "Benign"]
    print(f"\nFlagged windows: {len(flagged)} of {len(rows)} (>= {thr:.0%} and a non-benign stage)\n")
    header = f"{'window':<34} {'risk':>5} {'stage':<20} {'conf':>5}  {'forecast t+1..t+K':<22} {'P(progress)':>11}  top features"
    print(header)
    print("-" * len(header))
    shown = flagged if args.all is False else rows
    for wid, am, ex in shown[: args.limit]:
        curve = am.get("infiltration_curve") or []
        top = ", ".join(f["feature"] for f in (ex.get("top_features") or [])[:3])
        print(f"{wid[-34:]:<34} {_pct(am['infiltration_probability'])} {am['attack_stage']:<20} {_pct(am.get('confidence'))}  "
              f"{' '.join(f'{c * 100:3.0f}' for c in curve):<22} {_pct(am.get('progression_probability')):>11}  {top}")
    if len(shown) > args.limit:
        print(f"... {len(shown) - args.limit} more (use --limit)")
    stages = {}
    for _, am, _ in rows:
        stages[am["attack_stage"]] = stages.get(am["attack_stage"], 0) + 1
    print("\nStage annotations:", ", ".join(f"{k}: {v}" for k, v in sorted(stages.items(), key=lambda kv: -kv[1])))
    methods = {am.get("scoring_method") for _, am, _ in rows}
    print("Scoring:", ", ".join(sorted(m for m in methods if m)))
    if args.json:
        out = [{"window_id": wid, **{k: am.get(k) for k in ("infiltration_probability", "raw_probability", "attack_stage", "confidence",
                                                          "infiltration_curve", "infiltration_band", "infiltration_within_k",
                                                          "progression_probability", "predicted_stages", "scoring_method")},
                "top_features": ex.get("top_features"), "remediation": ex.get("remediation")} for wid, am, ex in rows]
        Path(args.json).write_text(json.dumps(out, indent=1, default=float))
        print(f"wrote {args.json}")
    return 0


def cmd_benchmark(args) -> int:
    from src.benchmark.evaluate import run_benchmark

    r = run_benchmark(refresh=args.refresh)
    print(json.dumps({k: r.get(k) for k in ("status", "message", "model_version", "val_size", "baseline", "netjepa", "temporal_baseline",
                                            "forecast", "generalization", "modality_ablation")}, indent=1, default=float))
    return 0


def cmd_verify(_args) -> int:
    from src.storage.blockchain_ledger import get_ledger
    from src.training.manifest import verify_training_data
    from src.orchestrator import model_registry

    v = get_ledger().verify()
    print(v["message"])
    t = verify_training_data(model_registry.get_model_meta().get("training_manifest_hash"))
    print(t["message"])
    return 0 if v["valid"] and t.get("verified", True) else 1


def cmd_info(_args) -> int:
    from src.orchestrator import model_registry

    meta = model_registry.get_model_meta()
    print(json.dumps({k: meta.get(k) for k in ("path", "sha256", "model_version", "feature_schema_version", "probabilistic", "context_len",
                                               "horizon", "heads_validated", "temperature", "decision_threshold", "num_parameters")},
                     indent=1, default=str))
    return 0


def cmd_auth(args) -> int:
    from src.auth import tiered

    if args.action == "users":
        print(json.dumps(tiered.policy(), indent=1))
        print(f"\nDemo password for all users: {tiered.DEMO_PASSWORD}")
        print(f"Hardware key files: {tiered.AUTH_DIR / 'keys'}")
        tiered._ensure_store()
    elif args.action == "totp":
        if not args.username:
            print("usage: python -m src.cli auth totp <username>")
            return 2
        print(json.dumps(tiered.demo_authenticator_code(args.username)))
    elif args.action == "sign":
        key = json.loads(open(args.username).read())
        print(tiered.key_response(key["secret"], args.nonce))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "train":
        from src.training.train_v2 import main as train_main

        train_main(argv[1:])
        return 0
    if argv and argv[0] == "build-pcap-windows":
        from src.training import build_pcap_windows

        sys.argv = ["build_pcap_windows", *argv[1:]]
        build_pcap_windows.main()
        return 0
    ap = argparse.ArgumentParser(prog="python -m src.cli", description="KAVACH offline CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="analyse a capture (PCAP or flow CSV)")
    p.add_argument("capture")
    p.add_argument("--max-windows", type=int, default=500)
    p.add_argument("--max-packets", type=int, default=2_000_000)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--all", action="store_true", help="list every window, not only flagged ones")
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--store", action="store_true", help="persist results + ledger blocks like a web upload")
    p.set_defaults(fn=cmd_predict)
    b = sub.add_parser("benchmark")
    b.add_argument("--refresh", action="store_true")
    b.set_defaults(fn=cmd_benchmark)
    sub.add_parser("verify-chain").set_defaults(fn=cmd_verify)
    sub.add_parser("info").set_defaults(fn=cmd_info)
    a = sub.add_parser("auth", help="tiered-login helpers: users | totp <user> | sign <keyfile> --nonce N")
    a.add_argument("action", choices=["users", "totp", "sign"])
    a.add_argument("username", nargs="?")
    a.add_argument("--nonce")
    a.set_defaults(fn=cmd_auth)
    sub.add_parser("train", help="train NetJEPA v2 (options: see src/training/train_v2.py)")
    sub.add_parser("build-pcap-windows", help="build joint windows from a raw capture")
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
