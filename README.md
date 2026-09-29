# Kavach backend

> **Sept 2026 enhancement pass:** retrained and calibrated scoring heads (F1 0.72 vs baseline 0.55, FPR 0.9%), a hash-chained forensic ledger with smart contracts and an AI audit of analyst feedback, deep explainability (SAE, causal tracing, tuned lens, probes, concept bottleneck), attack-chain reconstruction with remediation, and a four-layer validation benchmark. See **[ENHANCEMENTS.md](ENHANCEMENTS.md)** for what changed, run order and honest limitations.


Backend skeleton for the NetJEPA world-model pipeline (SIH 2026, problem statement 26153).
Matches the architecture from the technical approach slide: ingestion -> feature extraction ->
graph state -> NetJEPA world model -> scoring/stage mapping -> explainability -> API/streaming.

## Repository layout

The project is split into two explicit applications:

```text
KAVACH/
|-- backend/       Python API, model pipeline, training, tests, config, and data
|-- frontend/      React + TypeScript + Vite dashboard
|-- Images/        architecture diagrams and presentation assets
|-- README.md      setup and operating instructions
```

Run backend commands from `backend/` and frontend commands from `frontend/`.

## What's implemented vs. stubbed

**Implemented (runs as-is):**
- `src/ingestion/flow_parser.py` — streams and normalizes the CIC-IDS-2017 "MachineLearningCVE"
  flow CSVs (see Datasets below). Read the module docstring: this dataset release has no
  src/dst IP, no protocol column, and no real per-flow timestamp, so those are synthesized/left
  NaN. For Monday and Wednesday specifically, pass `pcap_path=` (or `pcap_dir=` to
  `load_flow_dir`) to anchor the synthesized timestamp to that day's real PCAP start time
  instead of a generic 9am guess — still not a real per-flow timestamp, just a better anchor.
  Tuesday/Thursday/Friday have no matching PCAP, so they stay on the guess.
- `src/ingestion/packet_parser.py` — streams PCAP/PCAPNG captures packet-by-packet via Scapy
  (never loads a file into memory, so it's safe against the 10-13 GB captures in this project).
  Extracts TTL, TCP window size, fragment flags, payload size, a retransmission flag (same
  5-tuple + seq number seen twice), and a sliding-window port-scan signature.
- `src/features/extract.py` — windows flow_parser/packet_parser output into one aggregate
  feature vector per `windowing.window_seconds` (tumbling windows), z-score normalized. Flow
  and packet records are windowed independently and merged on `window_start` only if you pass
  both — read the module docstring before doing that with two unrelated captures.
- `src/orchestrator/pipeline.py` — every stage is wired to real code. Ingestion streams one
  `stage:ingestion` event per batch; once it finishes, each window flows through
  feature_extraction -> state_representation -> forward_pass -> rollout -> attack_mapping ->
  explainability before the next window starts (one event per stage per window). See
  `src/orchestrator/model_registry.py` on the trained/untrained fallback below — every payload
  from state_representation onward carries a `trained: bool` so a consumer can tell whether
  a prediction is real. Explainability is skipped for low-infiltration-probability windows
  (Captum is too slow to run on every window of a large capture) — see
  `_EXPLAIN_INFILTRATION_THRESHOLD`.
- `src/orchestrator/model_registry.py` — loads `data/processed/netjepa_weights.pt` if it
  exists, hot-reloading if the file changes (training finishes, or saves a newer epoch) with
  no server restart needed. Falls back to a freshly-initialized (untrained, random) model if
  no checkpoint exists yet, so the pipeline still runs end-to-end either way.
- `src/storage/results_store.py` — the architecture doc's Results Store: SQLite, one row per
  processed window (predictions, attack-stage mapping, explainability output), written by
  `pipeline.py` as it runs. Backs `GET /flows`, `/flows/{id}`, `/alerts`, `/kill-chain`, and
  `/risk-score` (`src/api/adapters.py` maps rows into the frontend's shapes — read its
  docstring on which Flow fields are real vs. honest placeholders; this dataset has no real
  per-window port/protocol, and `bytes`/`duration`/`iat*` stay 0 rather than display a
  z-score-normalized value with a misleading "seconds"/"bytes" label).
- Live-pipeline safety caps: a CSV's time-based windowing can produce 100k+ windows on one
  file (flow_parser's synthesized-timestamp inflation — see its docstring), so
  `run_pipeline`'s `max_windows` (default 500) caps how many go through the full model
  pipeline live, the same way `max_packets` already capped PCAP parsing. Pass `None` for a
  real offline/batch pass over everything.
- Frontend pages now on real data (not mock): Ingestion, Benchmark, Operations (risk score,
  kill chain, alerts), Flow Explorer, and all three Model Internals pages (Pipeline/Network/
  Forecast — `frontend/src/api/websocket.ts` wires `/ws/pipeline` in directly; the Ingest
  page's "Inspect Pipeline 3D Internals" button carries the uploaded file's path through as a
  `capturePath` query param so the Internals pages watch that exact capture, not a placeholder).
  The Forecast page shows NetJEPA's actual rollout — one real autoregressive trajectory, not a
  branching tree with fabricated alternative-probability branches like the original mock. Still
  mocked: the infiltration timeline chart, network graph (Operations-level, different from the
  Internals live graph), provenance, audit trail, and feedback persistence.
- `src/api/main.py` — FastAPI app with a REST upload endpoint and a WebSocket that streams
  pipeline stage events, plus a `POST /ingest/upload` -> background job -> `GET /jobs/{id}/progress`
  flow backing the frontend's ingestion page.
- `src/graph/state_builder.py` — builds a graph snapshot from one feature window. Every window
  gets a "network" node (its own aggregate feature vector); "host" nodes only get added when the
  window carries real observed (src_ip, dst_ip) pairs, which only happens for PCAP-derived
  windows — MachineLearningCVE CSV windows have no real IPs (see flow_parser above), so their
  graphs are always network-node-only. Read the module docstring before expecting more.
- `src/models/netjepa.py` — HGT spatial encoder -> GRU temporal encoder -> K-step predictor,
  with an EMA target path for VICReg training. Falls back to a plain MLP when torch-geometric
  isn't installed. **BatchNorm on the encoder output is load-bearing, not decorative** — an
  early real training run collapsed to a near-constant embedding within ~20 steps regardless of
  VICReg's variance/covariance loss weight (a known BYOL/SimSiam failure mode: collapse is a
  cheaper way to satisfy the unbounded invariance term than the capped variance penalty can
  counteract). BatchNorm fixes it structurally instead of just penalizing it. See the module
  docstring and `tests/test_netjepa.py::test_batchnorm_keeps_embeddings_spread_out_not_collapsed`
  before removing it.
- `src/scoring/infiltration.py`, `src/scoring/attack_stage.py` — downstream heads mapping a
  NetJEPA embedding to an infiltration probability curve / a 6-class MITRE ATT&CK stage
  (Benign + the 5 kill-chain stages the labels in data/raw/CSV/MachineLearningCVE map to).
- `src/explainability/shap_explainer.py` — Captum IntegratedGradients (falls back to gradient
  x input) attributing a scoring head's output back to the window's own named features
  (`state_builder.CANONICAL_FEATURE_NAMES`) — real feature names, not `feature_0, feature_1, ...`.
- `src/models/losses.py` — VICReg loss (invariance + variance + covariance terms), used to
  prevent representation collapse in the encoder.
- `src/training/` — the training pipeline: `build_windows_cache.py` windows every CSV to
  `data/processed/windows/*.pkl` (row-count windows, not time windows — see its docstring on why:
  flow_parser's synthesized timestamp inflates wildly on unanchored days), `labels.py` derives
  infiltration/attack-stage labels from each window's *observed* labels (not the majority vote —
  rare attacks like Infiltration, 35/2887 windows, would never win a majority once windows span
  100+ rows), `dataset.py` builds context/target training examples with a per-file temporal
  train/val split, and `train.py` runs the actual training loop. Run:
  `python -m src.training.build_windows_cache` then `python -m src.training.train`.
- `src/benchmark/evaluate.py` — the architecture doc's Benchmark/Evaluation Module: trains a
  logistic regression baseline on the same windowed features, and reports F1/precision/recall/
  false-positive-rate for it alongside NetJEPA's infiltration head, on the *same* held-out
  validation windows `src.training.train` used (apples-to-apples). Binary infiltration task only
  (BENIGN vs any-attack) — FPR specifically is a binary-classification concept, and that's what
  the doc asks for; the 6-class MITRE stage head isn't benchmarked here. Exposed via
  `GET /benchmark`; cached on the checkpoint's mtime so repeated calls don't retrain the baseline
  and re-run inference over the whole validation set for nothing. Reports `"note"` instead of
  metrics for NetJEPA if no checkpoint exists yet.

## Datasets

`backend/data/raw/CSV/MachineLearningCVE/` (committed via Git LFS) has all 8 CSVs from the CIC-IDS-2017
"MachineLearningCVE" release — the full Monday-Friday capture week, pre-flowed with
CICFlowMeter. Between them they already cover the full attack taxonomy the architecture doc's
kill-chain mapping needs:

| Day | Labels |
|---|---|
| Monday | BENIGN only (baseline) |
| Tuesday | FTP-Patator, SSH-Patator (brute force / initial access) |
| Wednesday | DoS Hulk, GoldenEye, Slowloris, Slowhttptest, Heartbleed |
| Thursday | Web Attack (Brute Force/XSS/SQLi), Infiltration |
| Friday | Bot (C2), PortScan (recon), DDoS |

`backend/data/raw/PCAP/` holds the two raw captures (Monday = benign baseline, Wednesday = DoS/DDoS-heavy)
that back the packet-level ingestion path — too large for git, linked from here via Google Drive
once uploaded:
- Monday-WorkingHours.pcap — `<drive link>`
- Wednesday-workingHours.pcap — `<drive link>`

**Do you need more datasets?** No, not for the pipeline as designed. The 8 CSVs already give you
every attack category the MITRE stage-mapper needs, and the two PCAPs are enough to exercise the
raw-packet ingestion path (Scapy parsing, host graph, TTL/window/fragmentation/retransmission
features) on both a clean day and an attack-heavy day — you don't need PCAPs for every day, since
the flow-level CSVs already cover the rest. The one thing worth adding later, not now, is a
CTU-13 capture if you want a *real* botnet C2 pcap to demo — the "Bot" label above is flow-level
only, so there's no packet capture with genuine C2 beaconing to point the packet-level path at.
That's an enhancement, not a gap in the current build.

### Other datasets (auto-detected by `src/ingestion/flow_parser.detect_format`)

| Format | How it's recognised | Notes |
|---|---|---|
| CIC-IDS-2017 MachineLearningCVE | `Destination Port`, no IPs | synthetic 1 s timestamps, anchored to the matching PCAP if present |
| CIC-IDS-2017 full (with IPs) | `Flow ID` / `Source IP` | real timestamps |
| CSE-CIC-IDS-2018 | `Dst Port`, `Tot Fwd Pkts`, `TotLen Fwd Pkts`, `*_Flag_Cnt` | labels such as `DoS attacks-Hulk`, `Infilteration`, `Brute Force -Web` map onto the MITRE stages |
| CTU-13 (`.binetflow`) | `StartTime,Dur,Proto,SrcAddr,...,Label` | Argus `State` letters become a TCP-flag bitmask; `flow=From-Botnet*` → `Botnet` (C2), everything else BENIGN |

Drop any of these into `backend/data/raw/CSV/...` (or upload them) and run
`python -m src.training.build_windows_cache` from `backend/`.

### Joint flow + packet windows from raw PCAP

`python -m src.training.build_pcap_windows data/raw/PCAP/Wednesday-workingHours.pcap` reads the capture with a
dependency-free struct parser (`src/ingestion/fastpcap.py`, ~10x faster than scapy), builds 5-second windows that
carry *both* flow-level features (IPFIX-style flows cut at the window boundary) and packet-level features, and labels
them from the published CIC-IDS-2017 attack timeline (`src/training/attack_timeline.py`). These windows are part of
the v2 training set, so the model is trained on both levels of the network hierarchy.

## Setup (step by step)

These instructions take a fresh macOS or Linux machine from clone to a working dashboard. Windows
can use the same commands through WSL; native activation and path commands are noted where they differ.

### 1. Install system prerequisites

| Tool | Recommended version | Used for |
|---|---|---|
| Python | 3.14 | API, ingestion, model, training, tests |
| Node.js and npm | Node 22+ (the project is developed on Node 24) | React/Vite frontend |
| Git and Git LFS | current versions | source code and CIC-IDS-2017 CSVs |
| `tshark` | current version | optional PyShark packet inspection |

On macOS with Homebrew:

```bash
brew install python@3.14 node git git-lfs wireshark
git lfs install
```

On Ubuntu/Debian, install Python, Node, Git, and Git LFS with the system package manager. Install
`tshark` if you use the PyShark path; the fast PCAP training command uses the bundled parser and
does not require Wireshark.

Verify the tools before cloning:

```bash
python3 --version
node --version
npm --version
git --version
git lfs version
```

### 2. Clone the repository and fetch data

```bash
git lfs install
git clone https://github.com/Preetika-Khilnaney/KAVACH.git KAVACH
cd KAVACH
git lfs pull
```

The CIC-IDS-2017 flow CSVs are stored with Git LFS. Confirm that they are actual CSV files rather
than small LFS pointer files:

```bash
find backend/data/raw/CSV/MachineLearningCVE -maxdepth 1 -type f -exec ls -lh {} \;
```

Place any large PCAP files manually in `backend/data/raw/PCAP/`; they are intentionally not committed.
The expected captures and their purpose are documented in [Datasets](#datasets).

### 3. Create and verify the Python environment

Run these commands from `backend/`:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -c "import fastapi, numpy, pandas, torch; print('Python dependencies OK')"
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`. Keep the virtual environment
active in every terminal running backend commands. `torch` and `torch-geometric` are the largest
downloads. If `torch-geometric` cannot be installed, the model uses the plain-MLP fallback in
`src/models/netjepa.py`; a graph-encoder checkpoint may then be incompatible.

### 4. Install and verify the frontend

```bash
cd frontend
npm install
npm run build
npm run lint
cd ..
```

Run `npm install` again after pulling changes to `frontend/package.json` or
`frontend/package-lock.json`. The frontend has no required `.env` file: in development it connects
to `http://localhost:8000` and `ws://localhost:8000`.

### 5. Choose or create model weights

Weights are git-ignored. Without `backend/data/processed/netjepa_weights.pt`, the API still starts, but
responses report `trained: false` and inference uses the untrained/rule-based fallback.

**Use an existing checkpoint:** copy a compatible file to:

```text
backend/data/processed/netjepa_weights.pt
```

The model registry hot-reloads a newer checkpoint while the API is running. Verify its metadata:

```bash
python -m src.cli info
```

**Train locally:** first build row-count windows from the CSV data:

```bash
python -m src.training.build_windows_cache
```

For joint flow and packet training, add PCAPs to `backend/data/raw/PCAP/` and build packet windows:

```bash
python -m src.training.build_pcap_windows data/raw/PCAP/Monday-WorkingHours.pcap
python -m src.training.build_pcap_windows data/raw/PCAP/Wednesday-workingHours.pcap
```

Train with all available sources, or CSV data only:

```bash
python -m src.training.train_v2
python -m src.training.train_v2 --sources csv
```

The checkpoint is written to `backend/data/processed/netjepa_weights.pt`. Training metrics are written to
`backend/data/processed/training_metrics.jsonl`; inspect them before treating predictions as production-quality.

### 6. Start and verify the backend

From `backend/`, with `.venv` active:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Leave this terminal running. In another terminal, check the health response and API documentation:

```bash
curl http://localhost:8000/health
open http://localhost:8000/docs       # macOS; use a browser on Linux/Windows
```

The expected health response has `"status": "ok"`. Port 8000 is the frontend’s default backend
target; change both sides together if you must use another port.

### 7. Start the frontend

In a second terminal, from the repository root:

```bash
cd frontend
npm run dev
```

Open http://localhost:5173. For a production-like local preview, build first and then run
`npm run preview` from `frontend`.

### 8. Sign in and process a capture

1. Select **Sign in** and use one of the demo accounts listed in
  [Classification-tiered access](#classification-tiered-access). All demo accounts use the
  password `kavach2026`.
2. Open **Ingest** and upload a flow CSV or PCAP. The backend returns a job ID, processes the file
  in the background, and streams progress through the dashboard.
3. Open **Operations**, **Flows**, **Forensics**, or **Benchmark** after processing completes.
4. For a direct API check, upload a file with `POST /ingest/upload`, poll
  `GET /jobs/{job_id}/progress`, and connect to
  `ws://localhost:8000/ws/pipeline?capture_path=<path>` for stage events.

The first backend start creates local authentication state under `backend/data/auth/`. It is git-ignored;
do not commit generated passwords, TOTP secrets, or hardware-key files.

### 9. Run verification checks

Run backend tests from `backend/` and frontend checks from `frontend/`:

```bash
cd backend
source .venv/bin/activate
pytest
cd ../frontend
npm run build
npm run lint
```

For a quick offline smoke test without the web UI:

```bash
cd backend
python -m src.cli info
python -m src.cli predict data/raw/CSV/MachineLearningCVE/<file>.csv --limit 10
```

### Troubleshooting

| Symptom | Fix |
|---|---|
| `python` or `pip` uses the wrong interpreter | activate `.venv`, then use `python -m pip ...` and `python -m pytest` |
| CSV upload fails or produces no windows | run `git lfs install && git lfs pull`; verify the files are not LFS pointers |
| `trained` is `false` | add `backend/data/processed/netjepa_weights.pt` or complete the training steps above |
| `torch-geometric` installation fails | reinstall the PyTorch-compatible build, or continue with the documented plain-MLP fallback |
| PyShark cannot find a capture backend | install Wireshark/tshark; use `build_pcap_windows` for the bundled fast parser |
| Browser says `Backend unreachable` | start Uvicorn on port 8000 and check `curl http://localhost:8000/health` |
| Vite reports `Failed to resolve import` | run `cd frontend && npm install`, then restart Vite |
| Port 8000 or 5173 is busy | inspect with `lsof -nP -iTCP:8000 -sTCP:LISTEN` or `lsof -nP -iTCP:5173 -sTCP:LISTEN`; stop the old process or update the frontend API target |

## Command line (offline)

Everything runs without the web UI or any network access. Run these commands from `backend/`:

```bash
cd backend
python -m src.cli info                                    # loaded checkpoint, schema, threshold, ledger status
python -m src.cli predict capture.pcap --limit 20         # per-window risk, stage, t+1..t+K forecast band, top features
python -m src.cli predict flows.csv --json > out.json     # machine-readable
python -m src.cli predict capture.pcap --store            # also write to results.db + chain to the ledger
python -m src.cli benchmark [--refresh]                   # full evaluation report
python -m src.cli train --epochs 12                       # = python -m src.training.train_v2
python -m src.cli build-pcap-windows capture.pcap
python -m src.cli verify-chain
```

The frontend bundles its fonts (`@fontsource/*`), so the UI also works on an air-gapped network.

## Run the API

```bash
cd backend
uvicorn src.api.main:app --reload
```

POST a CSV or PCAP to `/ingest/upload`, then connect a WebSocket client to
`ws://localhost:8000/ws/pipeline?capture_path=<path from the upload response>` to stream real
events for every stage off that file, window by window. Connect to `/ws/pipeline` with no query
param to get the placeholder demo stream instead. Check each event's `payload.trained` from
state_representation onward — `false` until `python -m src.training.train` has produced
`data/processed/netjepa_weights.pt`.

## Classification-tiered access

Each KAVACH node guards one classification tier. Login strength and the authority to act on an
alert scale with it (`src/auth/tiered.py`, standard library only, fully offline):

| Tier | Zone | Sign-in factors | Response rule |
|---|---|---|---|
| Routine (low) | office · admin | password | approved playbook actions auto-execute |
| Sensitive (moderate) | finance · HR · procurement | password + authenticator code (TOTP, RFC 6238) | analyst confirms each action |
| Mission-critical (high) | classified · defence · grid control | password + TOTP + hardware security key | two-person approval, or escalate |

- Passwords are PBKDF2-SHA256 hashed; 5 failures lock the account for 5 minutes.
- The hardware key is a challenge/response: the node issues a nonce and the key answers with
  HMAC-SHA256(secret, nonce), computed in the browser, so the secret never crosses the wire. In the
  prototype the key is a `.kavachkey` file standing in for a FIDO2/PKI token.
- Every login, failure, lockout and logout is chained as an `ACCESS` block; every response action is
  chained as a `REMEDIATION` block with its tier, mode and approvers.
- Mission-critical actions wait in the two-person queue on the Operations page until a *different*
  cleared officer approves, rejects or escalates them.

Demo users (password `kavach2026`), created with fresh secrets in `data/auth/` on first start
(git-ignored):

| User | Tier |
|---|---|
| `l1.analyst` | Routine |
| `l2.responder` | Sensitive |
| `officer.a`, `officer.b` | Mission-critical (keys: `data/auth/keys/officer.*.kavachkey`) |

The login dialog has a "demo authenticator" link that shows the current TOTP code (prototype only;
disable with `KAVACH_DEMO_AUTH=0`). From the terminal: `python -m src.cli auth totp l2.responder`.
Endpoints: `/auth/policy`, `/auth/login`, `/auth/verify`, `/auth/me`, `/auth/logout`,
`/approvals/pending`, `/approvals/{id}`.

## Suggested build order

1. ~~`src/ingestion/flow_parser.py` + `src/ingestion/packet_parser.py`, wired into the
   orchestrator's ingestion stage.~~ Done — see `tests/test_ingestion.py`, `tests/test_pipeline.py`.
2. ~~`src/features/extract.py` — windowed feature matrix, wired into the orchestrator's
   `feature_extraction` stage.~~ Done — see `tests/test_extract.py`, `tests/test_pipeline.py`.
3. ~~`src/graph/state_builder.py` — turn a window into a `(V_t, E_t)` graph snapshot.~~ Done —
   see `tests/test_state_builder.py`.
4. ~~`src/models/netjepa.py` — HGT+GRU world model architecture.~~ Done — see
   `tests/test_netjepa.py` and the BatchNorm caveat above.
5. ~~`src/scoring/`, `src/explainability/` — downstream heads.~~ Done — see
   `tests/test_scoring.py`, `tests/test_shap_explainer.py`.
6. ~~Wire `state_representation` / `forward_pass` / `rollout` / `attack_mapping` /
   `explainability` in `src/orchestrator/pipeline.py`.~~ Done — see
   `src/orchestrator/model_registry.py` and `tests/test_pipeline.py`'s
   `test_run_pipeline_model_stages_*` tests.
7. ~~`src/training/` — training loop.~~ Done, code-wise; **run it** —
   `python -m src.training.build_windows_cache && python -m src.training.train` — and let it
   finish before trusting any `trained: true` payload's numbers. Next real step after that:
   look at `data/processed/training_metrics.jsonl` (val_vicreg, val_embed_std) and decide if
   more epochs / more data / actual hyperparameter tuning is worth it, since this training setup
   (small batch, no LR schedule, no real graph batching — see train.py's docstring) is a first
   working pass, not a tuned one.
8. ~~`src/benchmark/evaluate.py` — baseline vs. NetJEPA, `GET /benchmark`.~~ Done — see
   `tests/test_benchmark.py`. Still open: only the two `data/raw/PCAP` days get real IPs; a
   results store + replay mode (architecture doc's other spec'd mode) doesn't exist yet; and
   most of the frontend beyond the Ingest page is still mock data — see the loose-ends list from
   earlier in this conversation for the fuller picture.
