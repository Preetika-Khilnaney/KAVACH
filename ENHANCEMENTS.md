# KAVACH — Enhancement Pass (Sept 2026)

This pass implements the four design documents (`kavach_ai_blockchain.md`, `kavach_blockchain_deep_dive.md`,
`kavach_evaluation_validation.md`, `kavach_implementation_roadmap_2.md`) against the code that is actually in the repo.
Where the roadmap's description did not match the code, the code won. For example, the checkpoint was already named
`netjepa_weights.pt`, and several pages were already partly wired.

## Pass 2: world-model criteria (NetJEPA v2)

This pass closes the gaps found when the project was checked against the challenge brief: joint flow- and
packet-level training, a real probabilistic transition model P(S_t+1 | S_t), temporal-context and unseen-attack
tests, more datasets, and offline operation. The deployed checkpoint is `netjepa_weights.pt` (sha `baa45b20…`).
The previous one is kept as `netjepa_weights.before_v2.5fcb818a.pt`.

### Label bug found and fixed. The pass-1 numbers below are superseded

`flow_attack_ratio` was being z-scored along with the real features. The "is this window an attack" test
(`ratio > 0`) was therefore applied to a normalised value, and about half of the attack windows on Tuesday and
Wednesday were labelled benign. Pass 1's F1 0.72 / AUC 0.80 was measured on those partly wrong labels, and so were
the LR baseline's 0.55 / 0.80. The label field is now excluded from normalisation (`src/features/extract.py`), labels
use `flow_labels_present` (`src/training/labels.py`), and the cache was rebuilt as `data/processed/windows_v2`. The
corrected task is harder for simple models: the same logistic regression now scores F1 0.49 / AUC 0.73.

### Results: held-out validation, 8,043 windows (5,665 flow-CSV + 2,378 joint PCAP), temporal split

| Metric | LR, current window | LR, same 3-window context | **NetJEPA v2** |
|---|---|---|---|
| F1 | 0.49 | 0.46 | **0.86** |
| Precision / recall | 0.47 / 0.51 | 0.50 / 0.42 | **0.87 / 0.86** |
| False-positive rate | 29% | 21% | **6.3%** |
| ROC AUC | 0.73 | 0.76 | **0.96** |
| Forecast AUC t+1 / t+2 / t+3 (label of the future window) | 0.74 / 0.73 / 0.73 (current score) | — | **0.97 / 0.97 / 0.96** |
| Onset AUC t+1 / t+3 (benign now, attack later) | — | — | **0.93 / 0.90** |
| Stage macro-F1 now / t+3 | — | — | 0.75 / 0.70 |
| Calibration ECE raw → calibrated | — | — | 0.049 → 0.033 |
| 90% predictive-interval coverage of z_t+1 | — | — | 0.95 |

**Both network levels:** these results are on the joint PCAP validation windows, with one level hidden at inference:

| Input | F1 | AUC |
|---|---|---|
| flow + packet | **0.87** | **1.00** |
| flow level only | 0.62 | 0.98 |
| packet level only | 0.61 | 0.95 |

**Temporal context:** both runs were retrained with the same recipe for 4 epochs:

| Context | F1 | AUC | t+3 forecast AUC |
|---|---|---|---|
| 1 window | 0.735 | 0.917 | 0.893 |
| 3 windows | **0.772** | **0.924** | **0.917** |

The logistic regression gains nothing from the same three windows.

**Unseen attack families:** each family was retrained out of the data completely (4 epochs each). Recall and FPR are
measured at the deployed threshold, against benign validation windows.

| Held-out family (windows) | NetJEPA AUC / recall / FPR | LR AUC / recall / FPR |
|---|---|---|
| DDoS (1,303) | **0.98 / 95% / 7%** | 0.94 / 91% / 34% |
| Botnet C2 (463) | 0.73 / 13% / 6% | 0.74 / 61% / 30% |
| Brute force FTP/SSH (2,345) | 0.71 / 1% / 1% | 0.74 / 41% / 19% |
| Web attacks (461) | 0.65 / 2% / 10% | 0.78 / 67% / 29% |
| PortScan (1,721) | 0.58 / 5% / 8% | **0.90** / 94% / 27% |

**This is the model's main weakness.** NetJEPA transfers to an unseen family only when training contains a close
relative, as DoS is for DDoS. Behaviourally new families mostly get through at the deployed threshold. I also tested
a label-free zero-day score: kNN distance from benign in latent space, plus raw-feature kNN and world-model surprise.
It was at or below chance in every fold. The reason is that the unseen attacks are regular, machine-like traffic that
is often *more* predictable than benign. `src/scoring/novelty.py` documents this, and the LOAO JSONs keep the numbers.
The detector is not deployed.

**Adversarial:** this is still not robust. PGD flips a detection with about 0.06σ on one feature, with temporal
context held fixed. The fix is noise- or adversarial-augmented training.

### What was built

| Criterion gap | Change |
|---|---|
| Train on flow **and** packet levels | `src/ingestion/fastpcap.py` is a struct-based pcap/pcapng reader, about 10× faster than scapy, which remains the fallback. `src/features/joint_windows.py` builds 5 s windows carrying IPFIX-style flow features and packet features. `src/training/attack_timeline.py` labels PCAP windows from the published CIC-IDS-2017 schedule. `python -m src.training.build_pcap_windows`. The feature schema v2 has 29 features: bidirectional ratios, protocol mix and modality flags. Modality dropout during training lets one checkpoint score CSV, PCAP, or either level alone. Uploaded PCAPs now go through the model, not the packet heuristic. |
| Model P(S_t+1 \| S_t), not just a point prediction | The predictor outputs a diagonal Gaussian (mean + log-variance) trained with Gaussian NLL. `Predictor.sample_rollout` gives Monte-Carlo trajectories. The pipeline reports a p10–p90 infiltration band per future step, P(attack within K), P(stage progression) and a stage-frequency forecast. FlowDetail shows the band, and the CLI shows it too. |
| Learn transitions supervised, not only self-supervised | The heads are applied to the k-step rollout and trained against the labels of window t+k, which is why forecast AUC is 0.96–0.97. |
| Temporal context | context_len = 3. Inference, explanations (IG, causal tracing, tuned lens) and the benchmark all use the previous two windows from results.db. Batched PyG encoding makes this affordable. |
| Generalisation evidence | `src/benchmark/generalization.py` runs LOAO retraining and a context ablation. The Benchmark page shows the **Dynamics** and **Levels** panels. |
| More datasets | `flow_parser.detect_format` recognises CSE-CIC-IDS-2018 and CTU-13 `.binetflow` files alongside both CIC-IDS-2017 layouts. Labels map onto the MITRE stages, with encoding-proof matching. Neither dataset is in `data/raw` yet, so there is no cross-dataset score. |
| Fully offline | Fonts are bundled through `@fontsource/*`, so the build has no external requests; run `npm install` once. `python -m src.cli` runs predict, benchmark, train, build-pcap-windows, verify-chain and info with no UI or network. |

### Run order (v2)

```bash
python -m src.training.build_windows_cache                  # flow CSVs -> data/processed/windows_v2
python -m src.training.build_pcap_windows data/raw/PCAP/Monday-WorkingHours.pcap
python -m src.training.build_pcap_windows data/raw/PCAP/Wednesday-workingHours.pcap
python -m src.training.train_v2 --epochs 12                 # ~21 min on 2 CPUs; chains manifest + model
python -m src.training.train_interpretability               # ~3 min
python -m src.benchmark.generalization --epochs 4           # optional, ~55 min: LOAO + context ablation
python -m src.benchmark.evaluate --refresh                  # ~3 min
```

`pytest` passes all 104 tests.

---

## Pass 1 results (superseded: measured with the label bug described above)


| Metric | Logistic-regression baseline | NetJEPA (before) | **NetJEPA (now)** |
|---|---|---|---|
| F1 | 0.55 | ~0.28 (head saturated: recall 1.0, FPR 1.0) | **0.72** |
| Precision | 0.49 | — | **0.86** |
| False-positive rate | 6.1% | 100% | **0.9%** |
| ROC AUC | 0.80 | — | 0.80 |
| Calibration ECE | — | — | 0.128 → **0.065** (temperature scaling) |
| Forecast AUC t+1 / t+3 | — | — | 0.76 / 0.73 |

**What changed:** the jointly trained infiltration head had collapsed. It output about 1.0 for every window, which is
why the live pipeline fell back to a rule-based score and Integrated Gradients returned all zeros. The encoder itself
was healthy: its embeddings separate attack from benign at about 0.80 AUC. `src/training/train_heads.py` retrains
only the two scoring heads on the frozen encoder. It uses BCE-with-logits and class balancing, with early stopping,
the decision threshold and temperature scaling all chosen on a separate calibration split. The original weights are
kept at `data/processed/netjepa_weights.pre_heads_v2.pt`.

**Weak spots, reported as measured:**
- **Stage head:** macro-F1 is 0.43–0.53. Lateral Movement has only 35 training windows.
- **Adversarial robustness:** the model is fragile. PGD flips detections with perturbations of about 0.07σ. The
  Benchmark page flags this and recommends noise-augmented encoder training as the next step.
- **Packet-only captures:** PCAP files carry none of the flow features the model was trained on. They are scored by a
  transparent packet heuristic, and the API tags each score with `scoring_method`.

## Run order (pass 1, v1 model)

```bash
pip install -r requirements.txt
python -m src.training.build_windows_cache          # if data/processed/windows is empty
python -m src.training.train                        # only if you want to retrain the encoder (~50 min)
python -m src.training.train_heads                  # ~2 min: retrain + calibrate heads, chain manifest + model update
python -m src.training.train_interpretability       # ~2.5 min: SAE, tuned lens, probes, concept bottleneck
python -m src.benchmark.evaluate --refresh          # ~2 min: precompute the full validation pyramid (cached per model hash)
uvicorn src.api.main:app --reload
cd frontend && npm install && npm run dev
```

Demo login is `admin`, `analyst` or `analyst2`, all with password `kavach2026`. The login identifies the analyst, so
every chained verdict and remediation is attributable. It is not production auth.

The demo capture is `data/raw/uploads/demo_portscan_escalation.csv`, made from 2,000 benign rows followed by 1,500
PortScan rows of the Friday capture. It shows benign windows turning into Reconnaissance windows. Every PortScan
window is mapped to Reconnaissance, and the rollout forecasts Initial Access.

## Roadmap coverage

| Task | Where |
|---|---|
| 1.1 checkpoint lookup (candidates, key aliases, rollback override) | `src/orchestrator/model_registry.py` |
| 1.2 benchmark guard on a clean env | `src/benchmark/evaluate.py` returns `status: windows_cache_missing` |
| 1.3 SQLite WAL | `src/storage/results_store.py` and the ledger DB |
| 2.1 `GET /timeline` | falls back to `hackathon_demo_timeline.json` when that file exists |
| 2.2 `GET /provenance` | SHA-256, size, params, validation metrics, ledger block, model history |
| 2.3 `POST /feedback`, `GET /feedback/{id}`, `/feedback/stats` | persisted + chained as ANALYST_ACTION |
| 2.4 `GET /flows/{id}/audit` | real 5-stage chain + ledger blocks + analyst actions |
| 2.5 `/flows` filters | `risk_level, min_risk, max_risk, search, attack_stage, source, sort_by, sort_dir, offset` |
| 2.6 risk trend/delta | two most recent predictions |
| 3.1–3.5 frontend wiring | Operations, Benchmark, FlowDetail, Provenance drawer, new `MitreMatrixGrid` |
| 3.6 auth | `POST /auth/login` + `AuthContext` → analyst ID on every chained action |
| 4.1–4.3 ledger + API | `src/storage/blockchain_ledger.py`, `/blockchain/*` |
| 4.4 ledger dashboard | `BlockchainAuditTrail.tsx`, new **Forensics** page |
| 5.1 sparse autoencoder | `src/explainability/sparse_autoencoder.py` (TopK=32 of 4096; 1.4% variance unexplained) |
| 5.2 causal tracing | `src/explainability/causal_tracing.py` (subspace patching — whole-layer patching is trivially 1.0 in a serial net) |
| 5.3 probes | `src/training/train_interpretability.py::run_probes` → `probes.json` |
| 5.4 logit/tuned lens | `src/explainability/logit_lens.py` (ridge translators; a raw logit lens reads noise) |
| 5.5 `GET /explain/{id}` | `src/explainability/explain.py` + `DeepExplainPanel.tsx` |
| 6.1–6.4 attack chain + remediation | `src/scoring/attack_chain.py`, `src/scoring/remediation.py`, `/attack-chain/{id}`, `AttackChainTimeline.tsx` |
| 7.1 calibration | `src/scoring/calibration.py` (T=0.70; raw and calibrated both surfaced) |
| 7.2 adversarial | `src/benchmark/adversarial.py` (PGD; constant-in-training features excluded) |
| 7.3 extended benchmark | `src/benchmark/evaluate.py` + rewritten `Benchmark.tsx` (4-layer pyramid) |
| 8.1 concept bottleneck | `src/models/concept_bottleneck.py` (10 concepts, analyst overrides via `POST /explain/{id}/concepts`) |
| 8.3 replay | `WS /ws/replay/{source_file}?speed=5` |
| 8.4 heuristic attribution | `shap_explainer.heuristic_attribution` (IG only above 0.3; every window gets an explanation) |
| 8.2 streaming pipeline | **Not done on purpose.** Windows are z-scored across the whole capture, so emitting them incrementally would change the features the model sees. This needs running normalisation stats first. |

## AI ↔ blockchain synergy (design doc sections)

| Section | Implementation |
|---|---|
| 1.1 Model provenance | Every model load hashes the weights. When the hash changes, `record_model_if_changed` chains a MODEL_UPDATE block. Every PREDICTION block carries `model_sha256`. |
| 1.2 Training-data integrity | `src/training/manifest.py`: a per-file SHA-256 manifest is chained *before* training. `GET /blockchain/training-data` re-hashes the files on disk and reports any altered ones. |
| 1.3 Feedback-loop protection | Every verdict becomes an ANALYST_ACTION block. `src/blockchain/feedback_audit.py` checks volume, subnet/stage patterns, high-confidence dismissals, response time and post-update spikes. It also checks results.db rows against their blocks, which catches edits made after the fact. `verified_training_feedback()` feeds retraining. |
| 1.4 Prediction immutability | Hash chain, plus `GET /blockchain/evidence/{id}`, which re-derives a prediction from results.db and checks it against the chain or Merkle batch. |
| 1.5 Regression rollback | `model_rollback` contract: if FPR on reviewed alerts exceeds 2× the previous model's, it pins the previous weights by hash. |
| 2.1 Intelligent prioritisation | Risk-tiered chaining. Low-risk windows are batched 100 per block with a Merkle root. Medium-risk windows get a core block. High-risk windows get a full block with feature, graph and attribution hashes plus an ALERT block. Critical windows also get an ESCALATION block. |
| 2.2 Reliable smart contracts | `consensus_ip_block` fires on calibrated probability ≥ 0.85, C2 stage, confidence ≥ 0.8 and at least 3 independent nodes. `auto_escalation`: Recon → Initial Access within 10 min with no analyst acknowledgement within 5 min. Contracts are evaluated at capture end and every 60 s. |
| 2.3 Cross-node intelligence | `src/blockchain/intelligence.py`: peer blocks (hash-verified on import) are fused by noisy-OR for shared IPs, which can escalate a moderate local score. |
| 2.4 Audit-trail anomaly detection | The feedback audit above. Its result can itself be chained as a FEEDBACK_AUDIT block. |
| Air-gapped sync | `GET /blockchain/export` produces a bundle for secure media. `POST /blockchain/import` verifies it, detects forks and chains a SYNC receipt. |
| Tamper demo | `POST /blockchain/tamper-demo?mode=modify|delete` runs on a throw-away copy and shows verification catching both modification and deletion. |

This is a single-writer local SQLite ledger, which fits a prototype. A real multi-site deployment would use a
permissioned ledger with per-node signing keys, for example Hyperledger Fabric.

## Also fixed along the way

- IG and explanations previously ran through the SpatialEncoder's *fallback MLP*, which live inference never uses.
  All attribution now uses the real HGT path (`src/explainability/model_forward.py`). The IG baseline is now a benign
  reference window instead of zeros, because a zero baseline gives large attributions to features that are constant in
  training.
- The Flow Explorer used to overwrite every row with invented ports, flags and byte counts and a fixed `riskScore: 93`.
  It now shows real fields only: capture, stage, scoring method and records.
- The Benchmark page used to ignore the API and show hard-coded F1 0.93 / 0.85.
- CSV uploads now use 100-row windows, the same as the training cache (`windowing.csv_rows_per_window`).
- Window IDs are unique per capture run (`<capture>-<run>-<window>`), so `/flows/{id}` no longer collides.
- NaN attributions stored in old rows no longer 500 the API; a global safe JSON response handles them.

## Tests

`pytest` passes all 92 tests. That count includes new suites for the ledger (tamper, delete, Merkle, sync, fork),
contracts and feedback audit, calibration, remediation, concept bottleneck, and the new API endpoints.
`tests/conftest.py` isolates the ledger, results DB and benchmark cache so tests never touch real data.

## Still static / not done

- The Model Internals *Attention* page still uses its original static illustration.
- UNSW-NB15 cross-dataset evaluation can't run until that dataset is added under `data/raw`.
- Streaming windowing (Task 8.2): see the table above.
