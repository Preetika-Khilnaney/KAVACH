# Kavach Project Summary

## Overview

Kavach is a predictive network-intrusion analysis platform built for SIH 2026 problem statement 26153. It uses a NetJEPA-inspired world model to analyze network telemetry, represent traffic as a graph, forecast future intrusion behavior, map predictions to MITRE ATT&CK stages, and explain the features behind each prediction.

## Main Workflow

1. A user uploads a PCAP, PCAPNG, or flow CSV file through the web interface.
2. The FastAPI backend ingests the file in batches.
3. Flow and packet records are converted into fixed-size feature windows.
4. Each window is transformed into a graph containing aggregate network and, where available, host nodes.
5. NetJEPA encodes the graph spatially with HGT or an MLP fallback and temporally with a GRU.
6. The predictor rolls the latent representation forward for multiple future steps.
7. Scoring heads estimate infiltration probability and classify the predicted MITRE ATT&CK stage.
8. Captum Integrated Gradients or gradient-times-input generates feature attributions.
9. Results are saved in SQLite and exposed to the dashboard through REST and WebSocket APIs.

## Backend

- FastAPI REST API for uploads, jobs, predictions, alerts, risk score, kill-chain state, and benchmarks.
- WebSocket pipeline stream for ingestion and model-stage progress.
- Scapy packet parser for streamed PCAP processing.
- Pandas-based CSV and packet feature extraction.
- Graph state construction for network and host relationships.
- NetJEPA HGT plus GRU architecture with a K-step latent predictor and EMA target encoder.
- VICReg loss for self-supervised representation learning.
- Infiltration and six-class attack-stage scoring heads.
- SQLite results store for prediction history and explainability data.
- Training, window-cache, labeling, and benchmark modules.

## Frontend

The React and TypeScript dashboard includes:

- Landing page with product overview and live activity presentation.
- Authentication-style demo flow and protected navigation behavior.
- Upload and ingestion portal with progress tracking.
- Persistent ingestion session while navigating between pages; a full browser reload resets the session.
- Live graph visualization for the uploaded capture.
- Operations dashboard with risk score, kill-chain forecast, timeline, and security event feed.
- Flow Explorer for stored prediction windows.
- Model Internals views for pipeline, network, and forecast rollout visualization.
- Benchmark page comparing NetJEPA and logistic regression metrics.
- Explainability, provenance, accessibility, analyst feedback, and audit UI components.

## Datasets

The repository contains the eight CIC-IDS-2017 MachineLearningCVE flow CSVs under `data/raw/CSV/MachineLearningCVE/`, tracked with Git LFS.

Two large raw PCAP captures are hosted externally on Google Drive because they exceed GitHub file-size limits:

- Monday-WorkingHours.pcap: benign baseline capture.
- Wednesday-workingHours.pcap: attack-heavy capture.

Their Google Drive links and MD5 verification instructions are documented in `data/raw/PCAP/README.md`.

## Current Status

- Python test suite: 74 tests passing.
- Frontend TypeScript production build: passing.
- A trained checkpoint exists at `data/processed/netjepa_weights.pt` in the local development environment.
- The frontend can display processed backend predictions and live ingestion events.
- Some presentation values, including benchmark and Operations dashboard summaries, are intentionally hardcoded for a stable demonstration.

## Important Limitations

- The CIC-IDS-2017 flow CSVs do not contain complete real per-flow IP, port, protocol, or timestamp information. Missing values are represented honestly or synthesized where required.
- Flow Explorer rows represent aggregated feature windows, not literal individual network flows.
- The two PCAP files are external and are not stored in GitHub.
- Some Operations timeline, network, provenance, audit, and feedback behavior remains demo-oriented or mocked.
- Authentication is simulated and accepts demo credentials; it is not production authentication.
- Model output from a missing or untrained checkpoint must not be interpreted as meaningful detection results.
- The current training configuration is a first working pass rather than a fully tuned production model.

## How to Run

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the frontend URL shown by Vite, usually `http://localhost:5173`.

## Training

Build the cached training windows and train the model with:

```bash
python -m src.training.build_windows_cache
python -m src.training.train
```

The resulting checkpoint is consumed automatically by the backend model registry.
