"""FastAPI app: REST + WebSocket API for the KAVACH dashboard.

Run with: uvicorn src.api.main:app --reload

Groups of endpoints
- ingestion:      POST /ingest/upload, GET /jobs/{id}/progress, WS /ws/pipeline
- operations:     /flows, /flows/{id}, /flows/{id}/audit, /flows/{id}/neighbors, /alerts,
                  /kill-chain, /risk-score, /timeline, /sources, WS /ws/replay/{source}
- model:          /provenance, /benchmark, /explain/{id}, /attack-chain/{id}, /remediation/{stage}
- analyst:        /feedback (POST), /feedback/stats, /feedback/audit, /feedback/{target}
- access:         /auth/policy, /auth/login, /auth/verify, /auth/me, /auth/logout, /approvals/pending, /approvals/{id}
- blockchain:     /blockchain/* (chain, verify, block, evidence, tamper-demo, export/import, peers,
                  training-data, model-history), /contracts, /contracts/evaluate, /remediation/log
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Header, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api import adapters
from src.benchmark.evaluate import run_benchmark
from src.config import load_config
from src.orchestrator import model_registry
from src.orchestrator.pipeline import run_pipeline
from src.storage import blockchain_ledger, results_store

log = logging.getLogger(__name__)
DEMO_TIMELINE_PATH = Path("data/processed/hackathon_demo_timeline.json")
CONTRACT_INTERVAL_SECONDS = 60


async def _contract_loop():
    """Time-based contracts (e.g. 'no analyst ack within 5 minutes') need
    periodic evaluation, not just evaluation at the end of a capture."""
    from src.blockchain import contracts

    while True:
        await asyncio.sleep(CONTRACT_INTERVAL_SECONDS)
        try:
            await asyncio.to_thread(contracts.evaluate_all)
        except Exception:
            log.exception("periodic contract evaluation failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(_contract_loop())
    try:
        yield
    finally:
        task.cancel()


def _json_safe(value):
    """NaN/inf aren't valid JSON (and older stored attributions contain
    some); numpy scalars aren't serialisable. Sanitise every response."""
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return _json_safe(value.item())
    return value


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(_json_safe(content), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


app = FastAPI(title="Kavach backend", lifespan=lifespan, default_response_class=SafeJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/raw/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── background ingestion jobs ────────────────────────────────────────────────
_STAGE_ORDER = ["ingestion", "feature_extraction", "state_representation", "forward_pass", "rollout", "attack_mapping", "explainability"]
_STAGE_LABELS = {
    "ingestion": "Ingestion", "feature_extraction": "Feature Extraction", "state_representation": "State Building",
    "forward_pass": "Forecasting", "rollout": "Forecasting", "attack_mapping": "Attack Mapping", "explainability": "Explainability",
}
_INGESTION_BATCH_CAP = 30

JOBS: dict[str, dict] = {}
_background_tasks: set[asyncio.Task] = set()


def _stage_percent(stage: str, status: str, batches_seen: int, window_index: int | None = None, window_count: int | None = None) -> int:
    if window_count and stage != "ingestion":
        # model stages: progress through windows is the real signal
        return min(99, 15 + round((window_index or 0) / window_count * 85))
    idx = _STAGE_ORDER.index(stage)
    if stage == "ingestion" and status == "in_progress":
        frac = min(0.95, batches_seen / _INGESTION_BATCH_CAP)
        return min(15, round(frac * 15))
    frac = 1.0 if status == "complete" else 0.5
    return min(100, round((idx + frac) / len(_STAGE_ORDER) * 100))


async def _run_ingestion_job(job_id: str, capture_path: str) -> None:
    job = JOBS[job_id]
    batches_seen, window_index, window_count = 0, None, None
    try:
        async for event in run_pipeline(capture_path=capture_path):
            if event["stage"] == "ingestion" and event["status"] == "in_progress":
                batches_seen += 1
            if event["stage"] == "feature_extraction":
                window_index = event["payload"].get("window_index")
                window_count = event["payload"].get("window_count")
            job["stage"] = _STAGE_LABELS.get(event["stage"], event["stage"])
            job["percent"] = _stage_percent(event["stage"], event["status"], batches_seen, window_index, window_count)
            job["windows_processed"] = (window_index or 0) + (1 if event["stage"] == "explainability" else 0)
            job["window_count"] = window_count
            job["last_event"] = event
        job["complete"] = True
        job["percent"] = 100
    except Exception as exc:
        log.exception("ingestion job %s failed", job_id)
        job["error"] = str(exc)
        job["complete"] = True


@app.get("/health")
async def health():
    meta = model_registry.get_model_meta()
    return {"status": "ok", "trained": meta.get("trained"), "heads_validated": meta.get("heads_validated"),
            "node_id": blockchain_ledger.get_ledger().node_id}


@app.post("/ingest/upload")
async def upload_capture(file: UploadFile = File(...)):
    safe_name = Path(file.filename or "capture").name
    dest = UPLOAD_DIR / safe_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"stage": "Ingestion", "percent": 0, "complete": False, "error": None, "last_event": None, "source_file": safe_name}
    task = asyncio.create_task(_run_ingestion_job(job_id, str(dest)))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job_id, "filename": safe_name, "path": str(dest)}


@app.get("/jobs/{job_id}/progress")
async def job_progress(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id {job_id!r}")
    return {"stage": job["stage"], "percent": job["percent"], "complete": job["complete"], "error": job.get("error"),
            "windowsProcessed": job.get("windows_processed"), "windowCount": job.get("window_count"), "sourceFile": job.get("source_file")}


# ── operations ──────────────────────────────────────────────────────────────
@app.get("/flows")
async def get_flows(
    limit: int = Query(100, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    risk_level: str | None = None,
    min_risk: float | None = None,
    max_risk: float | None = None,
    search: str | None = None,
    attack_stage: str | None = None,
    source: str | None = None,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
):
    """Stored predictions (one per window) with server-side filtering
    (Task 2.5). risk_level: comma-separated low,medium,high,critical."""
    rows = results_store.query_predictions(limit=limit, offset=offset, risk_level=risk_level, min_risk=min_risk, max_risk=max_risk,
                                           search=search, attack_stage=attack_stage, source_file=source, sort_by=sort_by, sort_dir=sort_dir)
    verdicts = results_store.latest_verdicts()
    return [adapters.row_to_flow(r, verdicts.get(r["window_id"]) or verdicts.get(f"alert-{r['window_id']}")) for r in rows]


@app.get("/flows/{window_id}")
async def get_flow(window_id: str):
    row = results_store.get_prediction(window_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown window_id {window_id!r}")
    verdicts = results_store.latest_verdicts()
    return adapters.row_to_flow(row, verdicts.get(window_id) or verdicts.get(f"alert-{window_id}"))


@app.get("/flows/{window_id}/audit")
async def get_flow_audit(window_id: str):
    """Real decision chain for one window (Task 2.4) + its ledger evidence."""
    row = results_store.get_prediction(window_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown window_id {window_id!r}")
    blocks = blockchain_ledger.get_ledger().find_blocks_for_window(window_id)
    feedback = sorted(results_store.get_feedback(window_id) + results_store.get_feedback(f"alert-{window_id}"), key=lambda f: f["created_at"])
    return adapters.audit_trail(row, blocks, model_registry.get_model_meta(), feedback)


@app.get("/flows/{window_id}/neighbors")
async def get_flow_neighbors(window_id: str, before: int = 6, after: int = 3):
    rows = results_store.neighbors(window_id, before=before, after=after)
    if not rows:
        raise HTTPException(status_code=404, detail=f"Unknown window_id {window_id!r}")
    return [{"windowId": r["window_id"], "riskScore": round(r["infiltration_probability"] * 100), "attackStage": r["attack_stage"],
             "timestamp": adapters._iso(r["created_at"]), "isCurrent": r["window_id"] == window_id} for r in rows]


@app.get("/alerts")
async def get_alerts(min_probability: float | None = None, limit: int = 100):
    """Windows at or above the alert threshold. Defaults to the served
    model's calibrated decision threshold (0.5 if none)."""
    if min_probability is None:
        min_probability = min(0.5, model_registry.get_model_meta().get("decision_threshold") or 0.5)
    rows = results_store.alerts(min_probability=min_probability, limit=limit)
    verdicts = results_store.latest_verdicts()
    return [adapters.row_to_alert(r, verdicts.get(f"alert-{r['window_id']}") or verdicts.get(r["window_id"])) for r in rows]


@app.get("/kill-chain")
async def get_kill_chain():
    return adapters.kill_chain_state(results_store.summary_stats())


@app.get("/risk-score")
async def get_risk_score():
    latest_two = results_store.latest_predictions(2)
    stats = results_store.summary_stats()
    return adapters.risk_score_response(stats, latest_two[1] if len(latest_two) > 1 else None)


@app.get("/timeline")
async def get_timeline(limit: int = Query(200, ge=1, le=5000), source: str | None = None):
    """Infiltration probability over time (Task 2.1). Falls back to the
    pre-computed demo timeline file when nothing has been processed yet."""
    rows = results_store.timeline(limit=limit, source_file=source)
    if rows:
        return {"source": "results_store", "points": adapters.timeline_points(rows)}
    if DEMO_TIMELINE_PATH.exists():
        data = json.loads(DEMO_TIMELINE_PATH.read_text())
        points = data.get("windows", data) if isinstance(data, dict) else data
        return {"source": "demo_timeline_file", "points": points[:limit]}
    return {"source": "empty", "points": []}


@app.get("/sources")
async def get_sources():
    return [{"sourceFile": r["source_file"], "windows": r["windows"], "firstAt": adapters._iso(r["first_at"]), "lastAt": adapters._iso(r["last_at"]),
             "maxRiskScore": round((r["max_probability"] or 0) * 100)} for r in results_store.source_files()]


# ── model ────────────────────────────────────────────────────────────────────
@app.get("/provenance")
async def get_provenance():
    """Which model is serving predictions, cryptographically (Task 2.2)."""
    meta = model_registry.get_model_meta()
    cfg = load_config().get("model", {})
    ledger = blockchain_ledger.get_ledger()
    updates = ledger.iter_blocks([blockchain_ledger.MODEL_UPDATE])
    current_block = next((b for b in reversed(updates) if b["data"].get("new_weights_sha256") == meta.get("sha256")), None)
    base = {
        "trained": bool(meta.get("trained")),
        "architecture": "NetJEPA: HGT spatial encoder (2 layers) -> 2-layer GRU temporal encoder -> 3-layer MLP latent predictor, EMA target encoder",
        "trainingObjective": "VICReg(z_pred, z_target) = 25*invariance + 25*variance + 1*covariance; heads: BCE-with-logits (infiltration), CE (6 MITRE stages)",
        "embeddingDim": meta.get("embedding_dim", cfg.get("embedding_dim", 128)),
        "rolloutSteps": cfg.get("rollout_steps", 5),
        "datasets": [
            {"name": "CIC-IDS-2017", "description": "MachineLearningCVE flow CSVs, Monday-Friday (benign + PortScan, Patator, Web attacks, Infiltration, Bot, DoS/DDoS)", "size": "~2.8M flows / 28k windows", "year": 2017},
        ],
        "limitations": [
            "Flow CSVs carry no real IPs/ports, so CSV-derived graphs are network-node only.",
            "Packet-only PCAP windows are outside the flow-feature training distribution and are scored by a transparent packet heuristic.",
            "Stage classifier is weakest on rare stages (Lateral Movement has 35 training windows).",
            "Cross-dataset (UNSW-NB15) generalisation has not been evaluated: dataset not present.",
        ],
    }
    if not meta.get("trained"):
        return {**base, "version": "untrained", "lastTrained": None, "message": "No checkpoint found -- run python -m src.training.train"}
    vm = meta.get("validation_metrics") or {}
    return {
        **base,
        "version": f"netjepa-{meta['sha256'][:12]}",
        "lastTrained": adapters._iso(meta["mtime"]),
        "checkpointPath": meta["path"],
        "sha256": meta["sha256"],
        "sizeBytes": meta["size_bytes"],
        "numParameters": meta["num_parameters"],
        "headsVersion": meta.get("heads_version"),
        "headsValidated": meta.get("heads_validated"),
        "temperature": meta.get("temperature"),
        "decisionThreshold": meta.get("decision_threshold"),
        "validationMetrics": vm,
        "trainingManifestHash": meta.get("training_manifest_hash"),
        "parentSha256": meta.get("parent_sha256"),
        "ledgerBlock": current_block["index"] if current_block else None,
        "modelHistory": [{"block": b["index"], "timestamp": adapters._iso(b["timestamp"]), "sha256": b["data"].get("new_weights_sha256"),
                          "previous": b["data"].get("previous_weights_sha256")} for b in updates[-10:]],
    }


@app.get("/benchmark")
async def benchmark(full: bool = True, refresh: bool = False):
    """Full four-layer validation suite (Task 7.3); structured message
    instead of a crash when the windows cache is missing (Task 1.2)."""
    return await asyncio.to_thread(run_benchmark, 0.2, full, refresh)


@app.get("/explain/{window_id}")
async def explain(window_id: str, methods: str = "ig,sae,causal,logit_lens,concepts"):
    from src.explainability.explain import explain_window

    row = results_store.get_prediction(window_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown window_id {window_id!r}")
    return await asyncio.to_thread(explain_window, row, [m.strip() for m in methods.split(",") if m.strip()])


@app.post("/explain/{window_id}/concepts")
async def explain_concepts(window_id: str, overrides: dict[str, float] = Body(default={})):
    """Concept-bottleneck with analyst overrides (human-in-the-loop)."""
    from src.explainability.explain import explain_window

    row = results_store.get_prediction(window_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown window_id {window_id!r}")
    return await asyncio.to_thread(explain_window, row, ["concepts"], overrides)


@app.get("/attack-chain/{window_id}")
async def attack_chain(window_id: str):
    from src.scoring.attack_chain import reconstruct

    chain = await asyncio.to_thread(reconstruct, window_id)
    if chain is None:
        raise HTTPException(status_code=404, detail=f"Unknown window_id {window_id!r}")
    return chain


@app.get("/remediation/{stage}")
async def remediation(stage: str, src_ip: str | None = None, dst_ip: str | None = None):
    from src.scoring.remediation import playbook_stages, remediation_for_stage

    if stage not in playbook_stages():
        raise HTTPException(status_code=404, detail=f"No playbook for stage {stage!r}; have {playbook_stages()}")
    return remediation_for_stage(stage, {"src_ip": src_ip, "dst_ip": dst_ip})


class RemediationLog(BaseModel):
    action: str
    target: str
    analystId: str | None = None
    windowId: str | None = None
    command: str | None = None
    confirmed: bool = False


def _session_or_401(authorization: str | None) -> dict:
    from src.auth import tiered

    token = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else authorization
    sess = tiered.session(token)
    if sess is None:
        raise HTTPException(status_code=401, detail="Sign in to your zone's KAVACH console to act on alerts.")
    return sess


@app.post("/remediation/log")
async def log_remediation(body: RemediationLog, authorization: str | None = Header(None)):
    """Apply the analyst's tier response rule, then chain the action:
    low -> auto-executes; moderate -> needs explicit analyst confirmation;
    high -> queued until a *second* cleared officer approves (two-person rule)."""
    from src.auth import tiered

    sess = _session_or_401(authorization)
    act = body.model_dump()
    try:
        decision = tiered.authorise_action(sess, act)
    except tiered.AuthError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
    ledger = blockchain_ledger.get_ledger()
    extra = {"tier": sess["tier"], "response_mode": decision["mode"], "status": decision["status"], "approvers": decision.get("approvers", [])}
    if decision["status"] == "pending_approval":
        extra["request_id"] = decision["requestId"]
    block = ledger.log_remediation(action=body.action, target=body.target, authorized_by=sess["analystId"],
                                   window_id=body.windowId, command=body.command, automated=decision["mode"] == "auto", extra=extra)
    return {"ok": True, **decision, "block": block}


@app.get("/approvals/pending")
async def remediation_pending(authorization: str | None = Header(None)):
    from src.auth import tiered

    sess = _session_or_401(authorization)
    if sess["tier"] != "high":
        return {"requests": []}
    return {"requests": tiered.pending_approvals()}


class ApprovalIn(BaseModel):
    decision: str = Field("approve", description="approve | reject | escalate")


@app.post("/approvals/{request_id}")
async def remediation_approve(request_id: str, body: ApprovalIn, authorization: str | None = Header(None)):
    from src.auth import tiered

    sess = _session_or_401(authorization)
    try:
        req = tiered.decide(sess, request_id, body.decision)
    except tiered.AuthError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
    a = req["action"]
    status = {"approve": "executed", "reject": "rejected", "escalate": "escalated"}[body.decision]
    block = blockchain_ledger.get_ledger().log_remediation(
        action=a["action"], target=a["target"], authorized_by=req["requestedBy"], window_id=a.get("windowId"), command=a.get("command"),
        extra={"tier": "high", "response_mode": "two_person", "status": status, "request_id": request_id,
               "approvers": [req["requestedBy"], sess["analystId"]] if status == "executed" else [req["requestedBy"]],
               "decided_by": sess["analystId"]})
    return {"ok": True, "request": req, "status": status, "block": block}


# ── analyst feedback ─────────────────────────────────────────────────────────
class FeedbackIn(BaseModel):
    targetId: str
    targetType: str = "alert"
    verdict: str = Field(..., description="confirmed | false-positive | needs_review | escalated")
    note: str | None = None
    analystId: str = "analyst@kavach.demo"


@app.post("/feedback")
async def submit_feedback(body: FeedbackIn):
    """Persist a verdict (Task 2.3) and chain it as an ANALYST_ACTION block
    -- the immutable record the feedback audit checks before retraining."""
    wid = body.targetId[len("alert-"):] if body.targetId.startswith("alert-") else body.targetId
    row = results_store.get_prediction(wid)
    context = {}
    if row is not None:
        context = {"window_id": wid, "infiltration_probability": row["infiltration_probability"], "confidence": row["confidence"],
                   "attack_stage": row["attack_stage"], "model_sha256": row.get("model_sha256"),
                   "host_ips": [n["label"] for n in row.get("nodes", []) if n.get("type") == "host"]}
    try:
        fb = results_store.save_feedback(body.targetId, body.targetType, body.verdict, body.note, body.analystId, context)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    block = blockchain_ledger.get_ledger().log_analyst_action(
        analyst_id=fb["analyst_id"], action=fb["verdict"], target_id=fb["target_id"], target_type=fb["target_type"],
        justification=fb.get("note"), feedback_id=fb["id"],
        extra={"window_id": context.get("window_id"), "prediction_probability": context.get("infiltration_probability"),
               "model_sha256": context.get("model_sha256")})
    results_store.set_feedback_block(fb["id"], block["index"])
    fb["block_index"] = block["index"]
    return {"ok": True, "feedback": fb, "block": {"index": block["index"], "hash": block["hash"]}}


@app.get("/feedback/stats")
async def feedback_stats():
    return results_store.feedback_stats(min_probability=0.5)


@app.get("/feedback/audit")
async def feedback_audit(record: bool = False):
    """AI audit of the analyst trail (insider / poisoning detection)."""
    from src.blockchain import feedback_audit as fa

    return await asyncio.to_thread(fa.analyze, None, record)


@app.get("/feedback/trusted")
async def trusted_feedback():
    from src.blockchain import feedback_audit as fa

    items = await asyncio.to_thread(fa.verified_training_feedback)
    return {"count": len(items), "feedback": items}


@app.get("/feedback/{target_id}")
async def get_feedback(target_id: str):
    return results_store.get_feedback(target_id)


# ── classification-tiered authentication ─────────────────────────────────────
class LoginIn(BaseModel):
    username: str
    password: str


class VerifyIn(BaseModel):
    challengeId: str
    code: str | None = None
    signature: str | None = None


def _auth_call(fn, *args, **kw):
    from src.auth import tiered

    try:
        return fn(*args, **kw)
    except tiered.AuthError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))


@app.get("/auth/policy")
async def auth_policy():
    from src.auth import tiered

    return tiered.policy()


@app.post("/auth/login")
async def login(body: LoginIn):
    """Step 1 (password). Low tier gets a session straight away; moderate and
    high tiers get the next required factor (TOTP, then hardware key)."""
    from src.auth import tiered

    return await asyncio.to_thread(_auth_call, tiered.start_login, body.username, body.password)


@app.post("/auth/verify")
async def login_verify(body: VerifyIn):
    """Step 2/3: authenticator code or hardware-key signature over the nonce."""
    from src.auth import tiered

    return _auth_call(tiered.continue_login, body.challengeId, code=body.code, signature=body.signature)


@app.get("/auth/me")
async def auth_me(authorization: str | None = Header(None)):
    return _session_or_401(authorization)


@app.post("/auth/logout")
async def auth_logout(authorization: str | None = Header(None)):
    from src.auth import tiered

    token = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else authorization
    tiered.logout(token)
    return {"ok": True}


@app.get("/auth/demo/authenticator/{username}")
async def demo_authenticator(username: str):
    """PROTOTYPE ONLY: stands in for the analyst's authenticator app so the
    demo can be run on one machine. Disable with KAVACH_DEMO_AUTH=0."""
    import os
    from src.auth import tiered

    if os.environ.get("KAVACH_DEMO_AUTH", "1") == "0":
        raise HTTPException(status_code=404, detail="Demo authenticator disabled")
    return _auth_call(tiered.demo_authenticator_code, username)


# ── blockchain ───────────────────────────────────────────────────────────────
def _block_out(b: dict, full: bool = False) -> dict:
    out = {"index": b["index"], "timestamp": adapters._iso(b["timestamp"]), "blockType": b["block_type"], "typeLabel": b.get("type_label", b["block_type"]),
           "nodeId": b["node_id"], "hash": b["hash"], "prevHash": b["prev_hash"], "hashShort": b["hash"][:16], "prevHashShort": b["prev_hash"][:16],
           "data": b["data"]}
    if not full and b["block_type"] == blockchain_ledger.PREDICTION_BATCH:
        d = dict(b["data"])
        d["item_hashes"] = f"{len(d.get('item_hashes', []))} hashes"
        d["window_ids"] = (d.get("window_ids") or [])[:5] + (["..."] if len(d.get("window_ids") or []) > 5 else [])
        out["data"] = d
    return out


@app.get("/blockchain/chain")
async def blockchain_chain(limit: int = Query(50, ge=1, le=1000), block_type: str | None = None, before: int | None = None):
    ledger = blockchain_ledger.get_ledger()
    blocks = ledger.get_chain(limit=limit, block_type=block_type, before=before)
    return {"total": ledger.count(), "blocks": [_block_out(b) for b in blocks], "pendingLowRisk": ledger.pending_low_risk()}


@app.get("/blockchain/verify")
async def blockchain_verify():
    return await asyncio.to_thread(blockchain_ledger.get_ledger().verify)


@app.get("/blockchain/stats")
async def blockchain_stats():
    ledger = blockchain_ledger.get_ledger()
    return {"nodeId": ledger.node_id, "total": ledger.count(), "byType": ledger.type_counts(), "peers": ledger.peer_nodes(),
            "pendingLowRisk": ledger.pending_low_risk()}


@app.get("/blockchain/block/{index}")
async def blockchain_block(index: int):
    b = blockchain_ledger.get_ledger().get_block(index)
    if b is None:
        raise HTTPException(status_code=404, detail=f"No block {index}")
    return _block_out(b, full=True)


@app.get("/blockchain/evidence/{window_id}")
async def blockchain_evidence(window_id: str):
    """Re-derive a stored prediction from results.db and check it against
    the chain -- proves the operational record wasn't edited after the fact."""
    from src.graph.state_builder import build_graph_state
    from src.explainability.explain import _window_from_row

    row = results_store.get_prediction(window_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown window_id {window_id!r}")
    ledger = blockchain_ledger.get_ledger()
    blocks = ledger.find_blocks_for_window(window_id)
    fv = build_graph_state(_window_from_row(row))["feature_vector"]
    tier = row.get("chain_tier") or blockchain_ledger.risk_tier(row["infiltration_probability"], row["attack_stage"])
    core = blockchain_ledger.prediction_core(window_id=window_id, source_file=row.get("source_file"), infiltration_probability=row["infiltration_probability"],
                                             attack_stage=row["attack_stage"], confidence=row["confidence"], model_sha256=row.get("model_sha256"),
                                             scoring_method=row.get("scoring_method"), feature_vector=fv,
                                             calibrated_probability=row.get("calibrated_probability"), tier=tier)
    checks = []
    pred = next((b for b in blocks if b["block_type"] == blockchain_ledger.PREDICTION), None)
    if pred is not None:
        mismatched = [k for k, v in core.items() if pred["data"].get(k) != v]
        checks.append({"block": pred["index"], "kind": "prediction", "match": not mismatched, "mismatched_fields": mismatched})
    batch = next((b for b in blocks if b["block_type"] == blockchain_ledger.PREDICTION_BATCH), None)
    if batch is not None:
        h = blockchain_ledger.batch_item_hash(core)
        in_batch = h in batch["data"].get("item_hashes", [])
        root_ok = blockchain_ledger.merkle_root(batch["data"].get("item_hashes", [])) == batch["data"].get("merkle_root")
        checks.append({"block": batch["index"], "kind": "merkle_batch", "match": in_batch and root_ok, "item_hash": h, "merkle_root_valid": root_ok})
    if not checks:
        status = "pending" if ledger.pending_low_risk() else "not_chained"
        return {"windowId": window_id, "status": status, "checks": [], "blocks": [_block_out(b) for b in blocks],
                "message": "Low-risk window still in the batch buffer." if status == "pending" else "No ledger record for this window (processed before the ledger existed, or untrained)."}
    ok = all(c["match"] for c in checks)
    chain = ledger.verify()
    return {"windowId": window_id, "status": "verified" if ok and chain["valid"] else "mismatch" if not ok else "chain_invalid",
            "checks": checks, "chainValid": chain["valid"], "blocks": [_block_out(b) for b in blocks],
            "message": ("results.db matches the chained evidence and the chain is intact." if ok and chain["valid"] else
                        "results.db no longer matches the chained evidence -- the operational record was modified." if not ok else chain["message"])}


@app.post("/blockchain/tamper-demo")
async def blockchain_tamper_demo(index: int | None = None, mode: str = "modify"):
    """Simulate an intruder editing or deleting a block -- on a throw-away
    copy of the chain -- and show that verification catches it."""
    if mode not in ("modify", "delete"):
        raise HTTPException(status_code=422, detail="mode must be 'modify' or 'delete'")
    return await asyncio.to_thread(blockchain_ledger.simulate_tamper, blockchain_ledger.get_ledger(), index, mode)


@app.get("/blockchain/export")
async def blockchain_export(since: int = 0):
    """Chain delta for air-gapped sync (write to secure media)."""
    return blockchain_ledger.get_ledger().export_delta(since)


@app.post("/blockchain/import")
async def blockchain_import(bundle: dict = Body(...)):
    return await asyncio.to_thread(blockchain_ledger.get_ledger().import_bundle, bundle)


@app.get("/blockchain/peers")
async def blockchain_peers():
    return blockchain_ledger.get_ledger().peer_nodes()


@app.get("/blockchain/training-data")
async def blockchain_training_data():
    from src.training.manifest import verify_training_data

    meta = model_registry.get_model_meta()
    return await asyncio.to_thread(verify_training_data, meta.get("training_manifest_hash"))


@app.get("/blockchain/model-history")
async def blockchain_model_history():
    from src.blockchain import contracts

    ledger = blockchain_ledger.get_ledger()
    updates = ledger.iter_blocks([blockchain_ledger.MODEL_UPDATE])
    return {"current": model_registry.get_model_meta().get("sha256"), "updates": [_block_out(b) for b in reversed(updates)],
            "rollback": contracts.rollback_status()}


@app.get("/contracts")
async def contracts_overview():
    from src.blockchain import contracts

    return {"definitions": contracts.definitions(), "executions": [_block_out(b) for b in contracts.recent_executions()],
            "consensus": contracts.consensus_status(), "rollback": contracts.rollback_status()}


@app.post("/contracts/evaluate")
async def contracts_evaluate():
    from src.blockchain import contracts

    result = await asyncio.to_thread(contracts.evaluate_all)
    result["fired_blocks"] = [_block_out(b) for b in result["fired_blocks"]]
    return result


# ── websockets ──────────────────────────────────────────────────────────────
@app.websocket("/ws/pipeline")
async def pipeline_stream(websocket: WebSocket, capture_path: str | None = None):
    """Streams pipeline stage events as the orchestrator runs:
        {"stage", "window_id", "timestamp", "payload", "status"}"""
    await websocket.accept()
    try:
        async for event in run_pipeline(capture_path=capture_path):
            await websocket.send_json(event)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/replay/{source_file}")
async def replay_stream(websocket: WebSocket, source_file: str, speed: float = 5.0, max_gap: float = 2.0):
    """Replay mode (Task 8.3): stream a previously processed capture's stored
    predictions as pipeline-shaped events, with the original timing scaled
    by `speed` (gaps capped at max_gap seconds)."""
    await websocket.accept()
    try:
        n = 0
        for row, gap in results_store.replay_predictions(None if source_file == "_none_" else source_file):
            await asyncio.sleep(min(max_gap, gap / max(speed, 0.01)))
            flow = adapters.row_to_flow(row)
            await websocket.send_json({"stage": "attack_mapping", "window_id": row["window_id"], "timestamp": time.time(), "status": "in_progress",
                                       "replay": True, "payload": {"infiltration_probability": row["infiltration_probability"],
                                                                   "attack_stage": row["attack_stage"], "confidence": row["confidence"],
                                                                   "infiltration_curve": row.get("infiltration_curve"), "flow": flow,
                                                                   "trained": row["trained"]}})
            n += 1
        await websocket.send_json({"stage": "replay_complete", "status": "complete", "timestamp": time.time(), "payload": {"events": n}})
    except WebSocketDisconnect:
        pass
