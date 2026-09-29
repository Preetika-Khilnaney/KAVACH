/**
 * Data access layer for Kavach -- every function hits the FastAPI backend
 * (src/api/main.py). Empty/zeroed responses are honest until a capture has
 * been processed on the Ingestion page, not a bug.
 *
 * Still client-side only: getPipelineState, getNetworkGraph, getForecastTree
 * (the Model Internals pages drive those from the live WebSocket stream).
 */

export const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';

import type {
  Flow,
  FlowFilter,
  Alert,
  KillChainStage,
  TimelinePoint,
  PipelineStage,
  NetworkNode,
  NetworkEdge,
  ForecastNode,
  BenchmarkMetric,
  ModelProvenance,
  AuditTrail,
  FeedbackSubmission,
  RiskScoreResponse,
  JobProgress,
  FeedbackStats,
  ChainPage,
  ChainVerification,
  ChainStats,
  LedgerBlock,
  TamperDemoResult,
  EvidenceCheck,
  ContractsOverview,
  FeedbackAudit,
  TrainingDataVerification,
  AttackChain,
  ExplainResponse,
  BenchmarkReport,
  RemediationAction,
  ConceptPrediction,
  RemediationDecision,
  ApprovalRequest,
  LoginResponse,
  AuthUser,
  AuthPolicy,
} from './types';

import { MOCK_PIPELINE, MOCK_NETWORK, MOCK_FORECAST } from './mock-data';

// ── Simulated latency ────────────────────────────────────────────────────────

function delay(ms?: number): Promise<void> {
  const t = ms ?? 300 + Math.random() * 500;
  return new Promise(r => setTimeout(r, t));
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Session token from the tiered login (set by AuthContext). */
let authToken: string | null = null;
export function setAuthToken(token: string | null) {
  authToken = token;
}

async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (authToken) headers.set('Authorization', `Bearer ${authToken}`);
  const res = await fetch(`${BASE_URL}${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = '';
    try {
      const body = await res.json();
      detail = body?.detail ? `: ${typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)}` : '';
    } catch {
      /* not JSON */
    }
    throw new Error(`${path} failed: ${res.status} ${res.statusText}${detail}`);
  }
  return res.json();
}

function postJSON<T>(path: string, body?: unknown): Promise<T> {
  return getJSON<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

/** Analyst identity attached to feedback/remediation blocks (set by AuthContext). */
let currentAnalystId = 'analyst@kavach.demo';
export function setCurrentAnalyst(id: string | null | undefined) {
  currentAnalystId = id || 'analyst@kavach.demo';
}
export function getCurrentAnalyst() {
  return currentAnalystId;
}

// ── API Functions ────────────────────────────────────────────────────────────

export async function getRiskScore(): Promise<RiskScoreResponse> {
  // Real: derived from the most recent prediction in src/storage/results_store.py.
  // Zeroed out (score 0, activeStage "none") until at least one capture has
  // been processed via the Ingestion page.
  const res = await fetch(`${BASE_URL}/risk-score`);
  if (!res.ok) throw new Error(`Risk score fetch failed: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function getInfiltrationTimeline(limit = 200, source?: string): Promise<TimelinePoint[]> {
  // Real: GET /timeline -- stored predictions oldest-first (falls back to the
  // pre-computed demo timeline file on the backend when nothing is stored).
  const q = new URLSearchParams({ limit: String(limit) });
  if (source) q.set('source', source);
  const data = await getJSON<{ source: string; points: TimelinePoint[] }>(`/timeline?${q}`);
  return data.points;
}

const SORTABLE: Partial<Record<keyof Flow, string>> = {
  riskScore: 'infiltration_probability',
  timestamp: 'created_at',
  id: 'window_id',
};

export async function getFlows(filters?: FlowFilter): Promise<Flow[]> {
  // Real + server-side filtering (GET /flows?risk_level=&search=&sort_by=...).
  // One row per processed feature window (see src/api/adapters.py).
  const q = new URLSearchParams({ limit: String(filters?.limit ?? 500) });
  if (filters?.riskLevel?.length) q.set('risk_level', filters.riskLevel.join(','));
  if (filters?.ipSearch) q.set('search', filters.ipSearch);
  if (filters?.attackStage) q.set('attack_stage', filters.attackStage);
  if (filters?.sortBy && SORTABLE[filters.sortBy]) {
    q.set('sort_by', SORTABLE[filters.sortBy]!);
    q.set('sort_dir', filters.sortDir ?? 'desc');
  }
  let flows = await getJSON<Flow[]>(`/flows?${q}`);
  if (filters?.protocol) flows = flows.filter(f => f.protocol === filters.protocol);
  if (filters?.sortBy && !SORTABLE[filters.sortBy]) {
    const dir = filters.sortDir === 'desc' ? -1 : 1;
    const key = filters.sortBy;
    flows = [...flows].sort((a, b) => {
      const av = a[key];
      const bv = b[key];
      if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir;
      return String(av).localeCompare(String(bv)) * dir;
    });
  }
  return flows;
}

export async function getFlow(id: string): Promise<Flow> {
  return getJSON<Flow>(`/flows/${encodeURIComponent(id)}`);
}

export async function getFlowNeighbors(id: string): Promise<{ windowId: string; riskScore: number; attackStage: string; timestamp: string; isCurrent: boolean }[]> {
  return getJSON(`/flows/${encodeURIComponent(id)}/neighbors`);
}

export async function getKillChainState(): Promise<KillChainStage[]> {
  // Real: per-MITRE-stage counts across every stored prediction, active =
  // the most recent one's stage (src/api/adapters.py::kill_chain_state).
  const res = await fetch(`${BASE_URL}/kill-chain`);
  if (!res.ok) throw new Error(`Kill chain fetch failed: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function getAlerts(): Promise<Alert[]> {
  // Real: windows at/above the alert threshold, with persisted analyst verdicts.
  return getJSON<Alert[]>('/alerts');
}

export async function getPipelineState(): Promise<PipelineStage[]> {
  await delay();
  return [...MOCK_PIPELINE];
}

export async function getNetworkGraph(): Promise<{ nodes: NetworkNode[]; edges: NetworkEdge[] }> {
  await delay();
  return {
    nodes: [...MOCK_NETWORK.nodes],
    edges: [...MOCK_NETWORK.edges],
  };
}

export async function getForecastTree(): Promise<ForecastNode> {
  await delay();
  return { ...MOCK_FORECAST };
}

export async function getBenchmarks(): Promise<BenchmarkMetric[]> {
  const data = await getBenchmarkReport();
  return [data.baseline, data.netjepa].filter((r): r is BenchmarkMetric => !!r && r.f1 !== null) as BenchmarkMetric[];
}

export async function getBenchmarkReport(refresh = false): Promise<BenchmarkReport> {
  // Full validation pyramid (src/benchmark/evaluate.py). First computation
  // takes a minute or two; afterwards it's cached per model hash.
  return getJSON<BenchmarkReport>(`/benchmark${refresh ? '?refresh=true' : ''}`);
}

export async function getProvenance(): Promise<ModelProvenance> {
  return getJSON<ModelProvenance>('/provenance');
}

export async function getAuditTrail(flowId: string): Promise<AuditTrail> {
  return getJSON<AuditTrail>(`/flows/${encodeURIComponent(flowId)}/audit`);
}

export async function submitFeedback(fb: FeedbackSubmission): Promise<{ ok: boolean; block?: { index: number; hash: string } }> {
  return postJSON('/feedback', {
    targetId: fb.targetId,
    targetType: fb.targetType,
    verdict: fb.verdict,
    note: fb.note,
    analystId: fb.analystId ?? currentAnalystId,
  });
}

export async function getFeedbackStats(): Promise<FeedbackStats> {
  return getJSON<FeedbackStats>('/feedback/stats');
}

export async function getFeedbackAudit(record = false): Promise<FeedbackAudit> {
  return getJSON<FeedbackAudit>(`/feedback/audit${record ? '?record=true' : ''}`);
}

// ── Blockchain ───────────────────────────────────────────────────────────────

export async function getChain(limit = 50, blockType?: string): Promise<ChainPage> {
  const q = new URLSearchParams({ limit: String(limit) });
  if (blockType) q.set('block_type', blockType);
  return getJSON<ChainPage>(`/blockchain/chain?${q}`);
}

export async function verifyChain(): Promise<ChainVerification> {
  return getJSON<ChainVerification>('/blockchain/verify');
}

export async function getChainStats(): Promise<ChainStats> {
  return getJSON<ChainStats>('/blockchain/stats');
}

export async function getBlock(index: number): Promise<LedgerBlock> {
  return getJSON<LedgerBlock>(`/blockchain/block/${index}`);
}

export async function runTamperDemo(mode: 'modify' | 'delete' = 'modify', index?: number): Promise<TamperDemoResult> {
  const q = new URLSearchParams({ mode });
  if (index !== undefined) q.set('index', String(index));
  return postJSON<TamperDemoResult>(`/blockchain/tamper-demo?${q}`);
}

export async function getEvidence(windowId: string): Promise<EvidenceCheck> {
  return getJSON<EvidenceCheck>(`/blockchain/evidence/${encodeURIComponent(windowId)}`);
}

export async function getTrainingDataVerification(): Promise<TrainingDataVerification> {
  return getJSON<TrainingDataVerification>('/blockchain/training-data');
}

export async function exportChain(since = 0): Promise<unknown> {
  return getJSON(`/blockchain/export?since=${since}`);
}

export async function importChain(bundle: unknown): Promise<{ accepted: boolean; reason: string; blocks_imported?: number; peer_node_id?: string }> {
  return postJSON('/blockchain/import', bundle);
}

export async function getContracts(): Promise<ContractsOverview> {
  return getJSON<ContractsOverview>('/contracts');
}

export async function evaluateContracts(): Promise<{ fired: Record<string, number[]> }> {
  return postJSON('/contracts/evaluate');
}

// ── Attack chain, remediation, explainability ────────────────────────────────

export async function getAttackChain(windowId: string): Promise<AttackChain> {
  return getJSON<AttackChain>(`/attack-chain/${encodeURIComponent(windowId)}`);
}

export async function logRemediation(action: RemediationAction, target: string, windowId?: string, confirmed = false): Promise<RemediationDecision> {
  return postJSON('/remediation/log', { action: action.action, target, windowId, command: action.command, confirmed });
}

export async function getPendingApprovals(): Promise<ApprovalRequest[]> {
  const r = await getJSON<{ requests: ApprovalRequest[] }>('/approvals/pending');
  return r.requests;
}

export async function decideApproval(requestId: string, decision: 'approve' | 'reject' | 'escalate'): Promise<{ status: string; block: LedgerBlock }> {
  return postJSON(`/approvals/${encodeURIComponent(requestId)}`, { decision });
}

export async function getExplanation(windowId: string, methods = 'ig,sae,causal,logit_lens,concepts'): Promise<ExplainResponse> {
  return getJSON<ExplainResponse>(`/explain/${encodeURIComponent(windowId)}?methods=${methods}`);
}

export async function explainConcepts(windowId: string, overrides: Record<string, number>): Promise<{ concepts: ConceptPrediction }> {
  return postJSON(`/explain/${encodeURIComponent(windowId)}/concepts`, overrides);
}

// ── Classification-tiered login ───────────────────────────────────────────────
export async function login(username: string, password: string): Promise<LoginResponse> {
  return postJSON('/auth/login', { username, password });
}

export async function verifyLoginStep(challengeId: string, proof: { code?: string; signature?: string }): Promise<LoginResponse> {
  return postJSON('/auth/verify', { challengeId, ...proof });
}

export async function getSession(): Promise<AuthUser> {
  return getJSON<AuthUser>('/auth/me');
}

export async function logoutSession(): Promise<void> {
  await postJSON('/auth/logout');
}

export async function getAuthPolicy(): Promise<AuthPolicy> {
  return getJSON<AuthPolicy>('/auth/policy');
}

/** PROTOTYPE ONLY: stands in for the analyst's authenticator app. */
export async function getDemoAuthenticatorCode(username: string): Promise<{ code: string; validFor: number }> {
  return getJSON(`/auth/demo/authenticator/${encodeURIComponent(username)}`);
}

/** Hardware-key challenge/response: HMAC-SHA256(key secret, nonce), computed
 * locally with WebCrypto -- the key's secret never leaves this browser. */
export async function signWithKeyFile(keyFileText: string, nonce: string): Promise<{ signature: string; keyId: string; username: string }> {
  const key = JSON.parse(keyFileText) as { secret: string; key_id: string; username: string };
  if (!key.secret) throw new Error('Not a KAVACH security key file');
  const raw = new Uint8Array(key.secret.match(/.{2}/g)!.map(h => parseInt(h, 16)));
  const k = await crypto.subtle.importKey('raw', raw, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  const sig = new Uint8Array(await crypto.subtle.sign('HMAC', k, new TextEncoder().encode(nonce)));
  return { signature: Array.from(sig).map(b => b.toString(16).padStart(2, '0')).join(''), keyId: key.key_id, username: key.username };
}

export async function uploadFile(file: File): Promise<{ jobId: string; capturePath: string }> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE_URL}/ingest/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.status} ${res.statusText}`);
  const data = await res.json();
  // capturePath is the server-side path (e.g. data/raw/uploads/foo.csv) --
  // pass it to /ws/pipeline?capture_path=... to watch THIS file live on
  // the Model Internals pages, not the placeholder demo stream.
  return { jobId: data.job_id, capturePath: data.path };
}

export async function getJobProgress(jobId: string): Promise<JobProgress> {
  return getJSON<JobProgress>(`/jobs/${jobId}/progress`);
}
