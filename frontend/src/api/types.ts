// ── Risk & Status ────────────────────────────────────────────────────────────

export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';
export type Severity = 'info' | 'warning' | 'critical';
export type Trend = 'up' | 'down' | 'stable';
export type AnalystVerdict = 'confirmed' | 'false-positive' | null;

// ── Network Flow ─────────────────────────────────────────────────────────────

export interface FlowFeature {
  name: string;
  value: number;
  /** normalised 0-1 contribution magnitude (for heatmap) */
  contribution: number;
  /** which attribution produced it: integrated_gradients | heuristic */
  method?: string | null;
}

export interface PredictedStage {
  step: number;
  stage: string;
  probability: number;
  infiltration_probability?: number;
}

export interface Flow {
  id: string;
  srcIP: string;
  dstIP: string;
  srcPort: number;
  dstPort: number;
  protocol: 'TCP' | 'UDP' | 'ICMP' | 'DNS' | 'HTTP' | 'TLS';
  flags: string;
  bytes: number;
  duration: number; // seconds
  iatMean: number;  // inter-arrival time mean (ms)
  iatStd: number;   // inter-arrival time std dev (ms)
  riskScore: number; // 0-100
  riskLevel: RiskLevel;
  timestamp: string; // ISO 8601
  features: FlowFeature[];
  analystVerdict?: AnalystVerdict;
  verdictNote?: string;
  verdictAt?: string;
  verdictBy?: string;
  verdictBlockIndex?: number | null;
  // extended real fields from the backend (src/api/adapters.py)
  attackStage?: string;
  confidence?: number;
  rawProbability?: number | null;
  calibratedProbability?: number | null;
  scoringMethod?: 'netjepa' | 'packet_heuristic' | string | null;
  attributionMethod?: string | null;
  sourceFile?: string | null;
  hostIPs?: string[];
  flowCount?: number | null;
  packetCount?: number | null;
  infiltrationCurve?: number[];
  predictedStages?: PredictedStage[];
  stageDistribution?: Record<string, number> | null;
  modelSha256?: string | null;
  blockIndex?: number | null;
  chainTier?: string | null;
  groundTruthLabel?: string | null;
  forecast?: FlowForecast | null;
}

export interface FlowForecast {
  infiltration_band?: { p10: number[]; p90: number[]; mean: number[] };
  infiltration_within_k?: number[];
  progression_probability?: number;
  predicted_stage_frequency?: Record<string, number>[];
  forecast_samples?: number;
}

export interface FlowFilter {
  protocol?: string;
  riskLevel?: RiskLevel[];
  ipSearch?: string;
  timeRange?: { start: string; end: string };
  sortBy?: keyof Flow;
  sortDir?: 'asc' | 'desc';
  attackStage?: string;
  limit?: number;
}

// ── Alerts ───────────────────────────────────────────────────────────────────

export interface Alert {
  id: string;
  severity: Severity;
  title: string;
  description: string;
  timestamp: string;
  flowId: string;
  killChainStage: string;
  topFeatures?: FeatureAttribution[];
  analystVerdict?: AnalystVerdict;
  verdictNote?: string;
  verdictAt?: string;
  verdictBy?: string;
  mitreTactic?: string;
  mitreTechnique?: string;
  hostIPs?: string[];
  riskScore?: number;
  scoringMethod?: string | null;
  blockIndex?: number | null;
}

// ── Kill Chain ───────────────────────────────────────────────────────────────

export interface KillChainStage {
  name: string;
  probability: number; // 0-1
  isActive: boolean;
  isPredicted: boolean;
  isComplete: boolean;
  observedCount?: number;
  tactic?: string;
  technique?: string;
  predictedProbability?: number;
  predictedStepsAhead?: number;
}

// ── Timeline ─────────────────────────────────────────────────────────────────

export interface TimelinePoint {
  timestamp: string;
  riskScore: number;
  confidence: {
    low: number;
    high: number;
  };
  windowId?: string;
  attackStage?: string;
  sourceFile?: string | null;
  rawScore?: number | null;
  scoringMethod?: string | null;
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

export type PipelineStatus = 'idle' | 'active' | 'complete' | 'error';

export interface PipelineStage {
  name: string;
  status: PipelineStatus;
  description: string;
  startedAt?: string;
  completedAt?: string;
}

// ── Network Graph ────────────────────────────────────────────────────────────

export type NodeType = 'host' | 'server' | 'gateway';

export interface NetworkNode {
  id: string;
  ip: string;
  hostname: string;
  riskContribution: number; // 0-1
  type: NodeType;
  x?: number;
  y?: number;
}

export interface NetworkEdge {
  source: string;
  target: string;
  flowCount: number;
  riskScore: number;
  protocol: string;
}

// ── Forecast ─────────────────────────────────────────────────────────────────

export interface ForecastNode {
  id: string;
  step: number;
  riskScore: number;
  probability: number; // 0-1
  label?: string;
  children: ForecastNode[];
}

// ── Benchmarks ───────────────────────────────────────────────────────────────

export interface BenchmarkMetric {
  model: string;
  f1: number;
  precision: number;
  recall: number;
  fpr: number; // false positive rate
}

// ── Explainability ───────────────────────────────────────────────────────────

export interface FeatureAttribution {
  featureName: string;
  value: number;
  contribution: number; // signed: positive = pushes risk up
  direction: 'positive' | 'negative';
}

// ── Decision Audit Trail ─────────────────────────────────────────────────────

export type AuditStageName =
  | 'Raw Flow'
  | 'Feature Extraction'
  | 'Graph State'
  | 'Forecast'
  | 'Attack Stage Mapping'
  | 'Alert'
  | 'Blockchain'
  | 'Analyst Action';

export interface AuditStep {
  stepIndex: number;
  stageName: AuditStageName;
  summary: string; // plain-language one-liner
  detail: Record<string, unknown>; // stage-specific raw data
  timestamp: string;
}

export interface AuditTrail {
  flowId: string;
  alertId?: string;
  steps: AuditStep[];
}

// ── Analyst Feedback ─────────────────────────────────────────────────────────

export interface FeedbackSubmission {
  targetType: 'alert' | 'flow';
  targetId: string;
  verdict: 'confirmed' | 'false-positive';
  note?: string;
  analystId?: string;
}

export interface FeedbackStats {
  confirmed: number;
  false_positive: number;
  needs_review: number;
  escalated: number;
  total_feedback: number;
  total_alerts: number;
  pending: number;
}

// ── Model Provenance ─────────────────────────────────────────────────────────

export interface ModelProvenance {
  datasets: { name: string; description: string; size: string; year: number }[];
  limitations: string[];
  version: string;
  lastTrained: string | null;
  trained?: boolean;
  architecture?: string;
  trainingObjective?: string;
  embeddingDim?: number;
  rolloutSteps?: number;
  checkpointPath?: string;
  sha256?: string;
  sizeBytes?: number;
  numParameters?: number;
  headsValidated?: boolean;
  temperature?: number | null;
  decisionThreshold?: number | null;
  validationMetrics?: { netjepa?: Record<string, number>; baseline?: Record<string, number>; val_size?: number };
  trainingManifestHash?: string | null;
  parentSha256?: string | null;
  ledgerBlock?: number | null;
  modelHistory?: { block: number; timestamp: string; sha256: string; previous: string | null }[];
  message?: string;
}

// ── Risk Score Response ──────────────────────────────────────────────────────

export interface RiskScoreResponse {
  score: number;
  trend: Trend;
  delta: number;
  activeStage: string;
  explanation: string;
  confidence?: number;
  rawProbability?: number | null;
  calibratedProbability?: number | null;
  scoringMethod?: string | null;
  windowId?: string;
  forecast?: number[];
}

// ── Job Progress (ingestion) ─────────────────────────────────────────────────

export interface JobProgress {
  stage: string;
  percent: number;
  complete: boolean;
  error?: string | null;
  windowsProcessed?: number | null;
  windowCount?: number | null;
  sourceFile?: string | null;
}

// ── Blockchain forensic ledger ───────────────────────────────────────────────

export interface LedgerBlock {
  index: number;
  timestamp: string;
  blockType: string;
  typeLabel: string;
  nodeId: string;
  hash: string;
  prevHash: string;
  hashShort: string;
  prevHashShort: string;
  data: Record<string, unknown>;
}

export interface ChainIssue {
  index: number;
  kind: string;
  detail: string;
}

export interface ChainVerification {
  valid: boolean;
  message: string;
  total_blocks: number;
  first_bad_index: number | null;
  issues: ChainIssue[];
  tip_hash?: string;
  tip_index?: number;
  node_id?: string;
  verified_in_ms?: number;
}

export interface ChainPage {
  total: number;
  blocks: LedgerBlock[];
  pendingLowRisk: number;
}

export interface ChainStats {
  nodeId: string;
  total: number;
  byType: Record<string, number>;
  peers: { node_id: string; n: number; tip: number; last_sync: number }[];
  pendingLowRisk: number;
}

export interface TamperDemoResult {
  simulated: boolean;
  mode?: string;
  tampered_index?: number;
  action?: string;
  verification?: ChainVerification;
  reason?: string;
}

export interface EvidenceCheck {
  windowId: string;
  status: 'verified' | 'mismatch' | 'chain_invalid' | 'pending' | 'not_chained';
  message: string;
  checks: { block: number; kind: string; match: boolean; mismatched_fields?: string[] }[];
  blocks: LedgerBlock[];
  chainValid?: boolean;
}

export interface ContractDefinition {
  id: string;
  name: string;
  rule: string;
}

export interface RollbackStatus {
  state: string;
  current_weights_sha256?: string;
  previous_weights_sha256?: string | null;
  model_update_block?: number;
  reviewed_alerts_current?: number;
  required?: number;
  fpr_current?: number | null;
  fpr_previous?: number | null;
  multiplier?: number;
}

export interface ContractsOverview {
  definitions: ContractDefinition[];
  executions: LedgerBlock[];
  consensus: { ip: string; nodes: string[]; votes: number; required: number }[];
  rollback: RollbackStatus;
}

export interface FeedbackAuditFlag {
  kind: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  analyst_id: string | null;
  detail: string;
}

export interface FeedbackAudit {
  generated_at: number;
  total_feedback: number;
  trusted_feedback: number;
  excluded_feedback: number;
  status: 'clean' | 'review' | 'critical';
  flags: FeedbackAuditFlag[];
  analysts: {
    analyst_id: string;
    actions: number;
    dismissals: number;
    dismissal_rate: number;
    dismissals_this_week: number;
    median_response_seconds: number | null;
  }[];
}

export interface TrainingDataVerification {
  verified: boolean;
  status: string;
  message: string;
  manifest_block?: number;
  manifest_hash?: string;
  files_checked?: number;
  altered_files?: string[];
  missing_files?: string[];
}

// ── Attack chain ─────────────────────────────────────────────────────────────

export interface RemediationAction {
  priority: 'immediate' | 'short_term' | 'ongoing' | 'investigate';
  priority_label: string;
  action: string;
  rationale: string;
  command: string | null;
  mitigation: string;
  mitigation_name: string;
  stage: string;
}

export interface AttackChainStep {
  kind: 'observed' | 'predicted';
  stage: string;
  tactic?: string;
  technique?: string;
  timestamp?: string;
  end_timestamp?: string;
  window_count?: number;
  peak_probability?: number;
  confidence?: number;
  probability?: number | null;
  steps_ahead?: number | null;
  top_features?: { feature: string; attribution: number }[];
  affected_hosts?: string[];
  narrative: string;
  source?: string;
  block_index?: number | null;
}

export interface AttackChain {
  window_id: string;
  source_file: string | null;
  current_stage: string;
  infiltration_probability: number;
  observed: AttackChainStep[];
  predicted: AttackChainStep[];
  remediation: { current_stage: string; current: RemediationAction[]; next_stage?: string; next?: RemediationAction[] };
  affected_hosts: string[];
  narrative: string;
}

// ── Deep explainability ──────────────────────────────────────────────────────

export interface ExplainFeature {
  feature: string;
  attribution: number;
  value?: number;
  method?: string;
}

export interface CausalLayer {
  layer: string;
  component: string;
  scored_on: string;
  causal_importance: number | null;
  node_type_effect?: Record<string, number> | null;
}

export interface LensLayer {
  layer: string;
  label: string;
  infiltration_probability: number;
  top_stage: string;
  top_stage_probability: number;
  entropy: number;
  delta_infiltration?: number;
}

export interface ConceptPrediction {
  stage: string;
  stage_probability: number;
  distribution: Record<string, number>;
  concepts: { concept: string; description: string; probability: number }[];
  contributions: { concept: string; description: string; probability: number; weight: number; contribution: number }[];
  overrides_applied: Record<string, number>;
  explanation: string;
  available?: boolean;
  reason?: string;
}

export interface ExplainResponse {
  window_id: string;
  trained: boolean;
  caveat?: string;
  infiltration_probability: number;
  attack_stage: string;
  ig?: { source: string; features: ExplainFeature[] };
  sae?: { available: boolean; reason?: string; features?: { feature_id: number; activation: number; label: string; purity?: number | null }[]; dictionary_size?: number; k?: number };
  causal?: { layers: CausalLayer[]; critical_layers: string[]; malicious_probability: number; clean_probability: number; note: string | null; method: string };
  logit_lens?: { available: boolean; reason?: string; layers?: LensLayer[]; narrative?: string; decisive_layer?: string | null };
  concepts?: ConceptPrediction;
}

// ── Benchmark report ─────────────────────────────────────────────────────────

export interface BinaryMetrics extends BenchmarkMetric {
  auc?: number;
  n?: number;
  positives?: number;
  tp?: number;
  fp?: number;
  tn?: number;
  fn?: number;
  decision_threshold?: number;
  heads_validated?: boolean;
  note?: string;
}

export interface ReliabilityBin {
  bin: string;
  count: number;
  avg_confidence: number | null;
  observed_frequency: number | null;
}

export interface BenchmarkReport {
  status: string;
  message?: string;
  val_size: number;
  train_size: number;
  baseline: BinaryMetrics | null;
  netjepa: (Omit<BinaryMetrics, 'f1' | 'precision' | 'recall' | 'fpr'> & { f1: number | null; precision: number | null; recall: number | null; fpr: number | null }) | null;
  multiclass?: { macro_f1: number; accuracy: number | null; per_class: { stage: string; f1: number; support: number }[]; confusion_matrix: number[][]; labels: string[]; note: string };
  calibration?: { temperature: number | null; ece_raw: number; ece_calibrated: number; reliability_raw: ReliabilityBin[]; reliability_calibrated: ReliabilityBin[] };
  representation?: { embed_std: number; collapse_check: string };
  predictive?: { auc: Record<string, number | null>; note: string };
  interpretability?: {
    stale?: boolean;
    note?: string;
    probes?: { layers: string[]; concepts: { concept: string; layers: Record<string, { balanced_accuracy: number }>; best_layer: string }[] };
    sae?: Record<string, number>;
    concept_bottleneck?: { stage_macro_f1: number; stage_accuracy: number; concept_auc: Record<string, number | null> };
    tuned_lens?: { layers: { layer: string; infiltration_auc: number | null; stage_macro_f1: number | null }[] };
  };
  explanation_consistency?: { available: boolean; overall_mean_jaccard?: number | null; per_attack?: { attack_label: string; windows: number; mean_jaccard: number | null; most_common_features: string[] }[]; note?: string };
  causal_by_stage?: { available: boolean; mean_causal_importance?: Record<string, Record<string, number>>; note?: string };
  robustness?: { available: boolean; windows_attacked?: number; mean_min_flip_eps?: number | null; median_min_flip_eps?: number | null; robust_accuracy?: Record<string, number>; mean_min_features_to_flip?: number | null; most_targeted_features?: string[]; summary?: string; attackable_features?: number };
  cross_dataset?: { available: boolean; note: string };
  computed_in_seconds?: number;
  // v2 additions
  model_version?: number;
  context_len?: number;
  horizon?: number;
  temporal_baseline?: BinaryMetrics | null;
  detection_by_source?: Record<string, Record<string, BinaryMetrics> | undefined>;
  forecast?: Record<string, { auc_forecast_head: number | null; auc_current_score: number | null; auc_onset_forecast_head?: number; onset_positives?: number }>;
  stage_forecast?: Record<string, { macro_f1: number; accuracy: number }>;
  transition_model?: { mean_gaussian_nll_per_dim: number; coverage_of_90pct_interval: number; note: string };
  modality_ablation?: Record<string, BinaryMetrics>;
  generalization?: {
    loao: { family: string; labels: string[]; epochs: number; held_out?: { held_out_windows: number; benign_windows: number; netjepa?: { recall_at_threshold: number; fpr_on_benign: number; auc_vs_benign: number }; logreg?: { recall_at_threshold: number; fpr_on_benign: number; auc_vs_benign: number } } }[];
    context: Record<string, { context_len: number; epochs: number; detection: BinaryMetrics; forecast: Record<string, { auc_forecast_head: number | null }> }>;
  };
}


// ── Classification-tiered access ─────────────────────────────────────────────
export type Tier = 'low' | 'moderate' | 'high';
export type LoginFactor = 'password' | 'totp' | 'hardware_key';

export interface AuthUser {
  username: string;
  name: string;
  role: string;
  analystId: string;
  tier: Tier;
  tierLabel: string;
  zone: string;
  factors: LoginFactor[];
  responseRule: 'auto' | 'confirm' | 'two_person';
  responseLabel: string;
}

export type LoginResponse =
  | { status: 'ok'; token: string; expiresIn: number; user: AuthUser }
  | { status: 'step_required'; step: LoginFactor; challengeId: string; tier: Tier; tierLabel: string; factors: LoginFactor[];
      completed: LoginFactor[]; nonce?: string; keyId?: string };

export interface AuthPolicy {
  tiers: { tier: Tier; label: string; zone: string; examples: string; factors: LoginFactor[]; response_label: string; sessionMinutes: number }[];
  indicatorFlow: string;
  demoUsers: { username: string; tier: Tier; name: string }[];
}

export interface RemediationDecision {
  ok: boolean;
  status: 'executed' | 'pending_approval';
  mode: 'auto' | 'analyst_confirmed' | 'two_person';
  requestId?: string;
  detail?: string;
  approvers?: string[];
  block: LedgerBlock;
}

export interface ApprovalRequest {
  id: string;
  requestedBy: string;
  requestedByName: string;
  tier: Tier;
  status: string;
  createdAt: number;
  action: { action: string; target: string; command?: string | null; windowId?: string | null };
}
