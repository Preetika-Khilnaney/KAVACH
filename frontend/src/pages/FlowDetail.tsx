import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { X, ExternalLink, Activity, ShieldCheck, ShieldX, Link2 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import type { Flow, FeatureAttribution } from '../api/types';
import { RiskBadge } from '../components/RiskBadge';
import { FeatureHeatmapCell } from '../components/FeatureHeatmapCell';
import { ExplainabilityPanel } from '../components/ExplainabilityPanel';
import { DecisionAuditTrail } from '../components/DecisionAuditTrail';
import { FeedbackControls } from '../components/FeedbackControls';
import { AttackChainTimeline } from '../components/AttackChainTimeline';
import { DeepExplainPanel } from '../components/DeepExplainPanel';
import { useAuditTrail, useEvidence, useFlowNeighbors } from '../api/hooks';

interface FlowDetailProps {
  flow: Flow | null;
  onClose: () => void;
  onVerdictChange?: (verdict: 'confirmed' | 'false-positive', note?: string) => void;
}

type Tab = 'overview' | 'chain' | 'explain' | 'features' | 'audit';

function summarize(flow: Flow): string {
  const top = [...flow.features].sort((a, b) => b.contribution - a.contribution).slice(0, 3).map(f => f.name);
  const method = flow.scoringMethod === 'netjepa' ? 'The calibrated NetJEPA heads' : flow.scoringMethod === 'packet_heuristic' ? 'The packet heuristic' : 'The pipeline';
  const stage = flow.attackStage ?? 'an attack stage';
  const attrib = flow.features[0]?.method === 'integrated_gradients' ? 'Integrated Gradients' : 'deviation-from-baseline attribution';
  return `${method} scored this window ${flow.riskScore}% and mapped it to ${stage}${flow.confidence ? ` (${Math.round(flow.confidence * 100)}% confidence)` : ''}. ` +
    (top.length ? `Per ${attrib}, the strongest drivers were ${top.join(', ')}.` : 'No attribution was stored for this window.');
}

export function FlowDetail({ flow, onClose, onVerdictChange }: FlowDetailProps) {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const { data: auditTrail, loading: auditLoading } = useAuditTrail(flow && activeTab === 'audit' ? flow.id : '');
  const { data: neighbors } = useFlowNeighbors(flow?.id ?? null);
  const { data: evidence } = useEvidence(flow?.id ?? null);

  // Real history: neighbouring windows of the same capture, then the
  // NetJEPA rollout forecast (dashed) for the next K windows.
  const riskHistory = useMemo(() => {
    if (!flow) return [];
    const past = (neighbors ?? []).filter(n => !n.isCurrent && new Date(n.timestamp) <= new Date(flow.timestamp)).slice(-5);
    type Pt = { time: string; risk: number | null; forecast: number | null; p10?: number | null; p90?: number | null };
    const pts: Pt[] = past.map((n, i) => ({ time: `t-${past.length - i}`, risk: n.riskScore, forecast: null }));
    const band = flow.forecast?.infiltration_band;
    pts.push({ time: 't0', risk: flow.riskScore, forecast: flow.infiltrationCurve?.length ? flow.riskScore : null, p10: band ? flow.riskScore : null, p90: band ? flow.riskScore : null });
    (flow.infiltrationCurve ?? []).forEach((p, i) =>
      pts.push({ time: `t+${i + 1}`, risk: null, forecast: Math.round(p * 100), p10: band ? Math.round(band.p10[i] * 100) : null, p90: band ? Math.round(band.p90[i] * 100) : null }),
    );
    return pts;
  }, [flow, neighbors]);

  if (!flow) return null;

  // Prepare top features attribution
  const attributions: FeatureAttribution[] = flow.features.map(f => ({
    featureName: f.name,
    value: f.value,
    contribution: f.contribution,
    direction: f.contribution > 0.4 ? 'positive' : 'negative',
  }));

  const topAttributions = [...attributions]
    .sort((a, b) => b.contribution - a.contribution)
    .slice(0, 3);

  return (
    <div className="fixed inset-y-0 right-0 max-w-2xl w-full bg-surface border-l border-border-default shadow-panel z-50 flex flex-col">
      {/* Header with Title, Risk Badge, Analyst Feedback, and Close Button */}
      <div className="px-6 py-4 border-b border-border-default flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <RiskBadge level={flow.riskLevel} score={flow.riskScore} />
          <div>
            <h2 className="font-heading font-semibold text-text-primary text-base">
              Prediction Window Detail
            </h2>
            <p className="text-xs font-mono text-text-secondary">
              ID: {flow.id.slice(0, 16)}...
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Analyst human-in-the-loop feedback controls */}
          <FeedbackControls
            targetType="flow"
            targetId={flow.id}
            currentVerdict={flow.analystVerdict}
            onVerdictChange={onVerdictChange}
          />

          <button
            type="button"
            onClick={onClose}
            data-interactive
            className="p-1.5 rounded-lg text-text-tertiary hover:text-text-primary hover:bg-canvas transition-colors"
          >
            <X size={18} />
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center px-6 border-b border-border-default bg-canvas/60 overflow-x-auto">
        {([
          ['overview', 'Explainability & Metadata'],
          ['chain', 'Attack Chain & Remediation'],
          ['explain', 'Deep Explain'],
          ['features', `Feature Attributions (${flow.features.length})`],
          ['audit', 'Decision Audit Trail'],
        ] as [Tab, string][]).map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => setActiveTab(id)}
            data-interactive
            className={`px-4 py-2.5 text-xs font-medium border-b-2 whitespace-nowrap transition-colors ${
              activeTab === id ? 'border-accent-indigo text-accent-indigo font-semibold' : 'border-transparent text-text-secondary hover:text-text-primary'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Body Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {activeTab === 'overview' && (
          <>
            {/* Quick Metadata Grid (real fields only) */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="bg-canvas p-3 rounded-lg border border-border-default">
                <span className="text-[10px] text-text-tertiary font-mono uppercase block">Hosts in window</span>
                <span className="font-mono text-xs font-semibold text-text-primary break-all">{flow.hostIPs?.length ? flow.hostIPs.slice(0, 3).join(', ') : 'n/a (flow CSV)'}</span>
              </div>
              <div className="bg-canvas p-3 rounded-lg border border-border-default">
                <span className="text-[10px] text-text-tertiary font-mono uppercase block">Stage / confidence</span>
                <span className="font-mono text-xs font-semibold text-text-primary">{flow.attackStage ?? '—'} · {flow.confidence ? `${Math.round(flow.confidence * 100)}%` : '—'}</span>
              </div>
              <div className="bg-canvas p-3 rounded-lg border border-border-default">
                <span className="text-[10px] text-text-tertiary font-mono uppercase block">Model vs calibrated</span>
                <span className="font-mono text-xs font-semibold text-text-primary">
                  {flow.scoringMethod === 'netjepa' && flow.rawProbability != null
                    ? `${Math.round(flow.rawProbability * 100)}% → ${flow.riskScore}%`
                    : `${flow.riskScore}% (${flow.scoringMethod ?? 'n/a'})`}
                </span>
              </div>
              <div className="bg-canvas p-3 rounded-lg border border-border-default">
                <span className="text-[10px] text-text-tertiary font-mono uppercase block">Records</span>
                <span className="font-mono text-xs font-semibold text-text-primary">{flow.flowCount ?? 0} flows · {flow.packetCount ?? 0} pkts</span>
              </div>
            </div>

            {/* Ledger evidence */}
            {evidence && (
              <div className={`flex items-start gap-2 rounded-lg border p-3 text-xs ${evidence.status === 'verified' ? 'border-risk-green/40 bg-risk-green-subtle' : evidence.status === 'mismatch' || evidence.status === 'chain_invalid' ? 'border-risk-red/40 bg-risk-red-subtle' : 'border-border-default bg-canvas'}`}>
                {evidence.status === 'verified' ? <ShieldCheck size={15} className="text-risk-green shrink-0" /> : evidence.status === 'mismatch' ? <ShieldX size={15} className="text-risk-red shrink-0" /> : <Link2 size={15} className="text-text-tertiary shrink-0" />}
                <div>
                  <div className="font-semibold text-text-primary">Blockchain evidence: {evidence.status.replace('_', ' ')}</div>
                  <div className="text-text-secondary">{evidence.message} {evidence.checks.map(c => `block #${c.block}`).join(', ')}</div>
                  {flow.modelSha256 && <div className="font-mono text-[10px] text-text-tertiary mt-0.5">model sha256:{flow.modelSha256.slice(0, 24)}…</div>}
                </div>
              </div>
            )}

            {/* Explainability Panel */}
            <ExplainabilityPanel
              topFeatures={topAttributions}
              allFeatures={attributions}
              summary={summarize(flow)}
            />

            {/* Risk History Mini Sparkline */}
            <div className="bg-surface border border-border-default rounded-lg p-4 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-xs font-heading font-semibold text-text-primary">
                  <Activity size={14} className="text-accent-indigo" />
                  <span>Risk history (same capture) + NetJEPA {flow.infiltrationCurve?.length ?? 0}-step forecast</span>
                </div>
                <span className="font-mono text-[11px] text-text-tertiary">{flow.scoringMethod === 'packet_heuristic' ? 'heuristic: flat forecast' : 'dashed = forecast'}</span>
              </div>

              {flow.forecast?.progression_probability != null && (
                <div className="flex flex-wrap gap-3 text-[11px] font-mono text-text-secondary">
                  <span>P(progression to a later kill-chain stage): <b className="text-accent-violet">{Math.round(flow.forecast.progression_probability * 100)}%</b></span>
                  {flow.forecast.infiltration_within_k && (
                    <span>P(attack within {flow.forecast.infiltration_within_k.length} windows): <b className="text-accent-violet">{Math.round(flow.forecast.infiltration_within_k[flow.forecast.infiltration_within_k.length - 1] * 100)}%</b></span>
                  )}
                  <span className="text-text-tertiary">{flow.forecast.forecast_samples} sampled futures · dotted = 10–90% band</span>
                </div>
              )}
              <div className="h-28 w-full pt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={riskHistory}>
                    <XAxis dataKey="time" tick={{ fontSize: 10, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 10, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} width={25} />
                    <Tooltip
                      contentStyle={{
                        fontSize: 11,
                        fontFamily: 'JetBrains Mono',
                        borderRadius: 6,
                        border: '1px solid #E4E9EF',
                      }}
                    />
                    <ReferenceLine x="t0" stroke="#8B95A5" strokeDasharray="2 2" />
                    <Line type="monotone" dataKey="risk" stroke="#4F46E5" strokeWidth={2} dot={{ r: 3, fill: '#4F46E5' }} connectNulls={false} />
                    <Line type="monotone" dataKey="forecast" stroke="#7C3AED" strokeWidth={2} strokeDasharray="4 3" dot={{ r: 2, fill: '#7C3AED' }} connectNulls />
                    <Line type="monotone" dataKey="p90" stroke="#A78BFA" strokeWidth={1} strokeDasharray="1 3" dot={false} connectNulls />
                    <Line type="monotone" dataKey="p10" stroke="#A78BFA" strokeWidth={1} strokeDasharray="1 3" dot={false} connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Deep-link Action Button to Model Internals */}
            <div className="pt-2">
              <button
                type="button"
                onClick={() => {
                  onClose();
                  navigate(`/internals/pipeline?flowId=${flow.id}`);
                }}
                data-interactive
                className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-accent-indigo text-white font-heading font-semibold text-xs hover:bg-accent-indigo-light shadow-glow-indigo transition-all"
              >
                <span>Inspect in Model Internals (3D Pipeline & Graph)</span>
                <ExternalLink size={14} />
              </button>
            </div>
          </>
        )}

        {activeTab === 'features' && (
          <div className="space-y-3">
            <p className="text-xs text-text-secondary">
              Top attributed features for this window ({flow.attributionMethod ?? flow.features[0]?.method ?? 'n/a'}). Colour = normalised contribution; value = signed attribution.
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {flow.features.map(f => (
                <FeatureHeatmapCell
                  key={f.name}
                  name={f.name}
                  value={f.value}
                  contribution={f.contribution}
                />
              ))}
            </div>
          </div>
        )}

        {activeTab === 'chain' && <AttackChainTimeline windowId={flow.id} />}

        {activeTab === 'explain' && <DeepExplainPanel windowId={flow.id} />}

        {activeTab === 'audit' && (
          <div>
            {auditTrail ? (
              <DecisionAuditTrail trail={auditTrail} />
            ) : (
              <div className="p-8 text-center text-xs text-text-tertiary">
                {auditLoading ? 'Loading the recorded decision chain…' : 'No audit trail available.'}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
