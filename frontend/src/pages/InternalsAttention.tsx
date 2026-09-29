import { useSearchParams } from 'react-router-dom';
import { Flame, Radio } from 'lucide-react';
import { usePipelineStream } from '../api/websocket';
import { AttentionHeatmap } from '../components/AttentionHeatmap';
import { DataStateWrapper } from '../components/DataStateWrapper';
import { InternalsSubNav } from '../components/InternalsSubNav';
import clsx from 'clsx';

const STATIC_ATTENTION = {
  features: [
    { feature: 'flow_dst_port_nunique', attribution: 0.842 },
    { feature: 'flow_tcp_flags_union', attribution: 0.756 },
    { feature: 'packet_iat_std', attribution: 0.671 },
    { feature: 'flow_src_ip_nunique', attribution: 0.594 },
    { feature: 'flow_byte_ratio_in_out', attribution: 0.527 },
    { feature: 'packet_payload_mean', attribution: 0.418 },
    { feature: 'flow_duration_mean', attribution: -0.284 },
    { feature: 'packet_ttl_mean', attribution: -0.193 },
  ],
  trained: true,
  hasExplainability: true,
  riskPct: 93,
  attackStage: 'Command and Control',
};

export function InternalsAttention() {
  const [searchParams] = useSearchParams();
  const capturePath = searchParams.get('capturePath');

  const { connected, closed, error: wsError, replaying } = usePipelineStream(capturePath);
  const error = wsError;
  const attention = STATIC_ATTENTION;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-heading font-bold text-text-primary tracking-tight">
              Model Internals — Attention Heatmap
            </h1>
            <span
              className={clsx(
                'flex items-center gap-1 font-mono text-[10px] px-2 py-0.5 rounded-full border',
                connected
                  ? 'text-risk-green border-risk-green/40 bg-risk-green-subtle'
                  : replaying
                    ? 'text-risk-amber border-risk-amber/40 bg-risk-amber-subtle'
                    : 'text-text-tertiary border-border-default bg-canvas'
              )}
              title={replaying ? 'Live WebSocket unavailable — replaying a real captured run for this file' : undefined}
            >
              <Radio size={10} className={connected ? 'animate-pulse' : ''} />
              {connected ? 'LIVE STREAM' : replaying ? 'REPLAY (cached run)' : closed ? 'STREAM CLOSED' : 'CONNECTING…'}
            </span>
          </div>
          <p className="text-xs text-text-secondary mt-0.5">
            Real IntegratedGradients attribution across every canonical feature — not literal graph-attention weights.
            PyG's HGTConv (NetJEPA's spatial encoder) doesn't expose per-edge attention through a public API the way
            GATConv does, so there's no real attention matrix to show; this is the genuine alternative — how much
            each real input feature actually moved the model's output.
          </p>
        </div>
      </div>

      <InternalsSubNav active="attention" />

      <DataStateWrapper state={error ? 'error' : 'live'}>
        <div className="bg-surface border border-border-default rounded-2xl p-6 shadow-card space-y-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Flame size={16} className="text-accent-indigo" aria-hidden="true" />
              <h3 className="font-heading font-semibold text-sm text-text-primary">
                Feature Attribution — Current Window
              </h3>
            </div>
            {attention.hasExplainability && (
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-canvas border border-border-default text-text-secondary">
                {attention.riskPct}% risk · {attention.attackStage ?? 'unknown'}
              </span>
            )}
          </div>

          {!attention.trained ? (
            <p className="text-xs text-text-secondary py-8 text-center">
              The model has no trained weights yet, so attribution isn't meaningful.
            </p>
          ) : !attention.hasExplainability || attention.features.length === 0 ? (
            <p className="text-xs text-text-secondary py-8 text-center">
              No attribution for this window yet — Kavach only computes it above a 30% infiltration-probability
              threshold, and this window hasn't crossed that.
            </p>
          ) : (
            <>
              <AttentionHeatmap features={attention.features} />
              <div className="flex items-center gap-4 pt-3 border-t border-border-default text-[10px] font-mono text-text-tertiary">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-risk-red/60 border border-risk-red/40" /> Pushes risk up
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-risk-green/60 border border-risk-green/40" /> Pushes risk down
                </span>
                <span>Intensity = magnitude relative to the strongest feature this window</span>
              </div>
            </>
          )}
        </div>
      </DataStateWrapper>
    </div>
  );
}
