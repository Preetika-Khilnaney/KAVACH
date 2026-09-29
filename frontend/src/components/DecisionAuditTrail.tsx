import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, ChevronUp, Network, Cpu, Activity, ShieldAlert, FileText, CheckCircle2, Link2, UserCheck } from 'lucide-react';
import type { AuditTrail, AuditStep, AuditStageName } from '../api/types';
import { FeatureHeatmapCell } from './FeatureHeatmapCell';
import clsx from 'clsx';

interface DecisionAuditTrailProps {
  trail: AuditTrail;
  className?: string;
}

const STAGE_ICONS: Record<AuditStageName, React.ComponentType<{ className?: string }>> = {
  'Raw Flow': FileText,
  'Feature Extraction': Cpu,
  'Graph State': Network,
  'Forecast': Activity,
  'Attack Stage Mapping': ShieldAlert,
  'Alert': CheckCircle2,
  'Blockchain': Link2,
  'Analyst Action': UserCheck,
};

export function DecisionAuditTrail({ trail, className }: DecisionAuditTrailProps) {
  const [expandedStep, setExpandedStep] = useState<number | null>(0); // First step expanded by default

  const toggleStep = (index: number) => {
    setExpandedStep(prev => (prev === index ? null : index));
  };

  return (
    <div className={clsx('bg-surface border border-border-default rounded-xl p-5 shadow-card', className)}>
      <div className="flex items-center justify-between pb-4 mb-4 border-b border-border-default">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-accent-indigo animate-pulse" />
            <h3 className="font-heading font-semibold text-sm text-text-primary">
              Decision Audit Trail
            </h3>
          </div>
          <p className="text-xs text-text-secondary mt-0.5">
            Full end-to-end trace from raw packet ingestion to generated alert
          </p>
        </div>
        <span className="font-mono text-xs px-2.5 py-1 bg-canvas rounded border border-border-default text-text-secondary">
          Flow ID: {trail.flowId.slice(0, 8)}
        </span>
      </div>

      <div className="relative pl-6 space-y-6 before:content-[''] before:absolute before:left-3 before:top-3 before:bottom-3 before:w-0.5 before:bg-border-default">
        {trail.steps.map((step, idx) => {
          const Icon = STAGE_ICONS[step.stageName] || Activity;
          const isExpanded = expandedStep === idx;

          return (
            <div key={idx} className="relative group">
              {/* Timeline Indicator */}
              <div
                className={clsx(
                  'absolute -left-6 top-0.5 w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all duration-200',
                  isExpanded
                    ? 'bg-accent-indigo border-accent-indigo text-white shadow-glow-indigo'
                    : 'bg-surface border-border-default text-text-tertiary group-hover:border-accent-indigo group-hover:text-accent-indigo'
                )}
              >
                <Icon className="w-3 h-3" />
              </div>

              {/* Step Header */}
              <div
                onClick={() => toggleStep(idx)}
                data-interactive
                className="cursor-pointer bg-canvas/60 hover:bg-canvas rounded-lg p-3 border border-border-default/80 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-heading text-xs font-semibold text-text-primary">
                      Step {step.stepIndex + 1}: {step.stageName}
                    </span>
                    <span className="font-mono text-[10px] text-text-tertiary">
                      {new Date(step.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <button
                    type="button"
                    className="text-text-tertiary hover:text-text-primary transition-colors"
                    aria-label="Toggle details"
                  >
                    {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </button>
                </div>

                <p className="text-xs text-text-secondary mt-1.5 leading-relaxed font-body">
                  {step.summary}
                </p>

                {/* Expanded Details */}
                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden"
                    >
                      <div className="mt-3 pt-3 border-t border-border-default">
                        {renderStepDetail(step)}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function KV({ k, v, accent }: { k: string; v: React.ReactNode; accent?: boolean }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-text-secondary">{k}</span>
      <span className={clsx('font-mono font-semibold text-right break-all', accent && 'text-accent-indigo')}>{v}</span>
    </div>
  );
}

const pct = (v: unknown) => (typeof v === 'number' ? `${(v * 100).toFixed(0)}%` : '—');
const str = (v: unknown) => (v === null || v === undefined || v === '' ? '—' : String(v));

function renderStepDetail(step: AuditStep) {
  const d = step.detail as Record<string, any>;
  const box = 'bg-surface p-2.5 rounded border border-border-default text-xs space-y-1.5';
  switch (step.stageName) {
    case 'Raw Flow':
      return (
        <div className={box}>
          <KV k="Source capture" v={str(d.sourceFile)} />
          <KV k="Flow records" v={str(d.flowCount)} />
          <KV k="Packets" v={str(d.packetCount)} />
          <KV k="Window" v={`${str(d.windowStart)} → ${str(d.windowEnd)}`} />
          {d.groundTruthLabel && <KV k="Dataset label (ground truth)" v={str(d.groundTruthLabel)} accent />}
        </div>
      );

    case 'Feature Extraction': {
      const entries = Object.entries((d.window as Record<string, number>) || {}).filter(([k]) => k.startsWith('flow_') || k.startsWith('packet_')).slice(0, 12);
      return (
        <div className="space-y-2">
          <p className="text-[10px] text-text-tertiary font-mono uppercase tracking-wider">Window features (z-scored)</p>
          <div className="grid grid-cols-2 gap-1 text-[11px] font-mono">
            {entries.map(([k, v]) => (
              <div key={k} className="flex justify-between bg-surface px-2 py-1 rounded border border-border-default">
                <span className="text-text-tertiary truncate mr-2">{k}</span>
                <span>{Number(v).toFixed(2)}</span>
              </div>
            ))}
          </div>
        </div>
      );
    }

    case 'Graph State':
      return (
        <div className={box}>
          <KV k="Nodes" v={str(d.nodes)} />
          <KV k="Edges (host-to-host flows)" v={`${str(d.edges)} (${str(d.hostFlows)})`} />
          <KV k="Hosts" v={((d.hosts as string[]) || []).slice(0, 6).join(', ') || 'none (flow CSV windows carry no IPs)'} />
        </div>
      );

    case 'Forecast': {
      const curve = (d.forecastCurve as number[]) || [];
      const stages = (d.predictedStages as { step: number; stage: string; probability: number }[]) || [];
      return (
        <div className={box}>
          <KV k="Latent dimension" v={str(d.embeddingDim)} />
          <KV k="Rollout" v={curve.length ? curve.map(c => `${Math.round(c * 100)}`).join(' → ') : '—'} accent />
          {stages.length > 0 && <KV k="Predicted stages" v={stages.map(s => `t+${s.step}: ${s.stage}`).join(', ')} />}
          <KV k="Model" v={d.modelSha256 ? `sha256:${String(d.modelSha256).slice(0, 16)}…` : '—'} />
        </div>
      );
    }

    case 'Attack Stage Mapping': {
      const top = (d.topFeatures as Array<{ name: string; value: number; contribution: number }>) || [];
      const maxAbs = Math.max(1e-9, ...top.map(t => Math.abs(t.contribution)));
      return (
        <div className="space-y-2">
          <div className={box}>
            <KV k="Stage" v={str(d.attackStage)} accent />
            <KV k="Infiltration probability" v={pct(d.infiltrationProbability)} />
            {d.rawProbability !== undefined && d.calibratedProbability !== null && <KV k="Raw model output → calibrated" v={`${pct(d.rawProbability)} → ${pct(d.calibratedProbability)}`} />}
            <KV k="Stage confidence" v={pct(d.confidence)} />
            <KV k="Scoring method" v={str(d.scoringMethod)} />
            <KV k="Attribution" v={str(d.attributionMethod)} />
          </div>
          {top.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5">
              {top.map((f, i) => (
                <FeatureHeatmapCell key={i} name={f.name} value={f.contribution} contribution={Math.abs(f.contribution) / maxAbs} />
              ))}
            </div>
          )}
        </div>
      );
    }

    case 'Alert':
      return (
        <div className={box}>
          <KV k="Alert block" v={d.alertBlock !== null && d.alertBlock !== undefined ? `#${d.alertBlock}` : 'not chained'} accent />
          {d.alertHash && <KV k="Block hash" v={`${String(d.alertHash).slice(0, 24)}…`} />}
        </div>
      );

    case 'Blockchain': {
      const blocks = (d.blocks as Array<{ index: number; type: string; hash: string; prevHash: string }>) || [];
      return (
        <div className="space-y-1.5">
          <KV k="Chain tier" v={str(d.chainTier)} />
          {blocks.map(b => (
            <div key={b.index} className="bg-surface px-2 py-1.5 rounded border border-border-default font-mono text-[10px] flex items-center gap-2">
              <span className="text-accent-indigo font-semibold">#{b.index}</span>
              <span className="text-text-secondary">{b.type}</span>
              <span className="text-text-tertiary truncate">{b.hash.slice(0, 20)}… ← {b.prevHash.slice(0, 10)}…</span>
            </div>
          ))}
        </div>
      );
    }

    case 'Analyst Action':
      return (
        <div className={box}>
          <KV k="Analyst" v={str(d.analyst_id)} />
          <KV k="Verdict" v={str(d.verdict)} accent />
          <KV k="Justification" v={str(d.note)} />
          <KV k="Ledger block" v={d.block_index !== null && d.block_index !== undefined ? `#${d.block_index}` : '—'} />
        </div>
      );

    default:
      return (
        <pre className="text-[10px] font-mono bg-surface p-2 rounded border border-border-default overflow-x-auto text-text-secondary">
          {JSON.stringify(step.detail, null, 2)}
        </pre>
      );
  }
}
