import { useState } from 'react';
import clsx from 'clsx';
import { AnimatePresence, motion } from 'framer-motion';
import { ShieldCheck, ShieldX, Link2, X, RefreshCw, Skull, Trash2, Loader2, Boxes } from 'lucide-react';
import { useChain, useChainStats, useChainVerification } from '../api/hooks';
import { runTamperDemo } from '../api';
import type { LedgerBlock, TamperDemoResult } from '../api/types';

/**
 * Forensic ledger dashboard (roadmap Task 4.4): chain status, block list with
 * hash links, block detail modal, and a safe tamper simulation that proves
 * verification catches modification and deletion.
 */

const TYPE_STYLE: Record<string, string> = {
  GENESIS: 'bg-canvas text-text-secondary border-border-default',
  PREDICTION: 'bg-accent-indigo-subtle text-accent-indigo border-accent-indigo/30',
  PREDICTION_BATCH: 'bg-canvas text-text-secondary border-border-default',
  ALERT: 'bg-risk-red-subtle text-risk-red border-risk-red/30',
  ANALYST_ACTION: 'bg-accent-teal-subtle text-accent-teal border-accent-teal/30',
  ACCESS: 'bg-risk-amber-subtle text-risk-amber border-risk-amber/30',
  MODEL_UPDATE: 'bg-accent-violet-subtle text-accent-violet border-accent-violet/30',
  TRAINING_MANIFEST: 'bg-accent-violet-subtle text-accent-violet border-accent-violet/30',
  REMEDIATION: 'bg-risk-green-subtle text-risk-green border-risk-green/30',
  CONTRACT_EXECUTION: 'bg-risk-amber-subtle text-risk-amber border-risk-amber/30',
  ESCALATION: 'bg-risk-red-subtle text-risk-red border-risk-red/30',
  FEEDBACK_AUDIT: 'bg-accent-teal-subtle text-accent-teal border-accent-teal/30',
  SYNC: 'bg-canvas text-accent-indigo border-accent-indigo/30',
};

function blockSummary(b: LedgerBlock): string {
  const d = b.data as Record<string, any>;
  switch (b.blockType) {
    case 'PREDICTION':
      return `${d.window_id} · ${Math.round((d.infiltration_probability ?? 0) * 100)}% · ${d.attack_stage} · ${d.tier}`;
    case 'PREDICTION_BATCH':
      return `${d.count} low-risk windows · merkle ${String(d.merkle_root).slice(0, 10)}…`;
    case 'ALERT':
      return `${d.alert_id} · ${d.severity} · ${d.attack_stage} (${d.mitre_technique})`;
    case 'ANALYST_ACTION':
      return `${d.analyst_id} → ${d.action} on ${d.target_id}`;
    case 'MODEL_UPDATE':
      return `weights ${String(d.previous_weights_sha256 ?? 'none').slice(0, 8)}… → ${String(d.new_weights_sha256).slice(0, 8)}…`;
    case 'TRAINING_MANIFEST':
      return `${d.file_count} training files · manifest ${String(d.manifest_hash).slice(0, 10)}…`;
    case 'CONTRACT_EXECUTION':
      return `${d.contract}: ${d.action} ${d.subject ?? ''}`;
    case 'ESCALATION':
      return `${d.contract}: ${d.subject} → ${d.new_priority}`;
    case 'REMEDIATION':
      return `${d.action} on ${d.target} by ${d.authorized_by}${d.status ? ` · ${String(d.status).replace('_', ' ')}` : ''}${Array.isArray(d.approvers) && d.approvers.length > 1 ? ` (approved by ${d.approvers.length})` : ''}`;
    case 'ACCESS':
      return `${String(d.event).replace('_', ' ').toLowerCase()} · ${d.username}${d.tier ? ` · ${d.tier} tier` : ''}`;
    case 'FEEDBACK_AUDIT':
      return `analyst audit: ${d.status} (${d.flag_count} flags)`;
    case 'SYNC':
      return `sync from ${d.peer_node_id}: ${d.blocks_imported} blocks`;
    default:
      return String(d.message ?? '');
  }
}

export function BlockchainAuditTrail({ compact = false, limit = 30 }: { compact?: boolean; limit?: number }) {
  const [refreshKey, setRefreshKey] = useState(0);
  const [typeFilter, setTypeFilter] = useState<string>('');
  const { data: verification, loading: verifying } = useChainVerification(refreshKey);
  const { data: chain, loading } = useChain(limit, typeFilter || undefined, refreshKey);
  const { data: stats } = useChainStats(refreshKey);
  const [selected, setSelected] = useState<LedgerBlock | null>(null);
  const [tamper, setTamper] = useState<TamperDemoResult | null>(null);
  const [tampering, setTampering] = useState(false);

  const doTamper = async (mode: 'modify' | 'delete') => {
    setTampering(true);
    try {
      setTamper(await runTamperDemo(mode));
    } finally {
      setTampering(false);
    }
  };

  const valid = verification?.valid;
  return (
    <div className="bg-surface border border-border-default rounded-xl p-5 shadow-card space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className={clsx('w-10 h-10 rounded-xl flex items-center justify-center', verifying ? 'bg-canvas text-text-tertiary' : valid ? 'bg-risk-green-subtle text-risk-green' : 'bg-risk-red-subtle text-risk-red')}>
            {verifying ? <Loader2 size={20} className="animate-spin" /> : valid ? <ShieldCheck size={20} /> : <ShieldX size={20} />}
          </div>
          <div>
            <h3 className="font-heading font-semibold text-sm text-text-primary">
              {verifying ? 'Verifying chain…' : valid ? 'Chain Intact' : 'Tampering Detected'}
            </h3>
            <p className="text-xs text-text-secondary max-w-xl">{verification?.message ?? 'Walking genesis → tip, recomputing every SHA-256 link.'}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="px-2 py-1 rounded bg-canvas border border-border-default">{stats?.total ?? chain?.total ?? 0} blocks</span>
          {stats && <span className="px-2 py-1 rounded bg-canvas border border-border-default">node {stats.nodeId}</span>}
          {(chain?.pendingLowRisk ?? 0) > 0 && <span className="px-2 py-1 rounded bg-canvas border border-border-default">{chain?.pendingLowRisk} buffered</span>}
          <button type="button" data-interactive onClick={() => setRefreshKey(k => k + 1)} className="p-1.5 rounded border border-border-default hover:bg-canvas" aria-label="Re-verify chain">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {!compact && stats && (
        <div className="flex flex-wrap gap-1.5">
          <button type="button" data-interactive onClick={() => setTypeFilter('')} className={clsx('text-[10px] font-mono px-2 py-0.5 rounded border', !typeFilter ? 'bg-accent-indigo text-white border-accent-indigo' : 'border-border-default text-text-secondary')}>
            all
          </button>
          {Object.entries(stats.byType).map(([t, n]) => (
            <button key={t} type="button" data-interactive onClick={() => setTypeFilter(t)} className={clsx('text-[10px] font-mono px-2 py-0.5 rounded border', typeFilter === t ? 'bg-accent-indigo text-white border-accent-indigo' : TYPE_STYLE[t] ?? 'border-border-default')}>
              {t} · {n}
            </button>
          ))}
        </div>
      )}

      <div className={clsx('space-y-1 overflow-y-auto pr-1', compact ? 'max-h-64' : 'max-h-[480px]')} aria-busy={loading}>
        {chain?.blocks.map((b, i) => (
          <div key={b.index}>
            <button
              type="button"
              data-interactive
              onClick={() => setSelected(b)}
              className="w-full text-left flex items-center gap-2 px-2.5 py-2 rounded-lg border border-border-default hover:border-accent-indigo/50 hover:bg-canvas transition-colors"
            >
              <span className="font-mono text-[11px] font-semibold text-accent-indigo w-12 shrink-0">#{b.index}</span>
              <span className={clsx('text-[10px] font-mono px-1.5 py-0.5 rounded border shrink-0', TYPE_STYLE[b.blockType])}>{b.typeLabel}</span>
              <span className="text-[11px] text-text-secondary truncate flex-1">{blockSummary(b)}</span>
              <span className="font-mono text-[10px] text-text-tertiary hidden md:inline">{b.hashShort}</span>
              <span className="font-mono text-[10px] text-text-tertiary hidden lg:inline">{new Date(b.timestamp).toLocaleTimeString()}</span>
            </button>
            {i < (chain?.blocks.length ?? 0) - 1 && (
              <div className="flex items-center gap-1 pl-6 text-[9px] font-mono text-text-tertiary" aria-hidden>
                <Link2 size={9} /> prev {b.prevHashShort}
              </div>
            )}
          </div>
        ))}
        {!loading && !chain?.blocks.length && <p className="text-xs text-text-secondary text-center py-6">No blocks yet — process a capture.</p>}
      </div>

      {!compact && (
        <div className="border-t border-border-default pt-4 space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h4 className="font-heading font-semibold text-xs text-text-primary">Attack simulation: can an intruder with root erase the evidence?</h4>
              <p className="text-[11px] text-text-secondary">Runs on a throw-away copy of the chain — the real ledger is never touched.</p>
            </div>
            <div className="flex gap-2">
              <button type="button" data-interactive disabled={tampering} onClick={() => doTamper('modify')} className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-risk-red/40 text-risk-red hover:bg-risk-red-subtle">
                <Skull size={13} /> Rewrite a block
              </button>
              <button type="button" data-interactive disabled={tampering} onClick={() => doTamper('delete')} className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-risk-red/40 text-risk-red hover:bg-risk-red-subtle">
                <Trash2 size={13} /> Delete a block
              </button>
            </div>
          </div>
          {tamper && (
            <div className={clsx('rounded-lg border p-3 text-xs space-y-1', tamper.verification?.valid ? 'border-risk-amber bg-risk-amber-subtle' : 'border-risk-red/40 bg-risk-red-subtle')}>
              {tamper.simulated ? (
                <>
                  <div className="font-semibold text-text-primary">Attacker {tamper.action}</div>
                  <div className="text-risk-red font-mono">{tamper.verification?.message}</div>
                  {tamper.verification?.issues.slice(0, 3).map((iss, i) => (
                    <div key={i} className="font-mono text-[10px] text-text-secondary">• [{iss.kind}] {iss.detail}</div>
                  ))}
                  <div className="text-risk-green text-[11px]">Real chain untouched — still verifies.</div>
                </>
              ) : (
                <div>{tamper.reason}</div>
              )}
            </div>
          )}
        </div>
      )}

      <AnimatePresence>
        {selected && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-50 bg-black/30 flex items-center justify-center p-4" onClick={() => setSelected(null)}>
            <motion.div initial={{ scale: 0.97 }} animate={{ scale: 1 }} className="bg-surface rounded-2xl border border-border-default shadow-panel max-w-2xl w-full max-h-[85vh] flex flex-col" onClick={e => e.stopPropagation()} role="dialog" aria-label={`Block ${selected.index}`}>
              <div className="flex items-center justify-between p-4 border-b border-border-default">
                <div className="flex items-center gap-2">
                  <Boxes size={18} className="text-accent-indigo" />
                  <h3 className="font-heading font-semibold text-sm">Block #{selected.index} · {selected.typeLabel}</h3>
                </div>
                <button type="button" data-interactive onClick={() => setSelected(null)} aria-label="Close"><X size={18} /></button>
              </div>
              <div className="p-4 overflow-y-auto space-y-3 text-xs">
                <div className="grid grid-cols-1 gap-1.5 font-mono text-[11px]">
                  <div><span className="text-text-tertiary">timestamp </span>{new Date(selected.timestamp).toLocaleString()}</div>
                  <div><span className="text-text-tertiary">node </span>{selected.nodeId}</div>
                  <div className="break-all"><span className="text-text-tertiary">hash </span>{selected.hash}</div>
                  <div className="break-all"><span className="text-text-tertiary">prev_hash </span>{selected.prevHash}</div>
                </div>
                <pre className="bg-canvas border border-border-default rounded-lg p-3 text-[10px] font-mono overflow-x-auto whitespace-pre-wrap">{JSON.stringify(selected.data, null, 2)}</pre>
                <p className="text-[11px] text-text-tertiary">hash = SHA-256(index | timestamp | type | node | prev_hash | canonical JSON data). Change any byte above and every later block's link breaks.</p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
