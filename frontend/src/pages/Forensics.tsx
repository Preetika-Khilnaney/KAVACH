import { useRef, useState } from 'react';
import clsx from 'clsx';
import { Scale, Cpu, UserSearch, HardDriveDownload, HardDriveUpload, FileCheck2, Loader2, Play, Database } from 'lucide-react';
import { BlockchainAuditTrail } from '../components/BlockchainAuditTrail';
import { useContracts, useFeedbackAudit, useProvenance, useTrainingDataVerification } from '../api/hooks';
import { evaluateContracts, exportChain, getEvidence, getFeedbackAudit, importChain } from '../api';
import type { EvidenceCheck } from '../api/types';

/**
 * Forensics: the blockchain half of the AI <-> blockchain loop.
 * - tamper-evident evidence chain (+ tamper simulation)
 * - smart contracts (consensus block, escalation, regression rollback)
 * - model provenance + training-data integrity
 * - the AI auditing analyst feedback before it is trusted for retraining
 * - air-gapped multi-node sync bundles
 */

function Card({ icon: Icon, title, subtitle, children, action }: { icon: typeof Scale; title: string; subtitle: string; children: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="bg-surface border border-border-default rounded-xl p-5 shadow-card space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2">
          <Icon size={16} className="text-accent-indigo mt-0.5" />
          <div>
            <h3 className="font-heading font-semibold text-sm text-text-primary">{title}</h3>
            <p className="text-xs text-text-secondary">{subtitle}</p>
          </div>
        </div>
        {action}
      </div>
      {children}
    </div>
  );
}

const SEV: Record<string, string> = {
  critical: 'bg-risk-red text-white',
  high: 'bg-risk-red-subtle text-risk-red',
  medium: 'bg-risk-amber-subtle text-risk-amber',
  low: 'bg-canvas text-text-secondary',
};

export function Forensics() {
  const [refresh, setRefresh] = useState(0);
  const { data: contracts } = useContracts(refresh);
  const { data: audit } = useFeedbackAudit(refresh);
  const { data: provenance } = useProvenance();
  const { data: training } = useTrainingDataVerification();
  const [evaluating, setEvaluating] = useState(false);
  const [evalResult, setEvalResult] = useState<string | null>(null);
  const [syncMsg, setSyncMsg] = useState<string | null>(null);
  const [evidenceId, setEvidenceId] = useState('');
  const [evidence, setEvidence] = useState<EvidenceCheck | null>(null);
  const [evidenceErr, setEvidenceErr] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const runContracts = async () => {
    setEvaluating(true);
    try {
      const r = await evaluateContracts();
      const n = Object.values(r.fired).reduce((a, b) => a + b.length, 0);
      setEvalResult(n ? `${n} contract execution(s) chained.` : 'No contract conditions met right now.');
      setRefresh(k => k + 1);
    } finally {
      setEvaluating(false);
    }
  };

  const doExport = async () => {
    const bundle = await exportChain(0);
    const blob = new Blob([JSON.stringify(bundle)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `kavach-chain-${(bundle as { node_id?: string }).node_id ?? 'node'}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    setSyncMsg('Bundle exported — carry it across the air gap and import it on the peer node.');
  };

  const doImport = async (file: File) => {
    try {
      const r = await importChain(JSON.parse(await file.text()));
      setSyncMsg(r.accepted ? `Imported ${r.blocks_imported} verified blocks from ${r.peer_node_id}; receipt chained.` : `Rejected: ${r.reason}`);
      setRefresh(k => k + 1);
    } catch (e) {
      setSyncMsg(`Import failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const checkEvidence = async () => {
    setEvidenceErr(null);
    setEvidence(null);
    try {
      setEvidence(await getEvidence(evidenceId.trim()));
    } catch (e) {
      setEvidenceErr(e instanceof Error ? e.message : String(e));
    }
  };

  const rb = contracts?.rollback;
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-heading font-bold text-text-primary tracking-tight">Forensics & Trust Ledger</h1>
        <p className="text-xs text-text-secondary mt-0.5 max-w-3xl">
          Every prediction, alert, analyst verdict, model update and automated action is hash-chained. The AI decides what is worth chaining
          and audits the analysts; the chain makes the AI's history, training data and feedback tamper-evident.
        </p>
      </div>

      <BlockchainAuditTrail />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card
          icon={Scale}
          title="Smart contracts"
          subtitle="Pre-agreed rules over AI predictions + multi-node consensus. Executions are chained."
          action={
            <button type="button" data-interactive onClick={runContracts} disabled={evaluating} className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-accent-indigo text-white">
              {evaluating ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />} Evaluate now
            </button>
          }
        >
          {evalResult && <p className="text-[11px] font-mono text-accent-indigo">{evalResult}</p>}
          <div className="space-y-2">
            {contracts?.definitions.map(c => (
              <div key={c.id} className="rounded-lg border border-border-default p-2.5">
                <div className="font-heading font-semibold text-xs">{c.name}</div>
                <p className="text-[11px] text-text-secondary font-mono mt-0.5">{c.rule}</p>
              </div>
            ))}
          </div>
          {contracts && contracts.consensus.length > 0 && (
            <div>
              <div className="text-[10px] font-mono uppercase text-text-tertiary mb-1">C2 consensus votes (24h)</div>
              {contracts.consensus.slice(0, 5).map(v => (
                <div key={v.ip} className="flex items-center justify-between text-[11px] font-mono">
                  <span>{v.ip}</span>
                  <span className={v.votes >= v.required ? 'text-risk-red font-semibold' : 'text-text-secondary'}>{v.votes}/{v.required} nodes ({v.nodes.join(', ')})</span>
                </div>
              ))}
            </div>
          )}
          <div>
            <div className="text-[10px] font-mono uppercase text-text-tertiary mb-1">Recent executions</div>
            {contracts?.executions.length ? (
              contracts.executions.slice(0, 6).map(b => (
                <div key={b.index} className="text-[11px] font-mono flex gap-2">
                  <span className="text-accent-indigo">#{b.index}</span>
                  <span className="truncate">{String((b.data as Record<string, unknown>).contract)} → {String((b.data as Record<string, unknown>).subject ?? '')} {String((b.data as Record<string, unknown>).new_priority ?? (b.data as Record<string, unknown>).action ?? '')}</span>
                </div>
              ))
            ) : (
              <p className="text-[11px] text-text-tertiary">None yet.</p>
            )}
          </div>
        </Card>

        <Card icon={Cpu} title="Model provenance & training-data integrity" subtitle="Which weights produced every prediction, and whether the data they were trained on has changed since.">
          <div className="text-[11px] font-mono space-y-1">
            <div><span className="text-text-tertiary">serving </span>{provenance?.version ?? '—'} {provenance?.headsValidated ? <span className="text-risk-green">(heads validated)</span> : <span className="text-risk-amber">(heads not validated)</span>}</div>
            <div className="break-all"><span className="text-text-tertiary">sha256 </span>{provenance?.sha256 ?? '—'}</div>
            <div><span className="text-text-tertiary">parameters </span>{provenance?.numParameters?.toLocaleString() ?? '—'} · <span className="text-text-tertiary">ledger block </span>#{provenance?.ledgerBlock ?? '—'}</div>
          </div>
          <div>
            <div className="text-[10px] font-mono uppercase text-text-tertiary mb-1">Model history (MODEL_UPDATE blocks)</div>
            {(provenance?.modelHistory ?? []).slice().reverse().map(h => (
              <div key={h.block} className="text-[11px] font-mono">
                #{h.block} · {new Date(h.timestamp).toLocaleString()} · {h.previous ? `${h.previous.slice(0, 8)}… → ` : ''}{h.sha256.slice(0, 8)}…
              </div>
            ))}
          </div>
          {rb && (
            <div className="rounded-lg border border-border-default p-2.5 text-[11px]">
              <div className="font-heading font-semibold text-xs">Regression rollback contract: <span className={clsx(rb.state === 'regression_detected' ? 'text-risk-red' : 'text-accent-indigo')}>{rb.state.replace(/_/g, ' ')}</span></div>
              {rb.required != null && (
                <p className="text-text-secondary mt-0.5">
                  {rb.reviewed_alerts_current}/{rb.required} reviewed alerts from the current model · FPR {rb.fpr_current != null ? `${Math.round(rb.fpr_current * 100)}%` : 'n/a'} vs previous {rb.fpr_previous != null ? `${Math.round(rb.fpr_previous * 100)}%` : 'n/a'} (rollback if &gt; {rb.multiplier}×)
                </p>
              )}
            </div>
          )}
          {training && (
            <div className={clsx('rounded-lg border p-2.5 text-[11px] flex gap-2', training.verified ? 'border-risk-green/40 bg-risk-green-subtle' : 'border-risk-red/40 bg-risk-red-subtle')}>
              <Database size={14} className={training.verified ? 'text-risk-green' : 'text-risk-red'} />
              <div>
                <div className="font-semibold">{training.verified ? 'Training data intact' : training.status === 'no_manifest' ? 'No manifest yet' : 'Training data changed — possible poisoning'}</div>
                <div className="text-text-secondary">{training.message}</div>
                {!!training.altered_files?.length && <div className="font-mono text-risk-red">altered: {training.altered_files.join(', ')}</div>}
              </div>
            </div>
          )}
        </Card>
      </div>

      <Card
        icon={UserSearch}
        title="AI audit of analyst feedback"
        subtitle="Before any verdict is trusted for retraining, the model checks the chained feedback for insider patterns, poisoning and records edited after the fact."
        action={
          <button type="button" data-interactive onClick={async () => { await getFeedbackAudit(true); setRefresh(k => k + 1); }} className="text-xs px-3 py-1.5 rounded-lg border border-border-default hover:bg-canvas">
            Chain this audit
          </button>
        }
      >
        {audit && (
          <>
            <div className="flex flex-wrap gap-2 text-[11px] font-mono">
              <span className={clsx('px-2 py-0.5 rounded', audit.status === 'clean' ? 'bg-risk-green-subtle text-risk-green' : audit.status === 'critical' ? 'bg-risk-red text-white' : 'bg-risk-amber-subtle text-risk-amber')}>{audit.status}</span>
              <span className="px-2 py-0.5 rounded bg-canvas">{audit.total_feedback} verdicts</span>
              <span className="px-2 py-0.5 rounded bg-canvas">{audit.trusted_feedback} trusted for retraining</span>
              <span className="px-2 py-0.5 rounded bg-canvas">{audit.excluded_feedback} held back</span>
            </div>
            <div className="space-y-1.5">
              {audit.flags.map((f, i) => (
                <div key={i} className="flex gap-2 text-[11px]">
                  <span className={clsx('px-1.5 py-0.5 rounded font-mono text-[9px] uppercase h-fit shrink-0', SEV[f.severity])}>{f.severity}</span>
                  <span className="font-mono text-text-tertiary shrink-0">{f.kind}</span>
                  <span className="text-text-secondary">{f.detail}</span>
                </div>
              ))}
              {!audit.flags.length && <p className="text-[11px] text-text-tertiary">No anomalies in the analyst trail.</p>}
            </div>
            {audit.analysts.length > 0 && (
              <table className="w-full text-[11px] font-mono">
                <thead><tr className="text-text-tertiary text-left"><th className="font-normal">Analyst</th><th className="font-normal">Actions</th><th className="font-normal">Dismissal rate</th><th className="font-normal">Median response</th></tr></thead>
                <tbody>
                  {audit.analysts.map(a => (
                    <tr key={a.analyst_id} className="border-t border-border-default/60">
                      <td className="py-1">{a.analyst_id}</td>
                      <td>{a.actions}</td>
                      <td>{Math.round(a.dismissal_rate * 100)}%</td>
                      <td>{a.median_response_seconds != null ? `${Math.round(a.median_response_seconds / 60)} min` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card icon={HardDriveDownload} title="Air-gapped node sync" subtitle="Export this node's chain to secure media; import a peer's bundle (hash-verified before it is trusted).">
          <div className="flex gap-2">
            <button type="button" data-interactive onClick={doExport} className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-border-default hover:bg-canvas">
              <HardDriveDownload size={13} /> Export bundle
            </button>
            <button type="button" data-interactive onClick={() => fileRef.current?.click()} className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border border-border-default hover:bg-canvas">
              <HardDriveUpload size={13} /> Import peer bundle
            </button>
            <input ref={fileRef} type="file" accept="application/json" className="hidden" onChange={e => e.target.files?.[0] && doImport(e.target.files[0])} />
          </div>
          {syncMsg && <p className="text-[11px] font-mono text-text-secondary">{syncMsg}</p>}
        </Card>

        <Card icon={FileCheck2} title="Verify evidence for a window" subtitle="Re-derives the stored prediction from results.db and checks it against its chained hash / Merkle batch.">
          <div className="flex gap-2">
            <input value={evidenceId} onChange={e => setEvidenceId(e.target.value)} placeholder="window id (from Flow Explorer)" className="flex-1 text-xs font-mono px-2 py-1.5 rounded-lg border border-border-default bg-canvas" />
            <button type="button" data-interactive onClick={checkEvidence} disabled={!evidenceId.trim()} className="text-xs px-3 py-1.5 rounded-lg bg-accent-indigo text-white">Verify</button>
          </div>
          {evidenceErr && <p className="text-[11px] text-risk-red">{evidenceErr}</p>}
          {evidence && (
            <div className={clsx('rounded-lg border p-2.5 text-[11px]', evidence.status === 'verified' ? 'border-risk-green/40 bg-risk-green-subtle' : evidence.status === 'mismatch' || evidence.status === 'chain_invalid' ? 'border-risk-red/40 bg-risk-red-subtle' : 'border-border-default')}>
              <div className="font-semibold uppercase font-mono">{evidence.status.replace('_', ' ')}</div>
              <div className="text-text-secondary">{evidence.message}</div>
              {evidence.checks.map(c => (
                <div key={c.block} className="font-mono">block #{c.block} ({c.kind}): {c.match ? 'match' : `MISMATCH ${c.mismatched_fields?.join(', ') ?? ''}`}</div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
