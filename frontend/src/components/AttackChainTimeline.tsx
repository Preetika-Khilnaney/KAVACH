import { useState } from 'react';
import clsx from 'clsx';
import { Copy, Check, ChevronDown, ChevronUp, Crosshair, ShieldCheck, Loader2 } from 'lucide-react';
import { useAttackChain } from '../api/hooks';
import { logRemediation } from '../api';
import type { AttackChainStep, RemediationAction, RemediationDecision } from '../api/types';
import { useAuth } from '../contexts/AuthContext';

/**
 * Attack-chain narrative (roadmap Task 6.4): observed stages (solid) ->
 * current -> NetJEPA-predicted stages (dashed, fading), then prioritised
 * remediation with copyable commands. The action button follows the signed-in
 * analyst's tier: auto-execute (Routine), confirm & execute (Sensitive) or
 * request two-person approval (Mission-critical); each outcome is chained as a
 * REMEDIATION block recording who authorised it.
 */

const STAGE_COLOR: Record<string, string> = {
  Reconnaissance: 'bg-accent-indigo',
  'Initial Access': 'bg-risk-amber',
  'Lateral Movement': 'bg-accent-violet',
  'Command and Control': 'bg-risk-red',
  Exfiltration: 'bg-risk-red',
  Benign: 'bg-risk-green',
};

const PRIORITY_STYLE: Record<string, string> = {
  immediate: 'border-risk-red/40 bg-risk-red-subtle text-risk-red',
  short_term: 'border-risk-amber/40 bg-risk-amber-subtle text-risk-amber',
  ongoing: 'border-risk-green/40 bg-risk-green-subtle text-risk-green',
  investigate: 'border-accent-indigo/30 bg-accent-indigo-subtle text-accent-indigo',
};

function Step({ step, index, total }: { step: AttackChainStep; index: number; total: number }) {
  const [open, setOpen] = useState(false);
  const predicted = step.kind === 'predicted';
  const opacity = predicted ? Math.max(0.45, 1 - index * 0.2) : 1;
  return (
    <div className="relative pl-8" style={{ opacity }}>
      <span
        className={clsx(
          'absolute left-1.5 top-1.5 w-4 h-4 rounded-full border-2',
          predicted ? 'bg-surface border-dashed border-accent-violet' : clsx(STAGE_COLOR[step.stage] ?? 'bg-accent-indigo', 'border-surface shadow'),
        )}
      />
      {index < total - 1 && <span className={clsx('absolute left-[13px] top-6 bottom-[-12px] w-0.5', predicted ? 'border-l-2 border-dashed border-accent-violet/50' : 'bg-border-default')} />}
      <div className={clsx('rounded-lg border p-3', predicted ? 'border-dashed border-accent-violet/60 bg-accent-violet-subtle/40' : 'border-border-default bg-surface')}>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className={clsx('text-[10px] font-mono px-1.5 py-0.5 rounded text-white', predicted ? 'bg-accent-violet' : STAGE_COLOR[step.stage] ?? 'bg-accent-indigo')}>
              {predicted ? 'PREDICTED' : 'OBSERVED'}
            </span>
            <span className="font-heading font-semibold text-xs text-text-primary">{step.stage}</span>
            {step.technique && <span className="font-mono text-[10px] text-text-tertiary">{step.tactic} · {step.technique}</span>}
          </div>
          <span className="font-mono text-[10px] text-text-secondary">
            {predicted
              ? step.steps_ahead ? `t+${step.steps_ahead}${step.probability != null ? ` · ${Math.round(step.probability * 100)}%` : ''}` : 'kill-chain prior'
              : `${step.window_count} window(s) · peak ${Math.round((step.peak_probability ?? 0) * 100)}%`}
          </span>
        </div>
        <p className="text-xs text-text-secondary mt-1.5">{step.narrative}</p>
        {!predicted && (step.top_features?.length || step.affected_hosts?.length) ? (
          <button type="button" data-interactive onClick={() => setOpen(o => !o)} className="mt-1.5 inline-flex items-center gap-1 text-[11px] text-accent-indigo">
            {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />} evidence
          </button>
        ) : null}
        {open && (
          <div className="mt-2 grid sm:grid-cols-2 gap-2 text-[11px]">
            <div>
              <div className="text-text-tertiary font-mono uppercase text-[9px] mb-1">Top features</div>
              {(step.top_features ?? []).map(f => (
                <div key={f.feature} className="flex justify-between font-mono"><span className="truncate mr-2">{f.feature}</span><span>{f.attribution.toFixed(2)}</span></div>
              ))}
            </div>
            <div>
              <div className="text-text-tertiary font-mono uppercase text-[9px] mb-1">Affected hosts</div>
              <div className="font-mono">{(step.affected_hosts ?? []).join(', ') || 'no host IPs in this window'}</div>
              {step.block_index != null && <div className="font-mono text-text-tertiary mt-1">ledger block #{step.block_index}</div>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Action({ a, target, windowId }: { a: RemediationAction; target: string; windowId: string }) {
  const [copied, setCopied] = useState(false);
  const [result, setResult] = useState<RemediationDecision | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const { user } = useAuth();
  const [busy, setBusy] = useState(false);
  return (
    <div className={clsx('rounded-lg border p-3 space-y-1.5', PRIORITY_STYLE[a.priority])}>
      <div className="flex items-center justify-between gap-2">
        <span className="font-heading font-semibold text-xs">{a.action}</span>
        <span className="font-mono text-[9px] uppercase shrink-0">{a.priority_label} · {a.mitigation}</span>
      </div>
      <p className="text-[11px] text-text-secondary">{a.rationale}</p>
      {a.command && (
        <div className="flex items-center gap-2 bg-surface/80 rounded border border-border-default px-2 py-1">
          <code className="font-mono text-[10px] text-text-primary flex-1 overflow-x-auto whitespace-nowrap">{a.command}</code>
          <button
            type="button"
            data-interactive
            aria-label="Copy command"
            onClick={() => {
              void navigator.clipboard?.writeText(a.command!);
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            }}
          >
            {copied ? <Check size={12} className="text-risk-green" /> : <Copy size={12} className="text-text-tertiary" />}
          </button>
        </div>
      )}
      <div className="flex items-center justify-end gap-2 flex-wrap">
        {result ? (
          <span className={clsx('text-[10px] font-mono', result.status === 'executed' ? 'text-risk-green' : 'text-risk-amber')}>
            {result.status === 'executed'
              ? `${result.mode === 'auto' ? 'auto-executed' : 'executed (analyst confirmed)'} · block #${result.block.index}`
              : `awaiting second officer's approval · request ${result.requestId} · block #${result.block.index}`}
          </span>
        ) : !user ? (
          <span className="text-[10px] text-text-tertiary">Sign in to act on this alert</span>
        ) : (
          <>
            {err && <span className="text-[10px] text-risk-red">{err}</span>}
            <button
              type="button"
              data-interactive
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                setErr(null);
                try {
                  setResult(await logRemediation(a, target, windowId, user.responseRule !== 'auto'));
                } catch (e) {
                  setErr(e instanceof Error ? e.message.replace(/^.*: \d{3} [^:]*: /, '') : 'Failed');
                } finally {
                  setBusy(false);
                }
              }}
              className="inline-flex items-center gap-1 text-[10px] font-medium px-2 py-0.5 rounded border border-current"
            >
              {busy ? <Loader2 size={10} className="animate-spin" /> : <ShieldCheck size={10} />}{' '}
              {user.responseRule === 'auto' ? 'Execute (auto-approved)' : user.responseRule === 'confirm' ? 'Confirm & execute' : 'Request two-person approval'}
            </button>
          </>
        )}
      </div>
    </div>
  );
}

export function AttackChainTimeline({ windowId }: { windowId: string }) {
  const { data: chain, loading, error } = useAttackChain(windowId);
  if (loading) return <div className="p-6 text-center text-xs text-text-tertiary">Reconstructing attack chain…</div>;
  if (error || !chain) return <div className="p-6 text-center text-xs text-risk-red">{error?.message ?? 'No chain available.'}</div>;
  const steps = [...chain.observed, ...chain.predicted];
  const target = chain.affected_hosts[0] ?? '<target>';
  return (
    <div className="space-y-5">
      <div className="rounded-xl border border-border-default bg-canvas p-4">
        <div className="flex items-center gap-2 mb-1.5">
          <Crosshair size={14} className="text-risk-red" />
          <h4 className="font-heading font-semibold text-xs text-text-primary">Attack narrative</h4>
        </div>
        <p className="text-xs text-text-secondary leading-relaxed">{chain.narrative}</p>
      </div>
      <div className="space-y-3">
        {steps.map((s, i) => (
          <Step key={`${s.kind}-${s.stage}-${i}`} step={s} index={s.kind === 'predicted' ? i - chain.observed.length + 1 : i} total={steps.length} />
        ))}
      </div>
      {chain.remediation.current.length > 0 && (
        <div className="space-y-2">
          <h4 className="font-heading font-semibold text-xs text-text-primary">Remediation — current stage: {chain.remediation.current_stage}</h4>
          {chain.remediation.current.map(a => <Action key={a.action} a={a} target={target} windowId={windowId} />)}
        </div>
      )}
      {chain.remediation.next && chain.remediation.next.length > 0 && (
        <div className="space-y-2">
          <h4 className="font-heading font-semibold text-xs text-text-primary">Pre-empt the next stage: {chain.remediation.next_stage}</h4>
          {chain.remediation.next.filter(a => a.priority === 'immediate').map(a => <Action key={a.action} a={a} target={target} windowId={windowId} />)}
        </div>
      )}
    </div>
  );
}
