import { useCallback, useEffect, useState } from 'react';
import clsx from 'clsx';
import { ShieldCheck, Users, ArrowUpRight, X, Loader2, Lock } from 'lucide-react';
import { decideApproval, getPendingApprovals } from '../api';
import type { ApprovalRequest } from '../api/types';
import { useAuth, TIER_STYLE } from '../contexts/AuthContext';

const RULE_TEXT = {
  auto: 'Approved playbook actions execute automatically on this zone.',
  confirm: 'Each response action needs your explicit confirmation before it executes.',
  two_person: 'Every action needs approval from a second cleared officer, or can be escalated.',
} as const;

/** Current analyst's classification tier + (for mission-critical officers) the
 * two-person approval queue. Decisions are chained as REMEDIATION blocks. */
export function AccessTierPanel() {
  const { user } = useAuth();
  const [requests, setRequests] = useState<ApprovalRequest[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  const refresh = useCallback(() => {
    if (user?.tier !== 'high') return;
    getPendingApprovals().then(setRequests).catch(() => undefined);
  }, [user?.tier]);

  useEffect(() => {
    refresh();
    if (user?.tier !== 'high') return;
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh, user?.tier]);

  if (!user?.tier) {
    return (
      <div className="bg-surface border border-border-default rounded-xl p-4 shadow-card flex items-center gap-3 text-xs text-text-secondary">
        <Lock size={16} className="text-text-tertiary" /> Sign in to see your zone's access tier and act on alerts.
      </div>
    );
  }

  const st = TIER_STYLE[user.tier];
  const decide = async (r: ApprovalRequest, d: 'approve' | 'reject' | 'escalate') => {
    setBusy(r.id + d);
    setMsg(null);
    try {
      const res = await decideApproval(r.id, d);
      setMsg(`${r.action.action} → ${res.status} · chained as block #${res.block.index}`);
      refresh();
    } catch (e) {
      setMsg(e instanceof Error ? e.message.replace(/^.*: \d{3} [^:]*: /, '') : 'Failed');
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="bg-surface border border-border-default rounded-xl p-4 shadow-card space-y-3">
      <div className="flex flex-col md:flex-row md:items-center gap-3 justify-between">
        <div className="flex items-center gap-3">
          <div className={clsx('px-2.5 py-1 rounded-lg border text-[11px] font-semibold uppercase tracking-wide flex items-center gap-1.5', st.chip)}>
            <span className={clsx('w-2 h-2 rounded-full', st.dot)} aria-hidden="true" /> {st.label} zone
          </div>
          <div>
            <p className="text-xs font-medium text-text-primary">{user.zone}</p>
            <p className="text-[11px] text-text-secondary">{RULE_TEXT[user.responseRule ?? 'auto']}</p>
          </div>
        </div>
        <p className="text-[10px] text-text-tertiary md:text-right">
          Threat indicators flow upward only: Routine → Sensitive → Mission-critical.
        </p>
      </div>

      {user.tier === 'high' && (
        <div className="border-t border-border-default pt-3 space-y-2">
          <div className="flex items-center gap-2">
            <Users size={14} className="text-risk-red" />
            <h3 className="font-heading font-semibold text-xs text-text-primary">Two-person approval queue</h3>
            <span className="text-[10px] font-mono text-text-tertiary">{requests.length} pending</span>
          </div>
          {requests.length === 0 ? (
            <p className="text-[11px] text-text-tertiary">No actions awaiting a second officer.</p>
          ) : (
            requests.map(r => {
              const mine = r.requestedBy === user.analystId;
              return (
                <div key={r.id} className="rounded-lg border border-risk-red/30 bg-risk-red-subtle/40 p-2.5 flex flex-col md:flex-row md:items-center gap-2 justify-between">
                  <div className="min-w-0">
                    <p className="text-xs font-semibold text-text-primary">{r.action.action} <span className="font-mono font-normal text-text-secondary">→ {r.action.target}</span></p>
                    <p className="text-[10px] text-text-tertiary">
                      Requested by {r.requestedByName} · {new Date(r.createdAt * 1000).toLocaleTimeString()} · request {r.id}
                    </p>
                    {r.action.command && <code className="block font-mono text-[10px] text-text-secondary truncate">{r.action.command}</code>}
                  </div>
                  <div className="flex items-center gap-1.5 shrink-0">
                    {mine ? (
                      <span className="text-[10px] text-text-tertiary italic">Your request: another officer must approve</span>
                    ) : (
                      <button type="button" data-interactive disabled={!!busy} onClick={() => decide(r, 'approve')}
                        className="inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-1 rounded bg-risk-green text-white">
                        {busy === r.id + 'approve' ? <Loader2 size={10} className="animate-spin" /> : <ShieldCheck size={10} />} Approve
                      </button>
                    )}
                    <button type="button" data-interactive disabled={!!busy} onClick={() => decide(r, 'escalate')}
                      className="inline-flex items-center gap-1 text-[10px] font-medium px-2 py-1 rounded border border-risk-amber text-risk-amber">
                      <ArrowUpRight size={10} /> Escalate
                    </button>
                    <button type="button" data-interactive disabled={!!busy} onClick={() => decide(r, 'reject')}
                      className="inline-flex items-center gap-1 text-[10px] font-medium px-2 py-1 rounded border border-border-default text-text-secondary">
                      <X size={10} /> Reject
                    </button>
                  </div>
                </div>
              );
            })
          )}
          {msg && <p className="text-[10px] font-mono text-text-secondary">{msg}</p>}
        </div>
      )}
    </div>
  );
}
