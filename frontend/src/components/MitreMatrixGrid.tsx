import clsx from 'clsx';
import type { KillChainStage } from '../api/types';

/**
 * MITRE ATT&CK coverage matrix (roadmap Task 3.5).
 *
 * One column per kill-chain tactic KAVACH maps to. The technique the model
 * maps each stage to (src/scoring/attack_stage.py STAGE_META) is highlighted
 * solid when that stage has been observed in the processed traffic, dashed
 * when the NetJEPA rollout predicts it next. Neighbouring techniques in the
 * same tactic are shown for context (coverage gaps are visible, not hidden).
 */

const MATRIX: { stage: string; tactic: string; tacticId: string; techniques: { id: string; name: string }[] }[] = [
  {
    stage: 'Reconnaissance', tactic: 'Reconnaissance', tacticId: 'TA0043',
    techniques: [
      { id: 'T1046', name: 'Network Service Discovery' },
      { id: 'T1595', name: 'Active Scanning' },
      { id: 'T1590', name: 'Gather Victim Network Info' },
    ],
  },
  {
    stage: 'Initial Access', tactic: 'Initial Access', tacticId: 'TA0001',
    techniques: [
      { id: 'T1190', name: 'Exploit Public-Facing App' },
      { id: 'T1110', name: 'Brute Force' },
      { id: 'T1078', name: 'Valid Accounts' },
    ],
  },
  {
    stage: 'Lateral Movement', tactic: 'Lateral Movement', tacticId: 'TA0008',
    techniques: [
      { id: 'T1021', name: 'Remote Services' },
      { id: 'T1210', name: 'Exploitation of Remote Services' },
      { id: 'T1570', name: 'Lateral Tool Transfer' },
    ],
  },
  {
    stage: 'Command and Control', tactic: 'Command & Control', tacticId: 'TA0011',
    techniques: [
      { id: 'T1071', name: 'Application Layer Protocol' },
      { id: 'T1573', name: 'Encrypted Channel' },
      { id: 'T1095', name: 'Non-App Layer Protocol' },
    ],
  },
  {
    stage: 'Exfiltration', tactic: 'Exfiltration', tacticId: 'TA0010',
    techniques: [
      { id: 'T1041', name: 'Exfil Over C2 Channel' },
      { id: 'T1048', name: 'Exfil Over Alt Protocol' },
      { id: 'T1030', name: 'Data Transfer Size Limits' },
    ],
  },
];

interface Props {
  stages: KillChainStage[];
  className?: string;
}

export function MitreMatrixGrid({ stages, className }: Props) {
  const byName = new Map(stages.map(s => [s.name, s]));
  return (
    <div className={clsx('overflow-x-auto', className)}>
      <div className="grid grid-cols-5 gap-2 min-w-[640px]" role="table" aria-label="MITRE ATT&CK coverage matrix">
        {MATRIX.map(col => {
          const s = byName.get(col.stage);
          const observed = (s?.observedCount ?? 0) > 0;
          const mapped = s?.technique;
          return (
            <div key={col.stage} className="space-y-1.5" role="rowgroup">
              <div className={clsx('rounded-lg px-2 py-1.5 border text-center', s?.isActive ? 'bg-accent-indigo text-white border-accent-indigo' : 'bg-canvas border-border-default')}>
                <div className="font-heading font-semibold text-[11px] leading-tight">{col.tactic}</div>
                <div className={clsx('font-mono text-[9px]', s?.isActive ? 'text-white/80' : 'text-text-tertiary')}>
                  {col.tacticId} · {s?.observedCount ?? 0} windows
                </div>
              </div>
              {col.techniques.map(t => {
                const isMapped = t.id === mapped;
                const hit = isMapped && observed;
                const predicted = isMapped && !observed && s?.isPredicted;
                return (
                  <div
                    key={t.id}
                    role="cell"
                    title={isMapped ? 'Technique KAVACH maps this stage to' : 'Not directly modelled (context)'}
                    className={clsx(
                      'rounded-md px-2 py-1.5 border text-[10px] leading-tight transition-colors',
                      hit && 'bg-risk-red-subtle border-risk-red text-risk-red font-semibold',
                      predicted && 'border-dashed border-accent-violet text-accent-violet bg-accent-violet-subtle',
                      !hit && !predicted && isMapped && 'border-border-default text-text-primary bg-surface',
                      !isMapped && 'border-border-default/60 text-text-tertiary bg-surface/60',
                    )}
                  >
                    <div className="font-mono">{t.id}</div>
                    <div>{t.name}</div>
                    {hit && <div className="font-mono text-[9px] mt-0.5">detected</div>}
                    {predicted && <div className="font-mono text-[9px] mt-0.5">forecast t+{s?.predictedStepsAhead}</div>}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap gap-3 mt-3 text-[10px] text-text-secondary">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded border border-risk-red bg-risk-red-subtle" /> Detected in processed traffic</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded border border-dashed border-accent-violet bg-accent-violet-subtle" /> Forecast by NetJEPA rollout</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded border border-border-default" /> Modelled, not seen</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded border border-border-default/60 opacity-60" /> Context only (not modelled)</span>
      </div>
    </div>
  );
}
