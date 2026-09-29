import { useEffect, useState } from 'react';
import clsx from 'clsx';
import { Brain, Layers, Microscope, Sparkles, ToggleLeft, AlertTriangle } from 'lucide-react';
import { useExplanation } from '../api/hooks';
import { explainConcepts } from '../api';
import type { ConceptPrediction } from '../api/types';

/**
 * "Explain this prediction" (roadmap Task 5.5 frontend): Integrated
 * Gradients, sparse-autoencoder features, causal tracing, tuned lens and the
 * concept bottleneck -- with analyst overrides of individual concepts.
 */

function Bar({ value, max = 1, color = 'bg-accent-indigo' }: { value: number; max?: number; color?: string }) {
  const w = Math.min(100, Math.max(0, (Math.abs(value) / (max || 1)) * 100));
  return (
    <div className="h-1.5 bg-canvas rounded-full overflow-hidden flex-1">
      <div className={clsx('h-full rounded-full', color)} style={{ width: `${w}%` }} />
    </div>
  );
}

function Section({ icon: Icon, title, subtitle, children }: { icon: typeof Brain; title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-border-default p-4 space-y-3">
      <div className="flex items-start gap-2">
        <Icon size={15} className="text-accent-indigo mt-0.5" />
        <div>
          <h4 className="font-heading font-semibold text-xs text-text-primary">{title}</h4>
          <p className="text-[11px] text-text-secondary">{subtitle}</p>
        </div>
      </div>
      {children}
    </section>
  );
}

export function DeepExplainPanel({ windowId }: { windowId: string }) {
  const { data, loading, error } = useExplanation(windowId, 'ig,sae,causal,logit_lens,concepts');
  const [concepts, setConcepts] = useState<ConceptPrediction | null>(null);
  const [overrides, setOverrides] = useState<Record<string, number>>({});

  useEffect(() => {
    setConcepts(data?.concepts ?? null);
    setOverrides({});
  }, [data]);

  const toggle = async (name: string, current: number) => {
    const next = { ...overrides, [name]: current >= 0.5 ? 0 : 1 };
    setOverrides(next);
    const r = await explainConcepts(windowId, next);
    setConcepts(r.concepts);
  };

  if (loading) return <div className="p-6 text-center text-xs text-text-tertiary">Opening the model: running attribution, SAE, causal patching and the tuned lens…</div>;
  if (error || !data) return <div className="p-6 text-center text-xs text-risk-red">{error?.message ?? 'Explanation unavailable.'}</div>;

  const ig = data.ig?.features ?? [];
  const igMax = Math.max(1e-9, ...ig.map(f => Math.abs(f.attribution)));
  const causal = data.causal?.layers ?? [];
  const lens = data.logit_lens?.layers ?? [];

  return (
    <div className="space-y-4">
      {data.caveat && (
        <div className="flex gap-2 rounded-lg border border-risk-amber/40 bg-risk-amber-subtle p-3 text-[11px] text-text-secondary">
          <AlertTriangle size={14} className="text-risk-amber shrink-0 mt-0.5" />
          <span>{data.caveat}</span>
        </div>
      )}

      <Section icon={Sparkles} title="Integrated Gradients" subtitle={`Which input features moved the infiltration logit (${data.ig?.source ?? 'n/a'}). Red pushes risk up, green pushes it down.`}>
        <div className="space-y-1.5">
          {ig.slice(0, 8).map(f => (
            <div key={f.feature} className="flex items-center gap-2 text-[11px]">
              <span className="font-mono w-44 truncate">{f.feature}</span>
              <Bar value={f.attribution} max={igMax} color={f.attribution >= 0 ? 'bg-risk-red' : 'bg-risk-green'} />
              <span className="font-mono w-14 text-right">{f.attribution.toFixed(3)}</span>
            </div>
          ))}
          {!ig.length && <p className="text-[11px] text-text-tertiary">No attribution available.</p>}
        </div>
      </Section>

      <Section icon={Microscope} title="Sparse-autoencoder features" subtitle={data.sae?.available ? `The model's own vocabulary: top active features out of ${data.sae.dictionary_size} (TopK=${data.sae.k}), auto-labelled from their top-activating training windows.` : data.sae?.reason ?? ''}>
        <div className="space-y-1.5">
          {(data.sae?.features ?? []).map(f => (
            <div key={f.feature_id} className="flex items-center gap-2 text-[11px]">
              <span className="font-mono w-14 text-text-tertiary">#{f.feature_id}</span>
              <span className="flex-1 truncate">{f.label}</span>
              {f.purity != null && <span className="font-mono text-[10px] text-text-tertiary">purity {Math.round(f.purity * 100)}%</span>}
              <span className="font-mono w-12 text-right">{f.activation.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </Section>

      <Section icon={Layers} title="Causal tracing (activation patching)" subtitle={`Share of the gap between this window (${Math.round((data.causal?.malicious_probability ?? 0) * 100)}%) and a benign reference (${Math.round((data.causal?.clean_probability ?? 0) * 100)}%) closed by patching the most decisive subspace of each layer.`}>
        {data.causal?.note && <p className="text-[11px] text-text-tertiary">{data.causal.note}</p>}
        <div className="space-y-1.5">
          {causal.map(l => (
            <div key={l.layer} className="flex items-center gap-2 text-[11px]">
              <span className={clsx('font-mono w-28 truncate', data.causal?.critical_layers.includes(l.layer) && 'font-semibold text-accent-indigo')}>{l.layer}</span>
              <span className="text-[9px] font-mono text-text-tertiary w-24 truncate">{l.component}</span>
              <Bar value={l.causal_importance ?? 0} color={l.component === 'predictor' ? 'bg-accent-violet' : 'bg-accent-indigo'} />
              <span className="font-mono w-10 text-right">{l.causal_importance == null ? '—' : l.causal_importance.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </Section>

      <Section icon={Brain} title="Tuned lens: watching the model think" subtitle={data.logit_lens?.narrative ?? data.logit_lens?.reason ?? ''}>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] font-mono">
            <thead>
              <tr className="text-text-tertiary text-left">
                <th className="py-1 pr-2 font-normal">Stage of computation</th>
                <th className="py-1 pr-2 font-normal">Infiltration</th>
                <th className="py-1 font-normal">Top stage</th>
              </tr>
            </thead>
            <tbody>
              {lens.map(r => (
                <tr key={r.layer} className={clsx('border-t border-border-default/60', r.layer === data.logit_lens?.decisive_layer && 'bg-accent-indigo-subtle')}>
                  <td className="py-1 pr-2">{r.label}</td>
                  <td className="py-1 pr-2">
                    {Math.round(r.infiltration_probability * 100)}%
                    {r.delta_infiltration ? <span className={clsx('ml-1', r.delta_infiltration > 0 ? 'text-risk-red' : 'text-risk-green')}>({r.delta_infiltration > 0 ? '+' : ''}{Math.round(r.delta_infiltration * 100)})</span> : null}
                  </td>
                  <td className="py-1">{r.top_stage} {Math.round(r.top_stage_probability * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {concepts && concepts.concepts && (
        <Section icon={ToggleLeft} title="Concept bottleneck (human-verifiable reasoning)" subtitle="Stage = one weighted sum over named concepts. Click a concept to override it and see how the decision changes.">
          <p className="text-[11px] text-text-primary font-medium">{concepts.explanation}</p>
          <div className="grid sm:grid-cols-2 gap-1.5">
            {concepts.concepts.map(c => (
              <button
                key={c.concept}
                type="button"
                data-interactive
                onClick={() => toggle(c.concept, c.probability)}
                title={c.description}
                className={clsx(
                  'text-left rounded-lg border px-2 py-1.5 text-[11px] flex items-center gap-2',
                  c.probability >= 0.5 ? 'border-accent-indigo bg-accent-indigo-subtle' : 'border-border-default',
                  c.concept in overrides && 'ring-2 ring-risk-amber',
                )}
              >
                <span className="flex-1 truncate">{c.concept.replace(/_/g, ' ')}</span>
                <span className="font-mono">{c.probability.toFixed(2)}</span>
              </button>
            ))}
          </div>
          {Object.keys(overrides).length > 0 && (
            <p className="text-[10px] text-risk-amber font-mono">Analyst overrides applied: {Object.entries(overrides).map(([k, v]) => `${k}=${v}`).join(', ')}</p>
          )}
        </Section>
      )}
    </div>
  );
}
