import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, LineChart, Line, CartesianGrid } from 'recharts';
import clsx from 'clsx';
import { Award, BarChart3, ShieldCheck, Gauge, Layers, Brain, Swords, Timer, AlertTriangle, Loader2 } from 'lucide-react';
import { useBenchmarkReport } from '../api/hooks';
import type { BenchmarkReport } from '../api/types';

/**
 * Four-layer validation pyramid (evaluation doc; roadmap Tasks 3.2 + 7.3).
 * Every number here comes from GET /benchmark -- nothing is hard-coded.
 */

const pct = (v: number | null | undefined, d = 1) => (v == null ? '—' : `${(v * 100).toFixed(d)}%`);
const num = (v: number | null | undefined, d = 2) => (v == null ? '—' : v.toFixed(d));
const axis = { fontSize: 10, fontFamily: 'JetBrains Mono', fill: '#5B6472' };

function Layer({ n, title, question, icon: Icon, children }: { n: number; title: string; question: string; icon: typeof Brain; children: React.ReactNode }) {
  return (
    <section className="space-y-4">
      <div className="flex items-center gap-3">
        <span className="w-7 h-7 rounded-lg bg-accent-indigo text-white font-mono text-xs flex items-center justify-center">L{n}</span>
        <Icon size={16} className="text-accent-indigo" />
        <div>
          <h2 className="font-heading font-semibold text-sm text-text-primary">{title}</h2>
          <p className="text-[11px] text-text-secondary">{question}</p>
        </div>
      </div>
      {children}
    </section>
  );
}

function Panel({ title, children, className }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={clsx('bg-surface border border-border-default rounded-2xl p-5 shadow-card space-y-3', className)}>
      <h3 className="font-heading font-semibold text-xs text-text-primary">{title}</h3>
      {children}
    </div>
  );
}

function Stat({ label, value, sub, tone = 'indigo' }: { label: string; value: string; sub?: string; tone?: 'indigo' | 'teal' | 'green' | 'amber' | 'red' }) {
  const color = { indigo: 'text-accent-indigo', teal: 'text-accent-teal', green: 'text-risk-green', amber: 'text-risk-amber', red: 'text-risk-red' }[tone];
  return (
    <div className="bg-surface border border-border-default rounded-xl p-4 shadow-card">
      <span className="text-[11px] font-mono text-text-tertiary uppercase block">{label}</span>
      <div className={clsx('font-heading font-bold text-2xl mt-1', color)}>{value}</div>
      {sub && <div className="text-[11px] font-mono text-text-secondary mt-0.5">{sub}</div>}
    </div>
  );
}

function Output({ r }: { r: BenchmarkReport }) {
  const nj = r.netjepa!;
  const bl = r.baseline!;
  const delta = nj.f1 != null && bl.f1 ? ((nj.f1 - bl.f1) / bl.f1) * 100 : null;
  const chart = [bl, ...(r.temporal_baseline ? [r.temporal_baseline] : []), nj].map(b => ({ model: b.model.replace(' (baseline)', '').replace('Logistic Regression', 'LogReg'), F1: +(100 * (b.f1 ?? 0)).toFixed(1), Precision: +(100 * (b.precision ?? 0)).toFixed(1), Recall: +(100 * (b.recall ?? 0)).toFixed(1), FPR: +(100 * (b.fpr ?? 0)).toFixed(1), AUC: +(100 * (b.auc ?? 0)).toFixed(1) }));
  const rel = (r.calibration?.reliability_calibrated ?? []).map((b, i) => ({
    bin: b.bin,
    ideal: (i + 0.5) * 10,
    calibrated: b.observed_frequency == null ? null : +(b.observed_frequency * 100).toFixed(1),
    raw: r.calibration?.reliability_raw[i]?.observed_frequency == null ? null : +(r.calibration!.reliability_raw[i].observed_frequency! * 100).toFixed(1),
  }));
  const pred = Object.entries(r.predictive?.auc ?? {}).map(([k, v]) => ({ horizon: k, AUC: v == null ? null : +(v * 100).toFixed(1) }));
  return (
    <>
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <Stat label="NetJEPA F1" value={num(nj.f1)} sub={delta != null ? `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}% vs baseline (${num(bl.f1)})` : undefined} tone={delta != null && delta >= 0 ? 'indigo' : 'amber'} />
        <Stat label="Precision" value={num(nj.precision)} sub={`${nj.tp ?? 0} TP / ${nj.fp ?? 0} FP`} tone="teal" />
        <Stat label="Recall" value={num(nj.recall)} sub={`${nj.fn ?? 0} attacks missed`} tone="teal" />
        <Stat label="False positive rate" value={pct(nj.fpr)} sub={`baseline ${pct(bl.fpr)}`} tone={(nj.fpr ?? 1) < (bl.fpr ?? 0) ? 'green' : 'amber'} />
        <Stat label="ROC AUC" value={num(nj.auc)} sub={`baseline ${num(bl.auc)}`} />
      </div>
      {nj.note && <div className="p-3 rounded-xl border border-risk-amber/40 bg-risk-amber-subtle text-xs text-text-secondary">{nj.note}</div>}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <Panel title="Binary infiltration detection (%) — same held-out windows" className="lg:col-span-7">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chart} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                <XAxis dataKey="model" tick={axis} axisLine={{ stroke: '#E4E9EF' }} tickLine={false} />
                <YAxis domain={[0, 100]} tick={axis} axisLine={{ stroke: '#E4E9EF' }} tickLine={false} />
                <Tooltip contentStyle={{ fontSize: 11, fontFamily: 'JetBrains Mono', borderRadius: 8 }} formatter={v => [`${v}%`]} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="F1" fill="#4F46E5" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Precision" fill="#0EA5A0" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Recall" fill="#7C3AED" radius={[4, 4, 0, 0]} />
                <Bar dataKey="FPR" fill="#D97706" radius={[4, 4, 0, 0]} />
                <Bar dataKey="AUC" fill="#8B95A5" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-text-secondary">
            {r.val_size.toLocaleString()} validation windows (last 20% of every CIC-IDS-2017 day, never trained on). Decision threshold {num(nj.decision_threshold)} on the calibrated probability, chosen on a separate calibration split.
          </p>
        </Panel>
        <Panel title={`Calibration — ECE ${num(r.calibration?.ece_raw, 3)} → ${num(r.calibration?.ece_calibrated, 3)} (T = ${num(r.calibration?.temperature, 3)})`} className="lg:col-span-5">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rel} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                <CartesianGrid stroke="#E4E9EF" strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={{ ...axis, fontSize: 8 }} />
                <YAxis domain={[0, 100]} tick={axis} />
                <Tooltip contentStyle={{ fontSize: 11, fontFamily: 'JetBrains Mono', borderRadius: 8 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line dataKey="ideal" name="perfect" stroke="#8B95A5" strokeDasharray="4 4" dot={false} />
                <Line dataKey="raw" name="raw head" stroke="#D97706" connectNulls />
                <Line dataKey="calibrated" name="temperature-scaled" stroke="#4F46E5" strokeWidth={2} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-text-secondary">Observed attack frequency per predicted-probability bin. On the diagonal, "82%" really means ~82%.</p>
        </Panel>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <Panel title="Forecasting: AUC for 'attack present k windows later'" className="lg:col-span-5">
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={pred} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
                <CartesianGrid stroke="#E4E9EF" strokeDasharray="3 3" />
                <XAxis dataKey="horizon" tick={axis} />
                <YAxis domain={[50, 100]} tick={axis} />
                <Tooltip contentStyle={{ fontSize: 11, fontFamily: 'JetBrains Mono', borderRadius: 8 }} />
                <Line dataKey="AUC" stroke="#7C3AED" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-text-secondary">{r.predictive?.note}</p>
        </Panel>
        {r.multiclass && (
          <Panel title={`MITRE stage head — macro-F1 ${num(r.multiclass.macro_f1)}, accuracy ${pct(r.multiclass.accuracy)}`} className="lg:col-span-7">
            <div className="overflow-x-auto">
              <table className="text-[10px] font-mono w-full">
                <thead>
                  <tr className="text-text-tertiary">
                    <th className="text-left font-normal pr-2">true \ predicted</th>
                    {r.multiclass.labels.map(l => <th key={l} className="font-normal px-1">{l.split(' ').map(w => w[0]).join('')}</th>)}
                    <th className="font-normal px-1">F1</th>
                  </tr>
                </thead>
                <tbody>
                  {r.multiclass.confusion_matrix.map((row, i) => {
                    const total = row.reduce((a, b) => a + b, 0) || 1;
                    return (
                      <tr key={i} className="border-t border-border-default/60">
                        <td className="pr-2 py-0.5">{r.multiclass!.labels[i]} <span className="text-text-tertiary">({r.multiclass!.per_class[i].support})</span></td>
                        {row.map((v, j) => (
                          <td key={j} className="text-center px-1" style={{ background: v ? `rgba(79,70,229,${(0.08 + (0.8 * v) / total).toFixed(2)})` : undefined, color: v / total > 0.5 ? 'white' : undefined }}>{v}</td>
                        ))}
                        <td className="text-center px-1">{r.multiclass!.per_class[i].support ? num(r.multiclass!.per_class[i].f1) : '—'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="text-[11px] text-text-secondary">{r.multiclass.note}</p>
          </Panel>
        )}
      </div>
      <div className="p-3 rounded-xl border border-border-default bg-canvas text-[11px] text-text-secondary">
        Cross-dataset (zero-day) test: {r.cross_dataset?.note}
      </div>
    </>
  );
}

function M({ v, d = 2 }: { v: number | null | undefined; d?: number }) {
  return <>{v == null ? '—' : v.toFixed(d)}</>;
}

function Dynamics({ r }: { r: BenchmarkReport }) {
  const tb = r.temporal_baseline;
  const ctx = r.generalization?.context ?? {};
  const fc = r.forecast ?? {};
  const tm = r.transition_model;
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Panel title={`Does temporal context help? (NetJEPA sees ${r.context_len ?? 1} windows)`}>
        <table className="w-full text-[11px] font-mono">
          <thead><tr className="text-text-tertiary text-left"><th className="font-normal">model</th><th className="font-normal">F1</th><th className="font-normal">AUC</th><th className="font-normal">FPR</th></tr></thead>
          <tbody>
            {[r.baseline, tb, r.netjepa].filter(Boolean).map(b => (
              <tr key={b!.model} className={clsx('border-t border-border-default/60', b!.model.startsWith('NetJEPA') && 'text-accent-indigo font-semibold')}>
                <td className="py-1">{b!.model}</td><td><M v={b!.f1} /></td><td><M v={b!.auc} /></td><td>{pct(b!.fpr)}</td>
              </tr>
            ))}
            {Object.values(ctx).sort((a, b) => a.context_len - b.context_len).map(c => (
              <tr key={c.context_len} className="border-t border-border-default/60 text-text-secondary">
                <td className="py-1">NetJEPA retrained, context = {c.context_len} ({c.epochs} ep.)</td><td><M v={c.detection.f1} /></td><td><M v={c.detection.auc} /></td><td>{pct(c.detection.fpr)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="text-[11px] text-text-secondary">The temporal logistic regression sees the same {r.context_len ?? 1} windows NetJEPA does, so any gap is the value of learned dynamics, not of extra input.</p>
      </Panel>
      <Panel title="Forward simulation: predicting attack k windows ahead">
        <table className="w-full text-[11px] font-mono">
          <thead><tr className="text-text-tertiary text-left"><th className="font-normal">horizon</th><th className="font-normal">forecast head AUC</th><th className="font-normal">current score AUC</th><th className="font-normal">onset AUC (benign now)</th></tr></thead>
          <tbody>
            {Object.entries(fc).map(([k, v]) => (
              <tr key={k} className="border-t border-border-default/60">
                <td className="py-1">{k}</td><td className="text-accent-indigo font-semibold"><M v={v.auc_forecast_head} /></td><td><M v={v.auc_current_score} /></td>
                <td><M v={v.auc_onset_forecast_head} /> {v.onset_positives != null && <span className="text-text-tertiary">({v.onset_positives})</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {tm && (
          <p className="text-[11px] text-text-secondary">
            Transition model p(S<sub>t+1</sub>|S<sub>t</sub>): Gaussian NLL {tm.mean_gaussian_nll_per_dim.toFixed(3)} per dim; the predicted 90% interval covers the true next state {pct(tm.coverage_of_90pct_interval, 0)} of the time.
          </p>
        )}
        <p className="text-[11px] text-text-secondary">"Onset" restricts to windows that are benign now: can the model see an attack coming before it starts?</p>
      </Panel>
    </div>
  );
}

function Levels({ r }: { r: BenchmarkReport }) {
  const ab = r.modality_ablation ?? {};
  const bySrc = r.detection_by_source?.netjepa ?? {};
  const loao = r.generalization?.loao ?? [];
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Panel title="Flow level + packet level (joint PCAP windows)">
        {Object.keys(ab).length ? (
          <table className="w-full text-[11px] font-mono">
            <thead><tr className="text-text-tertiary text-left"><th className="font-normal">input</th><th className="font-normal">F1</th><th className="font-normal">AUC</th><th className="font-normal">recall</th><th className="font-normal">FPR</th></tr></thead>
            <tbody>
              {Object.entries(ab).map(([k, v]) => (
                <tr key={k} className={clsx('border-t border-border-default/60', k.startsWith('flow +') && 'text-accent-indigo font-semibold')}>
                  <td className="py-1">{k}</td><td><M v={v.f1} /></td><td><M v={v.auc} /></td><td><M v={v.recall} /></td><td>{pct(v.fpr)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p className="text-[11px] text-text-tertiary">No joint windows in the validation split.</p>}
        <div className="text-[11px] font-mono text-text-secondary">
          {Object.entries(bySrc).map(([src, m]) => <div key={src}>{src === 'pcap' ? 'PCAP (both levels)' : 'Flow CSV (flow level)'}: F1 <M v={m?.f1} /> · AUC <M v={m?.auc} /> · n={m?.n}</div>)}
        </div>
        <p className="text-[11px] text-text-secondary">Same model, same validation windows, with one traffic level hidden at inference -- measures what each level contributes.</p>
      </Panel>
      <Panel title="Generalisation to unseen attack families (leave-one-out retraining)">
        {loao.length ? (
          <table className="w-full text-[11px] font-mono">
            <thead><tr className="text-text-tertiary text-left"><th className="font-normal">held-out family</th><th className="font-normal">NetJEPA AUC</th><th className="font-normal">recall</th><th className="font-normal">FPR</th><th className="font-normal">LR AUC</th><th className="font-normal">LR recall</th><th className="font-normal">LR FPR</th></tr></thead>
            <tbody>
              {loao.map(f => (
                <tr key={f.family} className="border-t border-border-default/60">
                  <td className="py-1">{f.family}{f.held_out?.held_out_windows != null && <span className="text-text-tertiary"> ({f.held_out.held_out_windows})</span>}</td>
                  <td className="text-accent-indigo font-semibold"><M v={f.held_out?.netjepa?.auc_vs_benign} /></td>
                  <td>{pct(f.held_out?.netjepa?.recall_at_threshold, 0)}</td>
                  <td>{pct(f.held_out?.netjepa?.fpr_on_benign, 0)}</td>
                  <td><M v={f.held_out?.logreg?.auc_vs_benign} /></td>
                  <td>{pct(f.held_out?.logreg?.recall_at_threshold, 0)}</td>
                  <td>{pct(f.held_out?.logreg?.fpr_on_benign, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p className="text-[11px] text-text-tertiary">Run python -m src.benchmark.generalization to produce these.</p>}
        <p className="text-[11px] text-text-secondary">Each row retrains from scratch with every window of that family removed from training; scores are the family's windows vs benign validation windows. Neither model has ever seen the family.
          {' '}Finding: NetJEPA transfers to DDoS (it has seen the DoS family) but at its deployed threshold it misses most windows of families unlike anything in training (PortScan, brute force, web, botnet); the baseline catches more of them only by flagging 19-30% of benign windows. Latent-space novelty scoring did not close this gap (reported in the JSON as netjepa_novelty).</p>
      </Panel>
    </div>
  );
}

function Representation({ r }: { r: BenchmarkReport }) {
  const probes = r.interpretability?.probes;
  const ec = r.explanation_consistency;
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      <Panel title="Linear probes: what each layer knows (balanced accuracy)" className="lg:col-span-7">
        {probes ? (
          <div className="overflow-x-auto">
            <table className="text-[11px] font-mono w-full">
              <thead>
                <tr className="text-text-tertiary">
                  <th className="text-left font-normal">concept</th>
                  {probes.layers.map(l => <th key={l} className="font-normal px-1">{l}</th>)}
                </tr>
              </thead>
              <tbody>
                {probes.concepts.map(c => (
                  <tr key={c.concept} className="border-t border-border-default/60">
                    <td className="py-1 pr-2">{c.concept}</td>
                    {probes.layers.map(l => {
                      const v = c.layers[l]?.balanced_accuracy;
                      return (
                        <td key={l} className={clsx('text-center px-1', l === c.best_layer && 'font-bold')} style={{ background: v != null ? `rgba(14,165,160,${Math.max(0, (v - 0.4) * 1.4).toFixed(2)})` : undefined }}>
                          {v != null ? v.toFixed(2) : '—'}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-[11px] text-text-tertiary">{r.interpretability?.note ?? 'Run python -m src.training.train_interpretability.'}</p>
        )}
        <p className="text-[11px] text-text-secondary">Kill-chain stage and escalation become linearly decodable only after the GRU — the temporal encoder is adding them, as the architecture intends.</p>
      </Panel>
      <div className="lg:col-span-5 space-y-6">
        <Panel title="Embedding health">
          <div className="flex items-baseline gap-2">
            <span className="font-heading font-bold text-2xl text-accent-teal">{num(r.representation?.embed_std, 3)}</span>
            <span className="text-[11px] font-mono text-text-secondary">embed_std · {r.representation?.collapse_check}</span>
          </div>
        </Panel>
        <Panel title={`Explanation consistency — mean Jaccard ${num(ec?.overall_mean_jaccard)}`}>
          {(ec?.per_attack ?? []).map(a => (
            <div key={a.attack_label} className="flex items-center gap-2 text-[11px] font-mono">
              <span className="w-28 truncate">{a.attack_label}</span>
              <div className="flex-1 h-1.5 bg-canvas rounded-full overflow-hidden"><div className="h-full bg-accent-indigo" style={{ width: `${(a.mean_jaccard ?? 0) * 100}%` }} /></div>
              <span className="w-10 text-right">{num(a.mean_jaccard)}</span>
            </div>
          ))}
          <p className="text-[11px] text-text-secondary">{ec?.note}</p>
        </Panel>
      </div>
    </div>
  );
}

function Mechanistic({ r }: { r: BenchmarkReport }) {
  const causal = r.causal_by_stage?.mean_causal_importance ?? {};
  const layers = Object.keys(Object.values(causal)[0] ?? {});
  const sae = r.interpretability?.sae;
  const lens = r.interpretability?.tuned_lens?.layers ?? [];
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      <Panel title="Causal tracing by attack stage (subspace activation patching)" className="lg:col-span-7">
        {layers.length ? (
          <div className="overflow-x-auto">
            <table className="text-[11px] font-mono w-full">
              <thead>
                <tr className="text-text-tertiary"><th className="text-left font-normal">stage</th>{layers.map(l => <th key={l} className="font-normal px-1">{l}</th>)}</tr>
              </thead>
              <tbody>
                {Object.entries(causal).map(([stage, row]) => (
                  <tr key={stage} className="border-t border-border-default/60">
                    <td className="py-1 pr-2">{stage}</td>
                    {layers.map(l => <td key={l} className="text-center px-1" style={{ background: `rgba(79,70,229,${((row[l] ?? 0) * 0.7).toFixed(2)})`, color: (row[l] ?? 0) > 0.6 ? 'white' : undefined }}>{num(row[l])}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-[11px] text-text-tertiary">Not computed.</p>
        )}
        <p className="text-[11px] text-text-secondary">{r.causal_by_stage?.note}</p>
      </Panel>
      <div className="lg:col-span-5 space-y-6">
        <Panel title="Sparse autoencoder dictionary">
          {sae ? (
            <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
              <div>alive features <b>{sae.alive_features}</b> / {sae.hidden_dim}</div>
              <div>active per window <b>{num(sae.avg_active_features, 1)}</b></div>
              <div>variance unexplained <b>{pct(sae.val_fraction_variance_unexplained)}</b></div>
              <div>attack-pure features <b>{sae.attack_specific_features}</b></div>
            </div>
          ) : <p className="text-[11px] text-text-tertiary">Not trained.</p>}
        </Panel>
        <Panel title="Tuned lens: infiltration AUC by layer">
          {lens.map(l => (
            <div key={l.layer} className="flex items-center gap-2 text-[11px] font-mono">
              <span className="w-28">{l.layer}</span>
              <div className="flex-1 h-1.5 bg-canvas rounded-full overflow-hidden"><div className="h-full bg-accent-violet" style={{ width: `${Math.max(0, ((l.infiltration_auc ?? 0.5) - 0.5) * 200)}%` }} /></div>
              <span className="w-10 text-right">{num(l.infiltration_auc)}</span>
            </div>
          ))}
        </Panel>
      </div>
    </div>
  );
}

function Trust({ r }: { r: BenchmarkReport }) {
  const rb = r.robustness;
  const cb = r.interpretability?.concept_bottleneck;
  const fragile = (rb?.mean_min_flip_eps ?? 99) < 0.5;
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Panel title="Adversarial robustness (PGD, feature space)">
        {rb?.available ? (
          <>
            <div className="grid grid-cols-3 gap-2 text-[11px] font-mono">
              <div>mean flip ε <b>{num(rb.mean_min_flip_eps, 3)}σ</b></div>
              <div>features to flip <b>{num(rb.mean_min_features_to_flip, 1)}</b></div>
              <div>robust acc @0.3σ <b>{pct(rb.robust_accuracy?.['0.3'], 0)}</b></div>
            </div>
            {fragile && (
              <div className="flex gap-2 rounded-lg border border-risk-amber/40 bg-risk-amber-subtle p-2.5 text-[11px] text-text-secondary">
                <AlertTriangle size={14} className="text-risk-amber shrink-0" />
                <span>Honest finding: this checkpoint is not adversarially robust — small perturbations of volume/timing features flip detections. Next step: noise/adversarial augmentation when retraining the encoder.</span>
              </div>
            )}
            <p className="text-[11px] text-text-secondary">{rb.summary}</p>
          </>
        ) : <p className="text-[11px] text-text-tertiary">Not computed.</p>}
      </Panel>
      <Panel title="Concept bottleneck (transparent pathway)">
        {cb ? (
          <>
            <div className="text-[11px] font-mono">stage macro-F1 <b>{num(cb.stage_macro_f1)}</b> · accuracy <b>{pct(cb.stage_accuracy)}</b></div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[11px] font-mono">
              {Object.entries(cb.concept_auc).map(([k, v]) => (
                <div key={k} className="flex justify-between"><span className="truncate mr-2">{k}</span><span>{v == null ? 'n/a' : `AUC ${v.toFixed(2)}`}</span></div>
              ))}
            </div>
            <p className="text-[11px] text-text-secondary">Concepts are predicted from the embedding with high AUC; the stage decision is a single weighted sum over them, so analysts can inspect and override each one (Flow detail → Deep Explain). It trades accuracy for transparency versus the standard stage head.</p>
          </>
        ) : <p className="text-[11px] text-text-tertiary">Not trained.</p>}
      </Panel>
    </div>
  );
}

export function Benchmark() {
  const { data: report, loading, error } = useBenchmarkReport();

  return (
    <div className="space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-heading font-bold text-text-primary tracking-tight">Validation Pyramid</h1>
          <p className="text-xs text-text-secondary mt-0.5">
            Outputs → representations → mechanisms → trust. Every figure is computed live from the served checkpoint on held-out CIC-IDS-2017 windows.
          </p>
        </div>
        {report?.computed_in_seconds != null && <span className="text-[11px] font-mono text-text-tertiary flex items-center gap-1"><Timer size={12} /> computed in {report.computed_in_seconds}s (cached per model hash)</span>}
      </div>

      {loading && (
        <div className="flex items-center gap-2 p-6 bg-surface border border-border-default rounded-xl text-xs text-text-secondary">
          <Loader2 size={16} className="animate-spin" /> Running the benchmark suite (first run can take a couple of minutes: baseline fit, full validation pass, IG consistency, causal tracing, PGD)…
        </div>
      )}
      {error && <div className="p-4 rounded-xl border border-risk-red/40 bg-risk-red-subtle text-xs text-risk-red">{error.message}</div>}
      {report && report.status !== 'ok' && (
        <div className="p-4 rounded-xl border border-risk-amber/40 bg-risk-amber-subtle text-xs text-text-secondary">{report.message}</div>
      )}

      {report?.status === 'ok' && report.baseline && report.netjepa && (
        <>
          <Layer n={1} icon={Gauge} title="Output — do the outputs match reality?" question="Detection, calibration, forecasting and stage mapping vs a context-free baseline.">
            <Output r={report} />
          </Layer>
          {report.model_version === 2 && (
            <>
              <Layer n={1} icon={Timer} title="World-model dynamics — temporal context & forward simulation" question="Is this more than a static classifier? History, K-step forecasts and the probabilistic transition model.">
                <Dynamics r={report} />
              </Layer>
              <Layer n={1} icon={ShieldCheck} title="Both traffic levels & unseen attacks" question="What each traffic level contributes, and detection of attack families never seen in training.">
                <Levels r={report} />
              </Layer>
            </>
          )}
          <Layer n={2} icon={Layers} title="Representation — has the model learned the right things?" question="Embedding health, layer-by-layer probes, stable explanations.">
            <Representation r={report} />
          </Layer>
          <Layer n={3} icon={Brain} title="Mechanistic — how does it compute its decisions?" question="Causal tracing, sparse-autoencoder features, tuned lens.">
            <Mechanistic r={report} />
          </Layer>
          <Layer n={4} icon={Swords} title="Trust — can we rely on it?" question="Adversarial robustness and a human-verifiable concept pathway.">
            <Trust r={report} />
          </Layer>
          <div className="p-3.5 bg-canvas rounded-xl border border-border-default text-xs text-text-secondary flex gap-2">
            <ShieldCheck size={14} className="text-accent-teal shrink-0" />
            <span>
              Transparency: numbers that look weak (stage macro-F1, adversarial robustness) are shown as measured. <Award size={11} className="inline" /> <BarChart3 size={11} className="inline" /> The model hash these results belong to is in the Provenance drawer and on the forensic ledger.
            </span>
          </div>
        </>
      )}
    </div>
  );
}
