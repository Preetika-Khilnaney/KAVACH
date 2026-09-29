import { motion, AnimatePresence } from 'framer-motion';
import { X, ShieldAlert, Database, AlertTriangle, Calendar, Award } from 'lucide-react';
import type { ModelProvenance } from '../api/types';

interface ProvenanceDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  provenance: ModelProvenance | null;
}

export function ProvenanceDrawer({ isOpen, onClose, provenance }: ProvenanceDrawerProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/25 backdrop-blur-xs z-50 transition-opacity"
          />

          {/* Slide-over panel */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 28, stiffness: 240 }}
            className="fixed inset-y-0 right-0 max-w-md w-full bg-surface border-l border-border-default shadow-panel z-50 flex flex-col"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-border-default">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-accent-indigo-subtle text-accent-indigo">
                  <ShieldAlert size={18} />
                </div>
                <div>
                  <h2 className="font-heading font-semibold text-text-primary text-base">
                    Model Provenance & Limitations
                  </h2>
                  <p className="text-xs text-text-secondary">
                    No Black Box Disclosure Transparency Panel
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={onClose}
                data-interactive
                className="p-1.5 rounded-md text-text-tertiary hover:text-text-primary hover:bg-canvas transition-colors"
                aria-label="Close provenance drawer"
              >
                <X size={18} />
              </button>
            </div>

            {/* Scrollable Content */}
            <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6">
              {/* Architecture info */}
              <div className="bg-canvas rounded-lg p-4 border border-border-default space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-heading font-semibold text-text-primary">
                    Core World Model: NetJEPA
                  </span>
                  <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-surface border border-border-default text-accent-indigo font-medium">
                    {provenance?.version || 'loading…'}
                  </span>
                </div>
                <p className="text-xs text-text-secondary leading-relaxed">
                  {provenance?.architecture ?? 'A self-supervised Joint-Embedding Predictive Architecture over network flow windows.'}
                </p>
                <div className="flex items-center gap-4 pt-1 text-[11px] text-text-tertiary font-mono">
                  <span className="flex items-center gap-1">
                    <Calendar size={12} />
                    Trained: {provenance?.lastTrained ? new Date(provenance.lastTrained).toLocaleDateString() : 'untrained'}
                  </span>
                  <span className="flex items-center gap-1">
                    <Award size={12} />
                    Self-Supervised
                  </span>
                </div>
              </div>

              {/* Cryptographic identity + held-out metrics (GET /provenance) */}
              {provenance?.sha256 && (
                <div className="bg-canvas rounded-lg p-4 border border-border-default space-y-1.5 text-[11px] font-mono">
                  <div className="text-xs font-heading font-semibold text-text-primary mb-1">Model identity (chained on the ledger)</div>
                  <div className="break-all"><span className="text-text-tertiary">sha256 </span>{provenance.sha256}</div>
                  <div><span className="text-text-tertiary">parameters </span>{provenance.numParameters?.toLocaleString()} · <span className="text-text-tertiary">size </span>{((provenance.sizeBytes ?? 0) / 1e6).toFixed(2)} MB</div>
                  <div><span className="text-text-tertiary">ledger block </span>#{provenance.ledgerBlock ?? '—'} · <span className="text-text-tertiary">calibration T </span>{provenance.temperature?.toFixed(3) ?? '—'} · <span className="text-text-tertiary">threshold </span>{provenance.decisionThreshold?.toFixed(2) ?? '—'}</div>
                  {provenance.trainingManifestHash && <div className="break-all"><span className="text-text-tertiary">training data manifest </span>{provenance.trainingManifestHash.slice(0, 32)}…</div>}
                  {provenance.validationMetrics?.netjepa && (
                    <div className="pt-1">
                      <span className="text-text-tertiary">held-out </span>
                      F1 {provenance.validationMetrics.netjepa.f1?.toFixed(2)} · FPR {provenance.validationMetrics.netjepa.fpr?.toFixed(3)} · AUC {provenance.validationMetrics.netjepa.auc?.toFixed(2)}
                      {provenance.validationMetrics.baseline && <> · baseline F1 {provenance.validationMetrics.baseline.f1?.toFixed(2)}</>}
                    </div>
                  )}
                  <div className={provenance.headsValidated ? 'text-risk-green' : 'text-risk-amber'}>{provenance.headsValidated ? 'Scoring heads validated against the baseline' : 'Scoring heads NOT validated -- heuristic scoring in use'}</div>
                </div>
              )}

              {/* Training Datasets */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Database size={15} className="text-accent-teal" />
                  <h3 className="font-heading font-semibold text-sm text-text-primary">
                    Training Datasets Disclosed
                  </h3>
                </div>

                <div className="space-y-3">
                  {provenance?.datasets.map((dataset, idx) => (
                    <div
                      key={idx}
                      className="bg-surface border border-border-default rounded-lg p-3.5 space-y-1.5 shadow-xs"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-heading font-semibold text-xs text-text-primary">
                          {dataset.name} ({dataset.year})
                        </span>
                        <span className="font-mono text-[10px] text-text-tertiary">
                          {dataset.size}
                        </span>
                      </div>
                      <p className="text-xs text-text-secondary leading-relaxed font-body">
                        {dataset.description}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Known Limitations */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <AlertTriangle size={15} className="text-risk-amber" />
                  <h3 className="font-heading font-semibold text-sm text-text-primary">
                    Known Model Limitations
                  </h3>
                </div>

                <div className="bg-risk-amber-subtle/50 border border-risk-amber/20 rounded-lg p-3.5">
                  <ul className="space-y-2.5">
                    {provenance?.limitations.map((limitation, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-xs text-text-secondary leading-relaxed">
                        <span className="w-1.5 h-1.5 rounded-full bg-risk-amber mt-1.5 shrink-0" />
                        <span>{limitation}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="px-6 py-3.5 bg-canvas border-t border-border-default text-center">
              <p className="text-[11px] text-text-tertiary">
                Kavach Intrusion Forecasting · Smart India Hackathon 2026
              </p>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
