import type { AttentionFeature } from '../api/websocket';

interface AttentionHeatmapProps {
  features: AttentionFeature[];
}

/** Grid of every real canonical feature's IntegratedGradients attribution
 * for the current window, colored by direction (red = pushes infiltration
 * risk up, green = pushes it down) and intensity = magnitude relative to
 * the strongest feature in THIS window (attribution scale isn't absolute
 * across windows, so normalizing per-window is the honest choice here). */
export function AttentionHeatmap({ features }: AttentionHeatmapProps) {
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.attribution)), 1e-9);

  return (
    <div
      className="grid gap-2"
      style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(148px, 1fr))' }}
      role="img"
      aria-label="Heatmap of real feature attributions for the current window"
    >
      {features.map((f) => {
        const magnitude = Math.abs(f.attribution) / maxAbs; // 0-1, relative to this window
        const positive = f.attribution >= 0;
        const color = positive ? '#DC2626' : '#16A34A';
        const bgAlpha = 0.08 + magnitude * 0.42; // always at least faintly visible
        return (
          <div
            key={f.feature}
            className="rounded-lg border p-2.5 flex flex-col gap-1 transition-transform hover:scale-[1.03]"
            style={{
              borderColor: `${color}55`,
              background: `${color}${Math.round(bgAlpha * 255).toString(16).padStart(2, '0')}`,
            }}
            title={`${f.feature}: ${f.attribution >= 0 ? '+' : ''}${f.attribution.toFixed(4)} (${positive ? 'pushes risk up' : 'pushes risk down'})`}
          >
            <span className="font-mono text-[10px] leading-snug text-text-primary break-all">{f.feature}</span>
            <span className="font-mono text-[11px] font-semibold" style={{ color }}>
              {f.attribution >= 0 ? '+' : ''}{f.attribution.toFixed(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
