import { useState, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Search, ArrowUpDown, RefreshCw, Eye } from 'lucide-react';
import { useFlows } from '../api/hooks';
import type { Flow, RiskLevel } from '../api/types';
import { RiskBadge } from '../components/RiskBadge';
import { FlowDetail } from './FlowDetail';
import { DataStateWrapper } from '../components/DataStateWrapper';
import { SkeletonTableRow } from '../components/Skeleton';
import clsx from 'clsx';

export function Flows() {
  const [searchParams] = useSearchParams();
  const initialFlowId = searchParams.get('id');

  const [protocolFilter, setProtocolFilter] = useState<string>('ALL');
  const [riskFilter, setRiskFilter] = useState<RiskLevel | 'ALL'>('ALL');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<keyof Flow>('riskScore');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [selectedFlow, setSelectedFlow] = useState<Flow | null>(null);

  const { data: flows, loading, error, refetch } = useFlows();
  // Real rows only -- no decorative ports/flags/bytes (CSV windows carry none).
  const displayFlows = useMemo(() => flows || [], [flows]);

  // If URL has flowId, select it once data loads
  useMemo(() => {
    if (initialFlowId && flows && !selectedFlow) {
      const match = flows.find(f => f.id === initialFlowId);
      if (match) setSelectedFlow(match);
    }
  }, [initialFlowId, flows, selectedFlow]);

  // Filtering & Sorting
  const filteredFlows = useMemo(() => {
    if (!displayFlows.length) return [];
    return displayFlows
      .filter(f => {
        if (protocolFilter !== 'ALL' && f.protocol !== protocolFilter) return false;
        if (riskFilter !== 'ALL' && f.riskLevel !== riskFilter) return false;
        if (searchTerm) {
          const q = searchTerm.toLowerCase();
          const matches =
            f.srcIP.includes(q) ||
            f.dstIP.includes(q) ||
            f.id.toLowerCase().includes(q) ||
            (f.sourceFile ?? '').toLowerCase().includes(q) ||
            (f.attackStage ?? '').toLowerCase().includes(q) ||
            (f.hostIPs ?? []).some(ip => ip.includes(q));
          if (!matches) return false;
        }
        return true;
      })
      .sort((a, b) => {
        const valA = a[sortBy];
        const valB = b[sortBy];
        const multiplier = sortDir === 'asc' ? 1 : -1;
        if (typeof valA === 'number' && typeof valB === 'number') {
          return (valA - valB) * multiplier;
        }
        return String(valA).localeCompare(String(valB)) * multiplier;
      });
  }, [displayFlows, protocolFilter, riskFilter, searchTerm, sortBy, sortDir]);

  const handleSort = (field: keyof Flow) => {
    if (sortBy === field) {
      setSortDir(prev => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortBy(field);
      setSortDir('desc');
    }
  };

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-heading font-bold text-text-primary tracking-tight">
            Processed Prediction Windows
          </h1>
          <p className="text-xs text-text-secondary mt-0.5">
            Stored feature windows from the ingestion pipeline with NetJEPA risk predictions and explainability data.
          </p>
        </div>

        <button
          type="button"
          onClick={() => refetch()}
          data-interactive
          className="self-start sm:self-auto flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border-default bg-surface hover:bg-canvas text-xs font-medium text-text-secondary transition-colors"
        >
          <RefreshCw size={13} />
          <span>Refresh Flows</span>
        </button>
      </div>

      {/* Filter Toolbar */}
      <div className="bg-surface border border-border-default rounded-xl p-4 shadow-card flex flex-wrap items-center justify-between gap-3">
        {/* Search */}
        <div className="relative flex-1 min-w-[240px]">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary" />
          <input
            type="text"
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            placeholder="Search by IP, capture, window id, stage..."
            className="w-full pl-9 pr-3 py-1.5 bg-canvas border border-border-default rounded-lg text-xs font-mono text-text-primary focus:outline-none focus:border-accent-indigo"
          />
        </div>

        {/* Protocol Filter */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-secondary font-medium">Protocol:</span>
          <select
            value={protocolFilter}
            onChange={e => setProtocolFilter(e.target.value)}
            className="bg-canvas border border-border-default rounded-lg px-2.5 py-1.5 text-xs font-mono text-text-primary focus:outline-none focus:border-accent-indigo"
          >
            <option value="ALL">All Protocols</option>
            <option value="TCP">TCP</option>
            <option value="UDP">UDP</option>
            <option value="TLS">TLS</option>
            <option value="HTTP">HTTP</option>
            <option value="DNS">DNS</option>
            <option value="ICMP">ICMP</option>
          </select>
        </div>

        {/* Risk Filter */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-secondary font-medium">Risk:</span>
          <select
            value={riskFilter}
            onChange={e => setRiskFilter(e.target.value as RiskLevel | 'ALL')}
            className="bg-canvas border border-border-default rounded-lg px-2.5 py-1.5 text-xs font-mono text-text-primary focus:outline-none focus:border-accent-indigo"
          >
            <option value="ALL">All Risks</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>

        <span className="font-mono text-xs text-text-tertiary">
          Showing {filteredFlows.length} windows
        </span>
      </div>

      {/* Flows Table */}
      <div className="bg-surface border border-border-default rounded-xl shadow-card overflow-hidden">
        <DataStateWrapper
          state={loading ? 'loading' : error ? 'error' : filteredFlows.length === 0 ? 'empty' : 'stored'}
          emptyMessage="No processed windows match your filter criteria."
          skeleton={
            <div className="divide-y divide-border-default">
              {Array.from({ length: 6 }).map((_, i) => (
                <SkeletonTableRow key={i} cols={7} />
              ))}
            </div>
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-canvas/80 border-b border-border-default text-text-secondary font-mono text-[11px]">
                  <th
                    className="py-3 px-4 cursor-pointer hover:text-text-primary"
                    onClick={() => handleSort('timestamp')}
                  >
                    <div className="flex items-center gap-1">
                      <span>Time</span>
                      <ArrowUpDown size={11} />
                    </div>
                  </th>
                  <th className="py-3 px-4">Source Host</th>
                  <th className="py-3 px-4">Destination Host</th>
                  <th className="py-3 px-4">Capture</th>
                  <th className="py-3 px-4">MITRE Stage</th>
                  <th className="py-3 px-4">Scoring</th>
                  <th className="py-3 px-4">Records</th>
                  <th
                    className="py-3 px-4 cursor-pointer hover:text-text-primary"
                    onClick={() => handleSort('riskScore')}
                  >
                    <div className="flex items-center gap-1">
                      <span>Forecast Risk</span>
                      <ArrowUpDown size={11} />
                    </div>
                  </th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-default/60">
                {filteredFlows.map(flow => (
                  <tr
                    key={flow.id}
                    onClick={() => setSelectedFlow(flow)}
                    data-interactive
                    className={clsx(
                      'hover:bg-canvas transition-colors cursor-pointer group',
                      selectedFlow?.id === flow.id && 'bg-accent-indigo-subtle/50'
                    )}
                  >
                    <td className="py-3 px-4 font-mono text-text-tertiary whitespace-nowrap">
                      {new Date(flow.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-3 px-4 font-mono text-text-primary whitespace-nowrap">
                      {flow.srcIP}
                    </td>
                    <td className="py-3 px-4 font-mono text-text-primary whitespace-nowrap">
                      {flow.dstIP}
                    </td>
                    <td className="py-3 px-4 font-mono text-text-secondary whitespace-nowrap max-w-[180px] truncate" title={flow.sourceFile ?? ''}>
                      {flow.sourceFile ?? '—'}
                    </td>
                    <td className="py-3 px-4 whitespace-nowrap">
                      <span className="px-1.5 py-0.5 rounded bg-canvas border border-border-default text-text-secondary font-medium">
                        {flow.attackStage ?? '—'}
                      </span>
                      {flow.analystVerdict && <span className="ml-1 text-[10px] font-mono text-accent-teal">{flow.analystVerdict}</span>}
                    </td>
                    <td className="py-3 px-4 font-mono text-text-secondary whitespace-nowrap">
                      {flow.scoringMethod === 'netjepa' ? 'NetJEPA (calibrated)' : flow.scoringMethod === 'packet_heuristic' ? 'packet heuristic' : 'legacy'}
                    </td>
                    <td className="py-3 px-4 font-mono text-text-secondary whitespace-nowrap">
                      {flow.flowCount ? `${flow.flowCount} flows` : flow.packetCount ? `${flow.packetCount} pkts` : '—'}
                    </td>
                    <td className="py-3 px-4 whitespace-nowrap">
                      <RiskBadge level={flow.riskLevel} score={flow.riskScore} size="sm" />
                    </td>
                    <td className="py-3 px-4 text-right whitespace-nowrap">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedFlow(flow);
                        }}
                        className="p-1.5 rounded-md text-accent-indigo hover:bg-accent-indigo-subtle transition-colors"
                        title="View Window Detail & SHAP Attribution"
                      >
                        <Eye size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </DataStateWrapper>
      </div>

      {/* Flow Detail Slide-over */}
      {selectedFlow && (
        <FlowDetail
          flow={selectedFlow}
          onClose={() => setSelectedFlow(null)}
          onVerdictChange={() => refetch()}
        />
      )}
    </div>
  );
}
