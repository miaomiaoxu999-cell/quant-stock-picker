"use client";

import { useState, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import PageHeader from "@/components/layout/PageHeader";
import LoadingState from "@/components/shared/LoadingState";
import DataTable, { type Column } from "@/components/shared/DataTable";
import SliderInput from "@/components/shared/SliderInput";
import MetricCard from "@/components/shared/MetricCard";
import ProgressTracker from "@/components/shared/ProgressTracker";
import PBComparisonChart from "@/components/charts/PBComparisonChart";
import CombinedStockChart from "@/components/charts/CombinedStockChart";
import CorrelationChart from "@/components/charts/CorrelationChart";
import {
  getCrossSectorSummary,
  getSectorStockAnalysis,
  getWatchlist,
  addToWatchlist,
  removeFromWatchlist,
  getLegacyProfiles,
  getStockChartData,
  stockAnalyzeURL,
  stockRedoURL,
} from "@/lib/api";
import { createSSEConnection } from "@/lib/sse";
import type { StockRanking, WatchlistItem, SectorStockAnalysis } from "@/lib/types";

// ============ Watchlist Sidebar ============

function WatchlistPanel({
  items,
  onRemove,
  collapsed,
  onToggle,
}: {
  items: WatchlistItem[];
  onRemove: (code: string) => void;
  collapsed: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="rounded-lg border border-border bg-bg-surface">
      <button
        onClick={onToggle}
        className="w-full text-left px-4 py-2 flex items-center justify-between hover:bg-bg-hover transition-colors"
      >
        <span className="text-sm font-semibold text-text-primary">
          Watchlist ({items.length})
        </span>
        <svg
          className={`w-4 h-4 text-text-secondary transition-transform ${collapsed ? "" : "rotate-180"}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {!collapsed && (
        <div className="px-4 pb-3 space-y-1">
          {items.length === 0 ? (
            <p className="text-xs text-text-secondary py-2">No stocks in watchlist</p>
          ) : (
            items.map((item) => (
              <div key={item.code} className="flex items-center justify-between py-1">
                <div>
                  <span className="text-xs text-text-primary">{item.name}</span>
                  <span className="text-xs text-text-secondary ml-1">({item.code})</span>
                  <span className="text-xs text-text-secondary ml-1">{item.sector}</span>
                </div>
                <button
                  onClick={() => onRemove(item.code)}
                  className="text-xs text-red hover:text-red/80 transition-colors"
                >
                  Remove
                </button>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

// ============ Tab 1: Cross-Sector Summary ============

function CrossSectorTab({ onAddWatchlist }: { onAddWatchlist: (code: string, name: string, sector: string) => void }) {
  const summaryQuery = useQuery({
    queryKey: ["cross-sector-summary"],
    queryFn: getCrossSectorSummary,
  });

  if (summaryQuery.isLoading) return <LoadingState message="Loading summary..." />;
  if (summaryQuery.error) return <p className="text-sm text-red">Failed to load summary</p>;

  const stocks = summaryQuery.data?.stocks ?? [];

  const columns: Column<StockRanking>[] = [
    { key: "cycle_position", label: "Sector", sortable: true },
    { key: "rank", label: "Rank", sortable: true },
    { key: "name", label: "Name" },
    { key: "code", label: "Code" },
    { key: "price", label: "Price", sortable: true, render: (r) => r.price != null ? r.price.toFixed(2) : "-" },
    { key: "pb", label: "PB", sortable: true, render: (r) => r.pb != null ? r.pb.toFixed(2) : "-" },
    { key: "total_score", label: "Score", sortable: true, render: (r) => r.total_score.toFixed(1) },
    {
      key: "_action",
      label: "",
      render: (r) => (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onAddWatchlist(r.code, r.name, r.cycle_position);
          }}
          className="text-xs text-blue hover:underline"
        >
          + Watch
        </button>
      ),
    },
  ];

  return (
    <DataTable
      data={stocks as unknown as Record<string, unknown>[]}
      columns={columns as unknown as Column<Record<string, unknown>>[]}
      emptyMessage="No cross-sector data. Analyze sectors first."
    />
  );
}

// ============ Tab 2: Cycle Correlation Analysis ============

function CycleCorrelationTab() {
  const queryClient = useQueryClient();
  const [sector, setSector] = useState("");
  const [topN, setTopN] = useState(10);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [subTab, setSubTab] = useState(0);
  const [selectedStockIdx, setSelectedStockIdx] = useState(0);
  const abortRef = useRef<{ abort: () => void } | null>(null);

  const factorsQuery = useQuery({
    queryKey: ["all-factors"],
    queryFn: async () => {
      const { getAllFactors } = await import("@/lib/api");
      return getAllFactors();
    },
  });

  const analyzedSectors = factorsQuery.data?.sectors
    ? Object.entries(factorsQuery.data.sectors)
        .filter(([, v]) => v.factors && v.factors.length > 0)
        .map(([k]) => k)
    : [];

  const analysisQuery = useQuery({
    queryKey: ["sector-stock-analysis", sector],
    queryFn: () => getSectorStockAnalysis(sector),
    enabled: !!sector,
    retry: false,
  });

  const analysis: SectorStockAnalysis | null = analysisQuery.data ?? null;
  const stocks = analysis?.stocks ?? [];

  const handleAnalyze = (redo = false) => {
    if (!sector || isAnalyzing) return;
    setIsAnalyzing(true);
    setLogs([]);

    const url = redo ? stockRedoURL(sector) : stockAnalyzeURL(sector);
    const conn = createSSEConnection(
      url,
      { top_n: topN },
      {
        onChunk: (content) => setLogs((prev) => [...prev, content]),
        onDone: () => {
          setIsAnalyzing(false);
          queryClient.invalidateQueries({ queryKey: ["sector-stock-analysis", sector] });
        },
        onError: (msg) => {
          setIsAnalyzing(false);
          setLogs((prev) => [...prev, `[Error] ${msg}`]);
        },
        onStocksSaved: () => {
          setIsAnalyzing(false);
          queryClient.invalidateQueries({ queryKey: ["sector-stock-analysis", sector] });
        },
      },
    );
    abortRef.current = conn;
  };

  const subTabs = ["Ranking", "Score Details", "Correlation", "Valuation", "Data Quality"];

  // PB data for selected stock in correlation sub-tab
  const selectedStock = stocks[selectedStockIdx];
  const pbQuery = useQuery({
    queryKey: ["stock-pb-history", selectedStock?.code],
    queryFn: () => getStockChartData(selectedStock!.code),
    enabled: !!selectedStock && subTab === 2,
  });

  return (
    <div>
      {/* Controls */}
      <div className="flex items-end gap-4 mb-4">
        <div className="flex-1 max-w-xs">
          <label className="block text-xs text-text-secondary mb-1">Sector</label>
          <select
            value={sector}
            onChange={(e) => setSector(e.target.value)}
            className="w-full px-3 py-2 rounded-md border border-border bg-bg text-sm text-text-primary"
          >
            <option value="">Select sector...</option>
            {analyzedSectors.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
        <div className="w-48">
          <SliderInput label="Top N" value={topN} min={5} max={20} step={5} onChange={setTopN} />
        </div>
        <button
          onClick={() => handleAnalyze(false)}
          disabled={!sector || isAnalyzing}
          className="px-4 py-2 rounded-md bg-blue text-white text-sm hover:bg-blue/80 transition-colors disabled:opacity-40"
        >
          Analyze
        </button>
        {analysis && (
          <button
            onClick={() => handleAnalyze(true)}
            disabled={!sector || isAnalyzing}
            className="px-4 py-2 rounded-md bg-bg-hover text-text-secondary text-sm hover:text-text-primary transition-colors disabled:opacity-40"
          >
            Redo
          </button>
        )}
      </div>

      {/* Progress */}
      {isAnalyzing && (
        <div className="mb-4">
          <ProgressTracker
            title={`Analyzing ${sector} stocks`}
            isRunning={isAnalyzing}
            logs={logs}
            onCancel={() => { abortRef.current?.abort(); setIsAnalyzing(false); }}
          />
        </div>
      )}

      {/* Results */}
      {analysis && stocks.length > 0 && (
        <div>
          {/* Sub-tabs */}
          <div className="flex gap-1 mb-4 border-b border-border">
            {subTabs.map((label, i) => (
              <button
                key={label}
                onClick={() => setSubTab(i)}
                className={`px-3 py-2 text-xs transition-colors border-b-2 -mb-px ${
                  i === subTab
                    ? "border-blue text-blue"
                    : "border-transparent text-text-secondary hover:text-text-primary"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Sub-tab 0: Ranking */}
          {subTab === 0 && (
            <DataTable
              data={stocks as unknown as Record<string, unknown>[]}
              columns={[
                { key: "rank", label: "Rank", sortable: true },
                { key: "name", label: "Name" },
                { key: "code", label: "Code" },
                { key: "price", label: "Price", sortable: true, render: (r) => (r as unknown as StockRanking).price != null ? Number((r as unknown as StockRanking).price).toFixed(2) : "-" },
                { key: "pb", label: "PB", sortable: true, render: (r) => (r as unknown as StockRanking).pb != null ? Number((r as unknown as StockRanking).pb).toFixed(2) : "-" },
                { key: "total_score", label: "Score", sortable: true, render: (r) => (r as unknown as StockRanking).total_score.toFixed(1) },
                { key: "pb_anomaly", label: "Anomaly", render: (r) => (r as unknown as StockRanking).pb_anomaly ? "Yes" : "" },
              ] as Column<Record<string, unknown>>[]}
            />
          )}

          {/* Sub-tab 1: Score breakdown */}
          {subTab === 1 && (
            <div className="space-y-2">
              {stocks
                .filter((s) => !s.pb_anomaly)
                .sort((a, b) => b.total_score - a.total_score)
                .map((stock) => (
                  <details key={stock.code} className="rounded-lg border border-border bg-bg">
                    <summary className="px-4 py-2 cursor-pointer hover:bg-bg-hover text-sm text-text-primary">
                      #{stock.rank} {stock.name} - Score: {stock.total_score.toFixed(1)}
                    </summary>
                    <div className="px-4 pb-3 grid grid-cols-4 gap-2">
                      <MetricCard label="Upside" value={stock.scores.upside.toFixed(1)} color="green" />
                      <MetricCard label="Alignment" value={stock.scores.alignment.toFixed(1)} color="blue" />
                      <MetricCard label="Valuation" value={stock.scores.valuation.toFixed(1)} color="yellow" />
                      <MetricCard label="Momentum" value={stock.scores.momentum.toFixed(1)} color="red" />
                    </div>
                  </details>
                ))}
            </div>
          )}

          {/* Sub-tab 2: Correlation */}
          {subTab === 2 && (
            <div>
              <div className="mb-3">
                <label className="block text-xs text-text-secondary mb-1">Select Stock</label>
                <select
                  value={selectedStockIdx}
                  onChange={(e) => setSelectedStockIdx(Number(e.target.value))}
                  className="px-3 py-2 rounded-md border border-border bg-bg text-sm text-text-primary"
                >
                  {stocks.map((s, i) => (
                    <option key={s.code} value={i}>{s.name} ({s.code})</option>
                  ))}
                </select>
              </div>
              {selectedStock && pbQuery.data && (
                <CorrelationChart
                  factorName="Factor"
                  factorData={[]}
                  pbData={pbQuery.data.dates.map((d: string, i: number) => ({
                    date: d,
                    pb: pbQuery.data.pb[i] ?? null,
                    close: pbQuery.data.close[i] ?? null,
                  }))}
                  stockName={selectedStock.name}
                />
              )}
              {selectedStock && (
                <div className="mt-3">
                  <h4 className="text-xs font-semibold text-text-secondary mb-2">Correlation Details</h4>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(selectedStock.correlation).map(([factor, corr]) => (
                      <div key={factor} className="rounded-md bg-bg p-2 border border-border">
                        <span className="text-xs text-text-primary font-medium">{factor}</span>
                        <span className="text-xs text-text-secondary ml-2">
                          Pearson: {typeof corr === "number" ? corr.toFixed(3) : JSON.stringify(corr)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Sub-tab 3: Valuation comparison */}
          {subTab === 3 && (
            <PBComparisonChart
              stocks={stocks
                .filter((s) => !s.pb_anomaly && s.valuation.current_pb > 0)
                .map((s) => ({
                  name: s.name,
                  currentPB: s.valuation.current_pb,
                  histHighPB: s.valuation.cycle_peak_pb,
                  histLowPB: s.valuation.current_pb * 0.3,
                }))}
            />
          )}

          {/* Sub-tab 4: Data quality */}
          {subTab === 4 && (
            <div>
              {stocks.some((s) => s.pb_anomaly) ? (
                <div className="mb-3 rounded-md bg-red/10 border border-red/20 p-3">
                  <p className="text-sm text-red font-medium">
                    {stocks.filter((s) => s.pb_anomaly).length} stocks have PB data anomalies
                  </p>
                </div>
              ) : (
                <div className="mb-3 rounded-md bg-green/10 border border-green/20 p-3">
                  <p className="text-sm text-green font-medium">All stocks have normal PB data</p>
                </div>
              )}
              <DataTable
                data={stocks as unknown as Record<string, unknown>[]}
                columns={[
                  { key: "name", label: "Name" },
                  { key: "code", label: "Code" },
                  { key: "pb_months", label: "PB Months", sortable: true },
                  { key: "pb_anomaly", label: "Status", render: (r) => (r as unknown as StockRanking).pb_anomaly ? "Anomaly" : "Normal" },
                ] as Column<Record<string, unknown>>[]}
              />
            </div>
          )}
        </div>
      )}

      {!analysis && sector && !isAnalyzing && !analysisQuery.isLoading && (
        <div className="rounded-lg border border-border bg-bg-surface p-6 text-center text-text-secondary text-sm">
          No analysis results for this sector. Click &quot;Analyze&quot; to start.
        </div>
      )}
    </div>
  );
}

// ============ Tab 3: Legacy Profiles ============

function LegacyProfilesTab() {
  const legacyQuery = useQuery({
    queryKey: ["legacy-profiles"],
    queryFn: getLegacyProfiles,
  });

  const [expandedCode, setExpandedCode] = useState<string | null>(null);

  if (legacyQuery.isLoading) return <LoadingState message="Loading profiles..." />;

  const profilesDict = legacyQuery.data?.profiles ?? {};
  const profileEntries = Object.entries(profilesDict);

  if (profileEntries.length === 0) {
    return <p className="text-sm text-text-secondary">No legacy profiles found.</p>;
  }

  return (
    <div className="space-y-2">
      {profileEntries.map(([code, profile], i: number) => {
        const p = profile as Record<string, unknown>;
        const name = String(p.name ?? code);
        const expanded = expandedCode === code;

        return (
          <div key={i} className="rounded-lg border border-border bg-bg-surface">
            <button
              onClick={() => setExpandedCode(expanded ? null : code)}
              className="w-full text-left px-4 py-3 hover:bg-bg-hover transition-colors rounded-lg text-sm text-text-primary font-medium"
            >
              {name} ({code})
            </button>
            {expanded && (
              <div className="px-4 pb-4">
                <LegacyStockChart code={code} name={name} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function LegacyStockChart({ code, name }: { code: string; name: string }) {
  const pbQuery = useQuery({
    queryKey: ["stock-pb-history", code],
    queryFn: () => getStockChartData(code),
  });

  if (pbQuery.isLoading) return <LoadingState message="Loading chart..." />;
  if (!pbQuery.data) return <p className="text-xs text-text-secondary">No data available</p>;

  const chartData = pbQuery.data.dates.map((d: string, i: number) => ({
    date: d,
    close: pbQuery.data.close[i] ?? null,
    pb: pbQuery.data.pb[i] ?? null,
  }));

  return (
    <CombinedStockChart
      code={code}
      name={name}
      pbData={chartData}
    />
  );
}

// ============ Main Page ============

export default function StocksPage() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState(0);
  const [watchlistCollapsed, setWatchlistCollapsed] = useState(true);

  const watchlistQuery = useQuery({
    queryKey: ["watchlist"],
    queryFn: getWatchlist,
  });

  const handleAddWatchlist = async (code: string, name: string, sector: string) => {
    await addToWatchlist({ code, name, sector });
    queryClient.invalidateQueries({ queryKey: ["watchlist"] });
  };

  const handleRemoveWatchlist = async (code: string) => {
    await removeFromWatchlist(code);
    queryClient.invalidateQueries({ queryKey: ["watchlist"] });
  };

  const tabs = ["Cross-Sector Summary", "Cycle Correlation", "Legacy Profiles"];

  return (
    <div>
      <PageHeader
        title="Stock Screening"
        description="Filter and rank cyclical leader stocks by valuation metrics and cycle correlation."
      />

      {/* Watchlist */}
      <div className="mb-4">
        <WatchlistPanel
          items={watchlistQuery.data ?? []}
          onRemove={handleRemoveWatchlist}
          collapsed={watchlistCollapsed}
          onToggle={() => setWatchlistCollapsed(!watchlistCollapsed)}
        />
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-4 border-b border-border">
        {tabs.map((label, i) => (
          <button
            key={label}
            onClick={() => setActiveTab(i)}
            className={`px-4 py-2 text-sm transition-colors border-b-2 -mb-px ${
              i === activeTab
                ? "border-blue text-blue"
                : "border-transparent text-text-secondary hover:text-text-primary"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {activeTab === 0 && <CrossSectorTab onAddWatchlist={handleAddWatchlist} />}
      {activeTab === 1 && <CycleCorrelationTab />}
      {activeTab === 2 && <LegacyProfilesTab />}
    </div>
  );
}
