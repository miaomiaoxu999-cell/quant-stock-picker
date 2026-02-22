"use client";

import { useState, useCallback, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import PageHeader from "@/components/layout/PageHeader";
import LoadingState from "@/components/shared/LoadingState";
import ChatPanel from "@/components/shared/ChatPanel";
import ProgressTracker from "@/components/shared/ProgressTracker";
import MetricCard from "@/components/shared/MetricCard";
import CycleFactorChart from "@/components/charts/CycleFactorChart";
import {
  getAllFactors,
  getAllCycles,
  cycleAnalyzeURL,
  cycleChatURL,
} from "@/lib/api";
import { createSSEConnection } from "@/lib/sse";
import type { CycleAnalysis, CycleFactorData, ChatMessage } from "@/lib/types";

// ============ Saved Cycle Card ============

function CycleCard({
  sector,
  data,
}: {
  sector: string;
  data: CycleAnalysis;
}) {
  const [expanded, setExpanded] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(
    data.conversation ?? [],
  );

  const overall = data.overall;
  const pos = overall?.cycle_position ?? "Unknown";
  const prob = overall?.reversal_probability ?? 0;
  const timeStr = data.analyzed_at?.slice(0, 10) ?? "";

  const confidenceColor = (c: string) => {
    if (c === "high") return "text-green";
    if (c === "medium") return "text-warning";
    return "text-red";
  };

  return (
    <div className="rounded-lg border border-border bg-bg-surface">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left px-4 py-3 flex items-center justify-between hover:bg-bg-hover transition-colors rounded-lg"
      >
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-text-primary">{sector}</span>
          <span className="px-2 py-0.5 rounded-full bg-blue/10 text-blue text-xs font-medium">
            {pos}
          </span>
          <span className="text-xs text-text-secondary">
            Reversal {prob}%
          </span>
          <span className="text-xs text-text-secondary">{timeStr}</span>
        </div>
        <svg
          className={`w-4 h-4 text-text-secondary transition-transform ${expanded ? "rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-4">
          {/* Overall Summary */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <MetricCard label="Cycle Position" value={pos} color="blue" />
            <MetricCard
              label={`Reversal Probability (${overall?.probability_timeframe ?? "12m"})`}
              value={`${prob}%`}
              subtext={overall?.probability_rationale}
              color={prob > 50 ? "green" : "yellow"}
            />
            <MetricCard
              label="Key Signals"
              value={`${overall?.key_signals?.length ?? 0} signals`}
              color="blue"
            />
          </div>

          {/* Summary text */}
          {overall?.summary && (
            <div className="rounded-md bg-bg p-3 border border-border text-sm text-text-primary whitespace-pre-wrap">
              {overall.summary}
            </div>
          )}

          {/* Key signals */}
          {overall?.key_signals && overall.key_signals.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-text-secondary mb-2">Key Signals</h4>
              <ul className="space-y-1">
                {overall.key_signals.map((signal, i) => (
                  <li key={i} className="text-xs text-text-primary flex items-start gap-2">
                    <span className="text-blue mt-0.5">-</span>
                    {signal}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Factor charts */}
          {data.factors && data.factors.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-text-secondary mb-2">Factor Analysis</h4>
              <div className="space-y-4">
                {data.factors.map((factor: CycleFactorData, i: number) => (
                  <div key={i} className="rounded-lg border border-border bg-bg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <h5 className="text-sm font-medium text-text-primary">{factor.name}</h5>
                      <span className={`text-xs ${confidenceColor(factor.data_confidence)}`}>
                        Confidence: {factor.data_confidence}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-2 mb-2 text-xs text-text-secondary">
                      <span>Position: {factor.current_position}</span>
                      <span>Value: {factor.current_value}</span>
                      <span>Avg cycle: {factor.avg_cycle_length_months ?? "N/A"} months</span>
                    </div>
                    {factor.cycle_data && factor.cycle_data.length > 0 && (
                      <CycleFactorChart factor={factor} />
                    )}
                    {factor.analysis && (
                      <p className="text-xs text-text-secondary mt-2">{factor.analysis}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* News */}
          {data.news && data.news.length > 0 && (
            <div>
              <h4 className="text-xs font-semibold text-text-secondary mb-2">Industry News</h4>
              <div className="space-y-1">
                {data.news.slice(0, 8).map((item, i) => (
                  <a
                    key={i}
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block text-xs text-blue hover:underline"
                  >
                    {item.title} <span className="text-text-secondary">({item.source})</span>
                  </a>
                ))}
              </div>
            </div>
          )}

          {/* Chat */}
          <div className="border-t border-border pt-3">
            <h4 className="text-xs font-semibold text-text-secondary mb-2">Discussion</h4>
            <ChatPanel
              sseURL={cycleChatURL(sector)}
              messages={chatMessages}
              onMessagesChange={setChatMessages}
              placeholder={`Discuss "${sector}" cycle...`}
            />
          </div>
        </div>
      )}
    </div>
  );
}

// ============ Main Page ============

export default function CyclePage() {
  const queryClient = useQueryClient();
  const [selectedSector, setSelectedSector] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progressLogs, setProgressLogs] = useState<string[]>([]);
  const [progressStep, setProgressStep] = useState("");
  const [factorIndex, setFactorIndex] = useState(0);
  const [factorTotal, setFactorTotal] = useState(0);
  const [currentFactor, setCurrentFactor] = useState("");
  const abortRef = useRef<{ abort: () => void } | null>(null);

  const factorsQuery = useQuery({
    queryKey: ["all-factors"],
    queryFn: getAllFactors,
  });

  const cyclesQuery = useQuery({
    queryKey: ["all-cycles"],
    queryFn: getAllCycles,
  });

  const refresh = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["all-cycles"] });
  }, [queryClient]);

  const sectorsWithFactors = factorsQuery.data?.sectors
    ? Object.entries(factorsQuery.data.sectors)
        .filter(([, v]) => v.factors && v.factors.length > 0)
        .map(([k]) => k)
    : [];

  const handleAnalyze = () => {
    if (!selectedSector || isAnalyzing) return;

    setIsAnalyzing(true);
    setProgressLogs([]);
    setProgressStep("init");
    setFactorIndex(0);
    setFactorTotal(0);
    setCurrentFactor("");

    const conn = createSSEConnection(
      cycleAnalyzeURL(selectedSector),
      {},
      {
        onChunk: (content) => {
          setProgressLogs((prev) => [...prev, content]);
        },
        onDone: () => {
          setIsAnalyzing(false);
          refresh();
        },
        onError: (msg) => {
          setIsAnalyzing(false);
          setProgressLogs((prev) => [...prev, `[Error] ${msg}`]);
        },
        onProgress: (event) => {
          setProgressStep(event.step);
          setFactorIndex(event.factor_index);
          setFactorTotal(event.factor_total);
          setCurrentFactor(event.current_factor);
          if (event.log) {
            const logs = Array.isArray(event.log) ? event.log : [event.log];
            setProgressLogs((prev) => [...prev, ...logs]);
          }
        },
        onAnalysisSaved: () => {
          setIsAnalyzing(false);
          refresh();
        },
      },
    );

    abortRef.current = conn;
  };

  const handleCancel = () => {
    abortRef.current?.abort();
    setIsAnalyzing(false);
    setProgressLogs((prev) => [...prev, "Analysis cancelled"]);
  };

  if (factorsQuery.isLoading || cyclesQuery.isLoading) {
    return <LoadingState message="Loading cycle data..." />;
  }

  const cycles = cyclesQuery.data?.sectors ?? {};

  return (
    <div>
      <PageHeader
        title="Cycle Analysis"
        description="AI-driven industry cycle analysis with multi-source data collection and LLM synthesis."
      />

      {/* Progress tracker */}
      {isAnalyzing && (
        <div className="mb-5">
          <ProgressTracker
            title={`Analyzing "${selectedSector}"`}
            isRunning={isAnalyzing}
            logs={progressLogs}
            currentStep={progressStep}
            factorIndex={factorIndex}
            factorTotal={factorTotal}
            currentFactor={currentFactor}
            onCancel={handleCancel}
          />
        </div>
      )}

      {/* New analysis controls */}
      <div className="rounded-lg border border-border bg-bg-surface p-4 mb-5">
        <h2 className="text-sm font-semibold text-text-primary mb-3">New Cycle Analysis</h2>

        {sectorsWithFactors.length === 0 ? (
          <p className="text-sm text-text-secondary">
            No sectors with factors found. Generate factors on the Factors page first.
          </p>
        ) : (
          <div className="flex items-end gap-3">
            <div className="flex-1">
              <label className="block text-xs text-text-secondary mb-1">Select Sector</label>
              <select
                value={selectedSector}
                onChange={(e) => setSelectedSector(e.target.value)}
                className="w-full px-3 py-2 rounded-md border border-border bg-bg text-sm text-text-primary"
              >
                <option value="">Choose a sector...</option>
                {sectorsWithFactors.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
            <button
              onClick={handleAnalyze}
              disabled={!selectedSector || isAnalyzing}
              className="px-5 py-2 rounded-md bg-blue text-white text-sm font-medium hover:bg-blue/80 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {isAnalyzing ? "Analyzing..." : "Start Analysis"}
            </button>
          </div>
        )}

        {selectedSector && cycles[selectedSector] && (
          <p className="text-xs text-text-secondary mt-2">
            This sector has existing analysis ({cycles[selectedSector].analyzed_at?.slice(0, 10)}). Re-analyzing will overwrite.
          </p>
        )}
      </div>

      {/* Saved cycles */}
      {Object.keys(cycles).length > 0 && (
        <div>
          <h2 className="text-sm font-semibold text-text-primary mb-3">
            Analyzed Cycles ({Object.keys(cycles).length})
          </h2>
          <div className="space-y-2">
            {Object.entries(cycles).map(([sector, data]) => (
              <CycleCard key={sector} sector={sector} data={data} />
            ))}
          </div>
        </div>
      )}

      {Object.keys(cycles).length === 0 && sectorsWithFactors.length > 0 && !isAnalyzing && (
        <div className="rounded-lg border border-border bg-bg-surface p-8 text-center text-text-secondary text-sm">
          No cycle analyses yet. Select a sector and start analysis above.
        </div>
      )}
    </div>
  );
}
