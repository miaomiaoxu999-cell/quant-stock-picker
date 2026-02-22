"use client";

import { useState, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import PageHeader from "@/components/layout/PageHeader";
import LoadingState from "@/components/shared/LoadingState";
import ChatPanel from "@/components/shared/ChatPanel";
import MetricCard from "@/components/shared/MetricCard";
import ProgressTracker from "@/components/shared/ProgressTracker";
import SSEStreamText from "@/components/shared/SSEStreamText";
import {
  getAllFactors,
  getAllCycles,
  getAuditResults,
  auditRunURL,
  auditChatURL,
  auditApplyFeedback,
} from "@/lib/api";
import { createSSEConnection } from "@/lib/sse";
import type { AuditResult, AuditReport, AuditItem, AuditRedFlag, ChatMessage } from "@/lib/types";

const AUDIT_TYPES = [
  { key: "factors", label: "Factor Audit" },
  { key: "cycle", label: "Cycle Audit" },
  { key: "stock", label: "Stock Audit" },
  { key: "full", label: "Full Audit" },
];

const RISK_COLORS: Record<string, string> = {
  low: "bg-green/10 text-green border-green/20",
  medium: "bg-warning/10 text-warning border-warning/20",
  high: "bg-red/10 text-red border-red/20",
  critical: "bg-red/20 text-red border-red/30",
};

const RISK_LABELS: Record<string, string> = {
  low: "Low Risk",
  medium: "Medium Risk",
  high: "High Risk",
  critical: "Critical Risk",
};

const STATUS_COLORS: Record<string, string> = {
  good: "text-green",
  warning: "text-warning",
  alert: "text-red",
};

// ============ Audit Report Display ============

function AuditReportView({ report }: { report: AuditReport }) {
  return (
    <div className="space-y-4">
      {/* Risk level + Confidence */}
      <div className="grid grid-cols-2 gap-3">
        <div className={`rounded-lg border p-3 ${RISK_COLORS[report.risk_level] ?? "bg-bg-surface"}`}>
          <div className="text-xs opacity-80 mb-1">Risk Level</div>
          <div className="text-lg font-bold">{RISK_LABELS[report.risk_level] ?? report.risk_level}</div>
        </div>
        <MetricCard
          label="Confidence Score"
          value={`${report.confidence_score}%`}
          color={report.confidence_score >= 70 ? "green" : "yellow"}
        />
      </div>

      {/* Summary */}
      {report.summary && (
        <div className="rounded-md bg-bg p-3 border border-border text-sm text-text-primary whitespace-pre-wrap">
          {report.summary}
        </div>
      )}

      {/* Audit items */}
      {report.audit_items.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-text-secondary mb-2">Audit Items</h4>
          <div className="space-y-1">
            {report.audit_items.map((item: AuditItem, i: number) => (
              <div key={i} className="rounded-md bg-bg p-2 border border-border flex items-start gap-2">
                <span className={`text-xs font-bold mt-0.5 ${STATUS_COLORS[item.status] ?? "text-text-secondary"}`}>
                  [{item.status.toUpperCase()}]
                </span>
                <div className="flex-1">
                  <span className="text-xs text-text-primary">{item.item}</span>
                  {item.category && (
                    <span className="text-xs text-text-secondary ml-2">({item.category})</span>
                  )}
                  {item.details && (
                    <p className="text-xs text-text-secondary mt-0.5">{item.details}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Red flags */}
      {report.red_flags.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-red mb-2">Red Flags</h4>
          <div className="space-y-1">
            {report.red_flags.map((flag: AuditRedFlag, i: number) => (
              <div key={i} className="rounded-md bg-red/5 border border-red/20 p-2">
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-bold ${
                    flag.severity === "high" ? "text-red" : flag.severity === "medium" ? "text-warning" : "text-text-secondary"
                  }`}>
                    [{flag.severity.toUpperCase()}]
                  </span>
                  <span className="text-xs text-text-primary">{flag.flag}</span>
                </div>
                {flag.recommendation && (
                  <p className="text-xs text-text-secondary mt-1">Recommendation: {flag.recommendation}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Data quality issues */}
      {report.data_quality_issues.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-warning mb-2">Data Quality Issues</h4>
          <ul className="space-y-1">
            {report.data_quality_issues.map((issue: string, i: number) => (
              <li key={i} className="text-xs text-text-secondary">- {issue}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Parameter suggestions */}
      {report.parameter_suggestions && Object.keys(report.parameter_suggestions).length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-text-secondary mb-2">Parameter Suggestions</h4>
          <pre className="text-xs text-text-secondary bg-bg p-2 rounded-md border border-border overflow-x-auto">
            {JSON.stringify(report.parameter_suggestions, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

// ============ Audit Tab Content ============

function AuditTabContent({
  sector,
  auditType,
  auditResult,
  onRefresh,
}: {
  sector: string;
  auditType: string;
  auditResult: AuditResult | null;
  onRefresh: () => void;
}) {
  const [isRunning, setIsRunning] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [logs, setLogs] = useState<string[]>([]);
  const [applyingFeedback, setApplyingFeedback] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(
    auditResult?.conversation ?? [],
  );
  const abortRef = useRef<{ abort: () => void } | null>(null);

  const handleRun = () => {
    setIsRunning(true);
    setStreamText("");
    setLogs([]);

    const conn = createSSEConnection(
      auditRunURL(sector, auditType),
      {},
      {
        onChunk: (content) => setStreamText((prev) => prev + content),
        onDone: () => {
          setIsRunning(false);
          onRefresh();
        },
        onError: (msg) => {
          setIsRunning(false);
          setLogs((prev) => [...prev, `[Error] ${msg}`]);
        },
        onAuditSaved: () => {
          setIsRunning(false);
          onRefresh();
        },
      },
    );
    abortRef.current = conn;
  };

  const handleApplyFeedback = async () => {
    setApplyingFeedback(true);
    try {
      const feedback = auditResult?.report
        ? `Audit found ${auditResult.report.red_flags.length} red flags with risk level: ${auditResult.report.risk_level}. Please address the issues.`
        : "Please review and update based on audit findings.";
      await auditApplyFeedback(sector, auditType, feedback);
      onRefresh();
    } catch {
      // Error handled by UI state
    } finally {
      setApplyingFeedback(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Running state */}
      {isRunning && (
        <div className="space-y-3">
          <ProgressTracker
            title={`Running ${auditType} audit...`}
            isRunning={isRunning}
            logs={logs}
            onCancel={() => { abortRef.current?.abort(); setIsRunning(false); }}
          />
          <SSEStreamText text={streamText} streaming={isRunning} />
        </div>
      )}

      {/* Existing result */}
      {auditResult && (
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-xs text-text-secondary">
            <span>Last audited: {auditResult.audited_at?.slice(0, 19) ?? "Unknown"}</span>
          </div>

          {auditResult.report ? (
            <AuditReportView report={auditResult.report} />
          ) : (
            <div className="rounded-md bg-bg p-3 border border-border">
              <p className="text-xs text-text-secondary mb-1">Raw audit response:</p>
              <pre className="text-xs text-text-primary whitespace-pre-wrap">
                {auditResult.raw_response}
              </pre>
            </div>
          )}

          {/* Feedback button */}
          {auditResult.report && ["medium", "high", "critical"].includes(auditResult.report.risk_level) && (
            <button
              onClick={handleApplyFeedback}
              disabled={applyingFeedback}
              className="px-4 py-2 rounded-md bg-warning/20 text-warning text-sm hover:bg-warning/30 transition-colors disabled:opacity-40"
            >
              {applyingFeedback ? "Applying..." : "Apply Feedback (auto-trigger reanalysis)"}
            </button>
          )}

          {/* Chat */}
          <div className="border-t border-border pt-3">
            <h4 className="text-xs font-semibold text-text-secondary mb-2">Discuss Audit</h4>
            <ChatPanel
              sseURL={auditChatURL(sector, auditType)}
              messages={chatMessages}
              onMessagesChange={setChatMessages}
              placeholder="Ask about the audit findings..."
            />
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center gap-2">
        <button
          onClick={handleRun}
          disabled={isRunning}
          className="px-4 py-2 rounded-md bg-blue text-white text-sm hover:bg-blue/80 transition-colors disabled:opacity-40"
        >
          {isRunning ? "Running..." : auditResult ? "Re-run Audit" : "Start Audit"}
        </button>
      </div>
    </div>
  );
}

// ============ Main Page ============

export default function AuditPage() {
  const queryClient = useQueryClient();
  const [selectedSector, setSelectedSector] = useState("");
  const [activeAuditTab, setActiveAuditTab] = useState(0);

  const factorsQuery = useQuery({
    queryKey: ["all-factors"],
    queryFn: getAllFactors,
  });

  const cyclesQuery = useQuery({
    queryKey: ["all-cycles"],
    queryFn: getAllCycles,
  });

  const auditsQuery = useQuery({
    queryKey: ["audit-results", selectedSector],
    queryFn: () => getAuditResults(selectedSector),
    enabled: !!selectedSector,
  });

  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ["audit-results", selectedSector] });
  };

  // Collect sectors that have data
  const sectorSet = new Set<string>();
  if (factorsQuery.data?.sectors) {
    for (const s of Object.keys(factorsQuery.data.sectors)) {
      sectorSet.add(s);
    }
  }
  if (cyclesQuery.data?.sectors) {
    for (const s of Object.keys(cyclesQuery.data.sectors)) {
      sectorSet.add(s);
    }
  }
  const sectors = Array.from(sectorSet);

  if (factorsQuery.isLoading || cyclesQuery.isLoading) {
    return <LoadingState message="Loading audit data..." />;
  }

  const sectorAuditResults = auditsQuery.data?.results ?? {};

  return (
    <div>
      <PageHeader
        title="Audit"
        description="Independent AI audit of strategy assumptions, cycle analysis, and stock screening results."
      />

      {sectors.length === 0 ? (
        <div className="rounded-lg border border-border bg-bg-surface p-8 text-center text-text-secondary text-sm">
          No sectors with data found. Complete factor or cycle analysis first.
        </div>
      ) : (
        <>
          {/* Sector selector */}
          <div className="mb-5">
            <label className="block text-xs text-text-secondary mb-1">Select Sector</label>
            <select
              value={selectedSector}
              onChange={(e) => setSelectedSector(e.target.value)}
              className="max-w-sm px-3 py-2 rounded-md border border-border bg-bg text-sm text-text-primary"
            >
              <option value="">Choose a sector...</option>
              {sectors.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          {selectedSector && (
            <div>
              {/* Audit type tabs */}
              <div className="flex gap-1 mb-4 border-b border-border">
                {AUDIT_TYPES.map((at, i) => (
                  <button
                    key={at.key}
                    onClick={() => setActiveAuditTab(i)}
                    className={`px-4 py-2 text-sm transition-colors border-b-2 -mb-px ${
                      i === activeAuditTab
                        ? "border-blue text-blue"
                        : "border-transparent text-text-secondary hover:text-text-primary"
                    }`}
                  >
                    {at.label}
                  </button>
                ))}
              </div>

              {/* Data preview */}
              <div className="grid grid-cols-4 gap-3 mb-4">
                <MetricCard
                  label="Factors"
                  value={
                    factorsQuery.data?.sectors?.[selectedSector]?.factors?.length ?? 0
                  }
                  color="blue"
                />
                <MetricCard
                  label="Cycle Position"
                  value={cyclesQuery.data?.sectors?.[selectedSector]?.overall?.cycle_position ?? "None"}
                  color="green"
                />
                <MetricCard
                  label="Reversal Prob"
                  value={`${cyclesQuery.data?.sectors?.[selectedSector]?.overall?.reversal_probability ?? 0}%`}
                  color="yellow"
                />
                <MetricCard
                  label="Audit Results"
                  value={Object.keys(sectorAuditResults).length}
                  color="blue"
                />
              </div>

              {/* Tab content */}
              <AuditTabContent
                key={`${selectedSector}-${AUDIT_TYPES[activeAuditTab].key}`}
                sector={selectedSector}
                auditType={AUDIT_TYPES[activeAuditTab].key}
                auditResult={(sectorAuditResults[AUDIT_TYPES[activeAuditTab].key] as AuditResult) ?? null}
                onRefresh={refresh}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}
