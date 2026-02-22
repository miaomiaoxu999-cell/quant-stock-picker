"use client";

import { useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import PageHeader from "@/components/layout/PageHeader";
import LoadingState from "@/components/shared/LoadingState";
import StepIndicator from "@/components/shared/StepIndicator";
import ChatPanel from "@/components/shared/ChatPanel";
import SSEStreamText from "@/components/shared/SSEStreamText";
import SliderInput from "@/components/shared/SliderInput";
import DataTable from "@/components/shared/DataTable";
import PredictionChart from "@/components/charts/PredictionChart";
import { useAdvisorStore } from "@/stores/advisorStore";
import {
  getAdvisorSession,
  getAvailableStocks,
  getAnalyzedSectors,
  submitStep1,
  confirmStep2,
  confirmStep3,
  advisorDiagnoseURL,
  advisorDiagnoseChatURL,
  advisorAllocateURL,
  advisorAllocateChatURL,
  advisorPredictURL,
  advisorRiskPlanURL,
  saveAdvisorPlan,
  resetAdvisorSession,
  getPortfolioState,
} from "@/lib/api";
import type { Holding } from "@/lib/types";
import { createSSEConnection } from "@/lib/sse";
import type { ChatMessage, AvailableStock, Allocation } from "@/lib/types";

const STEP_LABELS = ["Information", "Diagnosis", "Allocation", "Prediction", "Risk Plan"];

// ============ Step 1: Information ============

function Step1() {
  const store = useAdvisorStore();

  const sectorsQuery = useQuery({
    queryKey: ["analyzed-sectors"],
    queryFn: getAnalyzedSectors,
  });

  const stocksQuery = useQuery({
    queryKey: ["available-stocks"],
    queryFn: getAvailableStocks,
  });

  const portfolioQuery = useQuery({
    queryKey: ["portfolio-state"],
    queryFn: getPortfolioState,
  });

  const [localCapital, setLocalCapital] = useState(store.totalCapital);
  const [localSectors, setLocalSectors] = useState<string[]>(store.bullishSectors);
  const [localStockCodes, setLocalStockCodes] = useState<string[]>(
    store.favoredStocks.map((s) => s.code),
  );

  const allStocks = stocksQuery.data?.stocks ?? [];
  const sectorOptions = sectorsQuery.data?.sectors ?? [];
  const holdings = portfolioQuery.data?.holdings ?? {};

  const handleNext = async () => {
    const favored = localStockCodes
      .map((code) => {
        const s = allStocks.find((st: AvailableStock) => st.code === code);
        return s ? { code: s.code, name: s.name } : null;
      })
      .filter(Boolean) as { code: string; name: string }[];

    store.setInputs({
      totalCapital: localCapital,
      bullishSectors: localSectors,
      favoredStocks: favored,
    });

    await submitStep1({
      total_capital: localCapital,
      bullish_sectors: localSectors,
      favored_stock_codes: localStockCodes,
    });

    store.setStep(2);
  };

  return (
    <div className="space-y-5">
      <h2 className="text-base font-semibold text-text-primary">Step 1: Information</h2>

      {/* Capital */}
      <div>
        <label className="block text-sm text-text-secondary mb-1">Total Capital</label>
        <input
          type="number"
          value={localCapital}
          onChange={(e) => setLocalCapital(Number(e.target.value))}
          min={10000}
          step={10000}
          className="max-w-xs px-3 py-2 rounded-md border border-border bg-bg text-sm text-text-primary"
        />
      </div>

      {/* Bullish sectors */}
      <div>
        <label className="block text-sm text-text-secondary mb-1">Bullish Sectors</label>
        <div className="flex flex-wrap gap-2">
          {sectorOptions.map((s) => (
            <button
              key={s}
              onClick={() =>
                setLocalSectors((prev) =>
                  prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s],
                )
              }
              className={`px-3 py-1 rounded-full text-xs transition-colors ${
                localSectors.includes(s)
                  ? "bg-blue/20 text-blue border border-blue/30"
                  : "bg-bg-hover text-text-secondary border border-border"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Favored stocks */}
      <div>
        <label className="block text-sm text-text-secondary mb-1">Favored Stocks</label>
        <div className="max-h-60 overflow-y-auto border border-border rounded-md bg-bg">
          {allStocks.map((s: AvailableStock) => (
            <label
              key={s.code}
              className="flex items-center gap-2 px-3 py-1.5 hover:bg-bg-hover cursor-pointer"
            >
              <input
                type="checkbox"
                checked={localStockCodes.includes(s.code)}
                onChange={(e) =>
                  setLocalStockCodes((prev) =>
                    e.target.checked
                      ? [...prev, s.code]
                      : prev.filter((c) => c !== s.code),
                  )
                }
                className="accent-blue"
              />
              <span className="text-xs text-text-primary">
                {s.name} ({s.code})
              </span>
              <span className="text-xs text-text-secondary ml-auto">
                {s.valuation_status} {s.total_score != null ? `Score: ${s.total_score}` : ""}
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* Current holdings */}
      {Object.keys(holdings).length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-text-primary mb-2">Current Holdings</h3>
          <DataTable
            data={Object.entries(holdings).map(([code, h]) => ({
              code,
              avg_cost: (h as Holding).avg_cost ?? 0,
              shares: (h as Holding).shares ?? 0,
              weight: (h as Holding).weight ?? 0,
            }))}
            columns={[
              { key: "code" as const, label: "Code" },
              { key: "avg_cost" as const, label: "Avg Cost" },
              { key: "shares" as const, label: "Shares" },
              { key: "weight" as const, label: "Weight" },
            ]}
          />
        </div>
      )}

      <button
        onClick={handleNext}
        disabled={localStockCodes.length === 0}
        className="px-5 py-2 rounded-md bg-blue text-white text-sm font-medium hover:bg-blue/80 transition-colors disabled:opacity-40"
      >
        Next: AI Diagnosis
      </button>
    </div>
  );
}

// ============ Step 2: Diagnosis ============

function Step2() {
  const store = useAdvisorStore();
  const [streamText, setStreamText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(store.diagnosisMessages);
  const abortRef = useRef<{ abort: () => void } | null>(null);

  useEffect(() => {
    if (!store.diagnosisResponse && !isStreaming) {
      setIsStreaming(true);
      const conn = createSSEConnection(advisorDiagnoseURL(), {}, {
        onChunk: (c) => setStreamText((prev) => prev + c),
        onDone: (fullText) => {
          setIsStreaming(false);
          store.setDiagnosis(fullText, null);
          store.addDiagnosisMessage({ role: "assistant", content: fullText });
        },
        onError: (msg) => {
          setIsStreaming(false);
          store.addDiagnosisMessage({ role: "assistant", content: `[Error] ${msg}` });
        },
      });
      abortRef.current = conn;
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const [confirmCodes, setConfirmCodes] = useState<string[]>(
    store.confirmedCodes.length > 0
      ? store.confirmedCodes
      : store.favoredStocks.map((s) => s.code),
  );

  const handleConfirm = async () => {
    store.setConfirmedCodes(confirmCodes);
    await confirmStep2({ confirmed_codes: confirmCodes });
    store.setStep(3);
  };

  return (
    <div className="space-y-4">
      <h2 className="text-base font-semibold text-text-primary">Step 2: AI Diagnosis</h2>

      {/* Streaming or display */}
      {isStreaming ? (
        <SSEStreamText text={streamText} streaming={true} />
      ) : store.diagnosisResponse ? (
        <div className="rounded-md bg-bg p-4 border border-border text-sm text-text-primary whitespace-pre-wrap max-h-80 overflow-y-auto">
          {store.diagnosisResponse}
        </div>
      ) : null}

      {/* Chat */}
      {store.diagnosisResponse && (
        <ChatPanel
          sseURL={advisorDiagnoseChatURL()}
          messages={chatMessages}
          onMessagesChange={(msgs) => {
            setChatMessages(msgs);
            // Sync to store
            const lastMsg = msgs[msgs.length - 1];
            if (lastMsg) store.addDiagnosisMessage(lastMsg);
          }}
          placeholder="Follow-up questions..."
        />
      )}

      {/* Stock confirmation */}
      {store.diagnosisResponse && (
        <div>
          <h3 className="text-sm font-semibold text-text-primary mb-2">Confirm Stock List</h3>
          <div className="flex flex-wrap gap-2 mb-3">
            {store.favoredStocks.map((s) => (
              <label key={s.code} className="flex items-center gap-1.5 text-xs text-text-primary">
                <input
                  type="checkbox"
                  checked={confirmCodes.includes(s.code)}
                  onChange={(e) =>
                    setConfirmCodes((prev) =>
                      e.target.checked
                        ? [...prev, s.code]
                        : prev.filter((c) => c !== s.code),
                    )
                  }
                  className="accent-blue"
                />
                {s.name} ({s.code})
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Navigation */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => store.setStep(1)}
          className="px-4 py-2 rounded-md bg-bg-hover text-text-secondary text-sm hover:text-text-primary transition-colors"
        >
          Back
        </button>
        <button
          onClick={handleConfirm}
          disabled={confirmCodes.length === 0 || !store.diagnosisResponse}
          className="px-5 py-2 rounded-md bg-blue text-white text-sm font-medium hover:bg-blue/80 transition-colors disabled:opacity-40"
        >
          Confirm & Next
        </button>
      </div>
    </div>
  );
}

// ============ Step 3: Allocation ============

function Step3() {
  const store = useAdvisorStore();
  const [streamText, setStreamText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(store.allocationMessages);
  const [localAlloc, setLocalAlloc] = useState<Allocation[]>(store.editableAllocation);
  const [cashRatio, setCashRatio] = useState(
    typeof store.cashReserve.ratio === "number" ? store.cashReserve.ratio * 100 : 10,
  );

  useEffect(() => {
    if (!store.allocationResponse && !isStreaming) {
      setIsStreaming(true);
      createSSEConnection(advisorAllocateURL(), {}, {
        onChunk: (c) => setStreamText((prev) => prev + c),
        onDone: (fullText) => {
          setIsStreaming(false);
          store.setAllocation(fullText, null);
          store.addAllocationMessage({ role: "assistant", content: fullText });
        },
        onError: (msg) => {
          setIsStreaming(false);
          store.addAllocationMessage({ role: "assistant", content: `[Error] ${msg}` });
        },
      });
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Initialize editable allocation from store response
  useEffect(() => {
    if (store.allocationJson && localAlloc.length === 0) {
      const allocations = (store.allocationJson as Record<string, unknown>).allocations as Allocation[] | undefined;
      if (allocations) setLocalAlloc(allocations);
    }
  }, [store.allocationJson]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRatioChange = (idx: number, newPct: number) => {
    const updated = [...localAlloc];
    updated[idx] = { ...updated[idx], ratio: newPct / 100 };
    // Normalize others
    const remaining = Math.max(0, 1 - newPct / 100);
    const othersTotal = updated.reduce(
      (sum, a, i) => (i !== idx ? sum + a.ratio : sum),
      0,
    );
    for (let i = 0; i < updated.length; i++) {
      if (i === idx) continue;
      updated[i] = {
        ...updated[i],
        ratio: othersTotal > 0 ? (updated[i].ratio / othersTotal) * remaining : remaining / (updated.length - 1),
      };
    }
    setLocalAlloc(updated);
  };

  const handleConfirm = async () => {
    store.setEditableAllocation(localAlloc);
    store.setCashReserve({ ratio: cashRatio / 100 });
    await confirmStep3({
      allocations: localAlloc.map((a) => ({ code: a.code, ratio: a.ratio })),
      cash_reserve: { ratio: cashRatio / 100 },
    });
    store.setStep(4);
  };

  return (
    <div className="space-y-4">
      <h2 className="text-base font-semibold text-text-primary">Step 3: Allocation</h2>

      {isStreaming ? (
        <SSEStreamText text={streamText} streaming={true} />
      ) : store.allocationResponse ? (
        <div className="rounded-md bg-bg p-4 border border-border text-sm text-text-primary whitespace-pre-wrap max-h-60 overflow-y-auto">
          {store.allocationResponse}
        </div>
      ) : null}

      {/* Editable allocation table */}
      {localAlloc.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-text-primary">Adjust Allocation</h3>
          {localAlloc.map((a, i) => (
            <div key={a.code} className="rounded-md bg-bg p-3 border border-border">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-text-primary font-medium">
                  {a.name} ({a.code})
                </span>
                <span className="text-sm text-blue font-bold">
                  {(a.ratio * 100).toFixed(1)}%
                </span>
              </div>
              <SliderInput
                label=""
                value={Math.round(a.ratio * 100)}
                min={0}
                max={100}
                step={1}
                unit="%"
                onChange={(v) => handleRatioChange(i, v)}
              />
              <div className="flex gap-4 text-xs text-text-secondary mt-1">
                <span>Action: {a.action}</span>
                <span>Price: {a.price_range}</span>
                <span>Amount: {(store.totalCapital * a.ratio).toFixed(0)}</span>
              </div>
            </div>
          ))}

          {/* Cash reserve */}
          <div className="rounded-md bg-bg p-3 border border-border">
            <SliderInput
              label="Cash Reserve"
              value={cashRatio}
              min={0}
              max={50}
              step={1}
              unit="%"
              onChange={setCashRatio}
            />
          </div>
        </div>
      )}

      {/* Chat */}
      {store.allocationResponse && (
        <ChatPanel
          sseURL={advisorAllocateChatURL()}
          messages={chatMessages}
          onMessagesChange={setChatMessages}
          placeholder="Adjust allocation..."
        />
      )}

      {/* Navigation */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => store.setStep(2)}
          className="px-4 py-2 rounded-md bg-bg-hover text-text-secondary text-sm hover:text-text-primary transition-colors"
        >
          Back
        </button>
        <button
          onClick={handleConfirm}
          disabled={localAlloc.length === 0 || !store.allocationResponse}
          className="px-5 py-2 rounded-md bg-blue text-white text-sm font-medium hover:bg-blue/80 transition-colors disabled:opacity-40"
        >
          Confirm & Next
        </button>
      </div>
    </div>
  );
}

// ============ Step 4: Prediction ============

function Step4() {
  const store = useAdvisorStore();
  const [streamText, setStreamText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    if (!store.predictionJson && !isStreaming) {
      setIsStreaming(true);
      createSSEConnection(advisorPredictURL(), {}, {
        onChunk: (c) => setStreamText((prev) => prev + c),
        onDone: (fullText) => {
          setIsStreaming(false);
          // Try to parse JSON from response
          try {
            const match = fullText.match(/```json\s*\n?([\s\S]*?)\n?\s*```/);
            if (match) {
              store.setPrediction(JSON.parse(match[1]));
            }
          } catch {
            // Not parseable
          }
        },
        onError: () => setIsStreaming(false),
      });
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const predictions = (store.predictionJson as Record<string, Record<string, { total_value: number; return_rate: number }>> | null)?.predictions;

  return (
    <div className="space-y-4">
      <h2 className="text-base font-semibold text-text-primary">Step 4: Prediction</h2>

      {isStreaming ? (
        <SSEStreamText text={streamText} streaming={true} />
      ) : streamText ? (
        <div className="rounded-md bg-bg p-4 border border-border text-sm text-text-primary whitespace-pre-wrap max-h-60 overflow-y-auto">
          {streamText}
        </div>
      ) : null}

      {predictions && (
        <PredictionChart
          predictions={predictions as unknown as Record<string, Record<string, { total_value: number; return_rate: number }>>}
          totalCapital={store.totalCapital}
        />
      )}

      {/* Detail table */}
      {predictions && (
        <div>
          <h3 className="text-sm font-semibold text-text-primary mb-2">Detail</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-bg-surface border-b border-border">
                  <th className="px-3 py-2 text-left text-text-secondary">Period</th>
                  <th className="px-3 py-2 text-left text-text-secondary">Scenario</th>
                  <th className="px-3 py-2 text-left text-text-secondary">Return</th>
                  <th className="px-3 py-2 text-left text-text-secondary">Total Value</th>
                </tr>
              </thead>
              <tbody>
                {["3m", "6m", "12m"].map((p) =>
                  ["optimistic", "baseline", "pessimistic"].map((s) => {
                    const d = (predictions as unknown as Record<string, Record<string, { total_value: number; return_rate: number }>>)?.[p]?.[s];
                    return (
                      <tr key={`${p}-${s}`} className="border-b border-border">
                        <td className="px-3 py-1.5 text-text-primary">{p}</td>
                        <td className="px-3 py-1.5 text-text-primary">{s}</td>
                        <td className="px-3 py-1.5 text-text-primary">
                          {d ? `${(d.return_rate * 100).toFixed(1)}%` : "-"}
                        </td>
                        <td className="px-3 py-1.5 text-text-primary">
                          {d ? d.total_value.toLocaleString() : "-"}
                        </td>
                      </tr>
                    );
                  }),
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="flex items-center gap-3">
        <button
          onClick={() => store.setStep(3)}
          className="px-4 py-2 rounded-md bg-bg-hover text-text-secondary text-sm hover:text-text-primary transition-colors"
        >
          Back
        </button>
        <button
          onClick={() => store.setStep(5)}
          disabled={isStreaming}
          className="px-5 py-2 rounded-md bg-blue text-white text-sm font-medium hover:bg-blue/80 transition-colors disabled:opacity-40"
        >
          Next: Risk Plan
        </button>
      </div>
    </div>
  );
}

// ============ Step 5: Risk Plan ============

function Step5() {
  const store = useAdvisorStore();
  const [streamText, setStreamText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [syncPortfolio, setSyncPortfolio] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!store.riskPlanJson && !isStreaming) {
      setIsStreaming(true);
      createSSEConnection(advisorRiskPlanURL(), {}, {
        onChunk: (c) => setStreamText((prev) => prev + c),
        onDone: (fullText) => {
          setIsStreaming(false);
          try {
            const match = fullText.match(/```json\s*\n?([\s\S]*?)\n?\s*```/);
            if (match) {
              store.setRiskPlan(JSON.parse(match[1]));
            }
          } catch {
            // Not parseable
          }
        },
        onError: () => setIsStreaming(false),
      });
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const riskData = store.riskPlanJson as Record<string, unknown[]> | null;

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveAdvisorPlan({ sync_portfolio: syncPortfolio });
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    await resetAdvisorSession();
    store.resetSession();
  };

  const handleDownload = () => {
    const data = {
      inputs: {
        total_capital: store.totalCapital,
        bullish_sectors: store.bullishSectors,
        favored_stocks: store.favoredStocks,
      },
      confirmed_stocks: store.confirmedCodes,
      allocation: store.editableAllocation,
      cash_reserve: store.cashReserve,
      prediction: store.predictionJson,
      risk_plan: store.riskPlanJson,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `advisor_plan_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      <h2 className="text-base font-semibold text-text-primary">Step 5: Risk Plan</h2>

      {isStreaming ? (
        <SSEStreamText text={streamText} streaming={true} />
      ) : streamText ? (
        <div className="rounded-md bg-bg p-4 border border-border text-sm text-text-primary whitespace-pre-wrap max-h-60 overflow-y-auto">
          {streamText}
        </div>
      ) : null}

      {/* Stock risks */}
      {riskData?.stock_risks && (riskData.stock_risks as Record<string, unknown>[]).length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-text-primary mb-2">Stock-Level Risks</h3>
          <DataTable
            data={riskData.stock_risks as Record<string, unknown>[]}
            columns={[
              { key: "name", label: "Stock" },
              { key: "code", label: "Code" },
              { key: "signal", label: "Signal" },
              { key: "threshold", label: "Threshold" },
              { key: "action", label: "Action" },
            ]}
          />
        </div>
      )}

      {/* Sector risks */}
      {riskData?.sector_risks && (riskData.sector_risks as Record<string, unknown>[]).length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-text-primary mb-2">Sector-Level Risks</h3>
          <DataTable
            data={riskData.sector_risks as Record<string, unknown>[]}
            columns={[
              { key: "sector", label: "Sector" },
              { key: "signal", label: "Signal" },
              { key: "threshold", label: "Threshold" },
              { key: "action", label: "Action" },
            ]}
          />
        </div>
      )}

      {/* Macro risks */}
      {riskData?.macro_risks && (riskData.macro_risks as Record<string, unknown>[]).length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-text-primary mb-2">Macro Risks</h3>
          <DataTable
            data={riskData.macro_risks as Record<string, unknown>[]}
            columns={[
              { key: "signal", label: "Signal" },
              { key: "threshold", label: "Threshold" },
              { key: "action", label: "Action" },
              { key: "source", label: "Source" },
            ]}
          />
        </div>
      )}

      {/* Actions */}
      <div className="border-t border-border pt-4 space-y-3">
        <label className="flex items-center gap-2 text-sm text-text-primary">
          <input
            type="checkbox"
            checked={syncPortfolio}
            onChange={(e) => setSyncPortfolio(e.target.checked)}
            className="accent-blue"
          />
          Sync allocation to portfolio
        </label>

        <div className="flex items-center gap-3">
          <button
            onClick={() => store.setStep(4)}
            className="px-4 py-2 rounded-md bg-bg-hover text-text-secondary text-sm hover:text-text-primary transition-colors"
          >
            Back
          </button>
          <button
            onClick={handleSave}
            disabled={saving || isStreaming}
            className="px-5 py-2 rounded-md bg-blue text-white text-sm font-medium hover:bg-blue/80 transition-colors disabled:opacity-40"
          >
            {saving ? "Saving..." : "Save Plan"}
          </button>
          <button
            onClick={handleDownload}
            className="px-4 py-2 rounded-md bg-bg-hover text-text-secondary text-sm hover:text-text-primary transition-colors"
          >
            Download JSON
          </button>
          <button
            onClick={handleReset}
            className="px-4 py-2 rounded-md bg-red/20 text-red text-sm hover:bg-red/30 transition-colors"
          >
            Reset Session
          </button>
        </div>
      </div>
    </div>
  );
}

// ============ Main Page ============

export default function AdvisorPage() {
  const store = useAdvisorStore();
  const [loaded, setLoaded] = useState(false);

  // Restore session from server on mount
  const sessionQuery = useQuery({
    queryKey: ["advisor-session"],
    queryFn: getAdvisorSession,
    enabled: !loaded,
  });

  useEffect(() => {
    if (sessionQuery.data && !loaded) {
      const serverSession = sessionQuery.data.session;
      if (serverSession) {
        store.restoreFromServer(serverSession);
      }
      setLoaded(true);
    } else if (sessionQuery.isError || (sessionQuery.isFetched && !sessionQuery.data)) {
      setLoaded(true);
    }
  }, [sessionQuery.data, sessionQuery.isError, sessionQuery.isFetched]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!loaded && sessionQuery.isLoading) {
    return <LoadingState message="Loading advisor session..." />;
  }

  return (
    <div>
      <PageHeader
        title="AI Advisor"
        description="5-step AI-driven investment advisor: information, diagnosis, allocation, prediction, and risk planning."
      />

      {/* Step indicator */}
      <div className="mb-6">
        <StepIndicator
          steps={STEP_LABELS}
          currentStep={store.step}
          onStepClick={(s) => {
            if (s <= store.step) store.setStep(s);
          }}
        />
      </div>

      {/* Step content */}
      {store.step === 1 && <Step1 />}
      {store.step === 2 && <Step2 />}
      {store.step === 3 && <Step3 />}
      {store.step === 4 && <Step4 />}
      {store.step === 5 && <Step5 />}
    </div>
  );
}
