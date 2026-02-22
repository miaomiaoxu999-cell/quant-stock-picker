"use client";

import { useState, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import PageHeader from "@/components/layout/PageHeader";
import LoadingState from "@/components/shared/LoadingState";
import ChatPanel from "@/components/shared/ChatPanel";
import SSEStreamText from "@/components/shared/SSEStreamText";
import {
  getPresetSectors,
  getAllFactors,
  updateFactorWeights,
  factorGenerateURL,
  factorChatURL,
} from "@/lib/api";
import { createSSEConnection } from "@/lib/sse";
import type { Factor, ChatMessage, SectorData } from "@/lib/types";

// ============ Sector Selector ============

function SectorSelector({
  presets,
  selected,
  onSelectedChange,
}: {
  presets: string[];
  selected: string[];
  onSelectedChange: (s: string[]) => void;
}) {
  const [custom, setCustom] = useState("");
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const toggle = (sector: string) => {
    if (selected.includes(sector)) {
      onSelectedChange(selected.filter((s) => s !== sector));
    } else if (selected.length < 5) {
      onSelectedChange([...selected, sector]);
    }
  };

  const addCustom = () => {
    const name = custom.trim();
    if (!name || selected.includes(name)) return;
    if (selected.length >= 5) return;
    onSelectedChange([...selected, name]);
    setCustom("");
  };

  return (
    <div className="rounded-lg border border-border bg-bg-surface p-4 mb-5">
      <h2 className="text-sm font-semibold text-text-primary mb-3">Select Sectors (max 5)</h2>
      <div className="flex gap-3 items-start">
        {/* Dropdown multiselect */}
        <div className="flex-1 relative">
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="w-full text-left px-3 py-2 rounded-md border border-border bg-bg text-sm text-text-primary hover:border-blue/50 transition-colors"
          >
            {selected.length > 0
              ? `${selected.length} selected`
              : "Choose from presets..."}
            <svg
              className="w-4 h-4 absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {dropdownOpen && (
            <div className="absolute z-30 mt-1 w-full max-h-60 overflow-y-auto rounded-md border border-border bg-bg-surface shadow-lg">
              {presets.map((s) => (
                <button
                  key={s}
                  onClick={() => toggle(s)}
                  className={`w-full text-left px-3 py-1.5 text-sm hover:bg-bg-hover transition-colors ${
                    selected.includes(s)
                      ? "text-blue bg-blue/5"
                      : "text-text-secondary"
                  }`}
                >
                  {selected.includes(s) ? "* " : "  "}
                  {s}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Custom input */}
        <div className="flex gap-2">
          <input
            type="text"
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && addCustom()}
            placeholder="Custom sector"
            className="w-36 text-sm"
          />
          <button
            onClick={addCustom}
            disabled={!custom.trim() || selected.length >= 5}
            className="px-3 py-2 rounded-md bg-bg-hover text-text-secondary text-sm hover:text-text-primary transition-colors disabled:opacity-40"
          >
            Add
          </button>
        </div>
      </div>

      {/* Selected tags */}
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-3">
          {selected.map((s) => (
            <span
              key={s}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-blue/10 text-blue text-xs"
            >
              {s}
              <button
                onClick={() => toggle(s)}
                className="hover:text-red transition-colors"
              >
                x
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ============ Factor Card ============

function FactorCard({
  factor,
  index,
  onWeightChange,
}: {
  factor: Factor;
  index: number;
  onWeightChange: (index: number, weight: number) => void;
}) {
  return (
    <div className="rounded-lg border border-border bg-bg p-4">
      <div className="flex items-start justify-between mb-2">
        <h4 className="text-sm font-semibold text-text-primary">{factor.name}</h4>
        <span className="text-lg font-bold text-blue">{factor.weight}%</span>
      </div>
      <p className="text-xs text-text-secondary mb-2">{factor.description}</p>
      <p className="text-xs text-text-secondary mb-3">
        Source: {factor.data_source}
      </p>
      <input
        type="range"
        min={0}
        max={100}
        value={factor.weight}
        onChange={(e) => onWeightChange(index, parseInt(e.target.value))}
        className="w-full h-1 bg-border rounded-lg appearance-none cursor-pointer accent-blue"
      />
    </div>
  );
}

// ============ Sector Tab Content ============

function SectorTab({
  sector,
  data,
  onRefresh,
}: {
  sector: string;
  data: SectorData | undefined;
  onRefresh: () => void;
}) {
  const queryClient = useQueryClient();
  const factors = data?.factors ?? [];
  const conversation = data?.conversation ?? [];

  const [localFactors, setLocalFactors] = useState<Factor[] | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(conversation);
  const [generating, setGenerating] = useState(false);
  const [genText, setGenText] = useState("");
  const [genError, setGenError] = useState("");

  const displayFactors = localFactors ?? factors;

  // Weight normalization (same logic as Streamlit)
  const handleWeightChange = (changedIdx: number, newWeight: number) => {
    const updated = displayFactors.map((f, i) => ({
      ...f,
      weight: i === changedIdx ? newWeight : f.weight,
    }));

    const n = updated.length;
    if (n <= 1) {
      updated[0].weight = 100;
      setLocalFactors(updated);
      return;
    }

    const remaining = Math.max(0, 100 - newWeight);
    const othersTotal = updated.reduce(
      (sum, f, i) => (i !== changedIdx ? sum + f.weight : sum),
      0,
    );

    for (let i = 0; i < n; i++) {
      if (i === changedIdx) continue;
      if (othersTotal > 0) {
        updated[i].weight = Math.round((updated[i].weight / othersTotal) * remaining);
      } else {
        updated[i].weight = Math.round(remaining / (n - 1));
      }
    }

    // Fix rounding
    const total = updated.reduce((s, f) => s + f.weight, 0);
    const diff = 100 - total;
    if (diff !== 0) {
      for (let i = 0; i < n; i++) {
        if (i !== changedIdx) {
          updated[i].weight += diff;
          break;
        }
      }
    }

    setLocalFactors(updated);
  };

  const weightsMutation = useMutation({
    mutationFn: () =>
      updateFactorWeights(sector, displayFactors.map((f) => f.weight)),
    onSuccess: () => {
      setLocalFactors(null);
      onRefresh();
    },
  });

  const handleGenerate = () => {
    setGenerating(true);
    setGenText("");
    setGenError("");

    createSSEConnection(factorGenerateURL(sector), {}, {
      onChunk: (content) => setGenText((prev) => prev + content),
      onDone: () => {
        setGenerating(false);
      },
      onError: (msg) => {
        setGenerating(false);
        setGenError(msg);
      },
      onFactors: () => {
        queryClient.invalidateQueries({ queryKey: ["all-factors"] });
        onRefresh();
      },
    });
  };

  const handleFactorsUpdated = useCallback(
    () => {
      queryClient.invalidateQueries({ queryKey: ["all-factors"] });
      onRefresh();
    },
    [queryClient, onRefresh],
  );

  // No factors yet - show generation prompt
  if (factors.length === 0 && !generating) {
    return (
      <div>
        <div className="rounded-lg border border-border bg-bg-surface p-6 text-center">
          <p className="text-sm text-text-secondary mb-4">
            No factors generated for &ldquo;{sector}&rdquo; yet. Let AI analyze the sector.
          </p>
          <button
            onClick={handleGenerate}
            className="px-5 py-2 rounded-md bg-blue text-white text-sm font-medium hover:bg-blue/80 transition-colors"
          >
            AI Generate Factors
          </button>
          {genError && (
            <p className="mt-3 text-sm text-red">{genError}</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div>
      {/* Generation in progress */}
      {generating && (
        <div className="mb-5">
          <h3 className="text-sm font-semibold text-text-primary mb-2">Generating factors...</h3>
          <SSEStreamText text={genText} streaming={generating} />
        </div>
      )}

      {/* Factor cards */}
      {displayFactors.length > 0 && (
        <>
          <h3 className="text-sm font-semibold text-text-primary mb-3">Core Driving Factors</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
            {displayFactors.map((factor, i) => (
              <FactorCard
                key={`${sector}-${i}`}
                factor={factor}
                index={i}
                onWeightChange={handleWeightChange}
              />
            ))}
          </div>

          <div className="flex items-center gap-3 mb-5">
            <button
              onClick={() => weightsMutation.mutate()}
              disabled={!localFactors || weightsMutation.isPending}
              className="px-4 py-2 rounded-md bg-blue text-white text-sm hover:bg-blue/80 transition-colors disabled:opacity-40"
            >
              {weightsMutation.isPending ? "Saving..." : "Save Weights"}
            </button>
            <button
              onClick={handleGenerate}
              disabled={generating}
              className="px-4 py-2 rounded-md bg-bg-hover text-text-secondary text-sm hover:text-text-primary transition-colors disabled:opacity-40"
            >
              Regenerate
            </button>
            {weightsMutation.isSuccess && (
              <span className="text-xs text-green">Saved</span>
            )}
          </div>

          {/* Chat */}
          <div className="border-t border-border pt-4">
            <h3 className="text-sm font-semibold text-text-primary mb-1">AI Chat</h3>
            <p className="text-xs text-text-secondary mb-3">
              Discuss and adjust factors through conversation. AI can modify factor count, weights, and descriptions.
            </p>
            <ChatPanel
              sseURL={factorChatURL(sector)}
              messages={chatMessages}
              onMessagesChange={setChatMessages}
              onFactorsUpdated={handleFactorsUpdated}
              placeholder={`Discuss "${sector}" factors...`}
            />
          </div>
        </>
      )}
    </div>
  );
}

// ============ Saved Sectors Overview ============

function SavedSectorsOverview({ sectors }: { sectors: Record<string, SectorData> }) {
  const [expandedSector, setExpandedSector] = useState<string | null>(null);

  const withFactors = Object.entries(sectors).filter(
    ([, data]) => data.factors && data.factors.length > 0,
  );

  if (withFactors.length === 0) return null;

  return (
    <div className="mb-5">
      <h2 className="text-sm font-semibold text-text-primary mb-3">
        Analyzed Sectors ({withFactors.length})
      </h2>
      <div className="space-y-2">
        {withFactors.map(([name, data]) => {
          const summary = data.factors
            .map((f) => `${f.name} ${f.weight}%`)
            .join(" / ");
          const expanded = expandedSector === name;

          return (
            <div
              key={name}
              className="rounded-lg border border-border bg-bg-surface"
            >
              <button
                onClick={() => setExpandedSector(expanded ? null : name)}
                className="w-full text-left px-4 py-3 flex items-center justify-between hover:bg-bg-hover transition-colors rounded-lg"
              >
                <div>
                  <span className="text-sm font-medium text-text-primary">{name}</span>
                  <span className="text-xs text-text-secondary ml-3">{summary}</span>
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
                <div className="px-4 pb-4 grid grid-cols-1 md:grid-cols-3 gap-3">
                  {data.factors.map((f, i) => (
                    <div key={i} className="rounded-md bg-bg p-3 border border-border">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm font-medium text-text-primary">{f.name}</span>
                        <span className="text-sm font-bold text-blue">{f.weight}%</span>
                      </div>
                      <p className="text-xs text-text-secondary">{f.description}</p>
                      <p className="text-xs text-text-secondary mt-1">Source: {f.data_source}</p>
                    </div>
                  ))}
                  {data.updated_at && (
                    <p className="text-xs text-text-secondary col-span-full">
                      Last updated: {data.updated_at}
                    </p>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ============ Main Page ============

export default function FactorsPage() {
  const queryClient = useQueryClient();
  const [selectedSectors, setSelectedSectors] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState(0);

  const presetsQuery = useQuery({
    queryKey: ["preset-sectors"],
    queryFn: getPresetSectors,
  });

  const factorsQuery = useQuery({
    queryKey: ["all-factors"],
    queryFn: getAllFactors,
  });

  const handleRefresh = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["all-factors"] });
  }, [queryClient]);

  if (presetsQuery.isLoading || factorsQuery.isLoading) {
    return <LoadingState message="Loading factors data..." />;
  }

  const presets = presetsQuery.data?.sectors ?? [];
  const allSectors = factorsQuery.data?.sectors ?? {};

  return (
    <div>
      <PageHeader
        title="Sector Factors"
        description="Select cyclical sectors and let AI generate core driving factors. Adjust weights and chat to refine."
      />

      <SectorSelector
        presets={presets}
        selected={selectedSectors}
        onSelectedChange={(s) => {
          setSelectedSectors(s);
          setActiveTab(0);
        }}
      />

      <SavedSectorsOverview sectors={allSectors} />

      {/* Active sector tabs */}
      {selectedSectors.length > 0 && (
        <div>
          {/* Tab buttons */}
          {selectedSectors.length > 1 && (
            <div className="flex gap-1 mb-4 border-b border-border">
              {selectedSectors.map((s, i) => (
                <button
                  key={s}
                  onClick={() => setActiveTab(i)}
                  className={`px-4 py-2 text-sm transition-colors border-b-2 -mb-px ${
                    i === activeTab
                      ? "border-blue text-blue"
                      : "border-transparent text-text-secondary hover:text-text-primary"
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          )}

          {/* Tab content */}
          <SectorTab
            key={selectedSectors[activeTab]}
            sector={selectedSectors[activeTab]}
            data={allSectors[selectedSectors[activeTab]]}
            onRefresh={handleRefresh}
          />
        </div>
      )}

      {selectedSectors.length === 0 && Object.keys(allSectors).length === 0 && (
        <div className="rounded-lg border border-border bg-bg-surface p-8 text-center text-text-secondary text-sm">
          Select at least one sector to begin analysis.
        </div>
      )}
    </div>
  );
}
