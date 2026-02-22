"use client";

import { useState, useRef } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import PageHeader from "@/components/layout/PageHeader";
import LoadingState from "@/components/shared/LoadingState";
import DataTable, { type Column } from "@/components/shared/DataTable";
import {
  getPortfolioState,
  updatePortfolioState,
  importPortfolio,
  getTargetPositions,
} from "@/lib/api";
import type { Holding, TargetPosition } from "@/lib/types";

// ============ Add Holding Form ============

function AddHoldingForm({ onAdd }: { onAdd: (code: string, holding: Holding) => void }) {
  const [code, setCode] = useState("");
  const [name, setName] = useState("");
  const [avgCost, setAvgCost] = useState("");
  const [shares, setShares] = useState("");
  const [weight, setWeight] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!code.trim()) return;
    onAdd(code.trim(), {
      name: name.trim() || undefined,
      avg_cost: Number(avgCost) || 0,
      shares: Number(shares) || 0,
      weight: Number(weight) || 0,
      updated_at: new Date().toISOString().slice(0, 16).replace("T", " "),
    });
    setCode("");
    setName("");
    setAvgCost("");
    setShares("");
    setWeight("");
  };

  return (
    <form onSubmit={handleSubmit} className="rounded-lg border border-border bg-bg-surface p-4">
      <h3 className="text-sm font-semibold text-text-primary mb-3">Add Holding</h3>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div>
          <label className="block text-xs text-text-secondary mb-1">Code *</label>
          <input
            type="text"
            value={code}
            onChange={(e) => setCode(e.target.value)}
            placeholder="600000"
            className="w-full px-2 py-1.5 rounded-md border border-border bg-bg text-sm text-text-primary"
            required
          />
        </div>
        <div>
          <label className="block text-xs text-text-secondary mb-1">Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full px-2 py-1.5 rounded-md border border-border bg-bg text-sm text-text-primary"
          />
        </div>
        <div>
          <label className="block text-xs text-text-secondary mb-1">Avg Cost</label>
          <input
            type="number"
            value={avgCost}
            onChange={(e) => setAvgCost(e.target.value)}
            step="0.01"
            className="w-full px-2 py-1.5 rounded-md border border-border bg-bg text-sm text-text-primary"
          />
        </div>
        <div>
          <label className="block text-xs text-text-secondary mb-1">Shares</label>
          <input
            type="number"
            value={shares}
            onChange={(e) => setShares(e.target.value)}
            step="100"
            className="w-full px-2 py-1.5 rounded-md border border-border bg-bg text-sm text-text-primary"
          />
        </div>
        <div>
          <label className="block text-xs text-text-secondary mb-1">Weight</label>
          <input
            type="number"
            value={weight}
            onChange={(e) => setWeight(e.target.value)}
            step="0.01"
            min="0"
            max="1"
            className="w-full px-2 py-1.5 rounded-md border border-border bg-bg text-sm text-text-primary"
          />
        </div>
      </div>
      <button
        type="submit"
        className="mt-3 px-4 py-2 rounded-md bg-blue text-white text-sm hover:bg-blue/80 transition-colors"
      >
        Add
      </button>
    </form>
  );
}

// ============ CSV Import ============

function CSVImport({ onImport }: { onImport: (csv: string) => void }) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [csvContent, setCsvContent] = useState("");

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setCsvContent(reader.result as string);
    };
    reader.readAsText(file);
  };

  return (
    <div className="rounded-lg border border-border bg-bg-surface p-4">
      <h3 className="text-sm font-semibold text-text-primary mb-3">CSV Import</h3>
      <div className="flex items-center gap-3">
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          onChange={handleFile}
          className="text-xs text-text-secondary"
        />
        <button
          onClick={() => {
            if (csvContent) onImport(csvContent);
          }}
          disabled={!csvContent}
          className="px-4 py-2 rounded-md bg-blue text-white text-sm hover:bg-blue/80 transition-colors disabled:opacity-40"
        >
          Import
        </button>
      </div>
      <p className="text-xs text-text-secondary mt-2">
        CSV format: code, name, avg_cost, shares, weight (with header row)
      </p>
    </div>
  );
}

// ============ Main Page ============

export default function PortfolioPage() {
  const queryClient = useQueryClient();

  const portfolioQuery = useQuery({
    queryKey: ["portfolio-state"],
    queryFn: getPortfolioState,
  });

  const targetsQuery = useQuery({
    queryKey: ["target-positions"],
    queryFn: getTargetPositions,
  });

  const holdings = portfolioQuery.data?.holdings ?? {};
  const targets = targetsQuery.data?.targets ?? [];

  const handleAddHolding = async (code: string, holding: Holding) => {
    const updated = { ...holdings, [code]: holding };
    await updatePortfolioState(updated);
    queryClient.invalidateQueries({ queryKey: ["portfolio-state"] });
  };

  const handleCSVImport = async (csv: string) => {
    // Parse CSV into structured items
    const lines = csv.trim().split("\n");
    const items = lines.slice(1).map((line) => {
      const parts = line.split(",").map((s) => s.trim());
      return {
        code: parts[0] ?? "",
        name: parts[1] ?? "",
        avg_cost: Number(parts[2]) || 0,
        shares: Number(parts[3]) || 0,
      };
    }).filter((item) => item.code);
    if (items.length > 0) {
      await importPortfolio(items);
      queryClient.invalidateQueries({ queryKey: ["portfolio-state"] });
    }
  };

  if (portfolioQuery.isLoading) {
    return <LoadingState message="Loading portfolio..." />;
  }

  // Build merged table: target vs actual
  const targetMap = new Map(targets.map((t: TargetPosition) => [t.code, t]));
  const allCodes = new Set([...Object.keys(holdings), ...targets.map((t: TargetPosition) => t.code)]);

  const rows = Array.from(allCodes).map((code) => {
    const h = holdings[code] as Holding | undefined;
    const t = targetMap.get(code) as TargetPosition | undefined;
    const actualWeight = h?.weight ?? 0;
    const targetWeight = t?.target_weight ?? 0;
    return {
      code,
      name: h?.name ?? t?.name ?? code,
      industry: t?.industry ?? "-",
      target_weight: targetWeight,
      actual_weight: actualWeight,
      avg_cost: h?.avg_cost ?? 0,
      shares: h?.shares ?? 0,
      difference: actualWeight - targetWeight,
    };
  });

  const columns: Column<Record<string, unknown>>[] = [
    { key: "code", label: "Code" },
    { key: "name", label: "Name" },
    { key: "industry", label: "Industry" },
    {
      key: "target_weight",
      label: "Target %",
      sortable: true,
      render: (r) => `${(Number(r.target_weight) * 100).toFixed(1)}%`,
    },
    {
      key: "actual_weight",
      label: "Actual %",
      sortable: true,
      render: (r) => `${(Number(r.actual_weight) * 100).toFixed(1)}%`,
    },
    {
      key: "avg_cost",
      label: "Avg Cost",
      render: (r) => Number(r.avg_cost) > 0 ? Number(r.avg_cost).toFixed(2) : "-",
    },
    { key: "shares", label: "Shares" },
    {
      key: "difference",
      label: "Diff",
      sortable: true,
      render: (r) => {
        const diff = Number(r.difference);
        const color = diff > 0.01 ? "text-green" : diff < -0.01 ? "text-red" : "text-text-secondary";
        return <span className={color}>{(diff * 100).toFixed(1)}%</span>;
      },
    },
  ];

  return (
    <div>
      <PageHeader
        title="Portfolio"
        description="Track positions, target vs actual allocation, and import holdings."
      />

      {/* Advisor sync indicator */}
      {Object.keys(holdings).length > 0 && (
        <div className="rounded-md bg-blue/10 border border-blue/20 p-2 mb-4">
          <p className="text-xs text-blue">
            Portfolio has {Object.keys(holdings).length} holdings.
            {targets.length > 0 && ` ${targets.length} target positions from advisor.`}
          </p>
        </div>
      )}

      {/* Target vs Actual table */}
      <div className="mb-5">
        <h2 className="text-sm font-semibold text-text-primary mb-3">
          Target vs Actual Positions
        </h2>
        <DataTable
          data={rows as Record<string, unknown>[]}
          columns={columns}
          emptyMessage="No holdings or target positions. Add holdings below."
        />
      </div>

      {/* Add holding form */}
      <div className="mb-5">
        <AddHoldingForm onAdd={handleAddHolding} />
      </div>

      {/* CSV import */}
      <CSVImport onImport={handleCSVImport} />
    </div>
  );
}
