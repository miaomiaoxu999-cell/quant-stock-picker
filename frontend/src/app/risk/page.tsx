"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import PageHeader from "@/components/layout/PageHeader";
import LoadingState from "@/components/shared/LoadingState";
import DataTable, { type Column } from "@/components/shared/DataTable";
import MetricCard from "@/components/shared/MetricCard";
import { getPortfolioRisk, getRealtimePrices } from "@/lib/api";
import type { RiskCheckItem } from "@/lib/types";

function drawdownColor(drawdown: number): string {
  if (drawdown >= -0.05) return "text-green";
  if (drawdown >= -0.15) return "text-warning";
  return "text-red";
}

function levelBadge(level: number): string {
  if (level >= 2) return "bg-red/20 text-red";
  if (level >= 1) return "bg-warning/20 text-warning";
  return "bg-bg-hover text-text-secondary";
}

function levelLabel(level: number): string {
  if (level >= 2) return "L2";
  if (level >= 1) return "L1";
  return "OK";
}

export default function RiskPage() {
  const queryClient = useQueryClient();
  const [refreshing, setRefreshing] = useState(false);

  const riskQuery = useQuery({
    queryKey: ["portfolio-risk"],
    queryFn: getPortfolioRisk,
  });

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await getRealtimePrices();
      queryClient.invalidateQueries({ queryKey: ["portfolio-risk"] });
    } finally {
      setRefreshing(false);
    }
  };

  if (riskQuery.isLoading) {
    return <LoadingState message="Loading risk data..." />;
  }

  const items = riskQuery.data?.items ?? [];

  const triggered = items.filter((a: RiskCheckItem) => a.stop_loss_triggered);
  const sellSignals = items.filter((a: RiskCheckItem) => a.should_sell);
  const safe = items.filter((a: RiskCheckItem) => !a.stop_loss_triggered && !a.should_sell);

  const columns: Column<Record<string, unknown>>[] = [
    { key: "name", label: "Stock" },
    { key: "code", label: "Code" },
    {
      key: "drawdown",
      label: "Drawdown",
      sortable: true,
      render: (r) => {
        const dd = Number(r.drawdown);
        return (
          <span className={drawdownColor(dd)}>
            {(dd * 100).toFixed(1)}%
          </span>
        );
      },
    },
    {
      key: "level",
      label: "Level",
      render: (r) => {
        const level = Number(r.level);
        return (
          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${levelBadge(level)}`}>
            {levelLabel(level)}
          </span>
        );
      },
    },
    {
      key: "action",
      label: "Action",
      render: (r) => (
        <span className="text-xs font-medium">{String(r.action || "-")}</span>
      ),
    },
    {
      key: "should_sell",
      label: "Sell Signal",
      render: (r) => {
        const shouldSell = Boolean(r.should_sell);
        return shouldSell ? (
          <span className="text-red text-xs font-medium">SELL ({(Number(r.sell_ratio) * 100).toFixed(0)}%)</span>
        ) : (
          <span className="text-text-secondary text-xs">-</span>
        );
      },
    },
    {
      key: "sell_reason",
      label: "Reason",
      render: (r) => (
        <span className="text-xs text-text-secondary">{String(r.sell_reason || "-")}</span>
      ),
    },
  ];

  return (
    <div>
      <PageHeader
        title="Risk Management"
        description="Stop-loss monitoring, drawdown alerts, and sell signal tracking."
        actions={
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="px-4 py-2 rounded-md bg-blue text-white text-sm hover:bg-blue/80 transition-colors disabled:opacity-40"
          >
            {refreshing ? "Refreshing..." : "Refresh Prices"}
          </button>
        }
      />

      {/* Summary metrics */}
      <div className="grid grid-cols-4 gap-3 mb-5">
        <MetricCard
          label="Total Monitored"
          value={items.length}
          color="blue"
        />
        <MetricCard
          label="Triggered Alerts"
          value={triggered.length}
          color={triggered.length > 0 ? "red" : "green"}
        />
        <MetricCard
          label="Safe Positions"
          value={safe.length}
          color="green"
        />
        <MetricCard
          label="Sell Signals"
          value={sellSignals.length}
          color={sellSignals.length > 0 ? "yellow" : "green"}
        />
      </div>

      {/* Stop-loss monitoring */}
      <div className="mb-5">
        <h2 className="text-sm font-semibold text-text-primary mb-3">Stop-Loss Monitoring</h2>

        {/* Explanation */}
        <div className="rounded-md bg-bg-surface border border-border p-3 mb-3">
          <div className="grid grid-cols-2 gap-3 text-xs text-text-secondary">
            <div>
              <span className="text-warning font-medium">L1</span>: -15% drawdown
              &rarr; Reduce 50%
            </div>
            <div>
              <span className="text-red font-medium">L2</span>: -25% drawdown
              &rarr; Liquidate
            </div>
          </div>
        </div>

        {items.length > 0 ? (
          <DataTable
            data={items as unknown as Record<string, unknown>[]}
            columns={columns}
          />
        ) : (
          <div className="rounded-lg border border-border bg-bg-surface p-6 text-center text-text-secondary text-sm">
            No positions to monitor. Add holdings in the Portfolio page.
          </div>
        )}
      </div>
    </div>
  );
}
