"use client";

import ReactECharts from "echarts-for-react";
import type { ChartDataPoint } from "@/lib/types";

interface CombinedStockChartProps {
  code: string;
  name: string;
  pbData: ChartDataPoint[];
  histPeakPB?: number;
  histTroughPB?: number;
}

export default function CombinedStockChart({
  name,
  pbData,
  histPeakPB,
  histTroughPB,
}: CombinedStockChartProps) {
  if (pbData.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-bg-surface p-4 text-center text-text-secondary text-xs">
        No historical data available
      </div>
    );
  }

  const dates = pbData.map((d) => d.date);
  const prices = pbData.map((d) => d.close);
  const pbs = pbData.map((d) => d.pb);

  const markLines = [];
  if (histPeakPB != null) {
    markLines.push({
      yAxis: histPeakPB,
      label: { formatter: `Peak PB ${histPeakPB.toFixed(1)}`, color: "#ff1744", fontSize: 10 },
      lineStyle: { color: "#ff1744", type: "dashed" as const },
    });
  }
  if (histTroughPB != null) {
    markLines.push({
      yAxis: histTroughPB,
      label: { formatter: `Low PB ${histTroughPB.toFixed(1)}`, color: "#00c853", fontSize: 10 },
      lineStyle: { color: "#00c853", type: "dashed" as const },
    });
  }

  const option = {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" as const },
    legend: {
      data: ["Price", "PB"],
      textStyle: { color: "#8b8fa3", fontSize: 11 },
      top: 0,
    },
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    xAxis: [
      {
        type: "category" as const,
        data: dates,
        gridIndex: 0,
        axisLabel: { show: false },
        axisLine: { lineStyle: { color: "#2a2d3a" } },
      },
      {
        type: "category" as const,
        data: dates,
        gridIndex: 1,
        axisLabel: { color: "#8b8fa3", fontSize: 10 },
        axisLine: { lineStyle: { color: "#2a2d3a" } },
      },
    ],
    yAxis: [
      {
        type: "value" as const,
        name: "Price",
        nameTextStyle: { color: "#8b8fa3" },
        gridIndex: 0,
        axisLabel: { color: "#8b8fa3", fontSize: 10 },
        splitLine: { lineStyle: { color: "#2a2d3a" } },
      },
      {
        type: "value" as const,
        name: "PB",
        nameTextStyle: { color: "#8b8fa3" },
        gridIndex: 1,
        axisLabel: { color: "#8b8fa3", fontSize: 10 },
        splitLine: { lineStyle: { color: "#2a2d3a" } },
      },
    ],
    grid: [
      { left: 60, right: 20, top: 30, height: "30%" },
      { left: 60, right: 20, top: "55%", height: "35%" },
    ],
    series: [
      {
        name: "Price",
        type: "line" as const,
        data: prices,
        xAxisIndex: 0,
        yAxisIndex: 0,
        lineStyle: { color: "#2196f3", width: 1.5 },
        itemStyle: { color: "#2196f3" },
        symbolSize: 0,
      },
      {
        name: "PB",
        type: "line" as const,
        data: pbs,
        xAxisIndex: 1,
        yAxisIndex: 1,
        lineStyle: { color: "#FF6B6B", width: 2 },
        itemStyle: { color: "#FF6B6B" },
        symbolSize: 0,
        markLine: markLines.length > 0 ? { data: markLines, silent: true } : undefined,
      },
    ],
  };

  return (
    <div>
      <p className="text-xs text-text-secondary mb-1">{name} - Price & PB History</p>
      <ReactECharts
        option={option}
        style={{ height: 450 }}
        opts={{ renderer: "canvas" }}
      />
    </div>
  );
}
