"use client";

import ReactECharts from "echarts-for-react";

interface StockPBData {
  name: string;
  currentPB: number;
  histHighPB: number;
  histLowPB: number;
}

interface PBComparisonChartProps {
  stocks: StockPBData[];
}

export default function PBComparisonChart({ stocks }: PBComparisonChartProps) {
  if (stocks.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-bg-surface p-4 text-center text-text-secondary text-xs">
        No valuation data available
      </div>
    );
  }

  const names = stocks.map((s) => s.name);

  const option = {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" as const },
    legend: {
      data: ["Current PB", "Historical High PB", "Historical Low PB"],
      textStyle: { color: "#8b8fa3", fontSize: 11 },
      top: 0,
    },
    xAxis: {
      type: "category" as const,
      data: names,
      axisLabel: { color: "#8b8fa3", fontSize: 10, rotate: names.length > 6 ? 30 : 0 },
      axisLine: { lineStyle: { color: "#2a2d3a" } },
    },
    yAxis: {
      type: "value" as const,
      name: "PB",
      nameTextStyle: { color: "#8b8fa3" },
      axisLabel: { color: "#8b8fa3", fontSize: 10 },
      splitLine: { lineStyle: { color: "#2a2d3a" } },
    },
    series: [
      {
        name: "Current PB",
        type: "bar" as const,
        data: stocks.map((s) => s.currentPB),
        itemStyle: { color: "#2196f3" },
        barGap: "10%",
        label: { show: true, position: "top" as const, color: "#8b8fa3", fontSize: 9 },
      },
      {
        name: "Historical High PB",
        type: "bar" as const,
        data: stocks.map((s) => s.histHighPB),
        itemStyle: { color: "#ff1744" },
        label: { show: true, position: "top" as const, color: "#8b8fa3", fontSize: 9 },
      },
      {
        name: "Historical Low PB",
        type: "bar" as const,
        data: stocks.map((s) => s.histLowPB),
        itemStyle: { color: "#00c853" },
        label: { show: true, position: "top" as const, color: "#8b8fa3", fontSize: 9 },
      },
    ],
    grid: { left: 50, right: 20, top: 40, bottom: names.length > 6 ? 60 : 30 },
  };

  return (
    <ReactECharts
      option={option}
      style={{ height: 400 }}
      opts={{ renderer: "canvas" }}
    />
  );
}
