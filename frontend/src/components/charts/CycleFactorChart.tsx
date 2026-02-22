"use client";

import ReactECharts from "echarts-for-react";
import type { CycleFactorData } from "@/lib/types";

interface CycleFactorChartProps {
  factor: CycleFactorData;
}

interface ChartPoint {
  date: string;
  value: number;
  type: "peak" | "trough" | "normal";
}

export default function CycleFactorChart({ factor }: CycleFactorChartProps) {
  const points: ChartPoint[] = [];

  for (const cycle of factor.cycle_data) {
    if (cycle.peak?.date && cycle.peak.value != null) {
      points.push({ date: cycle.peak.date, value: cycle.peak.value, type: "peak" });
    }
    if (cycle.trough?.date && cycle.trough.value != null) {
      points.push({ date: cycle.trough.date, value: cycle.trough.value, type: "trough" });
    }
    for (const kp of cycle.key_points ?? []) {
      if (kp.date && kp.value != null) {
        points.push({ date: kp.date, value: kp.value, type: "normal" });
      }
    }
  }

  if (points.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-bg-surface p-4 text-center text-text-secondary text-xs">
        No cycle data available for this factor
      </div>
    );
  }

  // Dedupe and sort
  const seen = new Map<string, ChartPoint>();
  const priority = { peak: 0, trough: 1, normal: 2 };
  for (const p of points) {
    const existing = seen.get(p.date);
    if (!existing || priority[p.type] < priority[existing.type]) {
      seen.set(p.date, p);
    }
  }
  const sorted = Array.from(seen.values()).sort((a, b) => a.date.localeCompare(b.date));

  const dates = sorted.map((p) => p.date);
  const values = sorted.map((p) => p.value);

  const peaks = sorted.filter((p) => p.type === "peak");
  const troughs = sorted.filter((p) => p.type === "trough");

  const markPoints = [
    ...peaks.map((p) => ({
      coord: [p.date, p.value],
      value: p.value,
      symbol: "triangle",
      symbolRotate: 180,
      symbolSize: 12,
      itemStyle: { color: "#ff1744" },
      label: { show: true, position: "top" as const, color: "#ff1744", fontSize: 10 },
    })),
    ...troughs.map((p) => ({
      coord: [p.date, p.value],
      value: p.value,
      symbol: "triangle",
      symbolSize: 12,
      itemStyle: { color: "#00c853" },
      label: { show: true, position: "bottom" as const, color: "#00c853", fontSize: 10 },
    })),
  ];

  const option = {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" as const },
    xAxis: {
      type: "category" as const,
      data: dates,
      axisLabel: { color: "#8b8fa3", fontSize: 10 },
      axisLine: { lineStyle: { color: "#2a2d3a" } },
    },
    yAxis: {
      type: "value" as const,
      name: factor.unit || undefined,
      nameTextStyle: { color: "#8b8fa3" },
      axisLabel: { color: "#8b8fa3", fontSize: 10 },
      splitLine: { lineStyle: { color: "#2a2d3a" } },
    },
    series: [
      {
        type: "line" as const,
        data: values,
        lineStyle: { color: "#2196f3", width: 2 },
        itemStyle: { color: "#2196f3" },
        symbolSize: 6,
        markPoint: { data: markPoints },
      },
    ],
    grid: { left: 50, right: 20, top: 30, bottom: 30 },
  };

  return (
    <ReactECharts
      option={option}
      style={{ height: 300 }}
      opts={{ renderer: "canvas" }}
    />
  );
}
