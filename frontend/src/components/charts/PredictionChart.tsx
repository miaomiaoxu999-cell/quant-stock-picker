"use client";

import ReactECharts from "echarts-for-react";

interface PredictionData {
  [timeframe: string]: {
    optimistic?: { total_value: number; return_rate: number };
    baseline?: { total_value: number; return_rate: number };
    pessimistic?: { total_value: number; return_rate: number };
  };
}

interface PredictionChartProps {
  predictions: PredictionData;
  totalCapital: number;
}

export default function PredictionChart({
  predictions,
  totalCapital,
}: PredictionChartProps) {
  const periods = ["3m", "6m", "12m"];
  const periodLabels = ["3 Months", "6 Months", "12 Months"];

  const optimistic = periods.map(
    (p) => predictions[p]?.optimistic?.total_value ?? totalCapital,
  );
  const baseline = periods.map(
    (p) => predictions[p]?.baseline?.total_value ?? totalCapital,
  );
  const pessimistic = periods.map(
    (p) => predictions[p]?.pessimistic?.total_value ?? totalCapital,
  );

  const option = {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" as const },
    legend: {
      data: ["Optimistic", "Baseline", "Pessimistic"],
      textStyle: { color: "#8b8fa3", fontSize: 11 },
      top: 0,
    },
    xAxis: {
      type: "category" as const,
      data: periodLabels,
      axisLabel: { color: "#8b8fa3", fontSize: 11 },
      axisLine: { lineStyle: { color: "#2a2d3a" } },
    },
    yAxis: {
      type: "value" as const,
      name: "Total Value",
      nameTextStyle: { color: "#8b8fa3" },
      axisLabel: {
        color: "#8b8fa3",
        fontSize: 10,
        formatter: (v: number) => `${(v / 10000).toFixed(0)}w`,
      },
      splitLine: { lineStyle: { color: "#2a2d3a" } },
    },
    series: [
      {
        name: "Optimistic",
        type: "bar" as const,
        data: optimistic,
        itemStyle: { color: "#00c853" },
        barGap: "10%",
        label: {
          show: true,
          position: "top" as const,
          color: "#8b8fa3",
          fontSize: 9,
          formatter: (p: { value: number }) => `${(p.value / 10000).toFixed(1)}w`,
        },
      },
      {
        name: "Baseline",
        type: "bar" as const,
        data: baseline,
        itemStyle: { color: "#2196f3" },
        label: {
          show: true,
          position: "top" as const,
          color: "#8b8fa3",
          fontSize: 9,
          formatter: (p: { value: number }) => `${(p.value / 10000).toFixed(1)}w`,
        },
      },
      {
        name: "Pessimistic",
        type: "bar" as const,
        data: pessimistic,
        itemStyle: { color: "#ff1744" },
        label: {
          show: true,
          position: "top" as const,
          color: "#8b8fa3",
          fontSize: 9,
          formatter: (p: { value: number }) => `${(p.value / 10000).toFixed(1)}w`,
        },
      },
    ],
    grid: { left: 60, right: 20, top: 40, bottom: 30 },
  };

  return (
    <ReactECharts
      option={option}
      style={{ height: 400 }}
      opts={{ renderer: "canvas" }}
    />
  );
}
