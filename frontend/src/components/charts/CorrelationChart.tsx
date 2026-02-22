"use client";

import ReactECharts from "echarts-for-react";
import type { ChartDataPoint } from "@/lib/types";

interface CorrelationChartProps {
  factorName: string;
  factorData: { date: string; value: number }[];
  pbData: ChartDataPoint[];
  stockName: string;
}

export default function CorrelationChart({
  factorName,
  factorData,
  pbData,
  stockName,
}: CorrelationChartProps) {
  const factorDates = factorData.map((d) => d.date);
  const pbDates = pbData.map((d) => d.date);

  // Merge all dates for a unified x-axis
  const allDates = Array.from(new Set([...factorDates, ...pbDates])).sort();

  const factorMap = new Map(factorData.map((d) => [d.date, d.value]));
  const pbMap = new Map(pbData.map((d) => [d.date, d.pb]));

  const factorSeries = allDates.map((d) => factorMap.get(d) ?? null);
  const pbSeries = allDates.map((d) => pbMap.get(d) ?? null);

  const option = {
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" as const },
    legend: {
      data: [factorName, `${stockName} PB`],
      textStyle: { color: "#8b8fa3", fontSize: 11 },
      top: 0,
    },
    xAxis: {
      type: "category" as const,
      data: allDates,
      axisLabel: { color: "#8b8fa3", fontSize: 10 },
      axisLine: { lineStyle: { color: "#2a2d3a" } },
    },
    yAxis: [
      {
        type: "value" as const,
        name: factorName,
        nameTextStyle: { color: "#FF9800" },
        axisLabel: { color: "#8b8fa3", fontSize: 10 },
        splitLine: { lineStyle: { color: "#2a2d3a" } },
      },
      {
        type: "value" as const,
        name: "PB",
        nameTextStyle: { color: "#2196f3" },
        axisLabel: { color: "#8b8fa3", fontSize: 10 },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: factorName,
        type: "line" as const,
        data: factorSeries,
        yAxisIndex: 0,
        lineStyle: { color: "#FF9800", width: 1.5, type: "dashed" as const },
        itemStyle: { color: "#FF9800" },
        symbolSize: 4,
        connectNulls: true,
      },
      {
        name: `${stockName} PB`,
        type: "line" as const,
        data: pbSeries,
        yAxisIndex: 1,
        lineStyle: { color: "#2196f3", width: 2 },
        itemStyle: { color: "#2196f3" },
        symbolSize: 0,
        connectNulls: true,
      },
    ],
    grid: { left: 60, right: 60, top: 40, bottom: 30 },
  };

  return (
    <ReactECharts
      option={option}
      style={{ height: 350 }}
      opts={{ renderer: "canvas" }}
    />
  );
}
