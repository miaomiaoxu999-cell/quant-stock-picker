"use client";

import { useRef, useEffect } from "react";

interface ProgressTrackerProps {
  title: string;
  isRunning: boolean;
  logs: string[];
  currentStep?: string;
  factorIndex?: number;
  factorTotal?: number;
  currentFactor?: string;
  onCancel?: () => void;
}

export default function ProgressTracker({
  title,
  isRunning,
  logs,
  currentStep,
  factorIndex = 0,
  factorTotal = 0,
  currentFactor,
  onCancel,
}: ProgressTrackerProps) {
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const progress = factorTotal > 0 ? Math.round((factorIndex / factorTotal) * 100) : 0;

  return (
    <div className="rounded-lg border border-border bg-bg-surface p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {isRunning && (
            <svg className="w-4 h-4 animate-spin text-blue" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          )}
          <span className="text-sm font-semibold text-text-primary">{title}</span>
        </div>
        {isRunning && onCancel && (
          <button
            onClick={onCancel}
            className="px-3 py-1 rounded-md bg-red/20 text-red text-xs hover:bg-red/30 transition-colors"
          >
            Cancel
          </button>
        )}
      </div>

      {/* Current step info */}
      {isRunning && (
        <div className="mb-3">
          {currentStep && (
            <p className="text-xs text-text-secondary mb-1">Step: {currentStep}</p>
          )}
          {factorTotal > 0 && (
            <div>
              <div className="flex items-center justify-between text-xs text-text-secondary mb-1">
                <span>
                  Factor {factorIndex}/{factorTotal}
                  {currentFactor && ` - ${currentFactor}`}
                </span>
                <span>{progress}%</span>
              </div>
              <div className="w-full h-1.5 bg-border rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Log list */}
      {logs.length > 0 && (
        <div className="max-h-48 overflow-y-auto rounded-md bg-bg p-2 border border-border">
          {logs.map((log, i) => (
            <p key={i} className="text-xs text-text-secondary py-0.5 font-mono">
              {log}
            </p>
          ))}
          <div ref={logEndRef} />
        </div>
      )}
    </div>
  );
}
