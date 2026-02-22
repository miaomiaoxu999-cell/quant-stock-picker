"use client";

import { useEffect, useRef } from "react";

interface SSEStreamTextProps {
  /** Accumulated text being streamed in. */
  text: string;
  /** Whether streaming is still in progress. */
  streaming: boolean;
}

/**
 * Renders streaming LLM text with an animated cursor.
 * Auto-scrolls to keep newest content visible.
 */
export default function SSEStreamText({ text, streaming }: SSEStreamTextProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [text]);

  if (!text && !streaming) return null;

  return (
    <div
      ref={containerRef}
      className="rounded-md bg-bg p-4 border border-border text-sm text-text-primary whitespace-pre-wrap max-h-80 overflow-y-auto font-mono leading-relaxed"
    >
      {text}
      {streaming && (
        <span className="inline-block w-2 h-4 ml-0.5 bg-blue animate-pulse align-text-bottom" />
      )}
    </div>
  );
}
