"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import type { ChatMessage } from "@/lib/types";
import { createSSEConnection } from "@/lib/sse";
import SSEStreamText from "./SSEStreamText";

interface ChatPanelProps {
  /** SSE endpoint URL to POST chat messages to. */
  sseURL: string;
  /** Extra body fields merged into each SSE request (e.g. history). */
  extraBody?: Record<string, unknown>;
  /** Chat history to display. */
  messages: ChatMessage[];
  /** Called when new messages are added (user + assistant). */
  onMessagesChange: (messages: ChatMessage[]) => void;
  /** Called when the AI reply includes updated factors. */
  onFactorsUpdated?: (factors: unknown) => void;
  /** Placeholder text for input. */
  placeholder?: string;
}

export default function ChatPanel({
  sseURL,
  extraBody,
  messages,
  onMessagesChange,
  onFactorsUpdated,
  placeholder = "Type a message...",
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [streamingText, setStreamingText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<{ abort: () => void } | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingText, scrollToBottom]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;

    const userMsg: ChatMessage = { role: "user", content: trimmed };
    const updated = [...messages, userMsg];
    onMessagesChange(updated);
    setInput("");
    setStreamingText("");
    setIsStreaming(true);

    const conn = createSSEConnection(
      sseURL,
      {
        message: trimmed,
        history: messages,
        ...extraBody,
      },
      {
        onChunk: (content) => {
          setStreamingText((prev) => prev + content);
        },
        onDone: (fullText) => {
          setIsStreaming(false);
          setStreamingText("");
          const assistantMsg: ChatMessage = { role: "assistant", content: fullText };
          onMessagesChange([...updated, assistantMsg]);
        },
        onError: (message) => {
          setIsStreaming(false);
          setStreamingText("");
          const errorMsg: ChatMessage = {
            role: "assistant",
            content: `[Error] ${message}`,
          };
          onMessagesChange([...updated, errorMsg]);
        },
        onFactors: (event) => {
          if ("factors" in event) {
            onFactorsUpdated?.(event.factors);
          }
        },
      },
    );

    abortRef.current = conn;
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleAbort = () => {
    abortRef.current?.abort();
    setIsStreaming(false);
    setStreamingText("");
  };

  return (
    <div className="flex flex-col h-full">
      {/* Message list */}
      <div className="flex-1 overflow-y-auto space-y-3 mb-4 max-h-96">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-blue/20 text-text-primary"
                  : "bg-bg-surface border border-border text-text-primary"
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}

        {/* Streaming response */}
        {isStreaming && (
          <div className="flex justify-start">
            <div className="max-w-[80%]">
              <SSEStreamText text={streamingText} streaming={true} />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="flex items-center gap-2 border-t border-border pt-3">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isStreaming}
          className="flex-1 text-sm disabled:opacity-50"
        />
        {isStreaming ? (
          <button
            onClick={handleAbort}
            className="px-4 py-2 rounded-md bg-red/20 text-red text-sm hover:bg-red/30 transition-colors"
          >
            Stop
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!input.trim()}
            className="px-4 py-2 rounded-md bg-blue text-white text-sm hover:bg-blue/80 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Send
          </button>
        )}
      </div>
    </div>
  );
}
