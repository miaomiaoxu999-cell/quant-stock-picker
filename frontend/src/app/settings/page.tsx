"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import PageHeader from "@/components/layout/PageHeader";
import LoadingState from "@/components/shared/LoadingState";
import { getLLMSettings, updateLLMSettings } from "@/lib/api";
import type { LLMSettings } from "@/lib/types";

function PasswordField({
  label,
  value,
  onChange,
  help,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  help?: string;
}) {
  const [visible, setVisible] = useState(false);
  return (
    <div>
      <label className="block text-sm text-text-primary mb-1">{label}</label>
      <div className="relative">
        <input
          type={visible ? "text" : "password"}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full pr-10"
        />
        <button
          type="button"
          onClick={() => setVisible(!visible)}
          className="absolute right-2 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary text-xs"
        >
          {visible ? "Hide" : "Show"}
        </button>
      </div>
      {help && <p className="text-xs text-text-secondary mt-1">{help}</p>}
    </div>
  );
}

function TextField({
  label,
  value,
  onChange,
  help,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  help?: string;
}) {
  return (
    <div>
      <label className="block text-sm text-text-primary mb-1">{label}</label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full"
      />
      {help && <p className="text-xs text-text-secondary mt-1">{help}</p>}
    </div>
  );
}

export default function SettingsPage() {
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ["llm-settings"],
    queryFn: getLLMSettings,
  });

  const [form, setForm] = useState<LLMSettings | null>(null);
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");

  // Initialize form from query data
  if (data && !form) {
    setForm({ ...data });
  }

  const mutation = useMutation({
    mutationFn: updateLLMSettings,
    onSuccess: (updated) => {
      queryClient.setQueryData(["llm-settings"], updated);
      setSaveStatus("saved");
      setTimeout(() => setSaveStatus("idle"), 2000);
    },
    onError: () => {
      setSaveStatus("error");
      setTimeout(() => setSaveStatus("idle"), 3000);
    },
  });

  const handleSave = () => {
    if (!form) return;
    setSaveStatus("saving");
    mutation.mutate(form);
  };

  const update = (field: keyof LLMSettings, value: string) => {
    if (!form) return;
    setForm({ ...form, [field]: value });
  };

  if (isLoading) return <LoadingState message="Loading settings..." />;

  if (error) {
    return (
      <div>
        <PageHeader title="Settings" />
        <div className="rounded-lg border border-red/30 bg-red/5 p-4 text-red text-sm">
          Failed to load settings. Is the backend running?
        </div>
      </div>
    );
  }

  if (!form) return null;

  return (
    <div className="max-w-2xl">
      <PageHeader
        title="Settings"
        description="Configure LLM API connection, search engines, and audit model."
      />

      {/* LLM API section */}
      <section className="rounded-lg border border-border bg-bg-surface p-5 mb-5">
        <h2 className="text-base font-semibold text-text-primary mb-1">LLM API</h2>
        <p className="text-xs text-text-secondary mb-4">
          SiliconFlow or other OpenAI-compatible API for AI factor generation and cycle analysis.
        </p>
        <div className="space-y-4">
          <PasswordField
            label="API Key"
            value={form.api_key}
            onChange={(v) => update("api_key", v)}
            help="SiliconFlow or OpenAI-compatible service API key"
          />
          <TextField
            label="API URL"
            value={form.base_url}
            onChange={(v) => update("base_url", v)}
            help="e.g. https://api.siliconflow.cn/v1"
          />
          <TextField
            label="Model"
            value={form.model}
            onChange={(v) => update("model", v)}
            help="e.g. Pro/zai-org/GLM-5"
          />
        </div>
      </section>

      {/* Search Engines section */}
      <section className="rounded-lg border border-border bg-bg-surface p-5 mb-5">
        <h2 className="text-base font-semibold text-text-primary mb-1">Search Engines & Data</h2>
        <p className="text-xs text-text-secondary mb-4">
          Used for cycle analysis page data retrieval. Leave empty to skip a source.
        </p>
        <div className="space-y-4">
          <PasswordField
            label="Tavily API Key"
            value={form.tavily_api_key}
            onChange={(v) => update("tavily_api_key", v)}
            help="Tavily AI search for industry data"
          />
          <PasswordField
            label="Jina Reader API Key"
            value={form.jina_api_key}
            onChange={(v) => update("jina_api_key", v)}
            help="Jina Reader for web content extraction"
          />
          <PasswordField
            label="Apify API Key"
            value={form.apify_api_key}
            onChange={(v) => update("apify_api_key", v)}
            help="Apify web scraper (fallback)"
          />
        </div>
      </section>

      {/* Audit model section */}
      <section className="rounded-lg border border-border bg-bg-surface p-5 mb-5">
        <h2 className="text-base font-semibold text-text-primary mb-1">Audit Model (Optional)</h2>
        <p className="text-xs text-text-secondary mb-4">
          Independent audit agent model. Leave API Key empty to reuse main LLM config.
        </p>
        <div className="space-y-4">
          <PasswordField
            label="Audit API Key"
            value={form.audit_api_key}
            onChange={(v) => update("audit_api_key", v)}
            help="Optional. Leave empty to reuse main LLM key"
          />
          <TextField
            label="Audit API URL"
            value={form.audit_base_url}
            onChange={(v) => update("audit_base_url", v)}
          />
          <TextField
            label="Audit Model"
            value={form.audit_model}
            onChange={(v) => update("audit_model", v)}
            help="Recommended: deepseek-ai/DeepSeek-V3"
          />
        </div>
      </section>

      {/* Save button */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleSave}
          disabled={saveStatus === "saving"}
          className="px-5 py-2 rounded-md bg-blue text-white text-sm font-medium hover:bg-blue/80 transition-colors disabled:opacity-50"
        >
          {saveStatus === "saving" ? "Saving..." : "Save Settings"}
        </button>
        {saveStatus === "saved" && (
          <span className="text-sm text-green">Settings saved</span>
        )}
        {saveStatus === "error" && (
          <span className="text-sm text-red">Failed to save</span>
        )}
      </div>
    </div>
  );
}
