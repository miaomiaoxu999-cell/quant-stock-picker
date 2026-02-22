import { create } from "zustand";
import type { ChatMessage, Allocation } from "@/lib/types";

interface AdvisorState {
  step: number;
  // Step 1 inputs
  totalCapital: number;
  bullishSectors: string[];
  favoredStocks: { code: string; name: string }[];
  // Step 2
  confirmedCodes: string[];
  diagnosisResponse: string;
  diagnosisJson: Record<string, unknown> | null;
  diagnosisMessages: ChatMessage[];
  // Step 3
  allocationResponse: string;
  allocationJson: Record<string, unknown> | null;
  allocationMessages: ChatMessage[];
  editableAllocation: Allocation[];
  cashReserve: Record<string, unknown>;
  // Step 4
  predictionJson: Record<string, unknown> | null;
  // Step 5
  riskPlanJson: Record<string, unknown> | null;

  // Actions
  setStep: (step: number) => void;
  setInputs: (inputs: {
    totalCapital?: number;
    bullishSectors?: string[];
    favoredStocks?: { code: string; name: string }[];
  }) => void;
  setConfirmedCodes: (codes: string[]) => void;
  setDiagnosis: (response: string, json: Record<string, unknown> | null) => void;
  addDiagnosisMessage: (msg: ChatMessage) => void;
  setAllocation: (response: string, json: Record<string, unknown> | null) => void;
  addAllocationMessage: (msg: ChatMessage) => void;
  setEditableAllocation: (alloc: Allocation[]) => void;
  setCashReserve: (cash: Record<string, unknown>) => void;
  setPrediction: (json: Record<string, unknown> | null) => void;
  setRiskPlan: (json: Record<string, unknown> | null) => void;
  resetSession: () => void;
  restoreFromServer: (session: Record<string, unknown>) => void;
}

const initialState = {
  step: 1,
  totalCapital: 500000,
  bullishSectors: [] as string[],
  favoredStocks: [] as { code: string; name: string }[],
  confirmedCodes: [] as string[],
  diagnosisResponse: "",
  diagnosisJson: null as Record<string, unknown> | null,
  diagnosisMessages: [] as ChatMessage[],
  allocationResponse: "",
  allocationJson: null as Record<string, unknown> | null,
  allocationMessages: [] as ChatMessage[],
  editableAllocation: [] as Allocation[],
  cashReserve: {} as Record<string, unknown>,
  predictionJson: null as Record<string, unknown> | null,
  riskPlanJson: null as Record<string, unknown> | null,
};

export const useAdvisorStore = create<AdvisorState>((set) => ({
  ...initialState,

  setStep: (step) => set({ step }),

  setInputs: (inputs) =>
    set((state) => ({
      totalCapital: inputs.totalCapital ?? state.totalCapital,
      bullishSectors: inputs.bullishSectors ?? state.bullishSectors,
      favoredStocks: inputs.favoredStocks ?? state.favoredStocks,
    })),

  setConfirmedCodes: (codes) => set({ confirmedCodes: codes }),

  setDiagnosis: (response, json) =>
    set({ diagnosisResponse: response, diagnosisJson: json }),

  addDiagnosisMessage: (msg) =>
    set((state) => ({
      diagnosisMessages: [...state.diagnosisMessages, msg],
    })),

  setAllocation: (response, json) =>
    set({ allocationResponse: response, allocationJson: json }),

  addAllocationMessage: (msg) =>
    set((state) => ({
      allocationMessages: [...state.allocationMessages, msg],
    })),

  setEditableAllocation: (alloc) => set({ editableAllocation: alloc }),

  setCashReserve: (cash) => set({ cashReserve: cash }),

  setPrediction: (json) => set({ predictionJson: json }),

  setRiskPlan: (json) => set({ riskPlanJson: json }),

  resetSession: () => set(initialState),

  restoreFromServer: (session) => {
    const inputs = (session.inputs ?? {}) as Record<string, unknown>;
    const conversations = (session.conversations ?? {}) as Record<string, ChatMessage[]>;
    const diagnosisConv = conversations.diagnosis ?? [];
    const allocationConv = conversations.allocation ?? [];

    // Extract last assistant message as the "response" text
    const diagnosisResponse = diagnosisConv.filter((m) => m.role === "assistant").pop()?.content ?? "";
    const allocationResponse = allocationConv.filter((m) => m.role === "assistant").pop()?.content ?? "";

    // favored_stocks could be enriched objects or simple {code, name}
    const rawFavored = (inputs.favored_stocks ?? []) as { code: string; name: string }[];
    const favored = rawFavored.map((s) => ({ code: s.code, name: s.name }));

    set({
      step: (session.current_step as number) ?? 1,
      totalCapital: (inputs.total_capital as number) ?? 500000,
      bullishSectors: (inputs.bullish_sectors as string[]) ?? [],
      favoredStocks: favored,
      confirmedCodes: (session.confirmed_stocks as string[]) ?? [],
      diagnosisResponse,
      diagnosisJson: (session.diagnosis_json as Record<string, unknown>) ?? null,
      diagnosisMessages: diagnosisConv,
      allocationResponse,
      allocationJson: null,
      allocationMessages: allocationConv,
      editableAllocation: (session.allocation as Allocation[]) ?? [],
      cashReserve: (session.cash_reserve as Record<string, unknown>) ?? {},
      predictionJson: (session.prediction as Record<string, unknown>) ?? null,
      riskPlanJson: (session.risk_plan as Record<string, unknown>) ?? null,
    });
  },
}));
