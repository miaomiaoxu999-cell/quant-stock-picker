"""Pydantic v2 schemas for cycle analysis endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CycleOverall(BaseModel):
    cycle_position: str = ""
    reversal_probability: int = 0
    probability_timeframe: str = ""
    probability_rationale: str = ""
    summary: str = ""
    key_signals: list[str] = []


class CycleSectorResponse(BaseModel):
    sector: str
    overall: CycleOverall = CycleOverall()
    factors: list[dict[str, Any]] = []
    news: list[dict[str, Any]] = []
    conversation: list[dict[str, str]] = []
    analyzed_at: str | None = None
    archive_path: str | None = None


class AllCyclesResponse(BaseModel):
    sectors: dict[str, Any]


class CycleChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []


class CycleProgressResponse(BaseModel):
    status: str = "unknown"
    current_step: str = ""
    current_factor: str = ""
    factor_index: int = 0
    factor_total: int = 0
    log: list[str] = []
    error: str | None = None
