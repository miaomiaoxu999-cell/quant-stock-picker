"""Pydantic v2 schemas for stocks endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StockSummaryItem(BaseModel):
    sector: str = ""
    rank: Any = "-"
    name: str = ""
    code: str = ""
    price: Any = None
    pb: Any = None
    cycle_position: str = ""
    upside_pct: Any = None
    total_score: float = 0


class StocksSummaryResponse(BaseModel):
    stocks: list[StockSummaryItem] = []


class SectorsListResponse(BaseModel):
    sectors: list[str]


class AnalyzeRequest(BaseModel):
    top_n: int = Field(default=10, ge=5, le=20)


class RedoRequest(BaseModel):
    top_n: int = Field(default=10, ge=5, le=20)
    weights: dict[str, float] | None = None
    excluded_codes: list[str] = []


class SectorResultsResponse(BaseModel):
    sector: str = ""
    analyzed_at: str = ""
    cycle_position: str = ""
    top_n: int = 0
    stocks: list[dict[str, Any]] = []


class ChartDataResponse(BaseModel):
    code: str
    dates: list[str] = []
    pb: list[float | None] = []
    close: list[float | None] = []


class ExtremesResponse(BaseModel):
    code: str
    high_price: Any = "-"
    high_pb: Any = "-"
    high_date: str = "-"
    low_price: Any = "-"
    low_pb: Any = "-"
    low_date: str = "-"


class WatchlistItem(BaseModel):
    code: str
    name: str
    sector: str = ""
    note: str = ""
    added_at: str = ""


class WatchlistAddRequest(BaseModel):
    code: str
    name: str
    sector: str = ""
    note: str = ""


class LegacyProfilesResponse(BaseModel):
    profiles: dict[str, Any]
