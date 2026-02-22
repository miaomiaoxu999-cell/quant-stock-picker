"""Pydantic v2 schemas for advisor endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AdvisorSessionResponse(BaseModel):
    session: dict[str, Any] | None = None


class Step1SubmitRequest(BaseModel):
    total_capital: float
    bullish_sectors: list[str] = []
    favored_stock_codes: list[str] = []


class Step2ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []


class Step2ConfirmRequest(BaseModel):
    confirmed_codes: list[str]


class Step3ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []


class Step3ConfirmRequest(BaseModel):
    allocations: list[dict[str, Any]]
    cash_reserve: dict[str, Any] = {}


class Step5SaveRequest(BaseModel):
    sync_portfolio: bool = False


class AvailableStocksResponse(BaseModel):
    stocks: list[dict[str, Any]]
